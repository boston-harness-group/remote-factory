"""Getting a machine (or a namespace) ready to run `factory contained`, in one pass.

`verify` reports; `setup` fixes. It ends in exactly one of two states — everything green with a
runnable command printed, or the full list of what is still missing with the command for each.
Never in between.

Two properties make it safe to run at any time:

- **Idempotent.** Re-running changes nothing that is already correct, so it is also the supported
  way to repair a partial setup and nothing needs to be torn down first.
- **Nothing silent.** Every step announces what it will do before doing it, and steps that touch
  credentials or a cluster ask first.

What is automated locally is deliberately narrow: starting a stopped podman machine and pulling the
runtime image. Inference is never automated — it is the one step that touches credential material,
so it is described and left to the user.
"""

from __future__ import annotations

import subprocess
import sys

import structlog

from factory.contained import style
from factory.contained.prereq import local_checks, render_checks
from factory.podman import build_pull_argv, resolve_image

log = structlog.get_logger()

# The local half has three steps, and it says so up front. A wizard that prints an unlabelled wall
# of lines gives the reader no way to tell "still working" from "finished" — numbering each step is
# what turns the same information into progress.
_LOCAL_STEPS = 3


def run_setup(
    target: str | None,
    *,
    interactive: bool,
    namespace: str | None = None,
    division: bool = False,
    assume_yes: bool = False,
    gateway: str | None = None,
) -> int:
    """Run setup for one target, or ask which when not told."""
    if target is None and interactive:
        target = _ask_target()

    from factory.contained.usage import record_target

    code = 0
    if target in (None, "local", "both"):
        record_target("local")
        if target == "both":
            print(style.section("Local runtime"))
        _setup_local()
        print(style.section("Result", step=_LOCAL_STEPS, total=_LOCAL_STEPS))
        checks = local_checks()
        print(render_checks(checks, setup_command=None))
        code = 0 if all(c.ok for c in checks) else 1

    if target in ("k8s", "both"):
        record_target("k8s")
        if target == "both":
            print(style.section("Cluster runtime"))
        from factory.contained.k8s_setup import setup_k8s

        k8s_code = setup_k8s(
            namespace=namespace,
            division=division,
            interactive=interactive,
            assume_yes=assume_yes,
        )
        code = code or k8s_code

    if target == "openshell":
        record_target("openshell")
        _setup_openshell(gateway)
        from factory.contained.openshell_prereq import openshell_checks

        checks = openshell_checks(gateway)
        print(render_checks(checks, setup_command=None))
        code = code or (0 if all(c.ok for c in checks) else 1)

    return code


def _ask_target() -> str:
    print(style.section("What are you setting up?"))
    print(style.note("Pass --target local, k8s or openshell to skip this question."))
    print()
    print(
        f"  {style.bold('1')}) {style.paint('local', 'cyan')}  a podman container on this machine"
    )
    print(f"  {style.bold('2')}) {style.paint('k8s', 'cyan')}    a pod on a cluster")
    print(f"  {style.bold('3')}) {style.paint('both', 'cyan')}  local and cluster")
    print(
        f"  {style.bold('4')}) {style.paint('openshell', 'cyan')}  a policy-governed sandbox, for "
        "runs whose input you do not trust"
    )
    print()
    try:
        choice = input(style.prompt("Choice", "1")).strip() or "1"
    except EOFError:
        # stdin closed before an answer arrived — a pipe, a CI job, or `< /dev/null`. The default
        # is the documented one; an unanswered prompt must not become a bare `Error:`.
        print("\nNo answer given; setting up the local runtime (the default).")
        return "local"
    # openshell is 4, not 3, so `both` keeps the number it has always had — anything scripted
    # against this menu (an answer piped in, a documented `echo 3`) keeps working.
    return {"1": "local", "2": "k8s", "3": "both", "4": "openshell"}.get(choice, "local")


def _setup_openshell(gateway: str | None = None) -> None:
    """Describe the openshell-target steps; automate none of them.

    Unlike the local half (which starts a stopped podman machine and pulls an image), every
    openshell prerequisite is either an install (the CLI, the SDK extra) or a credential-adjacent
    act (registering a gateway, creating the provider). The factory automates neither: an
    install script is the user's to choose, and the provider step is the one place this target
    touches credential material, so it is described and left to the user.
    """
    from factory.contained.openshell import PROVIDER_FIX, SDK_FIX
    from factory.contained.openshell_prereq import openshell_checks

    print(style.section("OpenShell runtime"))
    checks = {c.name: c for c in openshell_checks(gateway)}
    cli = checks.get("openshell_cli")
    if cli is not None and not cli.ok:
        print(style.line("Install the OpenShell CLI (one line, from the docs):"))
        print(style.line(f"  {cli.fix}"))
    else:
        print(style.note("openshell CLI present; nothing to do."))
    sdk = checks.get("openshell_sdk")
    if sdk is not None and not sdk.ok:
        print(style.line("Install the Python SDK into this environment:"))
        print(style.line(f"  {SDK_FIX}"))
    else:
        print(style.note("openshell SDK importable; nothing to do."))
    print(style.line("Register a local gateway, if you have not:"))
    print(style.line("  openshell gateway add --name local --local"))
    print(style.line("  openshell gateway select local"))
    # The one trap that turns a healthy gateway into `ControlSupervisorStartFailed` for every
    # sandbox: on macOS/WSL the supervisor lives in the Docker Desktop VM's host network and
    # cannot reach a gateway bound to 127.0.0.1. Worth surfacing here because the failure
    # names neither the gateway nor the VM.
    print(style.note("macOS/WSL with Docker Desktop: the sandbox supervisor cannot reach a"))
    print(style.note("gateway bound to 127.0.0.1. If sandboxes die at startup, point the"))
    print(style.note("docker driver at the host instead, in ~/.config/openshell/gateway.toml:"))
    print(style.line('  [openshell.drivers.docker]'))
    print(style.line('  grpc_endpoint = "https://host.docker.internal:17670"'))
    print(style.note("Inference is never automated — it touches credential material:"))
    print(style.line(f"  {PROVIDER_FIX}"))


def _setup_local() -> None:
    """Perform the local steps that are safe to automate; describe the ones that are not.

    Every branch announces before acting. The trailing `local_checks()`/`render_checks()` in
    `run_setup` is what reports the outcome, including for the cases handled here — so nothing in
    this function needs its own second, weaker copy of a check's message.
    """
    print(style.section("Container engine", step=1, total=_LOCAL_STEPS))
    engine = next((c for c in local_checks() if c.name == "container_engine"), None)
    if engine is not None and not engine.ok:
        _start_machine()
    else:
        print(style.note("podman is reachable; nothing to do."))

    print(style.section("Runtime image", step=2, total=_LOCAL_STEPS))
    image = resolve_image()
    if _image_present(image):
        print(style.line(style.dim(f"Image already present: {image}")))
    else:
        print(style.line(f"Pulling {style.value(image)}"))
        print(style.note("This takes a few minutes on a cold cache."))
        result = subprocess.run(build_pull_argv(image))
        if result.returncode != 0:
            print(
                f"\nCould not pull {image}.\n"
                "That usually means the image is not published yet, or the registry needs a login "
                "(`podman login ghcr.io`).\n"
                "\n"
                "Either way you have two options:\n"
                "  1. Use an image you already have:\n"
                "       export FACTORY_CONTAINED_IMAGE=<your-image-reference>\n"
                "  2. Build one from a checkout of this repository:\n"
                "       git clone https://github.com/akashgit/remote-factory\n"
                "       cd remote-factory\n"
                f"       podman build -f containers/factory/Containerfile -t {image} .\n"
                "     (the Containerfile ships in the git repository, not in the installed "
                "package)",
                file=sys.stderr,
            )


def _image_present(reference: str) -> bool:
    from factory.podman import build_image_exists_argv

    try:
        return (
            subprocess.run(build_image_exists_argv(reference), capture_output=True).returncode == 0
        )
    except (FileNotFoundError, PermissionError, OSError):
        return False


def _start_machine() -> None:
    """Start a stopped podman machine, announcing first.

    Automated because it mutates nothing durable and because on macOS it is the single most common
    reason a contained run fails — the machine stops quietly and every later error blames podman.
    """
    try:
        listed = subprocess.run(
            ["podman", "machine", "list", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError):
        return
    if listed.returncode != 0 or not listed.stdout.strip():
        # `line`, not `note`: this carries a command, and a wrapped command cannot be copied.
        print(style.line("No podman machine found. Create one with: podman machine init"))
        return
    print(style.note("The podman engine is not reachable. Starting the podman machine..."))
    subprocess.run(["podman", "machine", "start"])
