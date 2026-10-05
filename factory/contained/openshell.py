"""OpenShell integration — running the factory inside a policy-governed sandbox.

Everything that knows about OpenShell lives here: the Python SDK (lifecycle, exec, listing) and
the `openshell` CLI (file transfer, interactive attach — the SDK has neither). The module
**composes** and does not execute; the run path is `factory.cli.contained_openshell`.

Why this target exists when `local` and `k8s` already do: those bound accidents and give a run a
reproducible environment, and say so. This one additionally confines agent-authored code at the
kernel level — Landlock filesystem allowlists, seccomp on escape vectors, and deny-by-default
network through a supervisor proxy — which is what running the factory on untrusted input
(arbitrary issues, stranger's codebases) requires. The two consequences worth holding onto:

- **Credentials never enter the sandbox.** They attach as a gateway-held provider; the sandbox
  sees an opaque placeholder the supervisor resolves only for requests to the provider's
  endpoints. There is deliberately no env-var fallback — a fallback would make the insecure
  route the path of least resistance. What this does *not* mean: the key cannot be extracted,
  but **any descendant of `claude` can use it** — OpenShell binary rules match parent
  processes too, so every command the agent's Bash tool runs inherits claude's egress and the
  placeholder resolves for it. That residual capability is why the provider profile must drop
  `statsig.anthropic.com` and `sentry.io` (see `PROVIDER_FIX`) and why the allowlist stays
  small: the endpoints it names are reachable by exactly the untrusted code this target
  exists to confine.
- **The SDK is an optional dependency** (`contained-openshell` extra; grpc/protobuf do not
  belong in every install), so importing it raises a `ContainedError` carrying the fix rather
  than a bare `ImportError` at CLI startup.

Two shapes differ from the podman/k8s modules and deserve explanation up front.

**The main process is an idle command, not the run.** The provenance assertions have to run
after the workspace is in place and *before* the first agent call — the same ordering both other
targets enforce by launching the run after `podman exec`/`oc exec` probes. So the sandbox's
canonical main process is `sleep infinity` (it must outlive the run so a failed run stays
inspectable) and the run itself is started afterwards through the SDK's `exec()`.

**The run is a detached `nohup` process, not a tmux session.** The local and k8s targets hold
their runs in tmux; inside an OpenShell sandbox that is impossible — every tmux window needs a
pseudo-terminal, and PTY allocation (`/dev/ptmx`) is denied by the Landlock filesystem allowlist
(OpenShell issue #749, confirmed unfixed on `main`). The launch therefore writes the run's
output to a log file under the project's `.factory/`, records its pid, and records its exit code
when it ends; attach follows the log, liveness reads the pid. What tmux gave those targets for
free, the sandbox gives anyway: the run outlives the exec channel that started it (verified),
and the sandbox outlives the run by design. What is genuinely lost is a *live interactive*
terminal on the run — for a kernel-confined, unattended sandbox that reads-only interface is the
honest one. See `build_run_launch` for the exact properties.

**Dry-run cannot print argv for SDK calls.** The gateway API is gRPC, not a command line, so the
plan carries the SDK operations as *described* steps alongside the exact argv for the CLI parts.
`FACTORY_CONTAINED_DRY_RUN=1` prints that plan — the same ordered list the real path executes —
rather than a fabricated transcript. The CLI argv are still composed-only and printed verbatim,
so the parts that *are* commands keep the stronger guarantee.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from factory.contained.errors import ContainedError
from factory.contained.provenance import Probe
from factory.podman import LABEL_CONTAINED, LABEL_PROJECT, LABEL_SOURCE

# The OpenShell logical workspace sandboxes live in. The CLI targets `default` unless told
# otherwise, so a factory-created sandbox is visible to plain `openshell sandbox list` calls —
# the same discoverability rule the podman target gets by using podman's own default store.
SANDBOX_WORKSPACE = "default"

# Where the project lands inside the sandbox. The OpenShell docker/podman drivers derive the
# workspace root from the image's OCI WorkingDir (`resolve_oci_workspace_root`), and the
# factory-runtime image sets `/workspace` — so this matches the k8s target's constant rather
# than inventing a third location. A driver that ignored the image WorkingDir would fail the
# `workdir` provenance probe below loudly rather than extracting to the wrong place.
WORKSPACE_ROOT = "/workspace"

# What the sandbox's canonical main process runs. It has to outlive the factory (a failed run is
# exactly when the sandbox's state is worth reading) and die cleanly on stop; `sleep infinity`
# is what the local target's PID-1 payload runs for the same reasons.
IDLE_COMMAND: tuple[str, ...] = ("sleep", "infinity")

# The run's three on-disk artifacts, written by `build_run_launch` inside the project's
# `.factory/` so they ride the workspace copy home with `sync` — `.factory` is already the
# project-local, gitignored state dir. A sandbox cannot run tmux (see the module docstring), so
# these files *are* the run's interface: the log replaces scrollback, the pid names the process
# for anyone inspecting the sandbox, and the exit code file replaces the
# `[factory exited %s]` stamp — with the improvement that it is machine-readable.
RUN_LOG = ".factory/run.log"
RUN_PID = ".factory/run.pid"
RUN_EXIT = ".factory/run.exit"

# The provider profile the factory ships: the stock OpenShell profile with the two edits its
# own header asks for, made in `openshell_claude_code.yaml` next to this module instead of by
# hand on every machine. Two edits are mandatory, not cosmetic:
#
# - **Drop `statsig.anthropic.com` and `sentry.io`.** OpenShell binary rules match "the
#   executable that opens the connection *or any of its parent processes*", so every command
#   the agent's Bash tool runs — a descendant of `claude` — inherits this egress, and the
#   placeholder resolves for them too (profiles are endpoint-scoped, not yet binary-scoped).
#   sentry.io is a multi-tenant ingest service, which makes it an exfiltration channel if
#   untrusted code can reach it.
# - **Name the resolved claude path in the runtime image.** The kernel matches
#   `/proc/<pid>/exe`, which follows symlinks — `command -v claude` is a symlink to the
#   npm-packaged binary, so the stock profile's `/usr/local/bin/claude` never matches and
#   inference egress is silently denied.
#
# The file is the single source the verify check compares the gateway's profile against, so
# the shipped profile and the check cannot disagree.
PROVIDER_PROFILE_FILENAME = "openshell_claude_code.yaml"


def provider_profile_path() -> Path:
    """Where the shipped claude-code profile lives in *this* install — editable checkout or
    wheel alike. Printed in `PROVIDER_FIX` (the import command names a real file) and read
    by the verify check (its endpoints and binaries are the expectation)."""
    return Path(__file__).with_name(PROVIDER_PROFILE_FILENAME)


def load_provider_profile() -> dict:
    """The shipped profile as a mapping — the expectation the verify check holds the
    gateway's copy to. Parsed here rather than restated in code so the file is the one
    source of truth for both the import command and the check."""
    import yaml

    with open(provider_profile_path()) as f:
        return yaml.safe_load(f)


PROVIDER_FIX = (
    f"openshell profile import -f {provider_profile_path()} --global\n"
    "  openshell provider create --name claude-code --type claude-code --from-existing"
)
SDK_FIX = "uv sync --extra contained-openshell   # or: uv pip install openshell"

# The gateway enforces sandbox names itself and says so only as an INVALID_ARGUMENT at create
# time: at most 19 characters, lowercase ASCII alphanumeric and hyphens. 19 is tight, so the
# readable stem gives up most of podman's 32-char budget and the hash suffix is what keeps two
# same-named projects apart — it is never the part that is truncated.
MAX_SANDBOX_NAME = 19

# Stated once, as a set, so the generator and the validator cannot disagree about the alphabet —
# an earlier `c.islower() and c.isalnum() or c == "-"` formulation disagreed with itself through
# operator precedence and rejected digits, and `str.isalnum` would have admitted unicode letters
# the gateway refuses. ASCII, explicit, shared.
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def sandbox_name(project_path: Path) -> str:
    """A `container_name` sibling that fits the gateway's sandbox-name rules.

    Same shape (slugified stem + project-hash suffix) as `factory.podman.container_name`, with a
    tighter budget: underscores, dots, and everything outside the ASCII alphabet are not legal,
    so they all collapse to hyphens.
    """
    from factory.podman import project_hash

    digest = project_hash(project_path)[:6]
    stem = "".join(c if c in _NAME_CHARS else "-" for c in project_path.name.lower()).strip("-")
    stem = stem[: MAX_SANDBOX_NAME - 7].strip("-") or "factory"
    return f"{stem}-{digest}"


def validate_sandbox_name(name: str) -> str:
    """Reject a user-supplied `--name` the gateway would reject, naming the rule.

    Validating here turns a create-time INVALID_ARGUMENT (a gRPC traceback about a sandbox that
    never existed) into a parse-time error the user can fix before any workspace copy is made.
    """
    if not name:
        return name
    if len(name) > MAX_SANDBOX_NAME:
        raise ContainedError(
            f"--name {name!r} is {len(name)} characters; a sandbox name is at most "
            f"{MAX_SANDBOX_NAME} (lowercase alphanumeric and hyphens)"
        )
    illegal = {c for c in name if c not in _NAME_CHARS}
    if illegal:
        raise ContainedError(
            f"--name {name!r} contains {sorted(illegal)!r}; a sandbox name is lowercase "
            "alphanumeric and hyphens only"
        )
    return name

# Exit code `exec` reports when the command could not run at all.
_EXEC_LAUNCH_FAILURE = -1


def import_sdk():
    """Import the OpenShell SDK, raising a fix-carrying error when the extra is absent.

    The SDK is optional (grpc/protobuf/cloudpickle in every install is too heavy for a target
    most users never touch), so a missing import is a *prerequisite* to report, not a crash.
    Deferred to call time rather than module import so `factory contained --help` works on a
    machine that has never installed the extra.
    """
    try:
        import openshell  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ContainedError(
            "the openshell Python SDK is not installed; `factory contained --target openshell` "
            f"needs it.\n  {SDK_FIX}"
        ) from exc
    return openshell


def import_protos():
    """Import the generated protobuf modules the SDK's spec/policy messages live in."""
    try:
        from openshell._proto import (  # type: ignore[import-not-found]
            openshell_pb2,
            sandbox_pb2,
        )
    except ImportError as exc:
        raise ContainedError(
            "the openshell Python SDK is not installed; `factory contained --target openshell` "
            f"needs it.\n  {SDK_FIX}"
        ) from exc
    return openshell_pb2, sandbox_pb2


def connect(gateway: str | None = None):
    """Build a `SandboxClient` from the CLI's registered gateway state.

    `from_active_cluster` reads exactly what `openshell gateway select` wrote
    (`$OPENSHELL_GATEWAY` or `~/.config/openshell/active_gateway`), so the factory and the CLI
    always agree on which gateway is in play without the factory growing its own gateway
    configuration. Never called in dry-run: it opens a connection, and a promise to provision
    nothing is not kept by a network round trip.
    """
    sdk = import_sdk()
    try:
        return sdk.SandboxClient.from_active_cluster(cluster=gateway)
    except sdk.SandboxError as exc:
        raise ContainedError(
            f"cannot connect to an OpenShell gateway: {exc}\n"
            "  Register and select one first:\n"
            "    openshell gateway add --name local --local\n"
            "    openshell gateway select local"
        ) from exc


# --- plan ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenShellPlan:
    """Everything needed to provision one sandbox and start the run in it, in order.

    The environment here is configuration only — never credentials. The provider carries
    inference credentials, and a secret-looking key in this dict is a bug, not an escape hatch
    (the run path warns about `--forward`/`--env` the way k8s does, more loudly).
    """

    name: str
    image: str
    project_dir: str
    env: dict[str, str]
    labels: dict[str, str]
    provider: str
    policy_path: Path | None
    run_command: str
    factory_command: str
    # Which registered gateway every gateway-bound operation — SDK *and* CLI — must use. The
    # CLI composers take it too: without it, an upload with `--gateway X` while `Y` is active
    # lands in a same-named sandbox on Y (the name is deterministic per project), which is
    # worse than a clean failure.
    gateway: str | None = None
    warnings: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Step:
    """One provisioning action, named so a failure can say which stage broke.

    SDK operations have no argv to print, so a step carries either a description (executed
    through the SDK by the run path) or exact argv (executed as a subprocess). Dry-run prints
    the same list the real path walks, in the same order.
    """

    name: str
    description: str | None = None
    argv: list[str] | None = None

    def render(self) -> str:
        import shlex

        if self.argv is not None:
            return shlex.join(self.argv)
        return self.description or self.name


def plan_steps(plan: OpenShellPlan, tarball: Path, probes: list[Probe]) -> list[Step]:
    """The full provisioning sequence as ordered, named steps.

    Mirrors `podman.plan_steps`: create, assert each provenance probe, run. The upload and
    extract steps sit between create and the probes because the assertions read the workspace
    (that is the point of them); the workdir probe comes first among the asserts so a driver
    that ignored the image's WorkingDir fails before anything reads `/workspace`.
    """
    steps = [
        Step("create", description=f"sdk: create sandbox {plan.name} from {plan.image}"),
        Step("upload", argv=build_upload_argv(plan.name, tarball)),
        Step(
            "extract",
            description=f"sdk: exec tar xzf {tarball.name} -C {WORKSPACE_ROOT}",
        ),
        Step("assert:workdir", description=f"sdk: exec pwd (expect {WORKSPACE_ROOT})"),
    ]
    for probe in probes:
        steps.append(
            Step(
                f"assert:{probe.name}",
                description=f"sdk: exec {probe.argv!r} in {plan.project_dir}",
            )
        )
    steps.append(Step("run", description=f"sdk: exec detached run launch in {plan.project_dir}"))
    return steps


# --- policy -------------------------------------------------------------------------


def build_default_policy():
    """The base sandbox policy: the security core of this target.

    Rules, in the order a reader should weigh them:

    - The workspace (`/workspace`, which `include_workdir` also covers) is read-write; OpenShell
      adds its baseline read-only system paths (`/usr`, `/etc`, ...) on top, so the toolchain
      works without this policy granting system reads itself. **One path must be granted here
      anyway**: the runtime image installs the factory under `/opt/factory`, which is outside
      every baseline list, so without this entry the very first `factory` invocation dies with
      a `PermissionError` reading its own venv — Landlock denies the read no matter what the
      file mode says.
    - Egress is deny-by-default and the allowlist is deliberately small: read-only package
      registries for `uv` so eval environments can be built inside the sandbox. `pip` is
      deliberately absent — binary matching resolves `/proc/<pid>/exe`, and pip runs as the
      Python interpreter, so a `pip` path never matches (the docs say it directly: "Scripts
      run as their interpreter, so list the interpreter"); granting the *interpreter* PyPI
      access would grant it to every script the agent writes. `uv` is a native binary, so it
      matches. The defaults start slightly loose on purpose: loosening is a compatible
      change, silently tightening breaks running workflows. Every addition to this allowlist
      is a code review, the same as any other security-sensitive default.
    - **Inference egress is deliberately *absent*.** Attaching the `claude-code` provider (which
      `build_spec` always does) makes the gateway synthesize its own `_provider_claude_code`
      policy covering the provider profile's endpoints for the provider's declared binaries —
      restating those endpoints here is not redundancy but a hard create-time failure: the
      gateway's ambiguity validation rejects two rules for the same endpoint whose metadata
      differs (`transparent_tcp_eligible`), and the synthesized rule cannot be matched
      field-for-field from a user policy. The provider profile is the shipped one
      (`openshell_claude_code.yaml`): api.anthropic.com for the resolved claude binary, and
      *nothing else* — which matters because binary rules match parent processes, so those
      endpoints are reachable by every command the agent's Bash tool runs.
    - Process identity is stated explicitly — `run_as_user`/`run_as_group` 1001 — because an
      omitted identity falls back to the image's OCI `USER` (1001 with primary GID 0), and the
      gateway hard-rejects any workload identity containing GID 0. The runtime image's
      arbitrary-UID recipe (`chgrp 0` + `chmod g=u`, plus `o=u` on the container home and
      `/workspace` only) keeps the paths a sandboxed run writes open to this gid, while
      `/opt/factory` — read-only here — needs and gets no world-widening.

    `--policy` replaces this policy *entirely* — never merges (no negation semantics to
    maintain, WYSIWYG, and OpenShell already layers provider rules on top of the base).
    """
    _, sandbox_pb2 = import_protos()

    endpoints = sandbox_pb2.NetworkEndpoint
    binaries = sandbox_pb2.NetworkBinary

    def _rule(name: str, hosts: list[str], binary_paths: list[str]):
        return sandbox_pb2.NetworkPolicyRule(
            name=name,
            endpoints=[
                endpoints(host=host, port=443, protocol="tcp") for host in hosts
            ],
            binaries=[binaries(path=path) for path in binary_paths],
        )

    return sandbox_pb2.SandboxPolicy(
        version=1,
        filesystem=sandbox_pb2.FilesystemPolicy(
            include_workdir=True,
            # The runtime image's factory install — see the docstring's `/opt/factory` note.
            read_only=["/opt/factory"],
        ),
        process=sandbox_pb2.ProcessPolicy(run_as_user="1001", run_as_group="1001"),
        network_policies={
            "python_packages": _rule(
                "python_packages",
                ["pypi.org", "files.pythonhosted.org"],
                ["/usr/local/bin/uv", "/usr/bin/uv"],
            ),
        },
    )


# The documented YAML policy schema names this field `filesystem_policy`; the proto message
# calls it `filesystem`. This is the one rename between the format users write (what
# `openshell sandbox create --policy` accepts) and the proto the SDK takes.
_YAML_TO_PROTO_KEYS = {"filesystem_policy": "filesystem"}


def load_policy(path: Path):
    """Load a full-replacement policy from a YAML file in OpenShell's documented schema.

    The file is the whole policy — the default is not merged in (see `build_default_policy`).
    A user who needs the default plus one change copies the default and edits it (the default
    is documented in `docs/contained/`); the file they pass is everything they get, and an
    unknown key is an error rather than a silent drop — WYSIWYG cuts both ways.
    """
    import yaml

    _, sandbox_pb2 = import_protos()
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ContainedError(f"cannot read policy file {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContainedError(
            f"policy file {path} must be a YAML mapping (an OpenShell sandbox policy)"
        )
    translated = {_YAML_TO_PROTO_KEYS.get(key, key): value for key, value in document.items()}
    from google.protobuf import json_format  # type: ignore[import-untyped]

    policy = sandbox_pb2.SandboxPolicy()
    try:
        json_format.ParseDict(translated, policy)
    except json_format.ParseError as exc:
        raise ContainedError(
            f"policy file {path} is not a valid sandbox policy: {exc}"
        ) from exc
    return policy


def build_spec(plan: OpenShellPlan, policy) -> "object":
    """Compose the `SandboxSpec` the sandbox is created from.

    The command is the idle payload — see the module docstring for why the run is launched
    after the provenance probes rather than as the main process. Providers are attached at
    create time so the placeholder environment exists before any process could ask for it.
    """
    openshell_pb2, _ = import_protos()
    return openshell_pb2.SandboxSpec(
        environment=dict(plan.env),
        template=openshell_pb2.SandboxTemplate(image=plan.image),
        policy=policy,
        providers=[plan.provider],
        command=list(IDLE_COMMAND),
    )


# --- CLI argv (composed, never executed here) ----------------------------------------


def _cli(gateway: str | None, *rest: str) -> list[str]:
    """One `openshell` invocation, pinned to the gateway the user selected.

    `-g/--gateway` is a CLI-global flag. It is *omitted*, not passed empty, when no gateway
    was named — the flag with no value is an error, and the CLI's default (the active gateway,
    `$OPENSHELL_GATEWAY` or `~/.config/openshell/active_gateway`) is what an unnamed run
    wants. The SDK's `connect()` reads the same state, so the two transports stay in
    agreement either way.
    """
    return ["openshell", *(["--gateway", gateway] if gateway else []), *rest]


def build_upload_argv(name: str, tarball: Path, gateway: str | None = None) -> list[str]:
    """Compose the workspace upload. The tarball is a single file, so `.gitignore` filtering
    (which the CLI applies to directory uploads) has nothing to bite on — and the copy must
    carry gitignored state like `.factory/` the way the k8s target's tarball does."""
    return _cli(gateway, "sandbox", "upload", name, str(tarball))


def build_download_argv(
    name: str, sandbox_path: str, dest: Path, gateway: str | None = None
) -> list[str]:
    return _cli(gateway, "sandbox", "download", name, sandbox_path, str(dest))


def build_provider_list_argv(gateway: str | None = None) -> list[str]:
    """Compose the provider listing the provider check reads (shape only, never material)."""
    return _cli(gateway, "provider", "list", "--output", "json")


def build_profile_export_argv(profile_id: str, gateway: str | None = None) -> list[str]:
    """Compose the profile readback the provider check compares against the shipped profile.

    JSON rather than the human `profile describe` rendering, so the check parses a stable
    shape instead of scraping prose; the export carries endpoints and binaries, never
    credential material.
    """
    return _cli(gateway, "profile", "export", profile_id, "--output", "json")


def build_attach_argv(name: str, gateway: str | None = None) -> list[str]:
    """Compose the interactive attach.

    The SDK has no PTY, so the CLI's `exec --tty` is the transport — the supervisor allocates
    that outer PTY before the Landlock boundary applies, so it works where a *nested* one
    (tmux) cannot. The view is read-only: a finished run's exit stamp prints first (a log that
    is still growing tells its own story live), then the log is followed. Detaching is Ctrl-C —
    safe, because following a file cannot disturb the process writing it. The `sh -i` fallback
    covers a log that does not exist (nothing started) the way the tmux attach's fallback
    covered a missing session.
    """
    log, exit_code = shlex.quote(RUN_LOG), shlex.quote(RUN_EXIT)
    return _cli(
        gateway,
        "sandbox", "exec", "-n", name, "--tty", "--",
        "sh", "-lc",
        f"if [ -f {log} ]; then "
        f"[ -f {exit_code} ] && echo \"[factory exited $(cat {exit_code})]\"; "
        f"tail -n 200 -f {log}; "
        f"else exec sh -i; fi",
    )


def build_cli_binary_check_argv() -> list[str]:
    return ["openshell", "--version"]


# --- execution (the only place that touches the gateway) ------------------------------


def exec_argv(session, argv: list[str], *, workdir: str | None = None, timeout: int = 300):
    """Run one command in the sandbox through the SDK, mapping failures to `ContainedError`.

    A command that cannot run at all reports exit code -1, which is distinct from a command
    that ran and failed — the caller's message should not claim the probe ran.
    """
    try:
        result = session.exec(list(argv), workdir=workdir, timeout_seconds=timeout)
    except Exception as exc:  # SandboxError and grpc errors both arrive as plain exceptions
        raise ContainedError(f"exec in sandbox failed: {exc}") from exc
    if result.exit_code == _EXEC_LAUNCH_FAILURE:
        raise ContainedError(
            f"could not run {argv[0]!r} in the sandbox (it did not execute at all)"
        )
    return result


def run_probes(session, plan: OpenShellPlan, probes: list[Probe]) -> None:
    """Run the provenance assertions, aborting with the probe's own hint on failure.

    The probes are the same argv lists the other targets run (`factory.contained.provenance`);
    only the transport differs. The workdir assertion runs first so a driver that ignored the
    image's WorkingDir fails before a probe misreads `/workspace` as the project being absent.
    """
    result = exec_argv(session, ["pwd"], timeout=30)
    actual = result.stdout.strip()
    if result.exit_code != 0 or actual != WORKSPACE_ROOT:
        raise ContainedError(
            f"the sandbox's working directory is {actual or 'unknown'}, expected "
            f"{WORKSPACE_ROOT}. The openshell target derives it from the runtime image's "
            f"WORKDIR; a compute driver that ignores it is not supported. "
            f"Override the image with --image or FACTORY_CONTAINED_IMAGE."
        )
    for probe in probes:
        result = exec_argv(session, probe.argv, workdir=plan.project_dir)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            message = detail[0][:200] if detail else "no output"
            raise ContainedError(
                f"assertion {probe.name} failed in the sandbox: {message}\n  {probe.hint}"
            )


def build_run_launch(workdir: str, run_command: str) -> str:
    """Compose the shell script that starts the run as a detached, logged process.

    This is the openshell target's `build_tmux_launch`: same job (a run that outlives the exec
    channel that started it, whose output survives it, and whose exit is observable), no
    tmux — a sandbox cannot allocate the PTY every tmux window needs (module docstring,
    OpenShell #749). Property by property:

    - `rm -f` of the three run artifacts comes first: a fresh sandbox's workspace can carry
      them from an earlier run's `sync` (the upload packs the workspace copy, which the
      download wrote the previous run's artifacts into), and a stale `run.exit` makes `attach`
      print a phantom `[factory exited N]` before the new run has done anything.
    - `nohup … &` detaches, so the exec returns as soon as the run is started, and the run
      survives the exec session ending (verified against a live sandbox).
    - `> run.log 2>&1` is the scrollback. It lives under the project's `.factory/`, so `sync`
      downloads it with the workspace — a log the failed run wrote is *stronger* post-mortem
      state than a tmux pane, because it exists on the host too.
    - `echo $! > run.pid` records the run's pid.
    - `echo $? > run.exit` replaces tmux's `[factory exited %s]` stamp, machine-readably. A
      command that never *returns* (killed, `exec`ed away) writes no exit file — only its
      exit code is unknown, which is also the truth.
    """
    inner = f"{run_command}; echo $? > {RUN_EXIT}"
    log, pid = shlex.quote(RUN_LOG), shlex.quote(RUN_PID)
    stale = " ".join(shlex.quote(artifact) for artifact in (RUN_LOG, RUN_PID, RUN_EXIT))
    # The brace group is load-bearing: without it `cd … && nohup … & echo $! > pid` parses as
    # `(cd … && nohup …) & echo …` — the `&` backgrounds the whole `cd` chain, so the pidfile
    # lands in the *launcher's* cwd (nowhere, usually) rather than beside the log.
    return (
        f"cd {shlex.quote(workdir)} && mkdir -p .factory && rm -f {stale} && "
        f"{{ nohup sh -c {shlex.quote(inner)} > {log} 2>&1 & echo $! > {pid}; }}"
    )


def start_run(session, plan: OpenShellPlan) -> None:
    """Start the run as a detached process, per `build_run_launch`.

    The exec this issues returns immediately — the `&` in the script means the SDK session is
    only waiting for the shell that *started* the run, not the run itself.
    """
    script = build_run_launch(plan.project_dir, plan.run_command)
    exec_argv(session, ["sh", "-lc", script], timeout=60)


def extract_tarball(session, tarball_name: str) -> None:
    """Unpack the uploaded workspace tarball into the sandbox's workspace root.

    The tarball is packed as `<project>/...` (the k8s `_pack` convention), so it unpacks to
    `{WORKSPACE_ROOT}/<project>` — the path the plan, the rewritten payload and the probes
    already agree on.
    """
    exec_argv(
        session,
        ["tar", "xzf", tarball_name, "-C", WORKSPACE_ROOT],
        timeout=600,
    )


def create_sandbox(client, plan: OpenShellPlan, policy, tarball: Path) -> "object":
    """Create the sandbox and wait until it is ready. Returns the `SandboxSession`.

    Raises `ContainedError` on any gateway refusal (name collisions, provider unknown, image
    unpullable) rather than letting SDK exceptions escape as tracebacks.
    """
    spec = build_spec(plan, policy)
    try:
        client.create(
            workspace=SANDBOX_WORKSPACE,
            spec=spec,
            name=plan.name,
            labels=plan.labels,
        )
        client.wait_ready(name=plan.name, workspace=SANDBOX_WORKSPACE)
        return client.get_session(plan.name, workspace=SANDBOX_WORKSPACE)
    except ContainedError:
        raise
    except Exception as exc:
        raise ContainedError(f"creating sandbox {plan.name} failed: {exc}") from exc


def _phase_name(phase: int) -> str:
    """Map the proto phase number onto the word `ls` prints for the other targets."""
    return {
        0: "unknown",
        1: "provisioning",
        2: "running",       # READY: the idle main process is up; the run's own state is in
                            # `.factory/run.{log,pid,exit}` — `attach` shows it
        3: "error",
        4: "deleting",
        5: "unknown",
        6: "stopping",
        7: "stopped",
        8: "starting",
        9: "completed",
    }.get(phase, "unknown")


def list_runtimes(gateway: str | None = None) -> list[dict[str, object]]:
    """Every sandbox the factory created, running or not.

    Selection is the factory's own label — the same rule the podman target applies — so a tool
    that lists resources it did not create never invites the user to assume it manages them.
    """
    client = connect(gateway)
    try:
        pager = client.list(
            workspace=SANDBOX_WORKSPACE,
            label_selector=f"{LABEL_CONTAINED}=true",
        )
        entries = []
        for ref in pager.all():
            labels = dict(ref.labels or {})
            entries.append(
                {
                    "name": ref.name,
                    "phase": _phase_name(ref.phase),
                    "project": labels.get(LABEL_PROJECT, ""),
                    "source": labels.get(LABEL_SOURCE, "") or None,
                    "created": None,  # not carried on SandboxRef; `ls` renders `?`
                    "exit_code": ref.status.exit_code if ref.status else None,
                }
            )
        return entries
    except ContainedError:
        raise
    except Exception as exc:
        raise ContainedError(f"listing sandboxes failed: {exc}") from exc


def remove_runtime(name: str, gateway: str | None = None) -> None:
    """Delete one factory-created sandbox, waiting until the deletion completes."""
    client = connect(gateway)
    try:
        client.delete(name=name, workspace=SANDBOX_WORKSPACE, allow_missing=True)
        client.wait_deleted(name=name, workspace=SANDBOX_WORKSPACE)
    except ContainedError:
        raise
    except Exception as exc:
        raise ContainedError(f"deleting sandbox {name} failed: {exc}") from exc


def cli_available() -> bool:
    """Whether the `openshell` CLI is on PATH (a prerequisite check, so never raises)."""
    import shutil

    return shutil.which("openshell") is not None


def run_cli(argv: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run one `openshell` CLI command. Thin on purpose: composers live above, error wording
    lives in the caller, and this exists so tests have one seam to patch."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
