"""`ls`, `attach`, `rm`, `sync` — over runtimes the factory created, and only those.

A tool that lists resources it did not create invites the user to assume it manages them too, so
every subcommand here filters on the factory's own label and refuses a name that does not carry it.
`resolve_runtime` returning `None` is the enforcement point: `attach`, `remove`, and `sync` all go
through it before touching anything.

`ls` is the one command that spans both targets — one table, local and cluster together — because a
user asking "what is running?" does not want to ask it twice. Everything else acts on a single
named runtime and takes its target from `--target`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

from factory.contained.errors import ContainedError
from factory.contained.runtimes import LifecycleError, Runtime
from factory.contained.workspace import (
    Workspace,
    cleanup_hint,
    contained_home,
    merge_hint,
)
from factory.podman import (
    LABEL_CONTAINED,
    LABEL_PROJECT,
    LABEL_SOURCE,
    build_attach_argv,
    build_pane_liveness_argv,
    build_ps_argv,
    build_rm_argv,
)

log = structlog.get_logger()


def _podman_entries() -> list[dict[str, object]]:
    try:
        result = subprocess.run(build_ps_argv(), capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise LifecycleError(
            "`podman` is not installed or not on PATH. Install it and retry, or run "
            "`factory contained verify` for the full list of prerequisites."
        ) from exc
    if result.returncode != 0:
        raise LifecycleError(f"cannot reach podman ({_first_line(result.stderr)})")
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise LifecycleError(
            f"`podman ps` returned output that isn't JSON: {result.stdout.strip()[:200]!r}"
        ) from exc
    return payload if isinstance(payload, list) else []


def _first_line(stderr: str) -> str:
    """The first meaningful line of a CLI error. podman's connection failure runs to five."""
    for line in (stderr or "").splitlines():
        text = line.strip()
        if text:
            return text.removeprefix("Error: ")[:140]
    return "no details given"


def _labels_of(entry: dict[str, object]) -> dict[str, object]:
    raw = entry.get("Labels")
    return raw if isinstance(raw, dict) else {}


def _name_of(entry: dict[str, object]) -> str:
    names = entry.get("Names")
    if isinstance(names, list) and names:
        return str(names[0])
    return str(entry.get("Name", ""))


def _created_of(entry: dict[str, object]) -> datetime | None:
    """`podman ps --format json` reports Created as a unix timestamp in this podman line.

    Older builds emit an RFC-3339 string under the same key, so both are accepted and anything
    unparseable degrades to `None` (rendered as `?`) rather than raising inside a listing.
    """
    created = entry.get("Created")
    if isinstance(created, (int, float)):
        return datetime.fromtimestamp(created, tz=timezone.utc)
    if isinstance(created, str) and created.strip():
        try:
            return datetime.fromisoformat(created.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def local_runtimes() -> list[Runtime]:
    """Every container the factory created on this machine, running or not.

    `build_ps_argv` already selects on the factory's own label, so the label check below is a
    second, independent filter site rather than the only one.
    """
    runtimes = []
    for entry in _podman_entries():
        labels = _labels_of(entry)
        if str(labels.get(LABEL_CONTAINED, "")).lower() != "true":
            continue
        name = _name_of(entry)
        container_state = str(entry.get("State", "unknown"))
        runtimes.append(
            Runtime(
                name=name,
                target="local",
                project=str(labels.get(LABEL_PROJECT, "")),
                state=_run_state(name, container_state),
                created=_created_of(entry),
                source=str(labels.get(LABEL_SOURCE, "")) or None,
            )
        )
    return runtimes


def _run_state(name: str, container_state: str) -> str:
    """What the *run* is doing, which is not the same as what the container is doing.

    The container's PID 1 outlives the run on purpose, so a container whose run has finished still
    reports `running` — which is why a user is told a run is live and then finds nothing to attach
    to. When the container is up, the session is what says whether the run is.
    """
    if container_state.strip().lower() != "running":
        return container_state
    try:
        result = subprocess.run(
            build_pane_liveness_argv(name), capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return container_state
    if result.returncode != 0:
        return "finished"                       # no session left at all
    # `0` marks a pane whose process is still alive. All-dead means the run is over even though the
    # session is deliberately still there for its output.
    return "running" if "0" in result.stdout.split() else "finished"


def openshell_runtimes(gateway: str | None = None) -> list[Runtime]:
    """Every OpenShell sandbox the factory created, running or not.

    Selection is the factory's own label — OpenShell's list RPC takes a `label_selector`, so
    discovery works exactly like the podman target's rather than by name-prefix matching.
    """
    from factory.contained import openshell as os_mod

    try:
        entries = os_mod.list_runtimes(gateway)
    except ContainedError as exc:
        raise LifecycleError(str(exc)) from exc
    return [
        Runtime(
            name=str(entry["name"]),
            target="openshell",
            project=str(entry["project"]),
            # The sandbox's phase says the *sandbox* is up; whether the *run* is still going
            # is a different question (the sandbox deliberately outlives its run), answered
            # per-runtime only when something asks, because it costs an exec round trip.
            state=str(entry["phase"]),
            created=None,
            source=str(entry["source"]) if entry.get("source") else None,
        )
        for entry in entries
    ]


def list_runtimes(
    target: str | None = None, namespace: str | None = None, gateway: str | None = None
) -> tuple[list[Runtime], list[str], list[str]]:
    """Runtimes for one target, or all when `target` is None.

    Returns the runtimes, notes about a target that genuinely failed, and the names of targets that
    were simply not configured. Those last two are different facts: "your cluster is unreachable" is
    worth saying, "you have never used the cluster" is not, and `ls` on a laptop with no kubeconfig
    must still list the local containers without complaining about a target the user never asked
    for.
    """
    runtimes: list[Runtime] = []
    notes: list[str] = []
    unconfigured: list[str] = []
    if target in (None, "local"):
        try:
            runtimes += local_runtimes()
        except LifecycleError as exc:
            if target == "local":
                raise
            notes.append(f"local: {exc}")
    if target in (None, "k8s"):
        from factory.contained.usage import uses

        # Only reach for the cluster when there is reason to think it is wanted. Asking an
        # unreachable one costs a multi-second timeout and then reports an error about a target the
        # user may never have used — which is the common case for anyone who set up `local` only.
        if target is None and not uses("k8s"):
            unconfigured.append("k8s")
        else:
            try:
                from factory.contained.k8s import cluster_runtimes, has_cluster_context

                if target is None and not has_cluster_context():
                    unconfigured.append("k8s")
                else:
                    runtimes += cluster_runtimes(namespace)
            except LifecycleError as exc:
                if target == "k8s":
                    raise
                notes.append(f"k8s: {exc}")
    if target in (None, "openshell"):
        from factory.contained.usage import uses

        # Same rule as the cluster: only consult the gateway when there is reason to think it
        # is wanted, or `ls` on a laptop that never set it up waits on a connection error
        # about a target the user never asked for.
        if target is None and not uses("openshell"):
            unconfigured.append("openshell")
        else:
            try:
                runtimes += openshell_runtimes(gateway)
            except LifecycleError as exc:
                if target == "openshell":
                    raise
                notes.append(f"openshell: {exc}")
    return runtimes, notes, unconfigured


def _format_age(created: datetime | None) -> str:
    if created is None:
        return "?"
    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    seconds = int((now - created).total_seconds())
    if seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def render_table(
    runtimes: list[Runtime],
    notes: list[str] | None = None,
    unconfigured: list[str] | None = None,
) -> str:
    if not runtimes:
        # Three different facts, and only one of them is a problem: nothing is running, something
        # could not be reached, or a target was never set up. Reporting the second for the first
        # tells a user their fleet is empty when the engine is simply down.
        body = (
            "Could not list every runtime — see the note(s) below."
            if notes
            else "No contained runtimes. Start one with `factory contained -- ceo <path>`."
        )
    else:
        rows = [f"{'NAME':<34}{'TARGET':<8}{'PROJECT':<14}{'AGE':<6}{'STATE'}"]
        for runtime in runtimes:
            rows.append(
                f"{runtime.name:<34}"
                f"{runtime.target:<8}"
                f"{runtime.project:<14}"
                f"{_format_age(runtime.created):<6}"
                f"{runtime.state}"
            )
        body = "\n".join(rows)
    for note in notes or []:
        body += f"\n\nnote: {note}"
    return body


def resolve_runtime(name: str, runtimes: list[Runtime]) -> Runtime | None:
    """Find a factory-created runtime by name, or None when it is not one of ours."""
    return next((r for r in runtimes if r.name == name), None)


def _not_ours(name: str) -> int:
    print(
        f"contained: {name} is not a runtime `factory contained` created. "
        "`factory contained ls` shows the ones it manages.",
        file=sys.stderr,
    )
    return 1


def attach(
    name: str, target: str, namespace: str | None = None, gateway: str | None = None
) -> int:
    """Attach to the run's tmux session, blocking until the user detaches or it ends.

    `subprocess.call` forks and waits rather than exec'ing — this process resumes when the tmux
    client exits — but for the user at the terminal, that client *is* their terminal in the
    meantime: `Ctrl-b d` detaches without stopping the run.
    """
    runtimes, _, _ = list_runtimes(target, namespace, gateway)
    runtime = resolve_runtime(name, runtimes)
    if runtime is None:
        return _not_ours(name)
    if runtime.target == "local" and runtime.state == "finished":
        # The container is up but the run's session is gone — usually the run ended, or a stray
        # Ctrl-D closed it. Sessions created by current versions survive that; older ones do not,
        # and either way "no sessions" from tmux is not an answer a user can act on.
        print(
            f"contained: {name}'s run has finished and its session is gone, so there is nothing to "
            f"attach to.\n"
            f"  Look inside anyway:  podman exec -it {name} bash\n"
            f"  Get the work back:   factory contained sync {name}\n"
            f"  Remove it:           factory contained rm {name}",
            file=sys.stderr,
        )
        return 1
    if not runtime.active:
        print(
            f"contained: {name} is {runtime.state} — the container is not running, so there is "
            f"nothing to attach to.\n"
            f"  Its workspace is still on disk; `factory contained sync {name}` shows where.\n"
            f"  Remove it with:  factory contained rm {name}",
            file=sys.stderr,
        )
        return 1
    if runtime.target == "k8s":
        from factory.contained.k8s import build_pod_attach_argv

        return subprocess.call(build_pod_attach_argv(name, namespace))
    if runtime.target == "openshell":
        from factory.contained.openshell import build_attach_argv as build_sandbox_attach

        return subprocess.call(build_sandbox_attach(name, gateway=gateway))
    return subprocess.call(build_attach_argv(name))


def remove(
    name: str, target: str, namespace: str | None = None, *,
    assume_yes: bool, interactive: bool | None = None, gateway: str | None = None,
) -> int:
    """Delete a factory-created runtime.

    Prompts before deleting one that is still active (spec: "Prompts if the run is still
    active" — not a hard refusal). `--yes` skips the prompt for automation. When stdin is not a TTY
    and `--yes` was not passed, this refuses rather than hanging on an answer that will never come.
    """
    runtimes, _, _ = list_runtimes(target, namespace, gateway)
    runtime = resolve_runtime(name, runtimes)
    if runtime is None:
        return _not_ours(name)
    if runtime.active and not assume_yes:
        is_interactive = sys.stdin.isatty() if interactive is None else interactive
        if not is_interactive:
            print(
                f"contained: {name} is still active (state={runtime.state}). Re-run with --yes to "
                "delete it non-interactively.",
                file=sys.stderr,
            )
            return 1
        answer = input(f"{name} is still active (state={runtime.state}). Delete anyway? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print(f"contained: {name} was not deleted.", file=sys.stderr)
            return 1

    log.debug("contained_remove_requested", name=name, state=runtime.state, target=runtime.target)
    if runtime.target == "k8s":
        from factory.contained.k8s import remove_cluster_runtime

        return remove_cluster_runtime(name, namespace=namespace, assume_yes=assume_yes)
    if runtime.target == "openshell":
        from factory.contained.openshell import remove_runtime as remove_sandbox_runtime

        try:
            remove_sandbox_runtime(name, gateway=gateway)
        except ContainedError as exc:
            print(f"contained: {exc}", file=sys.stderr)
            return 1
        print(f"{name}: sandbox deleted.")
        ws = workspace_for(name)
        if ws is not None:
            print(f"{name}: deleted. Your work is kept — it is not removed with the runtime.")
            print(merge_hint(ws))
            print()
            print(cleanup_hint(ws))
        return 0

    # podman echoes the name it removed; we print our own report on the next line, and the doubled
    # name reads like a stutter.
    removed = subprocess.run(build_rm_argv(name), capture_output=True, text=True)
    if removed.returncode != 0:
        log.warning("contained_remove_failed", name=name, exit_code=removed.returncode,
                    stderr=removed.stderr.strip()[:200])
        print(f"contained: removing {name} failed: {removed.stderr.strip()}", file=sys.stderr)
        return removed.returncode
    log.debug("contained_remove_completed", name=name)
    # The division server is a *host* process the run depends on, so removing the run is what ends
    # it. Nothing else does: it is deliberately detached from the command that started it.
    from factory.contained.division import stop_recorded

    if stop_recorded(name):
        print(f"{name}: division endpoint stopped.")
    ws = workspace_for(name)
    if ws is not None:
        print(f"{name}: deleted. Your work is kept — it is not removed with the runtime.")
        print(merge_hint(ws))
        # The copy is a git worktree of the user's own repository, so it is registered in their
        # repo and its branch is in their refs. Removing the container does not touch either, and a
        # user who only deletes the directory leaves a stale registration that blocks the next run
        # of the same name.
        print()
        print(cleanup_hint(ws))
    else:
        print(f"{name}: deleted.")
    return 0


def reap_stale(name: str) -> tuple[bool, str]:
    """Delete `name` if — and only if — it is a factory-created container no longer active.

    A failed run that leaves its container behind otherwise blocks every later invocation of the
    same name behind a bare "name already in use", with nothing pointing at how to get unstuck.
    Reaping automatically is safe exactly when the two checks `remove()` applies interactively both
    hold: the label confirms the factory created it, and the state confirms it is not doing
    something a delete could interrupt. A still-running container is deliberately left alone —
    a name collision can equally mean "you meant to reattach".

    Returns `(reaped, detail)`; `detail` explains the outcome either way, so a caller that could
    not reap automatically still has something concrete to put in front of the user.
    """
    try:
        runtime = resolve_runtime(name, local_runtimes())
    except LifecycleError as exc:
        return False, str(exc)
    if runtime is None:
        return False, f"{name} is not a runtime `factory contained` created"
    if runtime.active:
        return False, f"{name} is still active (state={runtime.state})"
    removed = subprocess.run(build_rm_argv(name), capture_output=True, text=True)
    if removed.returncode != 0:
        return False, f"removing stale container {name} failed: {removed.stderr.strip()}"
    log.debug("contained_stale_reaped", name=name, state=runtime.state)
    return True, f"removed stale container {name} (was {runtime.state})"


def workspace_for(name: str) -> Workspace | None:
    """Reconstruct the `Workspace` `sync`/`rm` need for a named runtime.

    Nothing persists a run-name-to-source-path manifest, so this reconstructs it from the one place
    `materialize` leaves a record on disk: `contained_home()/<name>/` holds exactly one child
    directory — the workspace copy, named after the source project. For a git worktree, that copy's
    `.git` is a pointer file of the form `gitdir: <source>/.git/worktrees/<id>`, which is what lets
    the source path be recovered without ever having stored it.

    A plain rsync copy (non-git source) carries no such pointer, so for that case there is no way
    back to the source path from the copy alone, and this returns None. A worktree whose branch
    cannot be determined also returns None rather than a `Workspace` with an empty branch:
    `merge_hint` treats a worktree with a falsy branch as a plain copy and prints an rsync merge
    command for what is actually a git worktree, and wrong guidance is worse than "not found".
    """
    root = contained_home() / name
    if not root.is_dir():
        return None
    children = [child for child in root.iterdir() if child.is_dir()]
    if len(children) != 1:
        return None
    path = children[0]
    git_pointer = path / ".git"
    if not git_pointer.is_file():
        return None
    try:
        contents = git_pointer.read_text().strip()
    except OSError:
        return None
    if not contents.startswith("gitdir:"):
        return None
    worktree_git_dir = Path(contents.split(":", 1)[1].strip())
    if worktree_git_dir.parent.name != "worktrees":
        return None
    source = worktree_git_dir.parent.parent.parent
    branch_result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
    if not branch:
        return None
    return Workspace(source=source, path=path, kind="worktree", branch=branch)


def sync(
    name: str, target: str, namespace: str | None = None, gateway: str | None = None
) -> int:
    """Report how to get the workspace back. Nothing is ever merged automatically."""
    runtimes, _, _ = list_runtimes(target, namespace, gateway)
    runtime = resolve_runtime(name, runtimes)
    if runtime is None:
        return _not_ours(name)
    if runtime.target == "k8s":
        from factory.contained.k8s import sync_cluster_runtime

        return sync_cluster_runtime(name, namespace=namespace)
    if runtime.target == "openshell":
        return _sync_openshell(name, gateway)
    ws = workspace_for(name)
    if ws is None:
        print(
            f"contained: no local workspace found for {name} under {contained_home()}. The copy "
            "may have been removed, or the project was not a git repository (no source path is "
            "recoverable from a plain copy).",
            file=sys.stderr,
        )
        return 1
    print(f"{name}: the workspace is already on this machine — a bind mount, not a transfer.")
    print(merge_hint(ws))
    return 0


def _sync_openshell(name: str, gateway: str | None = None) -> int:
    """Fetch the sandbox's workspace back as one tarball, mirroring the cluster sync output.

    The download is run rather than merely suggested — unlike the cluster target, whose `oc cp`
    command the user is told to run, the openshell CLI is already a prerequisite of this target
    and the destination is ours to choose. The merge is still never automatic.
    """
    from factory.contained.openshell import build_download_argv, run_cli

    destination = contained_home() / name
    destination.mkdir(parents=True, exist_ok=True)
    # The project directory name is not recoverable from the sandbox listing alone (labels
    # carry a path hash, not a name), so the download covers the workspace root; the tarball
    # the run uploaded unpacked to `<root>/<project>`, so `<root>` is the faithful inverse.
    argv = build_download_argv(name, ".", destination, gateway=gateway)
    result = run_cli(argv, timeout=1800)
    if result.returncode != 0:
        print(
            f"contained: downloading the workspace failed: {result.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return 1
    # What the download left behind decides what the instructions say — a guessed
    # "tar xzf *.tar.gz" for a directory download is a command that cannot run. The
    # destination is also where the local workspace copy lives, which the hint below leans on.
    archives = sorted(destination.glob("*.tar.gz"))
    if archives:
        print(f"{name}: workspace fetched to {archives[0]}.")
        print(f"  Unpack:  tar xzf {archives[0]} -C <dir>")
    else:
        print(f"{name}: workspace fetched to {destination}.")
        print(f"  Review:  ls {destination}")
    ws = workspace_for(name)
    if ws is not None:
        print(merge_hint(ws))
    return 0


def dispatch_lifecycle(args: argparse.Namespace) -> int:
    """Route a parsed `factory contained` lifecycle subcommand to its handler."""
    name = getattr(args, "name", None)
    target = getattr(args, "target", "local")
    namespace = getattr(args, "namespace", None)
    gateway = getattr(args, "gateway", None)
    try:
        if args.subcommand == "ls":
            # No target filter: one table covering all targets, because a user asking "what is
            # running?" does not want to ask it three times.
            runtimes, notes, unconfigured = list_runtimes(None, namespace, gateway)
            print(render_table(runtimes, notes, unconfigured))
            # A target that failed to list is a failure, not an empty fleet — a script wrapping
            # `ls` must not read a dead engine as "nothing running".
            return 1 if notes else 0
        if args.subcommand in ("attach", "rm", "sync"):
            if not isinstance(name, str):
                # `interpret()` already enforces this on the real CLI path; this is
                # belt-and-suspenders for any other caller that constructs args by hand.
                print(
                    f"contained: `factory contained {args.subcommand}` needs a runtime name.",
                    file=sys.stderr,
                )
                return 2
            if args.subcommand == "attach":
                return attach(name, target, namespace, gateway)
            if args.subcommand == "rm":
                return remove(
                    name,
                    target,
                    namespace,
                    assume_yes=bool(getattr(args, "yes", False)),
                    interactive=sys.stdin.isatty(),
                    gateway=gateway,
                )
            return sync(name, target, namespace, gateway)
    except LifecycleError as exc:
        print(f"contained: {exc}", file=sys.stderr)
        return 1
    print(
        f"contained: `{args.subcommand}` is not implemented yet by lifecycle dispatch.",
        file=sys.stderr,
    )
    return 2
