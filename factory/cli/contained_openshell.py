"""Running the factory inside an OpenShell sandbox.

The sequence, and why it is this sequence (mirrors `contained_k8s` — both targets share the
shape "nothing mounted from the host, workspace travels as a copy, run unattended"):

1. **Materialize** the same self-contained workspace copy the cluster target uses — the run
   starts from the files on this machine, uncommitted changes included.
2. **Scan** it for secrets, because from here it leaves the machine — into a sandbox whose
   whole point is that what runs inside it is not trusted, which is exactly when a leaked
   key is least recoverable.
3. **Pack** it into one tarball (same excludes as k8s) and upload it with the OpenShell CLI;
   the SDK has no file transfer.
4. **Create** the sandbox with the policy and the `claude-code` provider attached.
5. **Assert** provenance inside the sandbox, before the factory starts — the packer copies
   what it is told, so the filtered-transfer trap that a bind mount removes locally is live
   here as on the cluster.
6. **Start** the run as a detached, logged process through the SDK's exec — a sandbox cannot
   run tmux (PTY allocation is denied by the Landlock allowlist), so the run writes
   `.factory/run.log`/`run.pid`/`run.exit` inside the project and those files are its interface.

What deliberately does *not* happen anywhere in this file: an inference credential crossing
into the sandbox. The provider holds it in the gateway; the sandbox sees a placeholder. A run
without a configured provider fails fast with the fix, because a fallback to `--env` would
make the insecure route the path of least resistance. What the provider does *not* buy: the
placeholder resolves for any descendant of `claude` — every command the agent's Bash tool
runs — so the key cannot be extracted but can be *used* against the provider's endpoints.
The profile edits in `openshell.PROVIDER_FIX` exist to keep that endpoint set as small as
the inference call itself.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import tarfile
from pathlib import Path

import structlog

from factory.contained import openshell
from factory.contained.env import CONTAINED_ENV_POLICY, is_secret_key
from factory.contained.errors import ContainedError
from factory.contained.paths import rewrite_argv
from factory.contained.provenance import content_probe, provenance_probes
from factory.contained.secrets import confirm_upload, scan
from factory.contained.workspace import (
    Workspace,
    WorkspaceError,
    contained_home,
    materialize,
    plan_workspace,
)
from factory.podman import (
    build_run_command,
    dry_run_enabled,
    growth_context_warning,
    project_hash,
    resolve_image,
)

log = structlog.get_logger()

# Same list as the cluster target, for the same reasons: large, host-shaped, or actively wrong
# (an arm64 .venv on an amd64 node) rather than merely wasteful. `.git` is *not* excluded.
PACK_EXCLUDES = frozenset({
    ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".factory-worktrees",
})

# The previous run's interface files. `sync` writes them into the local workspace copy, and
# `_materialize_copy` refreshes that copy without deleting, so without this exclusion the next
# run's upload carries them — and `attach` prints a stale `[factory exited N]` before the new
# run has done anything. They name the *previous* run by construction; a project file that
# happens to be called `run.log` outside `.factory` is not one of them and still packs.
RUN_ARTIFACTS = frozenset({"run.log", "run.pid", "run.exit"})

PROVIDER_NAME = "claude-code"


def run_openshell(args: argparse.Namespace) -> int:
    """Provision an OpenShell sandbox and start the run in it."""
    dry_run = dry_run_enabled()
    try:
        from factory.cli.contained_args import resolve_project, validate_env_args

        project = resolve_project(args.factory_args)
        extra, forwarded = validate_env_args(args)
        run_id = openshell.sandbox_name(project) if not args.name else openshell.validate_sandbox_name(args.name)
        ws = (
            plan_workspace(project, run_id, self_contained=True) if dry_run
            else materialize(project, run_id, self_contained=True)
        )
        plan = _build_plan(args, ws, run_id, extra, forwarded)
    except (ContainedError, WorkspaceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for warning in (growth_context_warning(), *plan.warnings):
        if warning:
            print(f"Warning: {warning}", file=sys.stderr)

    if dry_run:
        return _emit_dry_run(plan, ws)

    try:
        if not _scan_and_confirm(ws, assume_yes=args.yes):
            return 1
        from factory.contained.usage import record_target

        record_target("openshell")
        tarball = _pack(ws, run_id)
    except ContainedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # From here on, a failure distinguishes "nothing was created" (roll the workspace copy
    # back, so its worktree branch does not block the next run of the same name) from "a
    # sandbox exists" (keep it — a failed run is exactly when its state is worth reading).
    try:
        client = openshell.connect(args.gateway)
        session = openshell.create_sandbox(
            client, plan, _policy_for(plan), tarball
        )
    except ContainedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        from factory.cli.contained_local import _roll_back

        _roll_back(ws)
        return 2
    return _upload_and_start(plan, session, tarball, ws, project)


def _policy_for(plan: openshell.OpenShellPlan):
    return (
        openshell.load_policy(plan.policy_path) if plan.policy_path is not None
        else openshell.build_default_policy()
    )


def _build_plan(
    args: argparse.Namespace,
    ws: Workspace,
    run_id: str,
    extra: dict[str, str],
    forwarded: dict[str, str],
) -> openshell.OpenShellPlan:
    """Compose the sandbox plan. Pure: no gateway contact, so dry-run is honest by construction."""
    warnings: list[str] = []
    project_dir = f"{openshell.WORKSPACE_ROOT}/{ws.source.name}"

    # Configuration only. The provider carries credentials; a secret-looking key here means
    # --forward or --env was used to smuggle one past that boundary, so it is named loudly —
    # this is the one target whose stated purpose is being safe to point at untrusted code.
    env = CONTAINED_ENV_POLICY.resolve(dict(os.environ))
    env.update(forwarded)
    env.update(extra)
    # The run's stdout is a file (`build_run_launch`), where Python block-buffers; without this,
    # `attach`'s log tail shows nothing for whole minutes of live work — indistinguishable from
    # the hang it was reported as. The claude subprocess inherits it.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Claude Code's telemetry/statsig/sentry traffic is egress this target's policy story says
    # untrusted code should not have — and the provider profile may not even allow it (see
    # openshell.PROVIDER_FIX). Claude itself honours this switch, so its own non-essential
    # traffic stops rather than being denied connection-by-connection.
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    if any(is_secret_key(key) for key in env):
        warnings.append(
            "a credential-looking variable is being placed in the sandbox environment. The "
            f"claude-code provider is the supported route (see: {openshell.PROVIDER_FIX}); a "
            "key in the environment is readable by exactly the untrusted code this target "
            "exists to confine."
        )

    policy_path = Path(args.policy).expanduser() if args.policy else None
    if policy_path is not None and not policy_path.is_file():
        raise ContainedError(f"--policy {args.policy}: no such file")

    factory_argv, changes = rewrite_argv(args.factory_args, ws.source, project_dir)
    for before, after in changes:
        log.debug("contained_path_rewritten", before=before, after=after)
    inner = "factory " + " ".join(shlex.quote(token) for token in factory_argv)

    return openshell.OpenShellPlan(
        name=run_id,
        image=args.image or resolve_image(),
        project_dir=project_dir,
        env=env,
        labels={
            "factory.contained": "true",
            "factory.project": project_hash(ws.source),
            "factory.name": run_id,
            # No `factory.source` here, unlike the podman target: gateway label values must be
            # alphanumeric/-/_/. and a source *path* is not. The cluster target makes the same
            # omission, and `workspace_for` recovers the source from the local workspace copy,
            # so nothing that needs it goes through the label.
        },
        provider=PROVIDER_NAME,
        policy_path=policy_path,
        run_command=build_run_command(project_dir, inner),
        factory_command=inner,
        gateway=args.gateway,
        warnings=tuple(warnings),
    )


def _scan_and_confirm(ws: Workspace, *, assume_yes: bool) -> bool:
    """Nothing leaves the machine before this returns True."""
    result = scan(ws.path)
    return confirm_upload(result, assume_yes=assume_yes)


def _pack(ws: Workspace, run_id: str) -> Path:
    """Pack the workspace into one tarball, under its own directory name.

    Same convention as the cluster target: packed as `<project>/...` so it unpacks to
    `/workspace/<project>`, the path everything downstream already agrees on. A single file
    also sidesteps the upload CLI's `.gitignore` filtering, which would silently drop the
    `.factory/` state the whole experiment history lives in.
    """
    destination = contained_home() / run_id / "upload.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _filter(entry: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(entry.name).parts
        if set(parts) & PACK_EXCLUDES:
            return None
        # A previous run's artifacts, exactly as `sync` wrote them: `.factory/run.*`.
        if ".factory" in parts[:-1] and parts[-1] in RUN_ARTIFACTS:
            return None
        return entry

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(ws.path, arcname=ws.source.name, filter=_filter)
    log.debug("contained_packed", path=str(destination), bytes=destination.stat().st_size)
    return destination


def _upload_and_start(
    plan: openshell.OpenShellPlan,
    session,
    tarball: Path,
    ws: Workspace,
    project: Path,
) -> int:
    """Upload the workspace, assert provenance, start the run.

    The sandbox already exists on entry, so every failure path here keeps it — a failed run is
    exactly when its state is worth reading — and says how to inspect and remove it.
    """
    # The identifier first, before any long-running work: a run whose name the user cannot see
    # is a run they cannot manage.
    print(plan.name)

    uploaded = openshell.run_cli(
        openshell.build_upload_argv(plan.name, tarball, gateway=plan.gateway)
    )
    if uploaded.returncode != 0:
        return _sandbox_failure(
            plan, f"uploading the workspace failed: {uploaded.stderr.strip()[:300]}"
        )
    try:
        openshell.extract_tarball(session, tarball.name)
    except ContainedError as exc:
        return _sandbox_failure(plan, f"unpacking the workspace failed: {exc}")

    probes = provenance_probes(
        plan.project_dir,
        expect_factory_state=(project / ".factory" / "config.json").exists(),
        expect_git=(project / ".git").exists(),
        content=content_probe(ws.path),
    )
    try:
        openshell.run_probes(session, plan, probes)
    except ContainedError as exc:
        return _sandbox_failure(plan, str(exc))

    # No "already running" check here, unlike the tmux targets: a same-named sandbox was
    # already refused at `create` (the gateway rejects the collision — live-verified), so this
    # sandbox was created by *this* invocation and nothing can be mid-run in it yet. A check
    # here could only ever fire on stale pid/exit files from the uploaded copy, which
    # `build_run_launch` deletes before starting.

    try:
        openshell.start_run(session, plan)
    except ContainedError as exc:
        return _sandbox_failure(plan, f"starting the run failed: {exc}")
    print(f"  attach:  factory contained --target openshell attach {plan.name}")
    print(f"  result:  factory contained --target openshell sync {plan.name}")
    return 0


def _sandbox_failure(plan: openshell.OpenShellPlan, message: str) -> int:
    """Report a post-create failure, pointing at the sandbox that survives it."""
    print(
        f"contained: {message}\n"
        f"  The sandbox is still there for inspection:\n"
        f"    factory contained --target openshell attach {plan.name}\n"
        f"    factory contained --target openshell rm {plan.name}",
        file=sys.stderr,
    )
    return 1


def _emit_dry_run(plan: openshell.OpenShellPlan, ws: Workspace) -> int:
    """Print the plan and the exact steps the real path would execute, and execute none."""
    print(
        f"DRY RUN — sandbox {plan.name} from {plan.image}; nothing is provisioned, no gateway "
        f"is contacted."
    )
    print(f"provider: {plan.provider} (credentials stay in the gateway)")
    print(
        f"policy: {'full replacement from ' + str(plan.policy_path) if plan.policy_path else 'factory default (workspace rw, inference + PyPI egress only)'}"
    )
    for key, value in sorted(plan.env.items()):
        shown = "<redacted>" if is_secret_key(key) else value
        print(f"env: {key}={shown}")
    probes = provenance_probes(
        plan.project_dir,
        expect_factory_state=(ws.source / ".factory" / "config.json").exists(),
        expect_git=(ws.source / ".git").exists(),
        content=content_probe(ws.path),
    )
    for step in openshell.plan_steps(plan, contained_home() / plan.name / "upload.tar.gz", probes):
        print(f"[{step.name}] {step.render()}")
    return 0
