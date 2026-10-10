"""Git worktree lifecycle management for experiment isolation."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import structlog


log = structlog.get_logger()

# Telemetry files to preserve when cleaning up worktrees
_TELEMETRY_FILES = ("trace_id.txt",)

# .factory entries to seed into experiment worktrees so agents can read project
# config without sharing mutable eval state (like last_eval.json) across branches.
_EXPERIMENT_SEED_ENTRIES: Final[tuple[str, ...]] = (
    "config.json",
    "eval_profile.json",
    "strategy",
    "agents",
)

# .factory entries symlinked to main — shared, append-only/read-only project state.
_SHARED_SYMLINK_ENTRIES: Final[tuple[str, ...]] = (
    "config.json",
    "eval_profile.json",
    "results.tsv",
    "experiments",
    "archive",
    "events.jsonl",
    ".store.lock",
    "adversarial_state.json",
    "performance_report.json",
)

# .factory entries copied from main — read-only but agents may override per-run.
_COPY_ENTRIES: Final[tuple[str, ...]] = (
    "agents",
)


WORKTREE_VENV_MARKER: Final[str] = ".factory-managed"


def _setup_worktree_venv(worktree_path: Path) -> Path | None:
    """Create a per-worktree Python venv if pyproject.toml is present.

    Tries ``uv sync`` first (auto-creates ``.venv`` + editable install).
    Falls back to ``python -m venv`` + ``uv pip install -e``.
    Returns the venv path on success, ``None`` on skip/failure.
    """
    if not (worktree_path / "pyproject.toml").exists():
        return None

    result = subprocess.run(
        ["uv", "sync", "--directory", str(worktree_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        venv_path = worktree_path / ".venv"
        (venv_path / WORKTREE_VENV_MARKER).touch()
        log.info("worktree_venv_created", method="uv_sync", path=str(venv_path))
        return venv_path

    log.warning("worktree_venv_uv_sync_failed", stderr=result.stderr[:200])

    venv_path = worktree_path / ".venv"
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.warning("worktree_venv_fallback_venv_failed", stderr=result.stderr[:200])
        return None

    result = subprocess.run(
        ["uv", "pip", "install", "-e", str(worktree_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": str(venv_path)},
    )
    if result.returncode != 0:
        log.warning("worktree_venv_fallback_install_failed", stderr=result.stderr[:200])
        return None

    (venv_path / WORKTREE_VENV_MARKER).touch()
    log.info("worktree_venv_created", method="fallback", path=str(venv_path))
    return venv_path


def is_factory_venv(project_path: Path) -> bool:
    """Return True if the .venv at project_path was created by the factory."""
    return (project_path / ".venv" / WORKTREE_VENV_MARKER).exists()


def create_worktree(
    project_path: Path,
    base_branch: str = "main",
    run_id: str | None = None,
) -> tuple[Path, str]:
    """Create an isolated worktree for a factory run.

    Args:
        project_path: Path to the project root.
        base_branch: Branch to create the worktree from.
        run_id: Optional run identifier. If provided, uses the first 8 chars.
                If None, generates a random 8-char hex ID.

    Returns (worktree_path, branch_name).
    """
    project_path = project_path.resolve()

    # Resolve symbolic refs (HEAD, branch names) to commit SHAs so the
    # worktree always branches from a deterministic point — critical when
    # HEAD was just amended (e.g. FeatureBench mask-patch scenario).
    result = subprocess.run(
        ["git", "rev-parse", base_branch],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if _is_unborn_repo(project_path):
            _bootstrap_unborn_repo(project_path)
            result = subprocess.run(
                ["git", "rev-parse", base_branch],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            raise RuntimeError(
                f"Branch '{base_branch}' does not exist in {project_path}. "
                "Set `target_branch` in .factory/config.json or check your git state."
            )
    base_commit = result.stdout.strip()

    if run_id is not None:
        run_id = run_id[:8]
    else:
        run_id = secrets.token_hex(4)
    branch = f"factory/run-{run_id}"
    factory_dir = project_path / ".factory"
    wt_parent = project_path / ".factory-worktrees"
    wt_dir = wt_parent / f"run-{run_id}"

    log.info("worktree_create", branch=branch, base=base_commit[:12], path=str(wt_dir))

    wt_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", str(wt_dir), "-b", branch, base_commit],
        cwd=project_path,
        check=True,
        capture_output=True,
    )

    # Create independent .factory/ with selective sharing — shared append-only
    # state is symlinked, per-cycle mutable state gets fresh directories.
    wt_factory = wt_dir / ".factory"
    if wt_factory.exists() or wt_factory.is_symlink():
        if wt_factory.is_dir() and not wt_factory.is_symlink():
            shutil.rmtree(wt_factory)
        else:
            wt_factory.unlink()

    wt_factory.mkdir(parents=True, exist_ok=True)

    for entry in _SHARED_SYMLINK_ENTRIES:
        src = factory_dir / entry
        if src.exists():
            (wt_factory / entry).symlink_to(src)

    for entry in _COPY_ENTRIES:
        src = factory_dir / entry
        if src.exists():
            dst = wt_factory / entry
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    (wt_factory / "strategy").mkdir(exist_ok=True)
    (wt_factory / "reviews").mkdir(exist_ok=True)
    (wt_factory / "state").mkdir(exist_ok=True)

    backlog_src = factory_dir / "strategy" / "backlog.md"
    if backlog_src.exists():
        shutil.copy2(backlog_src, wt_factory / "strategy" / "backlog.md")

    # Copy remaining plugin-created subdirectories not already handled.
    _handled = set(_SHARED_SYMLINK_ENTRIES) | set(_COPY_ENTRIES)
    if factory_dir.is_dir():
        for child in factory_dir.iterdir():
            if child.name in _handled or not child.is_dir():
                continue
            dst = wt_factory / child.name
            if not dst.exists():
                shutil.copytree(child, dst)

    log.info("worktree_created", branch=branch, path=str(wt_dir))

    _setup_worktree_venv(wt_dir)

    try:
        from factory.events import emit_event

        emit_event(
            project_path,
            "worktree.created",
            data={
                "run_id": run_id,
                "worktree_path": str(wt_dir),
                "branch": branch,
                "base_branch": base_branch,
            },
        )
    except Exception:
        pass

    return wt_dir, branch


def create_experiment_worktree(
    project_path: Path,
    exp_id: int,
    base_commit: str,
) -> tuple[Path, str]:
    """Create an isolated worktree for a parallel experiment branch.

    Each worktree gets its own `.factory/` directory (not a symlink) seeded
    with read-only config from the project.  This ensures parallel branches
    write independent `last_eval.json` files so the selection node can
    compare genuinely separate scores.

    Returns (worktree_path, branch_name).
    """
    project_path = project_path.resolve()
    branch = f"factory/exp-{exp_id}"
    factory_dir = project_path / ".factory"
    wt_parent = project_path / ".factory-worktrees"
    wt_dir = wt_parent / f"exp-{exp_id}"

    log.info("experiment_worktree_create", branch=branch, base=base_commit[:12], exp_id=exp_id)

    wt_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", str(wt_dir), "-b", branch, base_commit],
        cwd=project_path,
        check=True,
        capture_output=True,
    )

    _seed_experiment_factory(factory_dir, wt_dir / ".factory")

    _setup_worktree_venv(wt_dir)

    log.info("experiment_worktree_created", branch=branch, path=str(wt_dir))

    try:
        from factory.events import emit_event

        emit_event(
            project_path,
            "experiment_worktree.created",
            data={
                "exp_id": exp_id,
                "worktree_path": str(wt_dir),
                "branch": branch,
                "base_commit": base_commit,
            },
        )
    except Exception:
        pass

    return wt_dir, branch


def _seed_experiment_factory(source: Path, dest: Path) -> None:
    """Copy config entries from the project .factory/ into an experiment worktree.

    Only copies entries listed in _EXPERIMENT_SEED_ENTRIES so that mutable
    runtime state (results.tsv, experiments/, last_eval.json) stays independent.
    """
    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not source.is_dir():
        return

    for entry_name in _EXPERIMENT_SEED_ENTRIES:
        src = source / entry_name
        dst = dest / entry_name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _sync_backlog_to_main(worktree_path: Path, project_path: Path) -> None:
    """Sync backlog changes from worktree back to main .factory/."""
    wt_backlog = worktree_path / ".factory" / "strategy" / "backlog.md"
    main_backlog = project_path / ".factory" / "strategy" / "backlog.md"
    if wt_backlog.exists() and not wt_backlog.is_symlink():
        main_backlog.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wt_backlog, main_backlog)
        log.info("backlog_synced", src=str(wt_backlog), dst=str(main_backlog))


_BOOTSTRAP_FACTORY_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "eval_profile.json",
)


def _sync_bootstrap_to_main(worktree_path: Path, project_path: Path) -> None:
    """Sync bootstrap artifacts from worktree back to main project.

    Only copies files that are real (not symlinks), meaning they were freshly
    created during this run rather than symlinked from main.
    """
    wt_factory = worktree_path / ".factory"
    main_factory = project_path / ".factory"

    if not wt_factory.exists():
        return

    main_factory.mkdir(parents=True, exist_ok=True)
    for filename in _BOOTSTRAP_FACTORY_FILES:
        src = wt_factory / filename
        if src.exists() and not src.is_symlink():
            dst = main_factory / filename
            if not dst.exists():
                shutil.copy2(src, dst)
                log.info("bootstrap_synced", file=filename, src=str(src), dst=str(dst))

    wt_factory_md = worktree_path / "factory.md"
    main_factory_md = project_path / "factory.md"
    if wt_factory_md.exists() and not wt_factory_md.is_symlink() and not main_factory_md.exists():
        shutil.copy2(wt_factory_md, main_factory_md)
        log.info("bootstrap_synced", file="factory.md", src=str(wt_factory_md), dst=str(main_factory_md))


def _preserve_telemetry(worktree_path: Path, project_path: Path) -> None:
    """Copy telemetry files from worktree .factory/ to main project .factory/."""
    wt_factory = worktree_path / ".factory"
    main_factory = project_path / ".factory"

    if not wt_factory.exists():
        return

    main_factory.mkdir(parents=True, exist_ok=True)
    for filename in _TELEMETRY_FILES:
        src = wt_factory / filename
        if src.exists():
            dst = main_factory / filename
            shutil.copy2(src, dst)
            log.info("telemetry_preserved", file=filename, src=str(src), dst=str(dst))


def _has_active_sessions(worktree_path: Path) -> bool:
    """Check if any Claude Code sessions are active in the worktree.

    Returns True if active sessions found, False otherwise.
    Fails open: returns False on any error so removal proceeds.
    """
    from factory.runners.claude import _claude_bin

    try:
        result = subprocess.run(
            [_claude_bin(), "agents", "--json", "--cwd", str(worktree_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        sessions = json.loads(result.stdout)
        if not isinstance(sessions, list):
            return False
        return any(
            isinstance(s, dict) and s.get("state") in ("working", "blocked") for s in sessions
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError):
        return False


def _should_remove_worktree(branch: str) -> bool:
    """Check whether a worktree should be removed based on config.

    Experiment branches (factory/exp-*) are always removed regardless of config.
    For run branches, consults FACTORY_REMOVE_WORKTREE (default: true).
    """
    if branch.startswith("factory/exp-"):
        return True

    from factory import user_config

    value = user_config.resolve(
        "remove_worktree", env_var="FACTORY_REMOVE_WORKTREE", default="true"
    )
    return (value or "true").lower() in ("true", "1", "yes")


def _is_greenfield_run(worktree_path: Path, project_path: Path) -> bool:
    """Return True if main has no factory.md but the worktree does (non-symlink)."""
    main_factory_md = project_path / "factory.md"
    wt_factory_md = worktree_path / "factory.md"
    return (
        not main_factory_md.exists()
        and wt_factory_md.exists()
        and not wt_factory_md.is_symlink()
    )


def _finalize_greenfield(worktree_path: Path, project_path: Path, branch: str) -> bool:
    """Ensure factory.md is tracked on the greenfield branch. Returns True on success."""
    wt_factory_md = worktree_path / "factory.md"
    if not wt_factory_md.exists():
        log.warning("finalize_greenfield_no_factory_md", path=str(worktree_path))
        return False

    result = subprocess.run(
        ["git", "ls-files", "factory.md"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        log.info("finalize_greenfield_already_tracked", branch=branch)
        return True

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Factory",
        "GIT_AUTHOR_EMAIL": "factory@localhost",
        "GIT_COMMITTER_NAME": "Factory",
        "GIT_COMMITTER_EMAIL": "factory@localhost",
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        add_result = subprocess.run(
            ["git", "add", "factory.md"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        if add_result.returncode != 0:
            log.warning("finalize_greenfield_add_failed", stderr=add_result.stderr[:200])
            return False

        commit_result = subprocess.run(
            ["git", "commit", "-m", "chore: track factory.md for greenfield initialization"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            env=env,
        )
        if commit_result.returncode == 0:
            break
        if "lock" in commit_result.stderr.lower() and attempt < max_attempts - 1:
            import time

            time.sleep(0.2 * (attempt + 1))
            continue
        log.warning("finalize_greenfield_commit_failed", stderr=commit_result.stderr[:200])
        return False

    verify = subprocess.run(
        ["git", "ls-files", "factory.md"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0 or not verify.stdout.strip():
        log.warning("finalize_greenfield_postcondition_failed", branch=branch)
        return False

    log.info("finalize_greenfield_committed", branch=branch)
    return True


def remove_worktree(project_path: Path, worktree_path: Path, branch: str) -> None:
    """Remove a worktree and its branch. Safe to call on already-removed paths."""
    log.info("worktree_remove", branch=branch, path=str(worktree_path))

    run_id = branch.removeprefix("factory/run-")

    if worktree_path.exists():
        if _has_active_sessions(worktree_path):
            log.warning(
                "worktree_remove_skipped",
                reason="active_sessions",
                path=str(worktree_path),
                branch=branch,
            )
            return
        if not _should_remove_worktree(branch):
            log.info(
                "worktree_remove_skipped",
                reason="retention_enabled",
                path=str(worktree_path),
                branch=branch,
            )
            try:
                from factory.events import emit_event

                emit_event(
                    project_path,
                    "worktree.retained",
                    data={
                        "run_id": run_id,
                        "branch": branch,
                        "worktree_path": str(worktree_path),
                    },
                )
            except Exception:
                pass

            print(
                f"Worktree retained: {worktree_path}\n"
                f"To clean up: git worktree remove {worktree_path} && git branch -D {branch}",
                file=sys.stderr,
            )
            return
        if worktree_path != project_path and _is_greenfield_run(worktree_path, project_path):
            if not _finalize_greenfield(worktree_path, project_path, branch):
                log.warning(
                    "greenfield_finalization_failed",
                    worktree=str(worktree_path),
                    branch=branch,
                    hint="factory.md not committed; retaining worktree for recovery",
                )
                print(
                    f"WARNING: Greenfield finalization failed — factory.md may not be tracked.\n"
                    f"Worktree retained: {worktree_path}\n"
                    f"To recover: cd {worktree_path} && git add factory.md && "
                    f"git commit -m 'chore: track factory.md'\n"
                    f"Then clean up: git worktree remove {worktree_path} && git branch -D {branch}",
                    file=sys.stderr,
                )
                return

        _sync_backlog_to_main(worktree_path, project_path)
        _sync_bootstrap_to_main(worktree_path, project_path)
        _preserve_telemetry(worktree_path, project_path)
        shutil.rmtree(worktree_path)

    try:
        from factory.events import emit_event

        emit_event(
            project_path,
            "worktree.removed",
            data={
                "run_id": run_id,
                "branch": branch,
            },
        )
    except Exception:
        pass

    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=project_path,
        capture_output=True,
    )

    subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=project_path,
        capture_output=True,
    )


_STANDARD_BRANCH_RE = re.compile(r"^factory/(run-[0-9a-f]+|exp-\d+)$")


def _is_standard_factory_branch(branch: str) -> bool:
    """Return True if *branch* matches the standard factory branch patterns.

    Standard patterns: ``factory/run-<hex>`` and ``factory/exp-<int>``.
    Human-named branches (e.g. ``fix/readme-content-regression``,
    ``factory/extract-skillopt-1342``) return False and are never GC'd.
    """
    return _STANDARD_BRANCH_RE.match(branch) is not None


def _get_branch_last_commit_ts(project_path: Path, branch: str) -> float | None:
    """Return the UNIX timestamp of the last commit on *branch*, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", branch],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


def _compute_idle_seconds(
    project_path: Path,
    worktree_path: Path,
    branch: str,
) -> float:
    """Return how many seconds the worktree has been idle.

    Idle age is the *minimum* of (now − last-commit-time) and (now − dir-mtime).
    Using the minimum means we only consider the worktree idle if BOTH signals
    indicate inactivity — this is conservative and catches uncommitted work that
    would only show up in the mtime.
    """
    now = time.time()
    ages: list[float] = []

    commit_ts = _get_branch_last_commit_ts(project_path, branch)
    if commit_ts is not None:
        ages.append(now - commit_ts)

    try:
        dir_mtime = worktree_path.stat().st_mtime
        ages.append(now - dir_mtime)
    except OSError:
        pass

    if not ages:
        return 0.0  # Can't determine age → treat as fresh (safe default)

    # Return the MINIMUM age — only reclaim when ALL signals agree it's idle
    return min(ages)


def prune_stale(project_path: Path) -> list[str]:
    """Clean up stale worktrees from crashed runs. Returns list of pruned entries.

    Two sweeps are performed:

    1. **Orphan sweep** — directories under ``.factory-worktrees/`` that git no
       longer tracks (after ``git worktree prune``).  Uses the real branch from
       porcelain output when available, falls back to dir-name reconstruction.

    2. **Idle-registered sweep** — worktrees that ARE still git-registered but
       are idle past a configurable threshold, have no active session, are on a
       standard ``factory/run-*`` or ``factory/exp-*`` branch, and pass the
       retention opt-out check.
    """
    project_path = project_path.resolve()
    if not project_path.exists():
        return []

    result = subprocess.run(
        ["git", "worktree", "prune", "--verbose"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    pruned = [line for line in result.stderr.splitlines() if "Removing" in line]

    # Full set of ALL registered worktree paths (including detached-HEAD) —
    # used for orphan detection in Sweep 1.
    active_paths = _list_active_worktrees(project_path)
    # Path→branch mapping (only worktrees WITH a branch) — used for branch
    # lookups in Sweep 2 (idle GC).
    wt_branch_map = _list_worktrees_with_branches(project_path)

    # --- Sweep 1: orphan directories (not in git worktree list) ---
    wt_parents = [
        project_path / ".factory-worktrees",
        project_path / ".factory" / "worktrees",
    ]
    for wt_parent in wt_parents:
        if not wt_parent.is_dir():
            continue
        for d in wt_parent.iterdir():
            if d.is_dir() and str(d.resolve()) not in active_paths:
                name = d.name
                # True orphans are not in git's worktree list at all, so
                # reconstruct the branch name from the directory name.
                if name.startswith("exp-"):
                    branch = f"factory/{name}"
                else:
                    branch = f"factory/run-{name.removeprefix('run-')}"
                if not name.startswith("exp-") and not _should_remove_worktree(branch):
                    log.info("worktree_prune_skipped", reason="retention_enabled", name=name)
                    continue
                shutil.rmtree(d)
                pruned.append(f"Removed orphaned directory: {name}")
                log.info("worktree_pruned_orphan", name=name)
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    cwd=project_path,
                    capture_output=True,
                )

    # --- Sweep 2: idle git-registered worktrees (worktree GC) ---
    from factory import user_config

    idle_hours_str = user_config.resolve(
        "worktree_idle_reclaim_hours",
        env_var="FACTORY_WORKTREE_IDLE_RECLAIM_HOURS",
        default="24",
    )
    try:
        idle_threshold_secs = float(idle_hours_str or "24") * 3600
    except (ValueError, TypeError):
        idle_threshold_secs = 24 * 3600

    wt_base = project_path / ".factory-worktrees"
    if wt_base.is_dir():
        for d in sorted(wt_base.iterdir()):
            if not d.is_dir():
                continue
            resolved = str(d.resolve())
            if resolved not in wt_branch_map:
                continue  # Already handled (or will be handled) by orphan sweep

            real_branch = wt_branch_map[resolved]

            # (c) Standard branch check
            if not _is_standard_factory_branch(real_branch):
                log.debug(
                    "worktree_gc_skip_nonstandard",
                    path=str(d),
                    branch=real_branch,
                )
                continue

            # (d) Retention opt-out
            if not _should_remove_worktree(real_branch):
                log.debug(
                    "worktree_gc_skip_retained",
                    path=str(d),
                    branch=real_branch,
                )
                continue

            # (b) Active session check
            if _has_active_sessions(d):
                log.debug(
                    "worktree_gc_skip_active_session",
                    path=str(d),
                    branch=real_branch,
                )
                continue

            # (a) Idle check
            idle_seconds = _compute_idle_seconds(project_path, d, real_branch)
            if idle_seconds < idle_threshold_secs:
                log.debug(
                    "worktree_gc_skip_fresh",
                    path=str(d),
                    branch=real_branch,
                    idle_seconds=idle_seconds,
                )
                continue

            # All conditions met — reclaim
            log.info(
                "worktree_gc_reclaiming",
                path=str(d),
                branch=real_branch,
                idle_seconds=idle_seconds,
            )

            _sync_backlog_to_main(d, project_path)
            _preserve_telemetry(d, project_path)
            shutil.rmtree(d)

            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=project_path,
                capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "-D", real_branch],
                cwd=project_path,
                capture_output=True,
            )

            try:
                from factory.events import emit_event

                emit_event(
                    project_path,
                    "worktree.gc_reclaimed",
                    data={
                        "branch": real_branch,
                        "idle_seconds": idle_seconds,
                        "worktree_path": str(d),
                        "reclaim_reason": "idle_standard_branch_no_session",
                    },
                )
            except Exception:
                pass

            pruned.append(
                f"GC reclaimed idle worktree: {d.name} (branch={real_branch}, "
                f"idle={idle_seconds:.0f}s)"
            )

    if pruned:
        log.info("worktree_prune_complete", pruned_count=len(pruned))

    return pruned


def _is_unborn_repo(project_path: Path) -> bool:
    """Return True if the repo exists but has no commits (unborn HEAD)."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0


def _bootstrap_unborn_repo(project_path: Path) -> None:
    """Create an initial empty commit so worktrees can branch from it."""
    log.info("bootstrap_unborn_repo", path=str(project_path))
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init (factory bootstrap)"],
        cwd=project_path,
        capture_output=True,
        check=True,
    )


def detect_default_branch(project_path: Path) -> str:
    """Detect the default branch for a git repository.

    Cascade: remote HEAD, then probe main/master, then current HEAD, then the
    symbolic HEAD of an unborn repo. Raises RuntimeError if none of these
    resolve, rather than guessing a branch that does not exist.
    """
    project_path = project_path.resolve()

    # Try remote default branch
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ref = result.stdout.strip()
        branch = ref.removeprefix("refs/remotes/origin/")
        if branch and branch != ref:
            log.debug("detect_default_branch", source="remote_head", branch=branch)
            return branch

    # Probe main then master
    for candidate in ("main", "master"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.debug("detect_default_branch", source="probe", branch=candidate)
            return candidate

    # Current branch (works on repos with commits)
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        branch = result.stdout.strip()
        if branch != "HEAD":
            log.debug("detect_default_branch", source="current_head", branch=branch)
            return branch

    # Unborn repo: rev-parse fails but symbolic-ref still resolves HEAD
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        branch = result.stdout.strip()
        log.debug("detect_default_branch", source="symbolic_ref", branch=branch)
        return branch

    # Every detection method failed: no origin/HEAD, no local main or master,
    # HEAD is detached, and the symbolic HEAD did not resolve either. Probing
    # for main or master already happened above, so returning a hardcoded
    # "main" here would only hand create_worktree a branch that does not
    # exist, which then surfaces as an opaque "git rev-parse main returned
    # non-zero exit status 128" error. Fail with something the caller can act
    # on instead.
    log.debug("detect_default_branch", source="fallback", branch=None)
    raise RuntimeError(
        f"Could not determine a default branch for {project_path}. "
        "There is no origin/HEAD, no local 'main' or 'master' branch, and "
        "HEAD is detached. Set `target_branch` in .factory/config.json or "
        "check out a branch before running."
    )


def _list_worktrees_with_branches(project_path: Path) -> dict[str, str]:
    """Return a mapping of resolved worktree path → branch name.

    Parses ``git worktree list --porcelain`` output.  Blocks are separated by
    blank lines.  Each block has a ``worktree <path>`` line and optionally a
    ``branch refs/heads/<name>`` line.  Detached-HEAD worktrees (no ``branch``
    line) are omitted from the result.
    """
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    mapping: dict[str, str] = {}
    current_path: str | None = None
    current_branch: str | None = None

    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            # If we had a previous block with both path and branch, record it
            if current_path is not None and current_branch is not None:
                mapping[current_path] = current_branch
            current_path = line.split(" ", 1)[1]
            current_branch = None
        elif line.startswith("branch refs/heads/"):
            current_branch = line.removeprefix("branch refs/heads/")
        elif line == "":
            # End of block
            if current_path is not None and current_branch is not None:
                mapping[current_path] = current_branch
            current_path = None
            current_branch = None

    # Handle the last block (porcelain output may not end with a blank line)
    if current_path is not None and current_branch is not None:
        mapping[current_path] = current_branch

    return mapping


def _list_active_worktrees(project_path: Path) -> set[str]:
    """Return set of absolute paths for ALL git-registered worktrees.

    Parses every ``worktree <path>`` line from ``git worktree list --porcelain``,
    including detached-HEAD worktrees and the main worktree.  This is the
    authoritative set of paths that git considers "registered" — used for
    orphan detection in :func:`prune_stale` so that detached-HEAD worktrees
    are never misclassified as orphans.
    """
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.add(line.split(" ", 1)[1])
    return paths
