"""Factory run command — single-shot and heartbeat loop execution."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import structlog

from factory.cli._helpers import (
    DESIGN_MODES,
    _emit_cli_event,
    _ensure_dashboard,
    _print_banner,
    _read_target_branch,
    _resolve_inactivity_timeout,
    _run,
    warn_deprecated_mode,
)
from factory.cli._mode_handlers import (
    _auto_detect_mode,
    _resolve_background,
    _resolve_bg_agents,
    _resolve_model,
    _resolve_tmux_persist,
)
from factory.cli._path_resolver import (
    _materialize_project,
    _read_prompt_file,
    _resolve_focus_issues,
    _resolve_input,
)
from factory.cli._task_builder import _build_ceo_task

log = structlog.get_logger()


def _resolve_clean_pr(args: argparse.Namespace, project_path: Path) -> bool:
    """Resolve clean_pr flag from CLI args or project config."""
    clean_pr_flag = getattr(args, "clean_pr", None)
    if clean_pr_flag is not None:
        return clean_pr_flag
    config_path = project_path / ".factory" / "config.json"
    if config_path.exists():
        try:
            _cfg = json.loads(config_path.read_text())
            return bool(_cfg.get("clean_pr", False))
        except (json.JSONDecodeError, OSError):
            return False
    return False


def _run_single_cycle(
    project_path: Path,
    mode: str,
    context: str | None = None,
    focus: str | None = None,
    prompt_file: str | None = None,
    min_growth: int | None = None,
    max_new: int | None = None,
    branch: str | None = None,
    discover_only: bool = False,
    no_github: bool = False,
    model: str | None = None,
    issue_number: int | None = None,
    issue_url: str | None = None,
    issue_numbers: list[int] | None = None,
    issue_urls: list[str] | None = None,
    use_profile: bool = False,
    clean_pr: bool = False,
    tmux_persist: bool = False,
    background: bool = False,
    run_id: str | None = None,
    no_worktree: bool = False,
    overwrite: str | None = None,
    auto_approve: bool = False,
) -> int:
    """Execute a single factory run cycle via the CEO agent. Returns 0 on success, 1 on error."""
    from factory.agents.runner import invoke_agent
    from factory.worktree import create_worktree, remove_worktree

    if focus:
        from factory.study import add_backlog_item

        add_backlog_item(project_path, focus)

    from factory.messages import mark_read, read_pending

    pending = read_pending(project_path)
    pending_ids = [m.id for m in pending]

    base_branch = branch or _read_target_branch(project_path)
    if no_worktree:
        wt_path = project_path
        wt_branch = None
    else:
        wt_path, wt_branch = create_worktree(project_path, base_branch, run_id=run_id)

    from factory.skill_cache import ensure_skills

    ensure_skills(wt_path)

    if overwrite and mode and mode != "auto":
        from factory.workflow.definitions import register_all
        from factory.workflow.overwrite import apply_overwrite, generate_session_skill

        workflows = register_all()
        if mode in workflows:
            mutated = apply_overwrite(workflows[mode], overwrite, wt_path)
            generate_session_skill(mutated, mode, wt_path)

    try:
        task = _build_ceo_task(
            wt_path,
            mode,
            context,
            focus=focus,
            prompt_file=prompt_file,
            min_growth=min_growth,
            max_new=max_new,
            branch=base_branch,
            discover_only=discover_only,
            no_github=no_github,
            messages=pending,
            issue_number=issue_number,
            issue_url=issue_url,
            issue_numbers=issue_numbers,
            issue_urls=issue_urls,
            clean_pr=clean_pr,
            auto_approve=auto_approve,
        )

        result, code = _run(
            invoke_agent(
                "ceo",
                task,
                wt_path,
                timeout=_resolve_inactivity_timeout(),
                dangerously_skip_permissions=True,
                model=model,
                use_profile=use_profile,
                tmux_persist=tmux_persist,
                background=background,
                workflow_mode=mode,
            )
        )

        if code == 0:
            if pending_ids:
                mark_read(project_path, pending_ids)

        print(result)
        return code
    finally:
        if not no_worktree:
            assert wt_branch is not None
            remove_worktree(project_path, wt_path, wt_branch)


def _run_heartbeat_loop(
    project_path: Path,
    mode: str,
    context: str | None,
    focus: str | None,
    prompt_file: str | None,
    discover_only: bool,
    no_github: bool,
    model: str | None,
    issue_number: int | None,
    issue_url: str | None,
    issue_numbers: list[int] | None,
    issue_urls: list[str] | None,
    use_profile_flag: bool,
    clean_pr_resolved: bool,
    tmux_persist: bool,
    background: bool,
    run_id: str | None,
    budget_kwargs: dict,
    interval: int,
    max_cycles: int | None,
    no_worktree: bool = False,
) -> int:
    """Continuous heartbeat loop with signal handling."""
    shutdown_event = threading.Event()

    def _shutdown_handler(signum: int, frame: object) -> None:
        shutdown_event.set()

    old_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)
    old_sigint = signal.signal(signal.SIGINT, _shutdown_handler)

    cycle = 0
    start_time = time.monotonic()

    try:
        while True:
            cycle += 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[factory] Cycle {cycle} started at {ts}")
            _emit_cli_event(project_path, "cycle.started", {"cycle": cycle, "mode": mode})

            _run_single_cycle(
                project_path,
                mode,
                context,
                focus=focus,
                prompt_file=prompt_file,
                discover_only=discover_only,
                no_github=no_github,
                model=model,
                issue_number=issue_number,
                issue_url=issue_url,
                issue_numbers=issue_numbers,
                issue_urls=issue_urls,
                use_profile=use_profile_flag,
                clean_pr=clean_pr_resolved,
                tmux_persist=tmux_persist,
                background=background,
                run_id=run_id,
                no_worktree=no_worktree,
                **budget_kwargs,
            )
            _emit_cli_event(project_path, "cycle.completed", {"cycle": cycle, "mode": mode})

            mode = _auto_detect_mode(project_path, has_prompt=bool(prompt_file or context))

            if shutdown_event.is_set():
                break

            if max_cycles is not None and cycle >= max_cycles:
                break

            print(f"[factory] Cycle {cycle} completed. Sleeping for {interval}s...")

            shutdown_event.wait(interval)

            if shutdown_event.is_set():
                break
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)

    elapsed = time.monotonic() - start_time
    print(f"[factory] Shutting down gracefully after {cycle} cycles. Total runtime: {elapsed:.0f}s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run factory cycle(s) via the CEO agent. Supports single-shot and heartbeat loop."""
    from factory.user_config import load_config

    profile = getattr(args, "profile", None)
    load_config(profile=profile)

    project_path, context = _resolve_input(args.path)
    prompt_file = getattr(args, "prompt", None)
    loop = getattr(args, "loop", False)
    focus = getattr(args, "focus", None)
    discover_only = getattr(args, "discover_only", False)
    no_github = getattr(args, "no_github", False)
    if no_github:
        os.environ["FACTORY_NO_GITHUB"] = "1"
    min_growth = getattr(args, "min_growth", None)
    max_new = getattr(args, "max_new", None)
    branch = getattr(args, "branch", None)
    run_id = getattr(args, "run_id", None)
    model = _resolve_model(args)
    use_profile_flag = getattr(args, "use_profile", False)
    tmux_persist = _resolve_tmux_persist(args)
    background = _resolve_background(args)
    bg_agents = _resolve_bg_agents(args)
    if bg_agents:
        background = False
    if background and tmux_persist:
        print("Error: --bg and --tmux-persist are mutually exclusive.", file=sys.stderr)
        return 1
    if background and bg_agents:
        print("Error: --bg and --bg-agents are mutually exclusive.", file=sys.stderr)
        return 1

    if bg_agents:
        os.environ["FACTORY_BG"] = "1"

    if prompt_file:
        context = _read_prompt_file(project_path, prompt_file)
    issue_number: int | None = None
    issue_url: str | None = None
    issue_numbers: list[int] = []
    issue_urls: list[str] = []
    if focus:
        from factory.issue import has_multi_issue_refs

        if has_multi_issue_refs(focus) and no_github:
            print(
                "Error: --focus resolved to an issue reference, but --no-github is set. "
                "Issue fetching requires GitHub/GitLab CLI access.",
                file=sys.stderr,
            )
            return 1
        multi_resolved = _resolve_focus_issues(focus, project_path)
        if multi_resolved:
            if len(multi_resolved) == 1:
                title, context, issue_number, issue_url = multi_resolved[0]
                focus = f"{title} (issue #{issue_number})"
            else:
                parts = []
                for title, ctx, num, url in multi_resolved:
                    parts.append(f"{title} (issue #{num})")
                    issue_numbers.append(num)
                    issue_urls.append(url)
                focus = " + ".join(parts)
                context = None
    mode = getattr(args, "mode", "auto")
    warn_deprecated_mode(mode)
    auto_approve: bool = getattr(args, "auto_approve", False)
    if auto_approve and mode not in DESIGN_MODES:
        print("Error: --auto-approve only applies to --mode design or design-v2", file=sys.stderr)
        return 1
    force_fresh = mode == "auto-fresh"
    if mode in ("auto", "auto-fresh"):
        mode = _auto_detect_mode(
            project_path,
            has_prompt=bool(prompt_file or context),
            force_fresh=force_fresh,
        )

    if focus and loop:
        print(
            "Error: --focus (targeted mode) and --loop are mutually exclusive. "
            "Targeted mode builds exactly one item and exits.",
            file=sys.stderr,
        )
        return 1
    if focus and prompt_file:
        print(
            "Error: --focus (targeted mode) and --prompt are mutually exclusive. "
            "--focus builds one backlog item; --prompt executes a spec file.",
            file=sys.stderr,
        )
        return 1
    if focus and mode not in (*DESIGN_MODES, "research", "task-setup"):
        print(
            f"Error: --focus (targeted mode) only works in design, research, or task-setup mode, got '{mode}'. "
            "The project must already be built before targeting specific items.",
            file=sys.stderr,
        )
        return 1

    no_worktree = getattr(args, "no_worktree", False)
    clean_pr_resolved = _resolve_clean_pr(args, project_path)

    _print_banner(mode)
    _ensure_dashboard(project_path)

    if context is not None and not (project_path / ".git").is_dir():
        _materialize_project(project_path, context)

    from factory.worktree import prune_stale

    if project_path.is_dir():
        pruned = prune_stale(project_path)
        if pruned:
            print(f"  Cleaned {len(pruned)} stale worktree(s)", file=sys.stderr)

    budget_kwargs = dict(min_growth=min_growth, max_new=max_new, branch=branch)

    overwrite = getattr(args, "overwrite", None)

    if not loop:
        code = _run_single_cycle(
            project_path,
            mode,
            context,
            focus=focus,
            prompt_file=prompt_file,
            discover_only=discover_only,
            no_github=no_github,
            model=model,
            issue_number=issue_number,
            issue_url=issue_url,
            issue_numbers=issue_numbers,
            issue_urls=issue_urls,
            use_profile=use_profile_flag,
            clean_pr=clean_pr_resolved,
            tmux_persist=tmux_persist,
            background=background,
            run_id=run_id,
            no_worktree=no_worktree,
            overwrite=overwrite,
            auto_approve=auto_approve,
            **budget_kwargs,
        )
        return code

    interval: int = getattr(args, "interval", 1800)
    max_cycles: int | None = getattr(args, "max_cycles", None)
    return _run_heartbeat_loop(
        project_path=project_path,
        mode=mode,
        context=context,
        focus=focus,
        prompt_file=prompt_file,
        discover_only=discover_only,
        no_github=no_github,
        model=model,
        issue_number=issue_number,
        issue_url=issue_url,
        issue_numbers=issue_numbers,
        issue_urls=issue_urls,
        use_profile_flag=use_profile_flag,
        clean_pr_resolved=clean_pr_resolved,
        tmux_persist=tmux_persist,
        background=background,
        run_id=run_id,
        budget_kwargs=budget_kwargs,
        interval=interval,
        max_cycles=max_cycles,
        no_worktree=no_worktree,
    )
