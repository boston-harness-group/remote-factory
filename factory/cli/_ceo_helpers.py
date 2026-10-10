"""CEO flag validation, project resolution, and execution logic."""

from __future__ import annotations

import argparse
import json
import os
import re
import structlog
import sys
import time
from pathlib import Path

from factory.cli._ceo_dispatch import _start_ceo_tailer, _stop_ceo_tailer
from factory.cli._helpers import (
    DESIGN_MODES,
    _emit_cli_event,
    _ensure_dashboard,
    get_all_ceo_modes,
    _print_banner,
    _read_target_branch,
    _resolve_inactivity_timeout,
    _resolve_runner,
    _run,
    _safe_is_dir,
    _safe_is_file,
    warn_deprecated_mode,
)
from factory.cli._mode_handlers import (
    _resolve_background,
    _resolve_bg_agents,
    _resolve_model,
    _resolve_tmux_persist,
)
from factory.cli._path_resolver import (
    PlanSource,
    _dedupe_project_path,
    _derive_session_name,
    _extract_project_name,
    _get_projects_dir,
    _has_research_target,
    _is_scaffold_only,
    _materialize_project,
    _read_prompt_file,
    _resolve_input,
    _resolve_plan_source,
    _slugify,
)
from factory.cli._task_builder import _build_ceo_task

log = structlog.get_logger()


def _tool_exec_protocol(wt_path: Path) -> str:
    """Return the tool-exec protocol section appended to the CEO prompt."""
    p = wt_path

    overview = ""
    try:
        from factory.workflow.tool import tool_overview

        overview = tool_overview(p, fmt="linear")
    except Exception:
        pass

    protocol = (
        "\n\n# Tool-Based Execution Protocol\n"
        "\n"
        "You are executing the workflow using factory tool commands instead of "
        "following a SKILL.md playbook.\n"
    )

    if overview:
        protocol += f"\n## Workflow Map\n\n{overview}\n"

    protocol += (
        "\n## Commands\n"
        "\n"
        f"  factory workflow tool next {p}\n"
        f"  factory workflow tool submit {p} --node <NODE_ID> <<'TOOL_OUTPUT'\n"
        "  <your output>\n"
        "  TOOL_OUTPUT\n"
        f"  factory workflow tool status {p}\n"
        f"  factory workflow tool curr {p}\n"
        "\n"
        "## Protocol\n"
        "\n"
        '1. Run "next" to see your current task — it tells you the node type, '
        "role, and what to do\n"
        "2. Execute the task:\n"
        '   - Agent nodes: run factory agent <role> --task "..." --project <path>\n'
        "   - Study nodes: run the study command shown\n"
        "   - Function nodes: run the command shown\n"
        '3. Run "next" again — the tool auto-detects that the previous node completed\n'
        "   (by checking for output files) and advances to the next task\n"
        "4. Repeat until GATE or DONE\n"
        "5. For GATE nodes: the tool asks you to evaluate — read the artifacts, then\n"
        '   call "submit" with your verdict (PROCEED, RETRY, or HALT)\n'
        '6. If RETRY: the tool rewinds — run "next" to get the retry task\n'
        "7. If DONE: report completion\n"
        "\n"
        "## Important\n"
        "\n"
        '- For most nodes, just run the command and call "next" — the tool handles tracking\n'
        '- Only call "submit" for gate verdicts (PROCEED/RETRY/HALT)\n'
        "- The tool auto-detects agent completion via .factory/reviews/ files\n"
        "- The tool auto-evaluates fn gates (precheck, guard) on your behalf\n"
        "- All Sacred Rules still apply — delegate to agents, review output, "
        "do not write code\n"
        '- Start by running "next" to get your first task\n'
        "\n"
        "## Loop Context\n"
        "\n"
        "For any node that is a RELOOP target in the workflow graph, the tool "
        "engine automatically injects a **## LOOP CONTEXT** section into the "
        "node's task description — starting from the very first invocation "
        "(iteration 0). This section shows:\n"
        "- The full loop topology (all nodes from this node through the gate) "
        "with reads/writes\n"
        "- The gate's criteria and evaluator command\n"
        "- The current iteration count (e.g. 0/3 on first pass, 1/3 after first reloop)\n"
        "\n"
        "After a RELOOP occurs, the section also includes:\n"
        "- Which gate triggered the reloop\n"
        "- Feedback history from prior iterations (last 2, truncated to 500 chars)\n"
        "\n"
        "Incorporate gate criteria from the LOOP CONTEXT section into your agent "
        "task prompts. When spawning a builder agent, include what downstream "
        "gates will check (e.g. health check criteria, code review expectations, "
        "QA scope) so the builder can proactively address them. This reduces "
        "reloops by making the builder aware of review criteria upfront.\n"
        "\n"
        "No separate command is needed — context is injected automatically by "
        "the tool engine.\n"
    )

    return protocol


# ── flag validation ───────────────────────────────────────────


def _validate_ceo_flags(
    args: argparse.Namespace,
) -> (
    tuple[
        str,
        bool,
        bool,
        bool,
        str | None,
        str | None,
        str | None,
        str | None,
        bool,
        str | None,
        bool,
    ]
    | int
):
    """Validate and resolve top-level CLI flags. Returns parsed values or an error code."""
    mode: str = getattr(args, "mode", "auto")
    if mode == "interactive":
        mode = "design"
    if mode.startswith("project:"):
        mode = mode[len("project:") :]
    all_modes = get_all_ceo_modes()
    if mode not in all_modes and mode != "auto":
        from factory.workflow.registry import WorkflowRegistry

        raw_path = getattr(args, "path", None)
        project_path = Path(raw_path).resolve() if raw_path else Path.cwd()
        entries = WorkflowRegistry.discover(project_path)
        all_workflow_entries = set(entries.keys())
        if mode not in all_workflow_entries:
            available = sorted(set(all_modes) | all_workflow_entries)
            print(
                f"Error: unknown mode '{mode}'. "
                f"Available modes: {', '.join(available)}",
                file=sys.stderr,
            )
            return 1
    warn_deprecated_mode(getattr(args, "mode", "auto"))
    bg: bool = getattr(args, "bg", False)
    bg_agents = _resolve_bg_agents(args)
    if bg and bg_agents:
        print("Error: --bg and --bg-agents are mutually exclusive.", file=sys.stderr)
        return 1
    headless: bool = getattr(args, "headless", False) or bg
    prompt_file: str | None = getattr(args, "prompt", None)
    focus: str | None = getattr(args, "focus", None)
    dir_name: str | None = getattr(args, "dir", None)
    auto_approve: bool = getattr(args, "auto_approve", False)
    from_plan: str | None = getattr(args, "from_plan", None)
    just_plan: bool = getattr(args, "just_plan", False)

    if auto_approve and mode not in DESIGN_MODES:
        print("Error: --auto-approve only applies to --mode design or design-v2", file=sys.stderr)
        return 1

    if just_plan:
        if mode not in DESIGN_MODES:
            print("Error: --just-plan requires --mode design or design-v2", file=sys.stderr)
            return 1
        if from_plan:
            print("Error: --just-plan and --from-plan are mutually exclusive.", file=sys.stderr)
            return 1
        if prompt_file:
            print("Error: --just-plan and --prompt are mutually exclusive.", file=sys.stderr)
            return 1

    if from_plan:
        if mode not in DESIGN_MODES:
            print("Error: --from-plan requires --mode design or design-v2", file=sys.stderr)
            return 1
        if focus:
            print("Error: --from-plan and --focus are mutually exclusive.", file=sys.stderr)
            return 1
        if prompt_file:
            print("Error: --from-plan and --prompt are mutually exclusive.", file=sys.stderr)
            return 1

    raw_path = getattr(args, "path", None)
    if not raw_path:
        from factory.plugins import get_registry

        plugin_registry = get_registry()
        has_pre_hooks = bool(plugin_registry.ceo_pre_hooks)
        if not has_pre_hooks:
            print(
                "Error: provide a project path, GitHub URL, idea file, or prompt",
                file=sys.stderr,
            )
            return 1

    no_github = getattr(args, "no_github", False)
    if no_github:
        os.environ["FACTORY_NO_GITHUB"] = "1"
    refine_request: str | None = getattr(args, "refine", None)

    if refine_request:
        if mode and mode != "auto":
            print(f"Error: --refine and --mode {mode} are mutually exclusive.", file=sys.stderr)
            return 1
        if prompt_file:
            print("Error: --refine and --prompt are mutually exclusive.", file=sys.stderr)
            return 1
        if focus:
            print("Error: --refine and --focus are mutually exclusive.", file=sys.stderr)
            return 1
        if not raw_path or not Path(raw_path).expanduser().resolve().is_dir():
            print(
                "Error: --refine requires an existing project directory, not a URL or idea.",
                file=sys.stderr,
            )
            return 1

    _design_is_existing = (
        mode in DESIGN_MODES
        and raw_path
        and _safe_is_dir(Path(raw_path).expanduser().resolve())
    )

    if mode in DESIGN_MODES:
        if auto_approve:
            headless = True
        elif headless:
            flag = "--bg" if bg else "--headless"
            print(
                f"Error: --mode design/design-v2 requires foreground mode (incompatible with {flag})",
                file=sys.stderr,
            )
            return 1
        if prompt_file:
            print(
                "Error: --mode design/design-v2 and --prompt are mutually exclusive. "
                "Design mode generates the spec; --prompt provides one.",
                file=sys.stderr,
            )
            return 1
        if focus and not _design_is_existing and not just_plan:
            print(
                "Error: --mode design/design-v2 and --focus are mutually exclusive "
                "for new ideas. To discuss a topic on an existing project, "
                'pass the project path: factory ceo /path --mode design --focus "topic"',
                file=sys.stderr,
            )
            return 1

    if mode == "create":
        if headless:
            flag = "--bg" if bg else "--headless"
            print(
                f"Error: --mode create requires foreground mode (incompatible with {flag})",
                file=sys.stderr,
            )
            return 1
        if prompt_file:
            print(
                "Error: --mode create and --prompt are mutually exclusive. "
                "Create mode generates the workflow from a description.",
                file=sys.stderr,
            )
            return 1

    if mode == "research" and prompt_file:
        print(
            "Error: --mode research and --prompt are mutually exclusive. "
            "Research ideation generates the spec; --prompt provides one.",
            file=sys.stderr,
        )
        return 1

    return (
        mode,
        headless,
        bg,
        bg_agents,
        prompt_file,
        focus,
        dir_name,
        refine_request,
        auto_approve,
        from_plan,
        just_plan,
    )


# ── project resolution ────────────────────────────────────────


def _resolve_ceo_project(
    raw_path: str,
    mode: str,
    headless: bool,
    bg: bool,
    focus: str | None,
    dir_name: str | None,
    prompt_file: str | None,
) -> (
    tuple[Path, str | None, str | None, str | None, str | None, bool, bool, str | None, str | None]
    | int
):
    """Resolve the project path and mode-specific context.

    Returns (project_path, context, design_idea, research_ideation, deferred_spec,
             needs_materialize, design_existing, create_description,
             update_existing_mode) or error code.
    """
    create_description: str | None = None
    update_existing_mode: str | None = None
    design_idea: str | None = None
    design_existing: bool = False
    research_ideation: str | None = None
    deferred_spec: str | None = None
    needs_materialize = False
    context: str | None = None

    _design_is_existing = (
        mode in DESIGN_MODES
        and raw_path
        and _safe_is_dir(Path(raw_path).expanduser().resolve())
    )

    if mode == "create":
        resolved_path = Path(raw_path).expanduser().resolve()
        if not _safe_is_dir(resolved_path):
            print(
                "Error: --mode create requires an existing project directory. "
                "Pass the factory project path: factory ceo /path/to/factory --mode create",
                file=sys.stderr,
            )
            return 1
        project_path, context = _resolve_input(raw_path, dir_name=dir_name)
        create_description = focus if focus else context
        if create_description and ":" in create_description:
            m = re.match(r"^([a-z_-]+):\s*(.+)$", create_description, re.DOTALL)
            if m:
                from factory.workflow.definitions import register_all

                registered = register_all()
                if m.group(1) in registered:
                    update_existing_mode = m.group(1)
                    create_description = m.group(2).strip()
    elif mode in DESIGN_MODES and _design_is_existing:
        project_path, context = _resolve_input(raw_path, dir_name=dir_name)
        design_existing = True
    elif mode in DESIGN_MODES:
        resolved_file = Path(raw_path).expanduser()
        if _safe_is_file(resolved_file):
            design_idea = resolved_file.read_text()
            slug = (
                _slugify(dir_name)
                if dir_name
                else _slugify(resolved_file.stem.split("—")[0].strip())
            )
            project_path = _dedupe_project_path(_get_projects_dir() / slug, design_idea)
            deferred_spec = design_idea
            needs_materialize = True
            print(f"Idea file: {resolved_file.name}")
            print(f"Project directory: {project_path}")
        else:
            design_idea = raw_path
            slug = _slugify(dir_name) if dir_name else _extract_project_name(raw_path)
            project_path = _dedupe_project_path(_get_projects_dir() / slug, raw_path)
            deferred_spec = raw_path
            needs_materialize = True
        context = None
    elif (
        mode == "research"
        and not _safe_is_dir(resolved := Path(raw_path).expanduser())
        and not _safe_is_file(resolved)
    ):
        if headless:
            flag = "--bg" if bg else "--headless"
            print(
                "Error: --mode research for new projects requires foreground mode "
                f"(incompatible with {flag})",
                file=sys.stderr,
            )
            return 1
        if focus:
            print(
                "Error: --focus cannot be used with research ideation for new projects. "
                "--focus targets existing backlog items.",
                file=sys.stderr,
            )
            return 1
        research_ideation = raw_path
        slug = _slugify(dir_name) if dir_name else _extract_project_name(raw_path)
        project_path = _dedupe_project_path(_get_projects_dir() / slug, raw_path)
        needs_materialize = True
        context = None
    else:
        project_path, context = _resolve_input(raw_path, dir_name=dir_name)
        if context is not None and not (project_path / ".git").is_dir():
            deferred_spec = context
            needs_materialize = True

    if prompt_file:
        context = _read_prompt_file(project_path, prompt_file)

    return (
        project_path,
        context,
        design_idea,
        research_ideation,
        deferred_spec,
        needs_materialize,
        design_existing,
        create_description,
        update_existing_mode,
    )


# ── late validation ───────────────────────────────────────────


def _validate_late_flags(
    mode: str,
    focus: str | None,
    prompt_file: str | None,
    research_ideation: str | None,
    design_existing: bool,
    project_path: Path,
    no_github: bool,
    issue_number: int | None,
    just_plan: bool = False,
) -> int | None:
    """Run validations that depend on resolved project state. Returns error code or None."""
    if mode == "research" and not research_ideation and not _has_research_target(project_path):
        print(
            "Error: --mode research requires research_target in factory.md. "
            "Either configure research_target manually, or pass an idea string "
            'to start research ideation: factory ceo "your idea" --mode research',
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

    if (
        focus
        and mode
        not in (
            "design",
            "design-v2",
            "research",
            "create",
            "create-v2",
            "evolve",
            "study",
            "frontend-design",
            "frontend-design-discover",
            "task-setup",
        )
        and not design_existing
        and not just_plan
    ):
        print(
            f"Error: --focus (targeted mode) only works in design, research, create, create-v2, evolve, study, frontend-design, "
            f"frontend-design-discover, task-setup, or design (with --just-plan) mode, "
            f"got '{mode}'. The project must already be built before targeting specific items.",
            file=sys.stderr,
        )
        return 1

    return None


# ── execution ─────────────────────────────────────────────────


def _execute_ceo(
    *,
    args: argparse.Namespace,
    project_path: Path,
    context: str | None,
    mode: str,
    banner_mode: str,
    headless: bool,
    bg: bool,
    bg_agents: bool,
    focus: str | None,
    prompt_file: str | None,
    design_idea: str | None,
    design_existing: bool,
    research_ideation: str | None,
    create_description: str | None,
    update_existing_mode: str | None,
    plugin_mode: bool = False,
    plugin_folder: str | None = None,
    data_node: bool = False,
    task_ref: str | None = None,
    task_context: dict | None = None,
    deferred_spec: str | None,
    needs_materialize: bool,
    refine_request: str | None,
    issue_number: int | None,
    issue_url: str | None,
    issue_numbers: list[int] | None = None,
    issue_urls: list[str] | None = None,
    no_github: bool = False,
    raw_path: str = "",
    from_plan: str | None = None,
    just_plan: bool = False,
) -> int:
    """Set up worktree, build task, and run the CEO agent."""
    from factory.agents.runner import begin_cycle_session, complete_cycle_session, resolve_prompt, resolve_prompt_core
    from factory.runners import get_runner
    from factory.runners.claude import _make_ceo_message_emitter
    from factory.worktree import create_worktree, prune_stale, remove_worktree

    discover_only = getattr(args, "discover_only", False)
    min_growth = getattr(args, "min_growth", None)
    max_new = getattr(args, "max_new", None)
    branch = getattr(args, "branch", None)
    run_id = getattr(args, "run_id", None)
    model = _resolve_model(args)
    runner_name = _resolve_runner(args)
    use_profile = getattr(args, "use_profile", False)
    tmux_persist = _resolve_tmux_persist(args)
    background = _resolve_background(args)
    if bg_agents:
        background = False
    if background and tmux_persist:
        print("Error: --bg and --tmux-persist are mutually exclusive.", file=sys.stderr)
        return 1
    clean_pr_flag = getattr(args, "clean_pr", None)
    no_worktree = getattr(args, "no_worktree", False)

    _print_banner(banner_mode)
    _ensure_dashboard(project_path)

    if needs_materialize:
        _materialize_project(project_path, deferred_spec)

    pruned = prune_stale(project_path)
    if pruned:
        print(f"  Cleaned {len(pruned)} stale worktree(s)", file=sys.stderr)

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

    auto_approve = getattr(args, "auto_approve", False)
    if auto_approve:
        _emit_cli_event(wt_path, "auto_approve.enabled", {"mode": mode})

    resolved_plan: PlanSource | None = None
    if from_plan:
        resolved_plan = _resolve_plan_source(from_plan, project_path)
        strategy_dir = wt_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "current.md").write_text(resolved_plan.plan)
        if resolved_plan.feedback:
            feedback_text = "\n\n---\n\n".join(resolved_plan.feedback)
            (strategy_dir / "thread-feedback.md").write_text(feedback_text)

    engine = getattr(args, "engine", "skill")

    if engine != "tool":
        from factory.skill_cache import ensure_skills

        ensure_skills(wt_path, mode=mode)

    from factory.graph import extract_graph, is_graphify_installed

    if is_graphify_installed():
        extract_graph(wt_path)

    overwrite = getattr(args, "overwrite", None)
    if overwrite and mode and mode != "auto":
        from factory.workflow.definitions import register_all
        from factory.workflow.overwrite import apply_overwrite, generate_session_skill

        workflows = register_all()
        if mode in workflows:
            mutated = apply_overwrite(workflows[mode], overwrite, wt_path)
            generate_session_skill(mutated, mode, wt_path)
        else:
            log.warning("overwrite.mode_not_found", mode=mode)

    verification_settings = wt_path / ".factory" / "hooks" / f"settings-{mode}.json"
    _verification_settings_file = (
        str(verification_settings) if verification_settings.exists() else None
    )

    interactive = (
        design_existing or bool(design_idea) or bool(research_ideation) or mode == "create"
    )
    if mode == "create":
        ceo_mode = "create"
    elif mode in DESIGN_MODES:
        ceo_mode = mode
    elif interactive:
        ceo_mode = "design"
    else:
        ceo_mode = mode

    headless_prompt_override: str | None = None
    if engine == "tool" and headless:
        base = resolve_prompt("ceo", wt_path, use_profile=use_profile, workflow_mode=None)
        headless_prompt_override = base + _tool_exec_protocol(wt_path)

    if engine == "deterministic":
        if not headless:
            print(
                "WARNING: --engine deterministic runs headless (no interactive CEO). "
                "Adding --headless implicitly.",
                file=sys.stderr,
            )
            headless = True

    if engine == "tool":
        from factory.workflow.tool import tool_init as _tool_init

        try:
            _tool_init(ceo_mode, wt_path)
        except Exception as e:
            log.warning("tool_exec.init_failed", error=str(e), mode=ceo_mode)
            engine = "skill"

    if clean_pr_flag is not None:
        clean_pr_resolved = clean_pr_flag
    else:
        config_path = project_path / ".factory" / "config.json"
        if config_path.exists():
            try:
                _cfg = json.loads(config_path.read_text())
                clean_pr_resolved = bool(_cfg.get("clean_pr", False))
            except (json.JSONDecodeError, OSError):
                clean_pr_resolved = False
        else:
            clean_pr_resolved = False

    task = _build_ceo_task(
        wt_path,
        ceo_mode,
        context,
        focus=focus,
        prompt_file=prompt_file,
        min_growth=min_growth,
        max_new=max_new,
        branch=base_branch,
        discover_only=discover_only,
        no_github=no_github,
        design_idea=design_idea,
        design_existing=design_existing,
        research_ideation=research_ideation,
        messages=pending,
        issue_number=issue_number,
        issue_url=issue_url,
        issue_numbers=issue_numbers,
        issue_urls=issue_urls,
        refine_request=refine_request,
        clean_pr=clean_pr_resolved,
        display_mode=banner_mode,
        create_description=create_description,
        update_existing_mode=update_existing_mode,
        plugin_mode=plugin_mode,
        plugin_folder=plugin_folder,
        data_node=data_node,
        task_ref=task_ref,
        task_context=task_context,
        from_plan=resolved_plan.plan if resolved_plan else None,
        from_plan_feedback=resolved_plan.feedback if resolved_plan else None,
        just_plan=just_plan,
        auto_approve=auto_approve,
    )

    session_name = _derive_session_name(
        focus=focus,
        design_idea=design_idea,
        research_ideation=research_ideation,
        raw_path=raw_path,
        project_path=project_path,
        mode=banner_mode,
    )

    if bg_agents:
        os.environ["FACTORY_BG"] = "1"

    cycle_span_id = begin_cycle_session(project_path, cycle_id=mode, model=model)
    _ceo_start = time.time()

    ceo_tailer = _start_ceo_tailer(
        wt_path,
        cycle_span_id,
        _ceo_start,
        on_line=_make_ceo_message_emitter(wt_path),
        is_headless=headless,
    )

    import uuid as _uuid

    from factory.ceo_completion import write_ceo_session_id

    ceo_session_id = str(_uuid.uuid4())
    write_ceo_session_id(wt_path, ceo_session_id, interactive=interactive, mode=mode)

    if headless:
        return _run_headless(
            wt_path=wt_path,
            project_path=project_path,
            task=task,
            mode=mode,
            runner_name=runner_name,
            model=model,
            session_name=session_name,
            ceo_session_id=ceo_session_id,
            use_profile=use_profile,
            tmux_persist=tmux_persist,
            background=background,
            ceo_tailer=ceo_tailer,
            cycle_span_id=cycle_span_id,
            pending_ids=pending_ids,
            needs_materialize=needs_materialize,
            wt_branch=wt_branch,
            no_worktree=no_worktree,
            ceo_mode=ceo_mode,
            verification_settings_file=_verification_settings_file,
            engine=engine,
            prompt_override=headless_prompt_override,
        )

    try:
        if pending_ids:
            print(
                f"Consuming {len(pending_ids)} message(s): {', '.join(pending_ids)}",
                file=sys.stderr,
            )
            mark_read(project_path, pending_ids)
        from factory.models import AgentRunRequest as _RunReq

        if engine == "tool":
            base_prompt = resolve_prompt(
                "ceo",
                wt_path,
                use_profile=use_profile,
                workflow_mode=None,
            )
            prompt = base_prompt + _tool_exec_protocol(wt_path)
        else:
            prompt = resolve_prompt(
                "ceo",
                wt_path,
                use_profile=use_profile,
                workflow_mode=ceo_mode,
            )
        runner = get_runner(runner_name)
        extras: dict[str, object] = {}
        if _verification_settings_file:
            extras["settings_file"] = _verification_settings_file
        prompt_core = resolve_prompt_core()
        return runner.interactive_run(
            _RunReq(
                prompt=prompt,
                prompt_core=prompt_core,
                task=task,
                cwd=wt_path,
                model=model,
                role="ceo",
                skip_permissions=True,
                session_name=session_name,
                session_id=ceo_session_id,
                extras=extras,
            )
        )
    finally:
        if engine == "tool":
            try:
                from factory.workflow.tool import tool_finalize

                finalize_result = tool_finalize(wt_path)
                log.info("tool_exec.finalized", result=finalize_result)
            except Exception:
                pass
        _stop_ceo_tailer(ceo_tailer)
        complete_cycle_session(project_path, cycle_span_id)
        from factory.ceo_completion import print_resume_hint

        print_resume_hint(project_path)
        if not no_worktree:
            assert wt_branch is not None
            remove_worktree(project_path, wt_path, wt_branch)
        if needs_materialize and _is_scaffold_only(project_path):
            import shutil

            shutil.rmtree(project_path, ignore_errors=True)


def _run_headless(
    *,
    wt_path: Path,
    project_path: Path,
    task: str,
    mode: str,
    runner_name: str | None,
    model: str | None,
    session_name: str,
    ceo_session_id: str,
    use_profile: bool,
    tmux_persist: bool,
    background: bool,
    ceo_tailer: object,
    cycle_span_id: str | None,
    pending_ids: list[str],
    needs_materialize: bool,
    wt_branch: str | None,
    no_worktree: bool,
    ceo_mode: str,
    verification_settings_file: str | None,
    engine: str = "skill",
    prompt_override: str | None = None,
) -> int:
    """Run the CEO in headless mode with completion guard."""
    from factory.ceo_completion import run_ceo_with_completion_guard
    from factory.messages import mark_read
    from factory.agents.runner import complete_cycle_session
    from factory.worktree import remove_worktree

    if engine == "deterministic":
        import asyncio
        from factory.workflow.executor import WorkflowExecutor
        from factory.workflow.registry import WorkflowRegistry
        from factory.workflow.primitives import DEFAULT_AGENT_POOL

        wf = WorkflowRegistry.get_workflow(ceo_mode, wt_path)
        if not wf:
            print(f'Error: workflow "{ceo_mode}" not found', file=sys.stderr)
            _stop_ceo_tailer(ceo_tailer)
            complete_cycle_session(project_path, cycle_span_id)
            return 1

        executor = WorkflowExecutor(wf, wt_path, agent_pool=DEFAULT_AGENT_POOL)
        try:
            exec_result = asyncio.run(executor.execute())
            print(
                json.dumps(
                    {
                        "workflow": ceo_mode,
                        "engine": "deterministic",
                        "success": exec_result.success,
                        "nodes_executed": exec_result.nodes_executed,
                        "duration_ms": round(exec_result.duration_ms, 1),
                    },
                    indent=2,
                )
            )
            return 0 if exec_result.success else 1
        finally:
            _stop_ceo_tailer(ceo_tailer)
            complete_cycle_session(project_path, cycle_span_id)
            from factory.ceo_completion import print_resume_hint

            print_resume_hint(project_path)
            if not no_worktree and wt_branch:
                remove_worktree(project_path, wt_path, wt_branch)
            if needs_materialize and _is_scaffold_only(project_path):
                import shutil

                shutil.rmtree(project_path, ignore_errors=True)

    try:
        result, code = _run(
            run_ceo_with_completion_guard(
                wt_path,
                task,
                mode=mode,
                runner_name=runner_name,
                model=model,
                timeout=_resolve_inactivity_timeout(),
                session_name=session_name,
                session_id=ceo_session_id,
                use_profile=use_profile,
                tmux_persist=tmux_persist,
                background=background,
                workflow_mode=ceo_mode,
                settings_file=verification_settings_file,
                prompt_override=prompt_override,
            )
        )
        print(result)
        if code == 0 and pending_ids:
            mark_read(project_path, pending_ids)
        return code
    finally:
        if engine == "tool":
            try:
                from factory.workflow.tool import tool_finalize

                tool_finalize(wt_path)
            except Exception:
                pass
        _stop_ceo_tailer(ceo_tailer)
        complete_cycle_session(project_path, cycle_span_id)
        from factory.ceo_completion import print_resume_hint

        print_resume_hint(project_path)
        if not no_worktree:
            assert wt_branch is not None
            remove_worktree(project_path, wt_path, wt_branch)
        if needs_materialize and _is_scaffold_only(project_path):
            import shutil

            shutil.rmtree(project_path, ignore_errors=True)
