"""CLI ceo commands — thin dispatcher delegating to extracted modules."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import structlog

from factory.cli._ceo_helpers import (
    _execute_ceo,
    _resolve_ceo_project,
    _validate_ceo_flags,
    _validate_late_flags,
)
from factory.cli._helpers import DESIGN_MODES
from factory.cli._mode_handlers import (
    _auto_detect_mode,
    handle_deep_qa_mode,
    handle_review_mode,
)
from factory.cli._path_resolver import _resolve_focus_issues


# ── subcommand handlers ──────────────────────────────────────


def cmd_ceo(args: argparse.Namespace) -> int:
    """Launch the Factory CEO agent to orchestrate a project."""
    from factory.user_config import load_config

    profile = getattr(args, "profile", None)
    load_config(profile=profile)

    raw_path: str | None = getattr(args, "path", None)

    validated = _validate_ceo_flags(args)
    if isinstance(validated, int):
        return validated
    mode, headless, bg, bg_agents, prompt_file, focus, dir_name, refine_request, auto_approve, from_plan, just_plan = validated

    from factory.plugins import get_registry

    _log = structlog.get_logger()
    registry = get_registry()
    for hook in registry.ceo_pre_hooks:
        try:
            override = hook(mode, args)
            if override is not None:
                raw_path = str(override)
                args.path = raw_path
        except Exception as exc:
            _log.warning("plugin_pre_hook_failed", error=str(exc))

    if raw_path is None:
        print(
            "Error: no project path provided and no plugin pre-hook supplied one.",
            file=sys.stderr,
        )
        return 1

    if mode == "review":
        return handle_review_mode(args, raw_path, headless)
    if mode == "deep-qa":
        return handle_deep_qa_mode(args, raw_path, headless)

    resolved = _resolve_ceo_project(raw_path, mode, headless, bg, focus, dir_name, prompt_file)
    if isinstance(resolved, int):
        return resolved
    (project_path, context, design_idea, research_ideation,
     deferred_spec, needs_materialize, design_existing, create_description,
     update_existing_mode) = resolved

    plugin_mode = getattr(args, "plugin", False)
    plugin_folder = getattr(args, "folder", None)
    data_node = getattr(args, "data_node", False)
    task_ref_raw: str | None = getattr(args, "task_ref", None)

    if plugin_mode and mode != "create":
        print(
            "Error: --plugin requires --mode create. "
            "Usage: factory ceo /path --mode create --focus 'my mode' --plugin",
            file=sys.stderr,
        )
        return 1

    if plugin_folder and not plugin_mode:
        print(
            "Warning: --folder is ignored without --plugin.",
            file=sys.stderr,
        )
        plugin_folder = None

    # --data-node requires --mode create or --mode create-v2
    if data_node and mode not in ("create", "create-v2"):
        print(
            "Error: --data-node requires --mode create or --mode create-v2. "
            "Usage: factory ceo /path --mode create --focus 'my pipeline' --data-node",
            file=sys.stderr,
        )
        return 1

    # --task also requires --mode create or --mode create-v2 (but NOT --data-node)
    if task_ref_raw and mode not in ("create", "create-v2"):
        print(
            "Error: --task requires --mode create or --mode create-v2. "
            "Usage: factory ceo /path --mode create --focus 'my pipeline' --task my-task",
            file=sys.stderr,
        )
        return 1

    # Resolve --task to extract context for directive injection
    task_context: dict | None = None
    if task_ref_raw:
        try:
            task_context = _resolve_task_for_directive(task_ref_raw, project_path)
        except (ImportError, KeyError, TypeError, ValueError) as exc:
            print(
                f"Error: --task {task_ref_raw!r} could not be resolved: {exc}",
                file=sys.stderr,
            )
            return 1

    no_github = getattr(args, "no_github", False)
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

    force_fresh = mode == "auto-fresh"
    if mode in ("auto", "auto-fresh"):
        mode = _auto_detect_mode(
            project_path,
            has_prompt=bool(prompt_file or context),
            force_fresh=force_fresh,
        )

    err = _validate_late_flags(
        mode, focus, prompt_file, research_ideation,
        design_existing, project_path, no_github, issue_number,
        just_plan=just_plan,
    )
    if err is not None:
        return err

    if design_existing:
        banner_mode = mode if mode == "design-v2" else "design"
    elif mode in (*DESIGN_MODES, "research") and (design_idea or research_ideation):
        banner_mode = "ideation"
    else:
        banner_mode = mode

    return _execute_ceo(
        args=args,
        project_path=project_path,
        context=context,
        mode=mode,
        banner_mode=banner_mode,
        headless=headless,
        bg=bg,
        bg_agents=bg_agents,
        focus=focus,
        prompt_file=prompt_file,
        design_idea=design_idea,
        design_existing=design_existing,
        research_ideation=research_ideation,
        create_description=create_description,
        update_existing_mode=update_existing_mode,
        plugin_mode=plugin_mode,
        plugin_folder=plugin_folder,
        data_node=data_node,
        task_ref=task_ref_raw,
        task_context=task_context,
        deferred_spec=deferred_spec,
        needs_materialize=needs_materialize,
        refine_request=refine_request,
        issue_number=issue_number,
        issue_url=issue_url,
        issue_numbers=issue_numbers,
        issue_urls=issue_urls,
        no_github=no_github,
        raw_path=raw_path,
        from_plan=from_plan,
        just_plan=just_plan,
    )


def _resolve_task_for_directive(
    task_ref_raw: str, project_path: Path
) -> dict:
    """Resolve a task reference and extract context for Builder directive injection."""
    from factory.task import TaskRef

    if ":" in task_ref_raw:
        # Qualified reference: module.path:ClassName
        task = TaskRef(ref=task_ref_raw).resolve()
        ref_string = task_ref_raw
    else:
        # Simple name: resolve via TaskRegistry
        from factory.task_registry import TaskRegistry

        task = TaskRegistry.load_task(task_ref_raw, project_path)
        ref_string = task_ref_raw

    # Extract context from the resolved Task
    defn = task.to_definition() if hasattr(task, "to_definition") else getattr(task, "_definition", None)

    # Get first instance for field names and sample prompt
    field_names: list[str] = []
    sample_prompt: str = ""
    try:
        instances = list(task.instances())
        if instances:
            first = instances[0]
            field_names = sorted(first.metadata.keys())
            sample_prompt = task.prompt(first)
    except Exception:
        pass  # Graceful degradation — directive works without sample data

    result: dict = {
        "name": defn.name if defn else task_ref_raw,
        "description": defn.description if defn else "",
        "field_names": field_names,
        "sample_prompt": sample_prompt,
        "ref_string": ref_string,
    }

    if defn:
        result["scoring_method"] = defn.scoring.method
        result["scoring_metric_path"] = defn.scoring.metric_path
        result["verify_command"] = defn.verify_config.command if defn.verify_config else ""
        result["timeout"] = defn.constraints.timeout if defn.constraints else 600

    return result


def cmd_refactory(args: argparse.Namespace) -> int:
    """Launch the re:factory persistent supervisor agent."""
    import shutil

    from factory.agents.runner import resolve_prompt
    from factory.refactory import get_session_id, setup_workspace
    from factory.user_config import load_config

    from factory.runners.claude import _claude_bin

    profile = getattr(args, "profile", None)
    if profile:
        load_config(profile=profile)

    bin_ = _claude_bin()
    claude_path = shutil.which(bin_)
    if not claude_path:
        print(f"Error: '{bin_}' CLI not found. Install Claude Code first.", file=sys.stderr)
        return 1

    project_path = Path(getattr(args, "path", None) or Path.cwd()).resolve()

    setup_workspace(project_path)

    loop = getattr(args, "loop", False)
    if loop:
        tune_skill_src = Path(__file__).parent.parent / "agents" / "skills" / "workflow-tune.md"
        if tune_skill_src.is_file():
            commands_dir = project_path / ".claude" / "commands"
            commands_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tune_skill_src, commands_dir / "workflow-tune.md")

    reset = getattr(args, "reset", False)
    session_file = project_path / ".refactory" / "session.json"
    is_new_session = reset or not session_file.exists()
    session_id = get_session_id(project_path, reset=reset)
    model = getattr(args, "model", None)

    prompt = resolve_prompt("refactory")
    prompt_tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="refactory-prompt-",
        delete=False,
    )
    prompt_tmp.write(prompt)
    prompt_tmp.close()

    if is_new_session:
        cmd = [
            bin_,
            "--session-id",
            session_id,
            "--append-system-prompt-file",
            prompt_tmp.name,
            "--disallowedTools",
            "Agent",
            "--dangerously-skip-permissions",
        ]
    else:
        cmd = [
            bin_,
            "--resume",
            session_id,
            "--append-system-prompt-file",
            prompt_tmp.name,
            "--disallowedTools",
            "Agent",
            "--dangerously-skip-permissions",
        ]

    if model:
        cmd.extend(["--model", model])

    mcp_config = project_path / ".refactory" / ".mcp.json"
    if mcp_config.exists():
        cmd.extend(["--mcp-config", str(mcp_config), "--strict-mcp-config"])

    os.chdir(project_path)
    os.execvp(bin_, cmd)
    return 0
