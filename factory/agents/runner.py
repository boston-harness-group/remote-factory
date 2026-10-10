"""Agent runner — load prompts and invoke Claude Code instances."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from factory.ace.injector import inject_playbook, load_playbook
from factory.runners import get_runner

logger = logging.getLogger(__name__)

AgentRole = str

# Consecutive failure tracking
_consecutive_failures: int = 0
_FAILURE_ABORT_THRESHOLD: int = 2


class ConsecutiveAgentFailureError(Exception):
    """Raised when too many consecutive agent spawns fail.

    This prevents the CEO from falling back to doing work itself when subagent
    infrastructure is broken. Instead, the cycle should abort with a clear error.
    """

    def __init__(self, failure_count: int, last_agent: str) -> None:
        self.failure_count = failure_count
        self.last_agent = last_agent
        super().__init__(
            f"Aborting after {failure_count} consecutive agent spawn failures. "
            f"Last failed agent: {last_agent}. "
            "Check .factory/events.jsonl for details. "
            "This usually means runner authentication is not configured correctly."
        )


IDENTITY_REANCHOR = """\

---

> **⚠ CEO IDENTITY RE-ANCHOR (Sacred Rule 8)**
> You are the Factory CEO. You orchestrate, delegate, and decide. You do NOT implement.
> If you are about to write code, run tests, do research, or fix bugs — STOP and spawn the appropriate agent.
> Re-read your Permitted/Forbidden Actions lists in the Identity section above.
"""

# Directory containing base agent prompts (shipped with the factory)
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_USER_PROMPTS_DIR = Path.home() / ".factory" / "agents" / "prompts"


def _is_plugin_role(role: str) -> bool:
    """True if the role was registered by a plugin (not a builtin role)."""
    try:
        from factory.plugins import get_registry

        return role in get_registry().agent_roles
    except Exception:
        return False


def resolve_prompt(
    role: AgentRole,
    project_path: Path | None = None,
    *,
    use_profile: bool = False,
    workflow_mode: str | None = None,
) -> str:
    """Resolve the prompt for an agent role.

    Resolution order:
    1. Project-specific override: <project>/.factory/agents/<role>.md
    2. User-global: ~/.factory/agents/prompts/<role>.md
    3. Factory default: factory/agents/prompts/<role>.md

    When *use_profile* is True, loads ~/.factory/profile.md and appends it
    after the ACE playbook injection.

    When *workflow_mode* is set and *role* is ``"ceo"``, the corresponding
    ``skills/workflow-{workflow_mode}/SKILL.md`` is appended to the prompt
    so it survives context compaction.

    Returns the prompt content as a string.
    """
    # Check for project-specific override
    if project_path is not None:
        override_path = project_path / ".factory" / "agents" / f"{role}.md"
        if override_path.exists():
            logger.info("Using project-specific prompt for %s: %s", role, override_path)
            prompt = override_path.read_text()
            # Auto-inject evolved playbook even with project overrides
            playbook = load_playbook(role)
            if playbook:
                prompt = inject_playbook(prompt, playbook)
                logger.info("Injected playbook for %s (project override)", role)
            if use_profile:
                prompt = _maybe_inject_profile(prompt, role)
            if role == "ceo" and workflow_mode and project_path is not None:
                prompt = _maybe_inject_skill(prompt, project_path, workflow_mode)
            return prompt

    # Check user-global prompts (~/.factory/agents/prompts/)
    user_path = _USER_PROMPTS_DIR / f"{role}.md"
    if user_path.exists():
        logger.info("Using user-global prompt for %s: %s", role, user_path)
        prompt = user_path.read_text()
        playbook = load_playbook(role)
        if playbook:
            prompt = inject_playbook(prompt, playbook)
            logger.info("Injected playbook for %s (user-global)", role)
        if use_profile:
            prompt = _maybe_inject_profile(prompt, role)
        if role == "ceo" and workflow_mode and project_path is not None:
            prompt = _maybe_inject_skill(prompt, project_path, workflow_mode)
        return prompt

    # Fall back to factory default
    default_path = _PROMPTS_DIR / f"{role}.md"
    if not default_path.exists():
        override_hint = (
            f" or {project_path / '.factory' / 'agents' / f'{role}.md'}" if project_path else ""
        )
        plugin_hint = ""
        if _is_plugin_role(role):
            plugin_hint = (
                f" Role '{role}' is plugin-registered; its prompt file likely did "
                "not ship with the plugin (check the built wheel's data files)."
            )
        raise FileNotFoundError(
            f"No prompt found for agent role '{role}'. "
            f"Expected at {default_path}, {_USER_PROMPTS_DIR / f'{role}.md'}{override_hint}"
            f"{plugin_hint}"
        )

    prompt = default_path.read_text()

    # Auto-inject evolved playbook if one exists for this role
    playbook = load_playbook(role)
    if playbook:
        prompt = inject_playbook(prompt, playbook)
        logger.info("Injected playbook for %s", role)

    if use_profile:
        prompt = _maybe_inject_profile(prompt, role)

    if role == "ceo" and workflow_mode and project_path is not None:
        prompt = _maybe_inject_skill(prompt, project_path, workflow_mode)

    return prompt


_PROMPT_CORE_TEMPLATE = """\
# Factory CEO Agent — Resume Identity

You ARE the Factory CEO — the executive orchestrator of the Software Factory. \
You delegate ALL technical work to specialist agents and review their output. \
You own the experiment lifecycle: `factory begin`, dispatch agents, `factory finalize`.

## Agent Dispatch

```bash
factory agent <role> --task "<description>" --project /path [--timeout 600]
```

Roles: researcher, strategist, builder, health_checker, code_reviewer, adversarial_tester, archivist.

## Permitted Actions

- `factory agent <role>` — spawn specialist agents
- `factory <cmd>` — CLI commands (`factory --help`)
- `git log/diff/status/add/commit/checkout/branch` — version control
- `gh issue/pr` — GitHub operations
- `cat/ls/head/grep` — read files for review
- Write verdict files to `.factory/reviews/`

## Forbidden Actions (Sacred Rule 8)

- Writing or editing source code files
- Running `python eval/score.py`, `pytest`, `ruff`, `mypy` directly
- Using Claude Code's native `Agent` tool
- Editing `CLAUDE.md`, `factory.md`, or project config files

## Sacred Rules

1. Do not delete or overwrite existing tests
2. Do not modify files outside the declared scope
3. Do not introduce secrets or credentials
4. Do not lower the eval threshold
5. Do not skip the eval step
6. Do not merge PRs
7. Do not skip archival
8. Do not do another agent's job — delegate, review, decide
9. Do not skip QA verification

## CEO Review Gate

After EVERY agent, review output at `.factory/reviews/<role>-latest.md`. \
Write verdict to `.factory/reviews/ceo-verdict-<role>.md`:
- **PROCEED** — satisfactory, continue
- **REDIRECT** — re-invoke with corrections (max 2)
- **ABORT** — log failure, finalize as error

## Keep/Revert Essentials

All must be true to keep: tests pass, lint clean, score improved, no guard violations, \
code readable. Use `factory finalize` with `--verdict keep` or `--verdict revert`.

## Error Recovery

On agent failure: re-invoke with adjusted params → try different agent → finalize as error. \
NEVER do the agent's work yourself.

## Mode Pointer

Full workflow playbook is injected via system prompt. On resume, read \
`.factory/strategy/current.md` for your plan and session state.
"""


def resolve_prompt_core() -> str:
    """Return a slim (~7-8KB) CEO identity prompt for CLAUDE.md resume resilience.

    This contains only the essential CEO identity, Sacred Rules, permitted/forbidden
    actions, agent dispatch syntax, keep/revert essentials, error recovery summary,
    and a pointer to the full playbook. The full prompt is delivered separately via
    --append-system-prompt-file.
    """
    return _PROMPT_CORE_TEMPLATE


def _maybe_inject_profile(prompt: str, role: str) -> str:
    """Load and inject user profile if it exists."""
    from factory.profile import inject_profile, load_profile

    profile = load_profile()
    if profile:
        prompt = inject_profile(prompt, profile)
        logger.info("Injected user profile for %s", role)
    return prompt


def _maybe_inject_skill(prompt: str, project_path: Path, workflow_mode: str) -> str:
    """Append the workflow SKILL.md to the CEO prompt so it survives compaction."""
    skill_path = project_path / "skills" / f"workflow-{workflow_mode}" / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(
            f"SKILL.md not found for mode {workflow_mode} at {skill_path}. "
            f"Run 'factory workflow export-skills' or check ensure_skills() was called."
        )
    skill_content = skill_path.read_text()
    logger.info("Injected SKILL.md for workflow-%s into CEO prompt", workflow_mode)
    return prompt + f"\n\n# Workflow Playbook ({workflow_mode})\n\n{skill_content}"


async def invoke_agent(
    role: AgentRole,
    task: str,
    project_path: Path,
    *,
    timeout: float = 600.0,
    dangerously_skip_permissions: bool = True,
    model: str | None = None,
    runner_name: str | None = None,
    _track_failures: bool = True,
    session_name: str | None = None,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    use_profile: bool = False,
    tmux_persist: bool = False,
    background: bool = False,
    review_tag: str | None = None,
    workflow_mode: str | None = None,
    settings_file: str | None = None,
    prompt_override: str | None = None,
    node_id: str | None = None,
) -> tuple[str, int]:
    """Invoke a Claude Code agent with the resolved prompt + task.

    Returns (stdout, return_code).

    Raises:
        ConsecutiveAgentFailureError: If too many consecutive agent spawns fail
            (only when _track_failures=True).
    """
    global _consecutive_failures

    if prompt_override:
        prompt = prompt_override
    else:
        prompt = resolve_prompt(
            role, project_path, use_profile=use_profile, workflow_mode=workflow_mode
        )

    if os.environ.get("FACTORY_NO_GITHUB") == "1":
        prompt += (
            "\n\n## GitHub Disabled\n\n"
            "GitHub integration is disabled for this session (--no-github). "
            "Do NOT run any gh CLI commands (gh issue, gh pr, gh api, etc.). "
            "Do NOT create pull requests or reference GitHub issues. "
            "Work locally only — create commits, run tests, but skip all GitHub operations. "
            "When a step would normally involve GitHub, skip it and note that it was skipped.\n"
        )

    logger.info("Invoking %s agent for %s", role, project_path.name)

    started_data: dict[str, object] = {"task": task[:200]}
    if review_tag:
        started_data["review_tag"] = review_tag
    _emit_safe(project_path, "agent.started", agent=role, data=started_data)

    sid = _begin_span_safe(project_path, role, model=model, task=task)

    runner = get_runner(runner_name, project_path=project_path)

    agent_session_name = session_name or f"factory: {project_path.resolve().name}/{role}"

    from factory.models import AgentRunRequest

    request = AgentRunRequest(
        prompt=prompt,
        task=task,
        cwd=project_path,
        timeout=timeout,
        model=model,
        skip_permissions=dangerously_skip_permissions,
        role=role,
        session_name=agent_session_name,
        session_id=session_id,
        resume_session_id=resume_session_id,
        project_path=project_path,
        extras={
            "tmux_persist": tmux_persist,
            "background": background,
            **({"settings_file": settings_file} if settings_file else {}),
        },
    )

    old_parent_span = os.environ.get("FACTORY_PARENT_SPAN_ID")
    if sid:
        os.environ["FACTORY_PARENT_SPAN_ID"] = sid
    try:
        try:
            result = await runner.headless(request)
            stdout = result.stdout
            return_code = result.return_code
            usage = result.usage
        except Exception as e:
            logger.error("%s agent failed: %s", role, e)
            _emit_safe(project_path, "agent.failed", agent=role, data={"error": str(e)[:200]})
            _complete_span_safe(project_path, sid, status="failed")
            if _track_failures:
                _consecutive_failures += 1
                _check_failure_threshold(project_path, role)
            return f"Error: {e}", 1

        if return_code != 0:
            logger.warning("%s agent exited with code %d", role, return_code)
            _emit_safe(
                project_path,
                "agent.failed",
                agent=role,
                data={"return_code": return_code, "stderr": stdout[:200] if stdout else ""},
            )
            _complete_span_safe(
                project_path,
                sid,
                status="failed",
                usage=usage,
                metadata=result.metadata,
                output=stdout,
            )
            if _track_failures:
                _consecutive_failures += 1
                _check_failure_threshold(project_path, role)
        else:
            completed_data: dict[str, object] = {"return_code": 0}
            if review_tag:
                completed_data["review_tag"] = review_tag
            if usage is not None:
                completed_data.update(
                    {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cache_read_tokens,
                        "total_cost_usd": usage.total_cost_usd,
                        "duration_ms": usage.duration_ms,
                        "num_turns": usage.num_turns,
                        "model": usage.model,
                    }
                )
            for meta_key in ("session_id", "stop_reason", "terminal_reason"):
                if result.metadata.get(meta_key) is not None:
                    completed_data[meta_key] = result.metadata[meta_key]
            _emit_safe(
                project_path,
                "agent.completed",
                agent=role,
                data=completed_data,
            )
            _complete_span_safe(
                project_path,
                sid,
                status="completed",
                usage=usage,
                metadata=result.metadata,
                output=stdout,
            )
            if _track_failures:
                _consecutive_failures = 0

        _save_review(project_path, role, stdout, return_code, review_tag=review_tag)

        return stdout, return_code
    finally:
        if old_parent_span is not None:
            os.environ["FACTORY_PARENT_SPAN_ID"] = old_parent_span
        elif sid:
            os.environ.pop("FACTORY_PARENT_SPAN_ID", None)


def _check_failure_threshold(project_path: Path, last_agent: str) -> None:
    """Check if consecutive failures have exceeded the threshold and abort if so."""
    global _consecutive_failures

    if _consecutive_failures >= _FAILURE_ABORT_THRESHOLD:
        # Emit cycle.aborted event before raising
        _emit_safe(
            project_path,
            "cycle.aborted",
            data={
                "reason": "consecutive_agent_failures",
                "failure_count": _consecutive_failures,
                "last_agent": last_agent,
            },
        )
        raise ConsecutiveAgentFailureError(_consecutive_failures, last_agent)


def _emit_safe(project_path: Path, event_type: str, **kwargs: object) -> None:
    """Emit an event, swallowing errors so agent invocation is never blocked."""
    try:
        from factory.events import emit_event

        emit_event(project_path, event_type, **kwargs)  # type: ignore[arg-type]
    except Exception:
        logger.debug("Failed to emit event %s", event_type, exc_info=True)


def _begin_span_safe(
    project_path: Path,
    role: str,
    *,
    model: str | None = None,
    task: str | None = None,
) -> str | None:
    """Begin a Langfuse span, swallowing errors so agent invocation is never blocked."""
    try:
        from factory.telemetry import begin_span, begin_trace, is_enabled

        if not is_enabled():
            return None
        trace_id = os.environ.get("FACTORY_TRACE_ID")
        parent_span_id = os.environ.get("FACTORY_PARENT_SPAN_ID")
        logger.debug(
            "Langfuse env: FACTORY_TRACE_ID=%s FACTORY_PARENT_SPAN_ID=%s",
            trace_id,
            parent_span_id,
        )
        if not trace_id:
            result = begin_trace(project_path.name, cycle_id=f"standalone-{role}")
            if result is None:
                return None
            trace_id, root_span_id = result
            os.environ["FACTORY_TRACE_ID"] = trace_id
            os.environ["FACTORY_PARENT_SPAN_ID"] = root_span_id
            parent_span_id = root_span_id
        return begin_span(trace_id, parent_span_id, role, model=model, task=task)
    except Exception:
        logger.debug("Failed to begin span for %s", role, exc_info=True)
        return None


def _complete_span_safe(
    project_path: Path,
    span_id: str | None,
    *,
    status: str = "completed",
    usage: object | None = None,
    metadata: dict[str, object] | None = None,
    output: str | None = None,
) -> None:
    """Complete a Langfuse span, swallowing errors so agent invocation is never blocked."""
    if span_id is None:
        return
    try:
        from factory.telemetry import end_span, ingest_transcript_to_span, is_enabled

        if not is_enabled():
            return
        trace_id = os.environ.get("FACTORY_TRACE_ID")
        if not trace_id:
            return

        usage_dict: dict | None = None
        if usage is not None:
            usage_dict = {}
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "total_cost_usd",
                "duration_ms",
                "num_turns",
                "model",
            ):
                val = getattr(usage, key, None)
                if val is not None:
                    usage_dict[key] = val

        meta = dict(metadata or {})
        claude_session_id = meta.pop("session_id", None)
        if isinstance(claude_session_id, str) and claude_session_id:
            ingest_transcript_to_span(trace_id, span_id, claude_session_id, project_path)

        end_span(
            trace_id,
            span_id,
            status=status,
            usage=usage_dict,
            metadata=meta or None,
            output=output[:4000] if output else None,
        )
        from factory.telemetry import flush as _flush

        _flush()
    except Exception:
        logger.debug("Failed to complete span %s", span_id, exc_info=True)


def _save_review(
    project_path: Path,
    role: str,
    output: str,
    return_code: int,
    review_tag: str | None = None,
) -> None:
    """Save agent output to .factory/reviews/<role>-latest.md for CEO review.

    When *review_tag* is provided the file is written as
    ``<role>-<tag>-latest.md`` instead, allowing multiple concurrent agents
    with the same role to produce distinct review files.

    Creates the reviews directory if needed. Errors are swallowed so they
    never block agent execution.
    """
    try:
        reviews_dir = project_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{role}-{review_tag}-latest.md" if review_tag else f"{role}-latest.md"
        review_path = reviews_dir / filename
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = f"# {role.title()} Agent Output\n\n- **timestamp:** {ts}\n- **exit_code:** {return_code}\n\n---\n\n"
        content = header + output
        if role != "ceo":
            content += IDENTITY_REANCHOR
        review_path.write_text(content)
        logger.debug("Saved review output for %s to %s", role, review_path)
    except Exception:
        logger.debug("Failed to save review for %s", role, exc_info=True)


def begin_cycle_session(
    project_path: Path,
    cycle_id: str | None = None,
    model: str | None = None,
) -> str | None:
    """Create a root Langfuse trace for a factory cycle.

    Sets FACTORY_TRACE_ID and FACTORY_PARENT_SPAN_ID env vars so child
    agents link to this trace. Returns the span_id, or None if Langfuse
    is not configured.
    """
    try:
        from factory.telemetry import begin_trace, is_enabled

        if not is_enabled():
            return None
        result = begin_trace(
            project_path.name,
            cycle_id or "unknown",
            model=model,
        )
        if result is None:
            return None
        trace_id, span_id = result
        os.environ["FACTORY_TRACE_ID"] = trace_id
        os.environ["FACTORY_PARENT_SPAN_ID"] = span_id
        try:
            factory_dir = project_path / ".factory"
            factory_dir.mkdir(parents=True, exist_ok=True)
            (factory_dir / "trace_id.txt").write_text(trace_id)
        except OSError:
            logger.debug("Failed to write trace_id.txt", exc_info=True)
        return span_id
    except Exception:
        logger.debug("Failed to begin cycle trace", exc_info=True)
        return None


def complete_cycle_session(
    project_path: Path,
    span_id: str | None,
) -> None:
    """Mark a root Langfuse trace as finished and flush."""
    if span_id is None:
        return
    try:
        from factory.telemetry import end_trace, flush, is_enabled

        if not is_enabled():
            return
        trace_id = os.environ.get("FACTORY_TRACE_ID", "")
        end_trace(trace_id, span_id=span_id)
        flush()
    except Exception:
        logger.debug("Failed to complete cycle trace", exc_info=True)
