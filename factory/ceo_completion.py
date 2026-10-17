"""CEO completion guard — auto-resume on premature exit.

Detects when the CEO exits before all planned work is complete and re-spawns
with a continuation task. This is a structural fix for model-side decisions
to "wrap up" early.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from factory.events import emit_event, load_events
from factory.models import CycleState

log = structlog.get_logger()

# Staleness threshold for cycle.json (24 hours)
CYCLE_STALENESS_HOURS = 24

# Hard cap on re-spawns per cycle (env-overridable)
DEFAULT_MAX_RESPAWNS = 5


# ── cycle state persistence ──────────────────────────────────────


def _cycle_state_path(project_path: Path) -> Path:
    """Return the path to .factory/state/cycle.json."""
    return project_path / ".factory" / "state" / "cycle.json"


def _session_state_path(project_path: Path) -> Path:
    """Return the path to .factory/state/session.json."""
    return project_path / ".factory" / "state" / "session.json"


def read_ceo_session_id(project_path: Path) -> str | None:
    """Read the CEO session ID from .factory/state/session.json."""
    path = _session_state_path(project_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("session_id")
    except (json.JSONDecodeError, ValueError):
        return None


def read_ceo_session(project_path: Path) -> dict | None:
    """Read the full CEO session metadata from .factory/state/session.json.

    Returns dict with keys: session_id, created, interactive, mode.
    Returns None if the file doesn't exist or is malformed.
    """
    path = _session_state_path(project_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def write_ceo_session_id(
    project_path: Path,
    session_id: str,
    *,
    interactive: bool = False,
    mode: str = "",
) -> None:
    """Write a CEO session ID and metadata to .factory/state/session.json."""
    path = _session_state_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "session_id": session_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "interactive": interactive,
        "mode": mode,
    }
    path.write_text(json.dumps(data, indent=2))
    log.info("ceo_session_id_written", session_id=session_id, interactive=interactive, mode=mode)


def read_cycle_state(project_path: Path) -> CycleState | None:
    """Read in-flight cycle state if it exists and is non-stale.

    Returns None if:
    - cycle.json doesn't exist
    - cycle.json is malformed
    - cycle.json is stale (older than CYCLE_STALENESS_HOURS)
    """
    path = _cycle_state_path(project_path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())

        # Parse datetime from ISO string (model_dump(mode="json") serializes as ISO)
        if "started_at" in data and isinstance(data["started_at"], str):
            data["started_at"] = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))

        state = CycleState.model_validate(data)

        # Check staleness
        now = datetime.now(timezone.utc)
        started = state.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_hours = (now - started).total_seconds() / 3600

        if age_hours > CYCLE_STALENESS_HOURS:
            log.info("cycle_state_stale", age_hours=age_hours, cycle_id=state.cycle_id)
            return None

        return state
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("cycle_state_parse_error", error=str(e))
        return None


def write_cycle_state(project_path: Path, state: CycleState) -> None:
    """Write cycle state to .factory/state/cycle.json."""
    path = _cycle_state_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use model_dump with mode="json" for proper datetime serialization
    data = state.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2))
    log.info(
        "cycle_state_written", cycle_id=state.cycle_id, mode=state.mode, respawns=state.respawns
    )


def delete_cycle_state(project_path: Path) -> bool:
    """Delete cycle.json and session.json on cycle completion. Returns True if deleted."""
    path = _cycle_state_path(project_path)
    deleted = False
    if path.exists():
        path.unlink()
        log.info("cycle_state_deleted", path=str(path))
        deleted = True

    session_path = _session_state_path(project_path)
    if session_path.exists():
        session_path.unlink()
        log.info("session_state_deleted", path=str(session_path))
        deleted = True

    return deleted


def print_resume_hint(project_path: Path) -> None:
    """Print session ID and resume instructions to stderr if the session is still active.

    Only prints when session.json still exists — if delete_cycle_state() already
    cleaned it up (cycle completed normally), this is a no-op.
    """
    import sys

    sid = read_ceo_session_id(project_path)
    if sid:
        print(f"\nSession: {sid}", file=sys.stderr)
        print(f"Resume with: factory resume {project_path}", file=sys.stderr)


def create_cycle_state(
    mode: str, initial_prompt: str = "", runner_name: str | None = None
) -> CycleState:
    """Create a new CycleState for a fresh cycle."""
    return CycleState(
        cycle_id=str(uuid.uuid4())[:8],
        started_at=datetime.now(timezone.utc),
        mode=mode,  # type: ignore[arg-type]
        initial_prompt=initial_prompt[:1000],  # Truncate to avoid bloat
        respawns=0,
        runner_name=runner_name,
    )


@dataclass
class IncompleteGap:
    """Describes what work is incomplete."""

    mode: str
    planned: int
    completed: int
    next_item: str
    reason: str


def _count_hypotheses(project_path: Path) -> int:
    """Count hypotheses in .factory/strategy/current.md."""
    strategy_file = project_path / ".factory" / "strategy" / "current.md"
    if not strategy_file.exists():
        return 0

    content = strategy_file.read_text()
    # Match headings like "#### H1:" or "### H2:" etc.
    matches = re.findall(r"^#{2,4}\s+H(\d+):", content, re.MULTILINE)
    return len(matches)


def _count_verdicts(project_path: Path, since_ts: datetime | None = None) -> int:
    """Count experiments with verdicts, optionally filtered by timestamp.

    If since_ts is provided, only counts experiments created after that time.
    This prevents counting stale experiments from previous cycles.

    Reads from results.tsv for timestamp-aware counting. Falls back to counting
    verdict.json files if results.tsv doesn't exist (backward compatibility).
    """
    import csv

    tsv_path = project_path / ".factory" / "results.tsv"

    # If results.tsv exists, use it for timestamp-aware counting
    if tsv_path.exists():
        count = 0
        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, dialect="excel-tab")
            for row in reader:
                verdict = row.get("verdict", "").strip().lower()
                if verdict not in ("keep", "revert", "error"):
                    continue

                if since_ts is not None:
                    try:
                        row_ts = datetime.fromisoformat(row["timestamp"])
                        if row_ts.tzinfo is None:
                            row_ts = row_ts.replace(tzinfo=timezone.utc)
                        since_ts_tz = since_ts
                        if since_ts_tz.tzinfo is None:
                            since_ts_tz = since_ts_tz.replace(tzinfo=timezone.utc)
                        if row_ts < since_ts_tz:
                            continue
                    except (KeyError, ValueError):
                        continue

                count += 1
        return count

    # Fallback: count verdict.json files (no timestamp filtering possible)
    # This path is used when results.tsv doesn't exist yet (e.g., fresh projects, tests)
    experiments_dir = project_path / ".factory" / "experiments"
    if not experiments_dir.exists():
        return 0

    count = 0
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "verdict.json").exists():
            count += 1
    return count


def _has_eval_profile(project_path: Path) -> bool:
    """Check if .factory/eval_profile.json exists and is non-empty."""
    profile = project_path / ".factory" / "eval_profile.json"
    if not profile.exists():
        return False
    return profile.stat().st_size > 10  # More than just "{}"


def _has_aborted(project_path: Path, since_ts: str | None = None) -> bool:
    """Check if cycle.aborted event exists in events.jsonl."""
    events = load_events(project_path)
    for event in reversed(events):
        if event.get("type") == "cycle.aborted":
            if since_ts is None:
                return True
            if event.get("timestamp", "") >= since_ts:
                return True
    return False


def _detect_incomplete(
    project_path: Path, mode: str, cycle_started_at: datetime | None = None
) -> IncompleteGap | None:
    """Detect if the cycle is incomplete for the given mode.

    Returns IncompleteGap if work is incomplete, None if complete.

    Args:
        project_path: Path to the project.
        mode: CEO mode (improve, build, discover, meta).
        cycle_started_at: If provided, only counts experiments created after this time.
            This prevents counting stale experiments from previous cycles.
    """
    if mode in ("design", "meta", "research"):
        planned = _count_hypotheses(project_path)
        completed = _count_verdicts(project_path, since_ts=cycle_started_at)

        if planned == 0:
            if mode == "design" and not _has_eval_profile(project_path):
                return IncompleteGap(
                    mode=mode,
                    planned=1,
                    completed=0,
                    next_item="discovery",
                    reason="design.incomplete: no eval profile yet",
                )
            return None

        if completed >= planned:
            return None

        next_h = completed + 1
        reason_prefix = "research" if mode == "research" else "design"
        return IncompleteGap(
            mode=mode,
            planned=planned,
            completed=completed,
            next_item=f"H{next_h}",
            reason=f"{reason_prefix}.incomplete: {completed}/{planned} hypotheses have verdicts",
        )

    # Unknown mode — assume complete
    return None


def _build_continuation_task(gap: IncompleteGap, cycle_state: CycleState | None = None) -> str:
    """Build the continuation task string for re-spawning the CEO.

    Includes explicit mode directive to prevent mode flipping on respawn.
    """
    # Mode directive header — prevents mode flip on respawn
    mode_directive = (
        f"## CRITICAL: Mode Override\n\n"
        f"This is a CONTINUATION of an in-flight {gap.mode.upper()} cycle. "
        f"Do NOT re-detect mode. Do NOT switch to a different mode. "
        f"The cycle mode is **{gap.mode}** — execute {gap.mode.upper()} mode only.\n\n"
    )

    if cycle_state:
        mode_directive += (
            f"Cycle ID: {cycle_state.cycle_id}\nRespawn count: {cycle_state.respawns}\n\n"
        )

    if gap.mode == "research":
        body = (
            f"Resume execution from hypothesis {gap.next_item}. "
            f"Strategy is already approved at .factory/strategy/current.md — "
            f"do not re-plan, do not re-run Failure Analyst, Researcher, or Strategist. "
            f"The baseline run (R0), failure analysis (R1), and research (R1.5) are already complete. "
            f"Spawn Builder for {gap.next_item} immediately, then continue the "
            f"research cycle (R3–R5) for each remaining hypothesis. "
            f"Progress so far: {gap.completed}/{gap.planned} hypotheses have verdicts."
        )
    elif gap.mode in ("design", "meta"):
        body = (
            f"Resume execution from {gap.next_item}. "
            f"Strategy is already approved at .factory/strategy/current.md — "
            f"do not re-plan, do not re-run Researcher or Strategist. "
            f"Spawn Builder for {gap.next_item} immediately. "
            f"Progress so far: {gap.completed}/{gap.planned} hypotheses have verdicts."
        )
    else:
        body = f"Resume from {gap.next_item}."

    return mode_directive + body


def _budget_allows_respawn(runner_name: str | None, project_path: Path) -> bool:
    """Check if budget/ceiling allows another spawn.

    With only per-cycle limits (no daily/session limit), we can always start
    a new cycle. Per-cycle limits are enforced within the runner during execution.
    """
    # All runners can respawn - per-cycle limits are enforced within the cycle
    return True


def _write_cycle_incomplete(project_path: Path, gap: IncompleteGap, reason: str) -> None:
    """Write .factory/strategy/cycle-incomplete.md describing what wasn't finished."""
    strategy_dir = project_path / ".factory" / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    incomplete_file = strategy_dir / "cycle-incomplete.md"
    content = f"""# Cycle Incomplete

**Mode:** {gap.mode}
**Reason:** {reason}
**Planned:** {gap.planned}
**Completed:** {gap.completed}
**Next item that wasn't started:** {gap.next_item}

## Details

{gap.reason}

This file is written when the CEO completion guard gives up after hitting
the respawn cap or budget limit. The cycle can be resumed manually with:

```bash
factory ceo /path/to/project --headless
```
"""
    incomplete_file.write_text(content)
    log.warning("cycle_incomplete", reason=reason, gap=gap)


def _extract_session_id(project_path: Path) -> str | None:
    """Extract the session_id from the most recent agent.completed event."""
    events = load_events(project_path)
    for event in reversed(events):
        if event.get("type") == "agent.completed" and event.get("agent") == "ceo":
            sid = event.get("data", {}).get("session_id")
            if isinstance(sid, str) and sid:
                return sid
    return None


def write_usage_json(project_path: Path, mode: str, since: datetime) -> None:
    """Write aggregated token usage to ``.factory/<mode>/usage.json``.

    Called after both the headless completion-guard path and the interactive
    CEO path complete, so every CEO run (not just ``--headless``) produces
    usage data for downstream consumers. Covers the CEO ``skill`` engine path,
    which does not go through WorkflowExecutor.execute().
    """
    try:
        from factory.events import sum_agent_usage

        usage = sum_agent_usage(project_path, since=since)
        if not usage:
            return

        usage_dir = project_path / ".factory" / mode
        usage_dir.mkdir(parents=True, exist_ok=True)
        (usage_dir / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
        log.info(
            "ceo.usage_written",
            mode=mode,
            cost_usd=usage["total_cost_usd"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
    except Exception:
        log.warning("ceo.usage_write_failed", exc_info=True)


async def run_ceo_with_completion_guard(
    project_path: Path,
    initial_task: str,
    *,
    mode: str,
    runner_name: str | None = None,
    model: str | None = None,
    timeout: float = 3600.0,
    max_respawns: int | None = None,
    session_name: str | None = None,
    session_id: str | None = None,
    use_profile: bool = False,
    tmux_persist: bool = False,
    background: bool = False,
    workflow_mode: str | None = None,
    settings_file: str | None = None,
    prompt_override: str | None = None,
) -> tuple[str, int]:
    """Spawn CEO; if it exits with planned work undone, re-spawn until done or cap hit.

    Mode is persisted in .factory/state/cycle.json to prevent mode flipping across
    respawns. The cycle state is created on first spawn and deleted on completion.

    Args:
        project_path: Path to the project.
        initial_task: Initial task string for the CEO.
        mode: CEO mode (improve, build, discover, meta).
        runner_name: Runner to use (default: claude).
        model: Optional model override.
        timeout: Timeout per CEO spawn in seconds.
        max_respawns: Max re-spawns (default from env or 5).
        session_name: Optional session name for /resume identification.
        use_profile: If True, inject user profile into the CEO prompt.
        tmux_persist: If True, run agents in tmux windows.
        background: If True, dispatch via claude --bg (single dispatch, no respawn).
        workflow_mode: If set, inject the SKILL.md for this mode into the CEO prompt.

    Returns:
        (final_output, exit_code)
    """
    from factory.agents.runner import invoke_agent

    if background:
        log.info("ceo_background_dispatch", reason="--bg: single dispatch, no respawn loop")
        return await invoke_agent(
            "ceo",
            initial_task,
            project_path,
            timeout=timeout,
            model=model,
            runner_name=runner_name,
            background=True,
            session_name=session_name,
            use_profile=use_profile,
            workflow_mode=workflow_mode,
            settings_file=settings_file,
            prompt_override=prompt_override,
        )

    # Check escape hatch
    from factory.user_config import resolve

    if resolve("ceo_respawn_disabled", env_var="FACTORY_CEO_RESPAWN_DISABLED") == "1":
        log.info("ceo_respawn_disabled", reason="FACTORY_CEO_RESPAWN_DISABLED=1")
        return await invoke_agent(
            "ceo",
            initial_task,
            project_path,
            timeout=timeout,
            model=model,
            runner_name=runner_name,
            session_name=session_name,
            use_profile=use_profile,
            tmux_persist=tmux_persist,
            workflow_mode=workflow_mode,
            settings_file=settings_file,
            prompt_override=prompt_override,
        )

    if max_respawns is None:
        max_respawns = int(
            resolve(
                "ceo_max_respawns",
                env_var="FACTORY_CEO_MAX_RESPAWNS",
                default=str(DEFAULT_MAX_RESPAWNS),
            )
            or DEFAULT_MAX_RESPAWNS
        )

    # Check for existing in-flight cycle (respawn scenario)
    cycle_state = read_cycle_state(project_path)
    if cycle_state:
        # Continuing an existing cycle — use its mode and runner, not the passed-in ones
        log.info(
            "cycle_state_found",
            cycle_id=cycle_state.cycle_id,
            original_mode=cycle_state.mode,
            passed_mode=mode,
        )
        mode = cycle_state.mode
        # Restore runner_name from persisted state (if set)
        if cycle_state.runner_name:
            runner_name = cycle_state.runner_name
    else:
        # Fresh cycle — create new state
        cycle_state = create_cycle_state(mode, initial_task, runner_name)
        write_cycle_state(project_path, cycle_state)
        log.info("cycle_state_created", cycle_id=cycle_state.cycle_id, mode=mode)

    task = initial_task
    final_output = ""
    gap: IncompleteGap | None = None
    captured_session_id: str | None = None

    for attempt in range(max_respawns + 1):
        log.info("ceo_spawn", attempt=attempt, task_preview=task[:100], mode=mode)

        resume_sid = captured_session_id if attempt > 0 else None
        spawn_sid = session_id if attempt == 0 else None

        result, code = await invoke_agent(
            "ceo",
            task,
            project_path,
            timeout=timeout,
            model=model,
            runner_name=runner_name,
            session_name=session_name,
            session_id=spawn_sid,
            resume_session_id=resume_sid,
            use_profile=use_profile,
            tmux_persist=tmux_persist,
            workflow_mode=workflow_mode,
            settings_file=settings_file,
            prompt_override=prompt_override,
        )
        final_output = result

        returned_sid = _extract_session_id(project_path)
        if returned_sid and returned_sid != captured_session_id:
            captured_session_id = returned_sid
            cycle_state.claude_session_id = returned_sid
            write_cycle_state(project_path, cycle_state)

        # User interrupt — respect it (but don't delete cycle state for later resume)
        if code in (130, 143) or code > 128:
            log.info("ceo_user_interrupt", code=code)
            print_resume_hint(project_path)
            return result, code

        # Explicit ABORT — respect it and clean up cycle state
        # Pass cycle start time to filter out stale abort events from previous cycles
        if _has_aborted(project_path, since_ts=cycle_state.started_at.isoformat()):
            log.info("ceo_aborted", reason="cycle.aborted event found")
            delete_cycle_state(project_path)
            return result, code

        # Check for incomplete work — pass cycle start time to filter stale experiments
        gap = _detect_incomplete(project_path, mode, cycle_started_at=cycle_state.started_at)
        if gap is None:
            log.info("ceo_complete", attempt=attempt)
            write_usage_json(project_path, workflow_mode or mode, cycle_state.started_at)
            # Cycle complete — delete cycle state so next invocation starts fresh
            delete_cycle_state(project_path)
            return result, code

        # Check budget before re-spawning
        if not _budget_allows_respawn(runner_name, project_path):
            log.warning("ceo_budget_exceeded", gap=gap)
            _write_cycle_incomplete(project_path, gap, "budget_exceeded")
            print_resume_hint(project_path)
            return result, 1

        # Update cycle state with incremented respawn count
        cycle_state.respawns += 1
        write_cycle_state(project_path, cycle_state)

        # Emit respawn event with cycle_id
        emit_event(
            project_path,
            "ceo.respawn",
            agent="ceo",
            data={
                "attempt": attempt + 1,
                "cycle_id": cycle_state.cycle_id,
                "mode": mode,
                "reason": gap.reason,
                "planned": gap.planned,
                "completed": gap.completed,
                "next": gap.next_item,
            },
        )

        # Build continuation task with explicit mode directive
        task = _build_continuation_task(gap, cycle_state)
        log.info("ceo_respawn", attempt=attempt + 1, next_item=gap.next_item, mode=mode)

    # Cap hit — don't delete cycle state to allow manual resume
    if gap:
        log.warning("ceo_respawn_cap_hit", attempts=max_respawns + 1, gap=gap)
        _write_cycle_incomplete(project_path, gap, "respawn_cap_hit")

    print_resume_hint(project_path)
    return final_output, 1
