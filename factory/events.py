"""Event system for factory observability — append-only JSONL event log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


def emit_event(
    project_path: Path,
    event_type: str,
    *,
    agent: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a structured event to .factory/events.jsonl. Returns the event dict."""
    project_path = project_path.resolve()
    event = {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": project_path.name,
        "agent": agent,
        "data": data or {},
    }

    from factory.store import ensure_factory_dir
    events_dir = project_path / ".factory"
    ensure_factory_dir(events_dir)
    events_file = events_dir / "events.jsonl"

    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    log.debug("event_emitted", type=event_type, project=project_path.name, agent=agent)
    return event


def load_events(
    project_path: Path,
    *,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Load events from .factory/events.jsonl, optionally filtered by timestamp."""
    project_path = project_path.resolve()
    events_file = project_path / ".factory" / "events.jsonl"
    if not events_file.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if since:
            event_ts = datetime.fromisoformat(event["timestamp"])
            if event_ts <= since:
                continue
        events.append(event)

    return events


def sum_agent_usage(
    project_path: Path,
    *,
    since: datetime | None = None,
) -> dict[str, int | float]:
    """Sum token counts and cost from agent.completed events.

    Returns a dict with input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, and total_cost_usd.  Returns an empty dict
    when no usage was found.
    """
    events = load_events(project_path, since=since)
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0
    total_cost_usd = 0.0

    for ev in events:
        if ev.get("type") != "agent.completed":
            continue
        data = ev.get("data", {})
        input_tokens += int(data.get("input_tokens", 0) or 0)
        output_tokens += int(data.get("output_tokens", 0) or 0)
        cache_read_tokens += int(data.get("cache_read_tokens", 0) or 0)
        cache_creation_tokens += int(data.get("cache_creation_tokens", 0) or 0)
        total_cost_usd += float(data.get("total_cost_usd", 0.0) or 0.0)

    if not any((input_tokens, output_tokens, total_cost_usd)):
        return {}

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
    }



def discover_factory_projects(projects_dir: Path, *, max_depth: int = 3) -> list[Path]:
    """Find all subdirectories with .factory/ directories, up to max_depth levels."""
    if not projects_dir.exists():
        return []
    results: list[Path] = []

    def _scan(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / ".factory").is_dir():
                results.append(child)
            # Always recurse into children — a parent with .factory/
            # may contain nested projects with their own .factory/
            _scan(child, depth + 1)

    _scan(projects_dir, 1)
    return sorted(results)
