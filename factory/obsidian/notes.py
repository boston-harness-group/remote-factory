"""Obsidian note creation — experiment logs, project dashboards, strategy notes."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import structlog

from factory.models import CompositeScore, ExperimentRecord

log = structlog.get_logger()

_PROJECTS_DIR = "10-Projects"
_KNOWLEDGE_DIR = "20-Knowledge"
_FACTORY_META_DIR = "00-Factory"
_TEMPLATES_DIR = "_templates"

# ── Template content ──────────────────────────────────────────

_TEMPLATE_EXPERIMENT = """\
---
tags:
  - factory
  - experiment
  - {{project}}
project: {{project}}
experiment_id: {{id}}
verdict: {{verdict}}
score_delta: {{delta}}
date: {{date}}
source: factory-qa
---

# Experiment #{{id}}: {{hypothesis}}

## Hypothesis
{{hypothesis}}

## Result
**{{verdict}}** — score changed from {{before}} to {{after}} ({{delta}})

## What Changed
{{summary}}

## Links
- [[{{project}}]]
"""

_TEMPLATE_DECISION = """\
---
tags:
  - factory
  - decision
  - {{project}}
project: {{project}}
date: {{date}}
context: {{context}}
outcome: {{outcome}}
source: factory-orchestrator
---

# Decision: {{title}}

## Context
{{context}}

## Alternatives Considered
{{alternatives}}

## Decision
{{decision}}

## Outcome
{{outcome}}
"""

_TEMPLATE_STRATEGY = """\
---
tags:
  - factory
  - strategy
  - {{project}}
date: {{date}}
source: factory-strategist
---

# Strategy: {{project}} — {{date}}

{{content}}
"""

_TEMPLATE_PROJECT = """\
---
tags:
  - factory
  - project
  - {{project}}
---

# Factory: {{project}}

## Status
- **State**: {{state}}
- **Current Score**: {{score}}

## Recent Experiments
(populated by archivist)
"""


def vault_path() -> Path | None:
    """Return the configured vault path, or ``None`` when unconfigured.

    Reads from ``FACTORY_VAULT_PATH`` only.  When the env var is unset the
    vault is considered *unavailable* and callers should skip vault operations
    gracefully (writes fall back to ``.factory/archive/``).
    """
    from factory.user_config import resolve

    raw = resolve("vault_path", env_var="FACTORY_VAULT_PATH")
    if raw:
        return Path(raw)
    return None


def _get_vault_path() -> Path | None:
    """Get the Obsidian vault path. Returns ``None`` when unconfigured."""
    return vault_path()


def _ensure_dir(path: Path) -> None:
    """Create directory and parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


# ── Obsidian CLI wrappers ────────────────────────────────────


def _obsidian_create(name: str, content: str, vault: str = "factory") -> bool:
    """Create a note via obsidian-cli. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "obsidian-cli",
                "create",
                f"vault={vault}",
                f"name={name}",
                f"content={content}",
                "silent",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _obsidian_search(query: str, vault: str = "factory", limit: int = 10) -> str | None:
    """Search the vault via obsidian-cli. Returns results or None."""
    try:
        result = subprocess.run(
            [
                "obsidian-cli",
                "search",
                f"vault={vault}",
                f"query={query}",
                f"limit={limit}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def obsidian_search_vault(
    query: str,
    vault: str = "factory",
    limit: int = 10,
) -> str | None:
    """Search the factory vault. Returns results from obsidian-cli, or None if unavailable."""
    return _obsidian_search(query, vault, limit)


def init_vault(vault_override: Path | None = None) -> Path | None:
    """Create the full factory vault structure. Returns the vault path.

    Returns ``None`` when no vault is configured and no explicit path is given.
    """
    vault = vault_override if vault_override is not None else _get_vault_path()
    if vault is None:
        log.debug("init_vault_skipped", reason="no vault path configured")
        return None
    log.info("init_vault", vault=str(vault))

    # .obsidian/
    _ensure_dir(vault / ".obsidian")

    # 00-Factory/
    factory_meta = vault / _FACTORY_META_DIR
    _ensure_dir(factory_meta)
    _ensure_dir(factory_meta / "Decisions")

    dashboard_path = factory_meta / "Dashboard.md"
    if not dashboard_path.exists():
        dashboard_path.write_text(
            "# Factory Dashboard\n\nCentral hub for all factory-managed projects.\n"
        )

    patterns_path = factory_meta / "Patterns.md"
    if not patterns_path.exists():
        patterns_path.write_text(
            "# Cross-Project Patterns\n\nRecurring patterns discovered across projects.\n"
        )

    # 10-Projects/
    _ensure_dir(vault / _PROJECTS_DIR)

    # 20-Knowledge/
    _ensure_dir(vault / _KNOWLEDGE_DIR / "Concepts")
    _ensure_dir(vault / _KNOWLEDGE_DIR / "Sources")

    # _templates/
    templates = vault / _TEMPLATES_DIR
    _ensure_dir(templates)

    template_files = {
        "experiment.md": _TEMPLATE_EXPERIMENT,
        "decision.md": _TEMPLATE_DECISION,
        "strategy.md": _TEMPLATE_STRATEGY,
        "project.md": _TEMPLATE_PROJECT,
    }
    for name, content in template_files.items():
        tpath = templates / name
        if not tpath.exists():
            tpath.write_text(content)

    # MEMORY.md
    memory_path = vault / "MEMORY.md"
    if not memory_path.exists():
        memory_path.write_text(
            "# Factory Memory Index\n\n"
            "Pointer file for factory agents.\n\n"
            "## Projects\n\n(none yet)\n"
        )

    log.debug("init_vault_complete", vault=str(vault))
    return vault


def _auto_init_vault() -> Path | None:
    """Get vault path and auto-create structure if needed.

    Returns ``None`` when no vault is configured, signalling that callers
    should skip vault writes.
    """
    vault = _get_vault_path()
    if vault is None:
        log.debug("auto_init_vault_skipped", reason="no vault path configured")
        return None
    if not (vault / ".obsidian").exists():
        log.info("auto_init_vault_triggered", vault=str(vault))
        init_vault(vault)
    return vault


def write_experiment_note(
    project_name: str,
    record: ExperimentRecord,
    score_before: CompositeScore | None = None,
    score_after: CompositeScore | None = None,
) -> Path | None:
    """Create an Obsidian note for a completed experiment.

    Returns ``None`` when the vault is not configured.
    """
    log.debug(
        "write_experiment_note", project=project_name, exp_id=record.id, verdict=record.verdict
    )
    vault = _auto_init_vault()
    if vault is None:
        log.debug("write_experiment_note_skipped", reason="vault not configured")
        return None
    experiments_dir = vault / _PROJECTS_DIR / project_name / "Experiments"
    _ensure_dir(experiments_dir)

    filename = f"{project_name}-{record.id:03d}.md"
    note_path = experiments_dir / filename

    delta_str = f"{record.delta:+.4f}" if record.delta is not None else "n/a"
    before_str = f"{record.score_before:.4f}" if record.score_before is not None else "n/a"
    after_str = f"{record.score_after:.4f}" if record.score_after is not None else "n/a"
    date_str = record.timestamp.strftime("%Y-%m-%d")

    lines = [
        "---",
        "tags:",
        "  - factory",
        "  - experiment",
        f"  - {project_name}",
        f"project: {project_name}",
        f"experiment_id: {record.id}",
        f"verdict: {record.verdict}",
        f"score_delta: {record.delta if record.delta is not None else 0.0}",
        f"date: {date_str}",
        "source: factory-qa",
        "---",
        "",
        f"# Experiment #{record.id}: {record.hypothesis[:80]}",
        "",
        "## Hypothesis",
        record.hypothesis,
        "",
        "## Result",
        f"**{record.verdict.upper()}** — score changed from {before_str} to {after_str} ({delta_str})",
        "",
        "## What Changed",
        record.change_summary or "No summary provided.",
        "",
    ]

    # Add eval details table if scores available
    if score_before and score_after:
        lines.extend(
            [
                "## Eval Details",
                "| Dimension | Before | After | Delta |",
                "|-----------|--------|-------|-------|",
            ]
        )
        before_map = {r.name: r.score for r in score_before.results}
        for r in score_after.results:
            b = before_map.get(r.name, 0.0)
            d = r.score - b
            lines.append(f"| {r.name} | {b:.2f} | {r.score:.2f} | {d:+.2f} |")
        lines.append("")

    if record.notes:
        lines.extend(["## Notes", record.notes, ""])

    lines.extend(
        [
            "## Links",
            f"- [[{project_name} Dashboard]]",
        ]
    )
    if record.issue_number:
        lines.append(f"- Issue: #{record.issue_number}")
    if record.pr_number:
        lines.append(f"- PR: #{record.pr_number}")

    content = "\n".join(lines) + "\n"

    # Try obsidian-cli first, fall back to direct write
    note_name = f"{_PROJECTS_DIR}/{project_name}/Experiments/{project_name}-{record.id:03d}"
    if not _obsidian_create(note_name, content):
        log.debug("write_experiment_note_fallback_to_file", path=str(note_path))
        note_path.write_text(content)

    return note_path


def write_project_dashboard(
    project_name: str,
    state: str,
    current_score: float | None,
    records: list[ExperimentRecord],
    eval_dimensions: list[dict] | None = None,
) -> Path | None:
    """Create or update the project dashboard note.

    Returns ``None`` when the vault is not configured.
    """
    log.debug(
        "write_project_dashboard",
        project=project_name,
        state=state,
        record_count=len(records),
    )
    vault = _auto_init_vault()
    if vault is None:
        log.debug("write_project_dashboard_skipped", reason="vault not configured")
        return None
    projects_dir = vault / _PROJECTS_DIR / project_name
    _ensure_dir(projects_dir)

    filename = f"{project_name}.md"
    note_path = projects_dir / filename

    kept = sum(1 for r in records if r.verdict == "keep")
    reverted = sum(1 for r in records if r.verdict == "revert")
    errored = sum(1 for r in records if r.verdict == "error")
    score_str = f"{current_score:.4f}" if current_score is not None else "n/a"

    lines = [
        "---",
        "tags:",
        "  - factory",
        "  - project",
        f"  - {project_name}",
        "---",
        "",
        f"# Factory: {project_name}",
        "",
        "## Status",
        f"- **State**: {state}",
        f"- **Current Score**: {score_str}",
        f"- **Experiments Run**: {len(records)}",
        f"- **Kept**: {kept}, **Reverted**: {reverted}, **Error**: {errored}",
        "",
    ]

    if eval_dimensions:
        lines.append("## Eval Dimensions")
        for dim in eval_dimensions:
            lines.append(
                f"- {dim.get('name', '?')} ({dim.get('weight', 0):.1%} weight)"
                f" — {dim.get('description', '')}"
            )
        lines.append("")

    # Recent experiments (last 5)
    lines.append("## Recent Experiments")
    recent = records[-5:] if records else []
    for r in reversed(recent):
        delta = f"{r.delta:+.4f}" if r.delta is not None else "n/a"
        lines.append(
            f"- [[{project_name}-{r.id:03d}]] — {r.hypothesis[:50]} ({r.verdict.upper()}, {delta})"
        )
    if not recent:
        lines.append("- No experiments yet")
    lines.append("")

    content = "\n".join(lines) + "\n"

    # Try obsidian-cli first, fall back to direct write
    note_name = f"{_PROJECTS_DIR}/{project_name}/{project_name}"
    if not _obsidian_create(note_name, content):
        log.debug("write_project_dashboard_fallback_to_file", path=str(note_path))
        note_path.write_text(content)

    return note_path


def write_strategy_note(
    project_name: str,
    strategy_content: str,
) -> Path | None:
    """Write a strategy snapshot to Obsidian.

    Returns ``None`` when the vault is not configured.
    """
    log.debug("write_strategy_note", project=project_name)
    vault = _auto_init_vault()
    if vault is None:
        log.debug("write_strategy_note_skipped", reason="vault not configured")
        return None
    strategies_dir = vault / _PROJECTS_DIR / project_name / "Strategies"
    _ensure_dir(strategies_dir)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{project_name}-{date_str}.md"
    note_path = strategies_dir / filename

    lines = [
        "---",
        "tags:",
        "  - factory",
        "  - strategy",
        f"  - {project_name}",
        f"date: {date_str}",
        "source: factory-strategist",
        "---",
        "",
        f"# Strategy: {project_name} — {date_str}",
        "",
        strategy_content,
    ]

    content = "\n".join(lines) + "\n"

    # Try obsidian-cli first, fall back to direct write
    note_name = f"{_PROJECTS_DIR}/{project_name}/Strategies/{project_name}-{date_str}"
    if not _obsidian_create(note_name, content):
        log.debug("write_strategy_note_fallback_to_file", path=str(note_path))
        note_path.write_text(content)

    return note_path


def update_memory_index(projects: list[dict] | None = None) -> Path | None:
    """Regenerate MEMORY.md at vault root with project listing.

    If *projects* is None, scans ``10-Projects/`` for subdirectories and
    reads each dashboard note for the latest score.

    Returns ``None`` when the vault is not configured.
    """
    vault = _get_vault_path()
    if vault is None:
        log.debug("update_memory_index_skipped", reason="vault not configured")
        return None
    log.debug("update_memory_index", vault=str(vault))

    if projects is None:
        projects = []
        projects_root = vault / _PROJECTS_DIR
        if projects_root.exists():
            for subdir in sorted(projects_root.iterdir()):
                if not subdir.is_dir():
                    continue
                name = subdir.name
                dashboard = subdir / f"{name}.md"
                score = "n/a"
                exp_count = 0
                if dashboard.exists():
                    content = dashboard.read_text(errors="replace")
                    score_match = re.search(r"\*\*Current Score\*\*:\s*(\S+)", content)
                    if score_match:
                        score = score_match.group(1)
                    exp_match = re.search(r"\*\*Experiments Run\*\*:\s*(\d+)", content)
                    if exp_match:
                        exp_count = int(exp_match.group(1))
                projects.append(
                    {
                        "name": name,
                        "score": score,
                        "experiments": exp_count,
                    }
                )

    lines = [
        "# Factory Memory Index",
        "",
        "Pointer file for factory agents.",
        "",
        "## Projects",
        "",
    ]

    if projects:
        for p in projects:
            lines.append(f"- [[{p['name']}]] — score: {p['score']}, {p['experiments']} experiments")
    else:
        lines.append("(none yet)")

    lines.append("")

    memory_path = vault / "MEMORY.md"
    content = "\n".join(lines)

    # Try obsidian-cli first, fall back to direct write
    if not _obsidian_create("MEMORY", content):
        log.debug("update_memory_index_fallback_to_file", path=str(memory_path))
        memory_path.write_text(content)

    log.info("update_memory_index_complete", project_count=len(projects))
    return memory_path
