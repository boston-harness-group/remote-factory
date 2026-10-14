"""Tests for enriched lifecycle events (issue #556)."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from factory.events import emit_event, load_events


def _setup_factory_dir(project: Path) -> None:
    """Create minimal .factory directory for event emission."""
    factory_dir = project / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)
    (factory_dir / "strategy").mkdir(parents=True, exist_ok=True)


# ── worktree events ──────────────────────────────────────────


@pytest.mark.real_worktree
def test_create_worktree_emits_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[0] == "git" and "worktree" in cmd and "add" in cmd:
            wt_path = Path(cmd[3])
            wt_path.mkdir(parents=True, exist_ok=True)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        from factory.worktree import create_worktree
        create_worktree(project, "main")

    events = load_events(project)
    wt_events = [e for e in events if e["type"] == "worktree.created"]
    assert len(wt_events) == 1
    data = wt_events[0]["data"]
    assert "run_id" in data
    assert "worktree_path" in data
    assert data["branch"].startswith("factory/run-")
    assert data["base_branch"] == "main"


@pytest.mark.real_worktree
def test_remove_worktree_emits_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    captured: list[dict] = []
    real_emit = emit_event

    def spy_emit(proj_path, event_type, *, agent=None, data=None):
        result = real_emit(proj_path, event_type, agent=agent, data=data)
        captured.append(result)
        return result

    wt_path = tmp_path / "fake-wt"

    with patch("factory.events.emit_event", side_effect=spy_emit), \
         patch("subprocess.run"):
        from factory.worktree import remove_worktree
        remove_worktree(project, wt_path, "factory/run-abc12345")

    events = load_events(project)
    rm_events = [e for e in events if e["type"] == "worktree.removed"]
    assert len(rm_events) == 1
    data = rm_events[0]["data"]
    assert data["run_id"] == "abc12345"
    assert data["branch"] == "factory/run-abc12345"


def test_worktree_created_event_schema(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    event = emit_event(project, "worktree.created", data={
        "run_id": "abc12345",
        "worktree_path": str(project / ".factory-worktrees" / "run-abc12345"),
        "branch": "factory/run-abc12345",
        "base_branch": "main",
    })

    assert event["type"] == "worktree.created"
    assert event["data"]["run_id"] == "abc12345"
    assert event["data"]["branch"] == "factory/run-abc12345"
    assert event["data"]["base_branch"] == "main"
    assert "worktree_path" in event["data"]


def test_worktree_removed_event_schema(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    event = emit_event(project, "worktree.removed", data={
        "run_id": "abc12345",
        "branch": "factory/run-abc12345",
    })

    assert event["type"] == "worktree.removed"
    assert event["data"]["run_id"] == "abc12345"
    assert event["data"]["branch"] == "factory/run-abc12345"


def test_worktree_events_persisted_to_jsonl(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    emit_event(project, "worktree.created", data={
        "run_id": "dead0001",
        "worktree_path": "/tmp/wt",
        "branch": "factory/run-dead0001",
        "base_branch": "main",
    })
    emit_event(project, "worktree.removed", data={
        "run_id": "dead0001",
        "branch": "factory/run-dead0001",
    })

    events = load_events(project)
    types = [e["type"] for e in events]
    assert "worktree.created" in types
    assert "worktree.removed" in types


# ── experiment.finalize enrichment ────────────────────────────


def test_finalize_event_includes_enriched_fields(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    event = emit_event(project, "experiment.finalize", data={
        "exp_id": 1,
        "verdict": "keep",
        "hypothesis": "test hypothesis",
        "pr_number": 42,
        "issue_number": 10,
        "score_before": 0.65,
        "score_after": 0.78,
        "delta": 0.13,
        "cost_usd": 1.23,
    })

    data = event["data"]
    assert data["pr_number"] == 42
    assert data["issue_number"] == 10
    assert data["score_before"] == 0.65
    assert data["score_after"] == 0.78
    assert data["delta"] == 0.13
    assert data["cost_usd"] == 1.23


def test_cmd_finalize_emits_enriched_event(tmp_path: Path) -> None:
    """cmd_finalize emits experiment.finalize with enriched fields."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    ns = argparse.Namespace(
        path=str(project),
        id=1,
        verdict="keep",
        hypothesis="Improve logging",
        summary="Added structlog",
        cost=2.50,
        issue=99,
        pr=55,
        score_before=0.60,
        score_after=0.75,
        notes="",
        force=True,
    )

    mock_store = MagicMock()

    with patch("factory.store.ExperimentStore", return_value=mock_store), \
         patch("factory.cli.store._run", return_value=None):
        from factory.cli import cmd_finalize
        cmd_finalize(ns)

    events = load_events(project)
    finalize_events = [e for e in events if e["type"] == "experiment.finalize"]
    assert len(finalize_events) == 1

    data = finalize_events[0]["data"]
    assert data["pr_number"] == 55
    assert data["issue_number"] == 99
    assert data["score_before"] == 0.60
    assert data["score_after"] == 0.75
    assert data["delta"] == 0.15
    assert data["cost_usd"] == 2.50


def test_finalize_autodetects_pr_number(tmp_path: Path) -> None:
    """cmd_finalize auto-detects PR number via gh when args.pr is None."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    ns = argparse.Namespace(
        path=str(project),
        id=1,
        verdict="keep",
        hypothesis="Auto PR detection",
        summary="Testing auto PR",
        cost=1.00,
        issue=42,
        pr=None,
        score_before=0.50,
        score_after=0.60,
        notes="",
        force=True,
    )

    mock_store = MagicMock()
    fake_gh_result = MagicMock(returncode=0, stdout=b"123\n")

    with patch("factory.store.ExperimentStore", return_value=mock_store), \
         patch("factory.cli.store._run", return_value=None), \
         patch("subprocess.run", return_value=fake_gh_result):
        from factory.cli import cmd_finalize
        cmd_finalize(ns)

    events = load_events(project)
    finalize_events = [e for e in events if e["type"] == "experiment.finalize"]
    assert len(finalize_events) == 1
    assert finalize_events[0]["data"]["pr_number"] == 123


def test_finalize_event_with_null_scores(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    ns = argparse.Namespace(
        path=str(project),
        id=2,
        verdict="revert",
        hypothesis="Bad idea",
        summary="Reverted",
        cost=None,
        issue=None,
        pr=None,
        score_before=None,
        score_after=None,
        notes="",
        force=True,
    )

    mock_store = MagicMock()

    with patch("factory.store.ExperimentStore", return_value=mock_store), \
         patch("factory.cli.store._run", return_value=None), \
         patch("factory.events.load_events", return_value=[]), \
         patch("factory.events.sum_agent_usage", return_value={}):
        from factory.cli import cmd_finalize
        cmd_finalize(ns)

    events_file = project / ".factory" / "events.jsonl"
    raw_events = []
    for line in events_file.read_text().splitlines():
        if line.strip():
            raw_events.append(json.loads(line))

    finalize_events = [e for e in raw_events if e["type"] == "experiment.finalize"]
    assert len(finalize_events) >= 1

    data = finalize_events[-1]["data"]
    assert data["pr_number"] is None
    assert data["issue_number"] is None
    assert data["score_before"] is None
    assert data["score_after"] is None
    assert data["delta"] is None


# ── backlog events ────────────────────────────────────────────


def test_backlog_add_emits_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    ns = argparse.Namespace(
        path=str(project),
        item="Add dark mode support",
    )

    from factory.cli import cmd_backlog_add
    result = cmd_backlog_add(ns)

    assert result == 0

    events = load_events(project)
    add_events = [e for e in events if e["type"] == "backlog.added"]
    assert len(add_events) == 1
    assert add_events[0]["data"]["item"] == "Add dark mode support"


def test_backlog_remove_emits_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    backlog_path = project / ".factory" / "strategy" / "backlog.md"
    backlog_path.write_text("- Fix login bug\n")

    ns = argparse.Namespace(
        path=str(project),
        item="Fix login bug",
    )

    from factory.cli import cmd_backlog_remove
    result = cmd_backlog_remove(ns)

    assert result == 0

    events = load_events(project)
    remove_events = [e for e in events if e["type"] == "backlog.removed"]
    assert len(remove_events) == 1
    assert remove_events[0]["data"]["item"] == "Fix login bug"


def test_backlog_add_duplicate_no_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    backlog_path = project / ".factory" / "strategy" / "backlog.md"
    backlog_path.write_text("- Existing item\n")

    ns = argparse.Namespace(
        path=str(project),
        item="Existing item",
    )

    from factory.cli import cmd_backlog_add
    result = cmd_backlog_add(ns)

    assert result == 1

    events = load_events(project)
    add_events = [e for e in events if e["type"] == "backlog.added"]
    assert len(add_events) == 0


def test_backlog_remove_missing_no_event(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    ns = argparse.Namespace(
        path=str(project),
        item="Nonexistent item",
    )

    from factory.cli import cmd_backlog_remove
    result = cmd_backlog_remove(ns)

    assert result == 1

    events = load_events(project)
    remove_events = [e for e in events if e["type"] == "backlog.removed"]
    assert len(remove_events) == 0


# ── exception-handler coverage ──────────────────────────────


@pytest.mark.real_worktree
def test_create_worktree_event_exception_swallowed(tmp_path: Path) -> None:
    """create_worktree silently swallows emit_event failures."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[0] == "git" and "worktree" in cmd and "add" in cmd:
            wt_path = Path(cmd[3])
            wt_path.mkdir(parents=True, exist_ok=True)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run), \
         patch("factory.events.emit_event", side_effect=RuntimeError("boom")):
        from factory.worktree import create_worktree
        wt_path, branch = create_worktree(project, "main")

    assert wt_path.exists()
    assert branch.startswith("factory/run-")


@pytest.mark.real_worktree
def test_remove_worktree_event_exception_swallowed(tmp_path: Path) -> None:
    """remove_worktree silently swallows emit_event failures."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    wt_path = tmp_path / "fake-wt"

    with patch("factory.events.emit_event", side_effect=RuntimeError("boom")), \
         patch("subprocess.run"):
        from factory.worktree import remove_worktree
        remove_worktree(project, wt_path, "factory/run-abc12345")


# ── auto-cost with experiment.begin event ────────────────────


def test_finalize_auto_cost_from_events(tmp_path: Path) -> None:
    """cmd_finalize auto-calculates cost from events when cost arg is None."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    emit_event(project, "experiment.begin", data={"exp_id": 3})
    emit_event(project, "agent.completed", data={
        "total_cost_usd": 0.75,
        "input_tokens": 100,
        "output_tokens": 50,
    })

    ns = argparse.Namespace(
        path=str(project),
        id=3,
        verdict="keep",
        hypothesis="Auto cost test",
        summary="Testing auto cost",
        cost=None,
        issue=None,
        pr=None,
        score_before=0.50,
        score_after=0.60,
        notes="",
        force=True,
    )

    mock_store = MagicMock()

    with patch("factory.store.ExperimentStore", return_value=mock_store), \
         patch("factory.cli.store._run", return_value=None):
        from factory.cli import cmd_finalize
        cmd_finalize(ns)

    events = load_events(project)
    finalize_events = [e for e in events if e["type"] == "experiment.finalize"]
    assert len(finalize_events) == 1
    data = finalize_events[0]["data"]
    assert data["cost_usd"] is not None


# ── _emit_cli_event exception handler ────────────────────────


def test_emit_cli_event_exception_swallowed(tmp_path: Path) -> None:
    """_emit_cli_event silently swallows emit_event failures."""
    from factory.cli._helpers import _emit_cli_event

    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    with patch("factory.events.emit_event", side_effect=RuntimeError("boom")):
        _emit_cli_event(project, "test.event", {"key": "value"})


# ── verdict override coverage ────────────────────────────────


def test_finalize_precheck_overrides_verdict_emits_event(tmp_path: Path) -> None:
    """cmd_finalize overrides keep→revert on precheck failure and emits verdict.overridden."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    config_data = {
        "goal": "test project",
        "scope": ["factory/"],
        "guards": [],
        "eval_command": "echo ok",
        "eval_threshold": 0.5,
        "constraints": [],
        "smoke_test": "",
        "hard_constraints": [],
    }
    config_path = project / ".factory" / "config.json"
    config_path.write_text(json.dumps(config_data))

    @dataclass
    class FakePreCheckResult:
        passed: bool
        checks: list = field(default_factory=list)
        blocking_failures: list = field(default_factory=list)

    failed_result = FakePreCheckResult(
        passed=False,
        blocking_failures=["score_regression"],
    )

    ns = argparse.Namespace(
        path=str(project),
        id=5,
        verdict="keep",
        hypothesis="Should be overridden",
        summary="Testing override",
        cost=1.00,
        issue=20,
        pr=30,
        score_before=0.80,
        score_after=0.70,
        notes="",
        force=False,
    )

    mock_store = MagicMock()
    mock_store.load_history = MagicMock(return_value=[])

    with patch("factory.store.ExperimentStore", return_value=mock_store), \
         patch("factory.cli.store._run", side_effect=[[], None]), \
         patch("factory.precheck.run_precheck", return_value=failed_result):
        from factory.cli import cmd_finalize
        cmd_finalize(ns)

    events = load_events(project)
    override_events = [e for e in events if e["type"] == "verdict.overridden"]
    assert len(override_events) == 1
    data = override_events[0]["data"]
    assert data["exp_id"] == 5
    assert data["original_verdict"] == "keep"
    assert data["new_verdict"] == "revert"
    assert "score_regression" in data["reason"]

    finalize_events = [e for e in events if e["type"] == "experiment.finalize"]
    assert len(finalize_events) == 1
    assert finalize_events[0]["data"]["verdict"] == "revert"


# ── worktree cleanup & removal path coverage ─────────────────


@pytest.mark.real_worktree
def test_create_worktree_cleans_existing_factory_dir(tmp_path: Path) -> None:
    """create_worktree removes an existing .factory dir in the worktree."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)
    (project / ".factory" / "config.json").write_text("{}")

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[0] == "git" and "worktree" in cmd and "add" in cmd:
            wt_path = Path(cmd[3])
            wt_path.mkdir(parents=True, exist_ok=True)
            (wt_path / ".factory").mkdir()
            (wt_path / ".factory" / "dummy.txt").write_text("stale")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        from factory.worktree import create_worktree
        wt_path, branch = create_worktree(project, "main")

    wt_factory = wt_path / ".factory"
    assert wt_factory.is_dir() and not wt_factory.is_symlink()
    assert (wt_factory / "config.json").is_symlink()
    assert (wt_factory / "strategy").is_dir()
    assert not (wt_factory / "dummy.txt").exists()


@pytest.mark.real_worktree
def test_create_worktree_cleans_existing_factory_symlink(tmp_path: Path) -> None:
    """create_worktree removes an existing .factory symlink in the worktree."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)
    (project / ".factory" / "config.json").write_text("{}")

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[0] == "git" and "worktree" in cmd and "add" in cmd:
            wt_path = Path(cmd[3])
            wt_path.mkdir(parents=True, exist_ok=True)
            stale_target = tmp_path / "stale-factory"
            stale_target.mkdir()
            (wt_path / ".factory").symlink_to(stale_target)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        from factory.worktree import create_worktree
        wt_path, branch = create_worktree(project, "main")

    wt_factory = wt_path / ".factory"
    assert wt_factory.is_dir() and not wt_factory.is_symlink()
    assert (wt_factory / "config.json").is_symlink()
    assert (wt_factory / "strategy").is_dir()


@pytest.mark.real_worktree
def test_remove_worktree_deletes_existing_path(tmp_path: Path) -> None:
    """remove_worktree calls rmtree when the worktree path exists on disk."""
    project = tmp_path / "proj"
    project.mkdir()
    _setup_factory_dir(project)

    wt_path = tmp_path / "existing-wt"
    wt_path.mkdir()
    (wt_path / "some_file.py").write_text("content")

    with patch("subprocess.run"):
        from factory.worktree import remove_worktree
        remove_worktree(project, wt_path, "factory/run-dead0002")

    assert not wt_path.exists()

    events = load_events(project)
    rm_events = [e for e in events if e["type"] == "worktree.removed"]
    assert len(rm_events) == 1
    assert rm_events[0]["data"]["run_id"] == "dead0002"
