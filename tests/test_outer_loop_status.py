"""Tests for outer-loop status in-progress candidate detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory.cli.outer_loop import (
    _format_elapsed,
    _get_last_agent_phase,
    _scan_eval_worktrees,
)


class TestScanEvalWorktrees:
    def test_no_directory(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        result = _scan_eval_worktrees(project)
        assert result == {}

    def test_empty_directory(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (tmp_path / ".eval-worktrees").mkdir()
        result = _scan_eval_worktrees(project)
        assert result == {}

    def test_valid_worktrees(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        (wt_base / "wt-evolve-gen0-abcd1234").mkdir()
        (wt_base / "wt-mymode-deadbeef").mkdir()
        (wt_base / "not-a-worktree").mkdir()
        (wt_base / "wt-bad").mkdir()  # no uuid suffix

        result = _scan_eval_worktrees(project)
        assert "evolve-gen0" in result
        assert "mymode" in result
        assert len(result) == 2

    def test_label_with_hyphens(self, tmp_path: Path) -> None:
        """Labels like 'evolve-gen0-0d4cc86a' where the label itself has hyphens."""
        project = tmp_path / "project"
        project.mkdir()
        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        (wt_base / "wt-evolve-gen0-0d4cc86a-7a8111fd").mkdir()

        result = _scan_eval_worktrees(project)
        assert "evolve-gen0-0d4cc86a" in result

    def test_race_condition_dir_disappears(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        wt = wt_base / "wt-test-12345678"
        wt.mkdir()
        result = _scan_eval_worktrees(project)
        assert "test" in result

    def test_permission_error(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        result = _scan_eval_worktrees(project)
        assert result == {}

    def test_file_not_dir_skipped(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        (wt_base / "wt-file-12345678").write_text("not a dir")
        result = _scan_eval_worktrees(project)
        assert result == {}


class TestGetLastAgentPhase:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert _get_last_agent_phase(tmp_path) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        events_dir = tmp_path / ".factory"
        events_dir.mkdir(parents=True)
        (events_dir / "events.jsonl").write_text("")
        assert _get_last_agent_phase(tmp_path) is None

    def test_valid_events(self, tmp_path: Path) -> None:
        events_dir = tmp_path / ".factory"
        events_dir.mkdir(parents=True)
        lines = [
            json.dumps({"type": "agent.started", "agent": "researcher"}),
            json.dumps({"type": "agent.completed", "agent": "researcher"}),
            json.dumps({"type": "agent.started", "agent": "builder"}),
        ]
        (events_dir / "events.jsonl").write_text("\n".join(lines) + "\n")
        assert _get_last_agent_phase(tmp_path) == "builder"

    def test_no_agent_started_events(self, tmp_path: Path) -> None:
        events_dir = tmp_path / ".factory"
        events_dir.mkdir(parents=True)
        lines = [
            json.dumps({"type": "cycle.started"}),
            json.dumps({"type": "agent.completed", "agent": "builder"}),
        ]
        (events_dir / "events.jsonl").write_text("\n".join(lines) + "\n")
        assert _get_last_agent_phase(tmp_path) is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        events_dir = tmp_path / ".factory"
        events_dir.mkdir(parents=True)
        lines = [
            json.dumps({"type": "agent.started", "agent": "researcher"}),
            "this is not json{{{",
            json.dumps({"type": "agent.started", "agent": "builder"}),
        ]
        (events_dir / "events.jsonl").write_text("\n".join(lines) + "\n")
        assert _get_last_agent_phase(tmp_path) == "builder"

    def test_blank_lines(self, tmp_path: Path) -> None:
        events_dir = tmp_path / ".factory"
        events_dir.mkdir(parents=True)
        content = (
            json.dumps({"type": "agent.started", "agent": "strategist"})
            + "\n\n\n"
        )
        (events_dir / "events.jsonl").write_text(content)
        assert _get_last_agent_phase(tmp_path) == "strategist"


class TestFormatElapsed:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (5, "5s"),
            (59, "59s"),
            (60, "1m00s"),
            (61, "1m01s"),
            (192, "3m12s"),
            (3599, "59m59s"),
            (3600, "1h00m"),
            (3900, "1h05m"),
            (7261, "2h01m"),
            (-5, "0s"),
        ],
    )
    def test_format(self, seconds: float, expected: str) -> None:
        assert _format_elapsed(seconds) == expected


class TestStatusClassification:
    """Test the classification logic: completed vs in-progress vs pending."""

    def _setup_project(self, tmp_path: Path, gen: int = 0) -> Path:
        project = tmp_path / "project"
        project.mkdir()
        ol_dir = project / ".factory" / "outer_loop"
        ol_dir.mkdir(parents=True)
        return project

    def test_completed_mode(self, tmp_path: Path) -> None:
        project = self._setup_project(tmp_path)
        mode = "evolve-gen0-abc12345"
        runs_dir = project / ".factory" / "outer_loop" / "runs" / mode
        runs_dir.mkdir(parents=True)
        (runs_dir / "cycle_summary.json").write_text(json.dumps({"score": 0.5}))

        summary_path = runs_dir / "cycle_summary.json"
        assert summary_path.exists()

    def test_in_progress_mode(self, tmp_path: Path) -> None:
        project = self._setup_project(tmp_path)
        mode = "evolve-gen0-abc12345"

        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        wt = wt_base / "wt-evolve-gen0-deadbeef"
        wt.mkdir()
        events_dir = wt / ".factory"
        events_dir.mkdir(parents=True)
        (events_dir / "events.jsonl").write_text(
            json.dumps({"type": "agent.started", "agent": "builder"}) + "\n"
        )

        worktrees = _scan_eval_worktrees(project)
        assert len(worktrees) == 1
        label = list(worktrees.keys())[0]
        assert label == "evolve-gen0"

        runs_dir = project / ".factory" / "outer_loop" / "runs" / mode
        runs_dir.mkdir(parents=True)
        summary_path = runs_dir / "cycle_summary.json"
        assert not summary_path.exists()

        wt_path = worktrees[label]
        phase = _get_last_agent_phase(wt_path)
        assert phase == "builder"

    def test_pending_mode(self, tmp_path: Path) -> None:
        project = self._setup_project(tmp_path)
        mode = "evolve-gen0-abc12345"

        runs_dir = project / ".factory" / "outer_loop" / "runs" / mode
        runs_dir.mkdir(parents=True)
        summary_path = runs_dir / "cycle_summary.json"
        assert not summary_path.exists()

        worktrees = _scan_eval_worktrees(project)
        assert len(worktrees) == 0

    def test_multi_mode_single_worktree_consume_match(self, tmp_path: Path) -> None:
        """One worktree must match at most one mode; unmatched modes stay pending."""
        import time

        project = self._setup_project(tmp_path)
        runs_dir = project / ".factory" / "outer_loop" / "runs"

        modes = [
            "evolve-gen0-aaa11111",
            "evolve-gen0-bbb22222",
            "evolve-gen0-ccc33333",
        ]

        # ccc33333 is completed (has cycle_summary)
        (runs_dir / modes[2]).mkdir(parents=True)
        (runs_dir / modes[2] / "cycle_summary.json").write_text(
            json.dumps({"score": 0.75})
        )
        # aaa11111 and bbb22222 have no cycle_summary
        (runs_dir / modes[0]).mkdir(parents=True)
        (runs_dir / modes[1]).mkdir(parents=True)

        # One worktree whose label is "evolve-gen0" — shared prefix for all 3 modes
        wt_base = tmp_path / ".eval-worktrees"
        wt_base.mkdir()
        wt = wt_base / "wt-evolve-gen0-deadbeef"
        wt.mkdir()
        events_dir = wt / ".factory"
        events_dir.mkdir(parents=True)
        (events_dir / "events.jsonl").write_text(
            json.dumps({"type": "agent.started", "agent": "builder"}) + "\n"
        )

        # Reproduce the classification logic from _cmd_status
        worktrees = _scan_eval_worktrees(project)
        assert len(worktrees) == 1

        completed: list[str] = []
        in_progress: list[tuple[str, str | None, float, str]] = []
        pending: list[str] = []
        now = time.time()

        for mode_name in modes:
            summary_path = runs_dir / mode_name / "cycle_summary.json"
            if summary_path.exists():
                completed.append(mode_name)
                continue
            matched_wt: str | None = None
            for label, wt_path in worktrees.items():
                if mode_name.startswith(label) or label.startswith(mode_name[:12]):
                    matched_wt = label
                    break
            if matched_wt is not None:
                wt_path = worktrees[matched_wt]
                del worktrees[matched_wt]
                phase = _get_last_agent_phase(wt_path)
                try:
                    elapsed = now - wt_path.stat().st_mtime
                except OSError:
                    elapsed = 0.0
                wt_dir_name = wt_path.name
                in_progress.append((mode_name, phase, elapsed, wt_dir_name))
            else:
                pending.append(mode_name)

        assert len(completed) == 1, f"Expected 1 completed, got {completed}"
        assert completed[0] == "evolve-gen0-ccc33333"
        assert len(in_progress) == 1, f"Expected 1 in-progress, got {in_progress}"
        assert in_progress[0][0] == "evolve-gen0-aaa11111"
        assert in_progress[0][1] == "builder"
        assert len(pending) == 1, f"Expected 1 pending, got {pending}"
        assert pending[0] == "evolve-gen0-bbb22222"
