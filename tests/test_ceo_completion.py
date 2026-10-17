"""Tests for factory/ceo_completion.py — CEO completion guard."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestCycleState:
    """Tests for cycle state persistence (read, write, delete, staleness)."""

    def test_write_and_read_cycle_state(self, tmp_path: Path) -> None:
        """Cycle state can be written and read back."""
        from factory.ceo_completion import (
            create_cycle_state,
            read_cycle_state,
            write_cycle_state,
        )

        state = create_cycle_state("design", "Build a CLI tool")
        write_cycle_state(tmp_path, state)

        loaded = read_cycle_state(tmp_path)
        assert loaded is not None
        assert loaded.cycle_id == state.cycle_id
        assert loaded.mode == "design"
        assert loaded.initial_prompt == "Build a CLI tool"
        assert loaded.respawns == 0

    def test_read_cycle_state_nonexistent(self, tmp_path: Path) -> None:
        """read_cycle_state returns None if cycle.json doesn't exist."""
        from factory.ceo_completion import read_cycle_state

        assert read_cycle_state(tmp_path) is None

    def test_delete_cycle_state(self, tmp_path: Path) -> None:
        """delete_cycle_state removes the file and returns True."""
        from factory.ceo_completion import (
            create_cycle_state,
            delete_cycle_state,
            read_cycle_state,
            write_cycle_state,
        )

        state = create_cycle_state("design")
        write_cycle_state(tmp_path, state)
        assert read_cycle_state(tmp_path) is not None

        deleted = delete_cycle_state(tmp_path)
        assert deleted is True
        assert read_cycle_state(tmp_path) is None

    def test_delete_cycle_state_nonexistent(self, tmp_path: Path) -> None:
        """delete_cycle_state returns False if file doesn't exist."""
        from factory.ceo_completion import delete_cycle_state

        assert delete_cycle_state(tmp_path) is False

    def test_stale_cycle_state_ignored(self, tmp_path: Path) -> None:
        """Cycle state older than 24 hours is treated as stale and ignored."""
        from factory.ceo_completion import (
            CYCLE_STALENESS_HOURS,
            read_cycle_state,
            _cycle_state_path,
        )

        # Write a cycle state with old timestamp
        state_path = _cycle_state_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        old_time = datetime.now(timezone.utc) - timedelta(hours=CYCLE_STALENESS_HOURS + 1)
        state_data = {
            "cycle_id": "old123",
            "started_at": old_time.isoformat(),
            "mode": "design",
            "initial_prompt": "",
            "respawns": 5,
        }
        state_path.write_text(json.dumps(state_data))

        # Should return None due to staleness
        loaded = read_cycle_state(tmp_path)
        assert loaded is None

    def test_malformed_cycle_state_ignored(self, tmp_path: Path) -> None:
        """Malformed cycle.json returns None instead of crashing."""
        from factory.ceo_completion import read_cycle_state, _cycle_state_path

        state_path = _cycle_state_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("NOT VALID JSON {{{")

        assert read_cycle_state(tmp_path) is None

    def test_cycle_state_truncates_long_prompt(self, tmp_path: Path) -> None:
        """Initial prompt is truncated to avoid bloat."""
        from factory.ceo_completion import create_cycle_state, write_cycle_state, read_cycle_state

        long_prompt = "x" * 5000
        state = create_cycle_state("design", long_prompt)
        write_cycle_state(tmp_path, state)

        loaded = read_cycle_state(tmp_path)
        assert loaded is not None
        assert len(loaded.initial_prompt) <= 1000


class TestBudgetAllowsRespawn:
    """Tests for _budget_allows_respawn().

    With only per-cycle limits (no daily/session limit), respawn is always allowed.
    """

    def test_claude_always_allowed(self, tmp_path: Path) -> None:
        from factory.ceo_completion import _budget_allows_respawn

        assert _budget_allows_respawn("claude", tmp_path) is True
        assert _budget_allows_respawn(None, tmp_path) is True


class TestDetectIncomplete:
    """Tests for _detect_incomplete()."""

    def test_design_incomplete_no_eval_profile(self, tmp_path: Path) -> None:
        """Build mode without strategy needs eval profile."""
        from factory.ceo_completion import _detect_incomplete

        (tmp_path / ".factory").mkdir()

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is not None
        assert gap.mode == "design"
        assert gap.next_item == "discovery"
        assert "no eval profile" in gap.reason

    def test_improve_complete_when_all_verdicts(self, tmp_path: Path) -> None:
        """Improve mode is complete when verdict count >= hypothesis count."""
        from factory.ceo_completion import _detect_incomplete

        # Setup: 2 hypotheses in strategy
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: First\n\n#### H2: Second\n"
        )

        # Setup: 2 verdicts
        for i in (1, 2):
            exp_dir = tmp_path / ".factory" / "experiments" / f"00{i}"
            exp_dir.mkdir(parents=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is None

    def test_design_incomplete_when_missing_verdicts(self, tmp_path: Path) -> None:
        """Improve mode is incomplete when verdict count < hypothesis count."""
        from factory.ceo_completion import _detect_incomplete

        # Setup: 3 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: First\n\n#### H2: Second\n\n#### H3: Third\n"
        )

        # Setup: only 1 verdict
        exp_dir = tmp_path / ".factory" / "experiments" / "001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is not None
        assert gap.planned == 3
        assert gap.completed == 1
        assert gap.next_item == "H2"
        assert "design.incomplete" in gap.reason

    def test_design_no_strategy_no_eval_profile_returns_gap(self, tmp_path: Path) -> None:
        """No strategy + no eval profile means discovery is needed."""
        from factory.ceo_completion import _detect_incomplete

        (tmp_path / ".factory").mkdir()

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is not None
        assert "no eval profile" in gap.reason

    def test_design_no_strategy_with_eval_profile_returns_none(self, tmp_path: Path) -> None:
        """No strategy but eval profile exists means nothing planned — not incomplete."""
        from factory.ceo_completion import _detect_incomplete

        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        (factory_dir / "eval_profile.json").write_text('{"dimensions": []}')

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is None

    def test_discover_complete_when_profile_exists(self, tmp_path: Path) -> None:
        """Discover mode is complete when eval_profile.json exists."""
        from factory.ceo_completion import _detect_incomplete

        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        (factory_dir / "eval_profile.json").write_text('{"dimensions": []}')

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is None

    def test_design_incomplete_when_no_eval_profile_discover(self, tmp_path: Path) -> None:
        """Design mode is incomplete without eval_profile.json when no hypotheses exist."""
        from factory.ceo_completion import _detect_incomplete

        (tmp_path / ".factory").mkdir()

        gap = _detect_incomplete(tmp_path, "design")
        assert gap is not None
        assert gap.mode == "design"
        assert "no eval profile" in gap.reason


class TestCountVerdictsWithResultsTsv:
    """Tests for _count_verdicts() using results.tsv with timestamp filtering.

    These tests verify the primary code path (results.tsv) rather than the
    fallback path (verdict.json files). The timestamp filtering is critical
    for preventing cross-cycle contamination.
    """

    def _write_results_tsv(self, tmp_path: Path, rows: list[dict]) -> None:
        """Helper to write a results.tsv with the given rows."""
        import csv
        from factory.store import TSV_COLUMNS

        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = factory_dir / "results.tsv"

        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, dialect="excel-tab")
            writer.writeheader()
            for row in rows:
                # Fill in defaults for required columns
                full_row = {col: "" for col in TSV_COLUMNS}
                full_row.update(row)
                writer.writerow(full_row)

    def test_counts_all_verdicts_when_no_since_ts(self, tmp_path: Path) -> None:
        """Without since_ts, all verdicts are counted."""
        from factory.ceo_completion import _count_verdicts

        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "revert"},
                {"id": "3", "timestamp": "2026-04-28T12:00:00+00:00", "verdict": "keep"},
            ],
        )

        count = _count_verdicts(tmp_path)
        assert count == 3

    def test_filters_by_since_ts(self, tmp_path: Path) -> None:
        """With since_ts, only verdicts after that time are counted."""
        from factory.ceo_completion import _count_verdicts

        # Two old rows from a previous cycle, one new row from current cycle
        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "revert"},
                {"id": "3", "timestamp": "2026-04-29T14:00:00+00:00", "verdict": "keep"},
            ],
        )

        # Filter to only count after noon on Apr 29
        since = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        count = _count_verdicts(tmp_path, since_ts=since)
        assert count == 1

    def test_ignores_pending_verdicts(self, tmp_path: Path) -> None:
        """Rows without keep/revert/error verdict are not counted."""
        from factory.ceo_completion import _count_verdicts

        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "pending"},
                {"id": "3", "timestamp": "2026-04-28T12:00:00+00:00", "verdict": ""},
            ],
        )

        count = _count_verdicts(tmp_path)
        assert count == 1

    def test_handles_error_verdict(self, tmp_path: Path) -> None:
        """Error verdicts are counted (they are finalized experiments)."""
        from factory.ceo_completion import _count_verdicts

        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "error"},
            ],
        )

        count = _count_verdicts(tmp_path)
        assert count == 2

    def test_handles_naive_timestamps(self, tmp_path: Path) -> None:
        """Timestamps without timezone are treated as UTC."""
        from factory.ceo_completion import _count_verdicts

        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T10:00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-29T14:00:00", "verdict": "keep"},
            ],
        )

        since = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        count = _count_verdicts(tmp_path, since_ts=since)
        assert count == 1

    def test_cross_cycle_scenario(self, tmp_path: Path) -> None:
        """Realistic scenario: 3 experiments from old cycle, 2 from current cycle."""
        from factory.ceo_completion import _count_verdicts

        # Old cycle started at 2026-04-28T08:00:00
        # Current cycle started at 2026-04-29T10:00:00
        self._write_results_tsv(
            tmp_path,
            [
                # Old cycle experiments
                {"id": "1", "timestamp": "2026-04-28T09:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "revert"},
                {"id": "3", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "keep"},
                # Current cycle experiments
                {"id": "4", "timestamp": "2026-04-29T11:00:00+00:00", "verdict": "keep"},
                {"id": "5", "timestamp": "2026-04-29T12:00:00+00:00", "verdict": "revert"},
            ],
        )

        # Current cycle started at 10:00 on Apr 29
        current_cycle_start = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)
        count = _count_verdicts(tmp_path, since_ts=current_cycle_start)
        assert count == 2


class TestDetectIncompleteWithTimestampFiltering:
    """Tests for _detect_incomplete() with cycle_started_at parameter.

    These tests verify that _detect_incomplete correctly scopes verdict counting
    to the current cycle, preventing cross-cycle contamination.
    """

    def _write_results_tsv(self, tmp_path: Path, rows: list[dict]) -> None:
        """Helper to write a results.tsv with the given rows."""
        import csv
        from factory.store import TSV_COLUMNS

        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = factory_dir / "results.tsv"

        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, dialect="excel-tab")
            writer.writeheader()
            for row in rows:
                full_row = {col: "" for col in TSV_COLUMNS}
                full_row.update(row)
                writer.writerow(full_row)

    def test_improve_filters_by_cycle_start(self, tmp_path: Path) -> None:
        """Improve mode only counts verdicts from the current cycle."""
        from factory.ceo_completion import _detect_incomplete

        # Setup: 2 hypotheses in current strategy
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: First\n\n#### H2: Second\n"
        )

        # 3 old verdicts from previous cycle, 1 from current cycle
        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T09:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "3", "timestamp": "2026-04-28T11:00:00+00:00", "verdict": "keep"},
                {"id": "4", "timestamp": "2026-04-29T11:00:00+00:00", "verdict": "keep"},
            ],
        )

        # Current cycle started at 10:00 on Apr 29 — only 1 verdict should count
        cycle_start = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)
        gap = _detect_incomplete(tmp_path, "design", cycle_started_at=cycle_start)

        # Should be incomplete: 2 hypotheses, only 1 current-cycle verdict
        assert gap is not None
        assert gap.planned == 2
        assert gap.completed == 1
        assert gap.next_item == "H2"

    def test_improve_complete_with_cycle_filtering(self, tmp_path: Path) -> None:
        """Improve mode is complete when current-cycle verdicts match hypotheses."""
        from factory.ceo_completion import _detect_incomplete

        # Setup: 2 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: First\n\n#### H2: Second\n"
        )

        # 1 old verdict, 2 current-cycle verdicts
        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T09:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-29T11:00:00+00:00", "verdict": "keep"},
                {"id": "3", "timestamp": "2026-04-29T12:00:00+00:00", "verdict": "revert"},
            ],
        )

        cycle_start = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)
        gap = _detect_incomplete(tmp_path, "design", cycle_started_at=cycle_start)

        # Should be complete: 2 hypotheses, 2 current-cycle verdicts
        assert gap is None

    def test_build_filters_by_cycle_start(self, tmp_path: Path) -> None:
        """Build mode only counts verdicts from the current cycle."""
        from factory.ceo_completion import _detect_incomplete

        # Setup: 3 phases in build plan
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Build Plan\n\n#### H1: Phase 1\n\n#### H2: Phase 2\n\n#### H3: Phase 3\n"
        )

        # 2 old verdicts, 1 current
        self._write_results_tsv(
            tmp_path,
            [
                {"id": "1", "timestamp": "2026-04-28T09:00:00+00:00", "verdict": "keep"},
                {"id": "2", "timestamp": "2026-04-28T10:00:00+00:00", "verdict": "keep"},
                {"id": "3", "timestamp": "2026-04-29T11:00:00+00:00", "verdict": "keep"},
            ],
        )

        cycle_start = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)
        gap = _detect_incomplete(tmp_path, "design", cycle_started_at=cycle_start)

        # Should be incomplete: 3 phases, only 1 current-cycle verdict
        assert gap is not None
        assert gap.planned == 3
        assert gap.completed == 1
        assert gap.next_item == "H2"


class TestDetectIncompleteResearchMode:
    """Tests for _detect_incomplete() with research mode."""

    def test_research_incomplete_when_missing_verdicts(self, tmp_path: Path) -> None:
        """Research mode is incomplete when verdict count < hypothesis count."""
        from factory.ceo_completion import _detect_incomplete

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: Fix localization\n\n#### H2: Improve planning\n"
        )

        exp_dir = tmp_path / ".factory" / "experiments" / "001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        gap = _detect_incomplete(tmp_path, "research")
        assert gap is not None
        assert gap.mode == "research"
        assert gap.planned == 2
        assert gap.completed == 1
        assert gap.next_item == "H2"
        assert "research.incomplete" in gap.reason

    def test_research_complete_when_all_verdicts(self, tmp_path: Path) -> None:
        """Research mode is complete when all hypotheses have verdicts."""
        from factory.ceo_completion import _detect_incomplete

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text(
            "### Hypotheses\n\n#### H1: Fix localization\n\n#### H2: Improve planning\n"
        )

        for i in (1, 2):
            exp_dir = tmp_path / ".factory" / "experiments" / f"00{i}"
            exp_dir.mkdir(parents=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        gap = _detect_incomplete(tmp_path, "research")
        assert gap is None

    def test_research_no_strategy_returns_none(self, tmp_path: Path) -> None:
        """No strategy file means nothing planned — not incomplete."""
        from factory.ceo_completion import _detect_incomplete

        (tmp_path / ".factory").mkdir()

        gap = _detect_incomplete(tmp_path, "research")
        assert gap is None


class TestBuildContinuationTask:
    """Tests for _build_continuation_task()."""

    def test_improve_continuation(self) -> None:
        """Improve mode continuation tells CEO to spawn Builder for next H."""
        from factory.ceo_completion import _build_continuation_task, IncompleteGap

        gap = IncompleteGap(
            mode="design",
            planned=5,
            completed=2,
            next_item="H3",
            reason="design.incomplete",
        )

        task = _build_continuation_task(gap)
        assert "Resume execution from H3" in task
        assert "do not re-plan" in task
        assert "Spawn Builder for H3" in task
        assert "2/5" in task

    def test_discover_continuation(self) -> None:
        """Discover mode continuation tells CEO to resume discovery."""
        from factory.ceo_completion import _build_continuation_task, IncompleteGap

        gap = IncompleteGap(
            mode="design",
            planned=1,
            completed=0,
            next_item="eval_profile",
            reason="design.incomplete",
        )

        task = _build_continuation_task(gap)
        assert "design" in task.lower()

    def test_design_continuation_with_phases(self) -> None:
        """Design mode continuation tells CEO to resume from next hypothesis."""
        from factory.ceo_completion import _build_continuation_task, IncompleteGap

        gap = IncompleteGap(
            mode="design",
            planned=6,
            completed=3,
            next_item="H4",
            reason="design.incomplete",
        )

        task = _build_continuation_task(gap)
        assert "Resume execution from H4" in task
        assert "3/6" in task

    def test_research_continuation(self) -> None:
        """Research mode continuation tells CEO to spawn Builder for next H."""
        from factory.ceo_completion import _build_continuation_task, IncompleteGap

        gap = IncompleteGap(
            mode="research",
            planned=3,
            completed=1,
            next_item="H2",
            reason="research.incomplete",
        )

        task = _build_continuation_task(gap)
        assert "Resume execution from hypothesis H2" in task
        assert "RESEARCH" in task
        assert "do not re-plan" in task
        assert "1/3" in task
        assert "R1.5" in task

    def test_continuation_includes_mode_directive(self) -> None:
        """Continuation task includes explicit mode directive to prevent flip."""
        from factory.ceo_completion import (
            _build_continuation_task,
            IncompleteGap,
            create_cycle_state,
        )

        gap = IncompleteGap(
            mode="design",
            planned=6,
            completed=3,
            next_item="Phase4",
            reason="design.incomplete",
        )
        cycle_state = create_cycle_state("design", "Build a CLI")

        task = _build_continuation_task(gap, cycle_state)
        assert "## CRITICAL: Mode Override" in task
        assert "CONTINUATION" in task
        assert "DESIGN" in task
        assert "Do NOT re-detect mode" in task
        assert cycle_state.cycle_id in task


class TestRunCeoWithCompletionGuard:
    """Tests for run_ceo_with_completion_guard()."""

    @pytest.fixture(autouse=True)
    def enable_respawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enable respawn for all tests in this class (disabled globally in conftest)."""
        monkeypatch.delenv("FACTORY_CEO_RESPAWN_DISABLED", raising=False)

    async def test_complete_on_first_try_no_respawn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If CEO completes all work in one spawn, no respawn occurs."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        # Setup: 2 hypotheses, 2 verdicts (complete)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n\n#### H2: B\n")

        for i in (1, 2):
            exp_dir = tmp_path / ".factory" / "experiments" / f"00{i}"
            exp_dir.mkdir(parents=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        mock_invoke = AsyncMock(return_value=("CEO output", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        assert code == 0
        assert mock_invoke.call_count == 1

    async def test_respawns_when_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If CEO exits with work undone, it respawns with continuation task."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        # Setup: 3 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n\n#### H2: B\n\n#### H3: C\n")
        (tmp_path / ".factory" / "experiments").mkdir(parents=True)

        call_count = 0

        async def mock_invoke(role, task, path, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call: create 1 verdict
            if call_count == 1:
                exp_dir = path / ".factory" / "experiments" / "001"
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
                return "First run", 0

            # Second call: create 2nd verdict
            if call_count == 2:
                exp_dir = path / ".factory" / "experiments" / "002"
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
                return "Second run", 0

            # Third call: create 3rd verdict (complete)
            exp_dir = path / ".factory" / "experiments" / "003"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
            return "Third run", 0

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        assert code == 0
        assert call_count == 3

        # Check respawn events were emitted
        events_file = tmp_path / ".factory" / "events.jsonl"
        assert events_file.exists()
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        respawn_events = [e for e in events if e["type"] == "ceo.respawn"]
        assert len(respawn_events) == 2

    async def test_respects_user_interrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit code 130 (SIGINT) stops respawning."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        # Setup incomplete
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        mock_invoke = AsyncMock(return_value=("Interrupted", 130))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        assert code == 130
        assert mock_invoke.call_count == 1

    async def test_respects_abort_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cycle.aborted event stops respawning."""
        from factory.ceo_completion import run_ceo_with_completion_guard
        from factory.events import emit_event

        # Setup incomplete
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        async def mock_invoke(role, task, path, **kwargs):
            # CEO emits abort event
            emit_event(path, "cycle.aborted", data={"reason": "unrecoverable"})
            return "Aborted", 1

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        assert code == 1
        # Only one call because abort was respected
        events_file = tmp_path / ".factory" / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        respawn_events = [e for e in events if e["type"] == "ceo.respawn"]
        assert len(respawn_events) == 0

    async def test_cap_hit_writes_incomplete_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After max respawns, writes cycle-incomplete.md and exits non-zero."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        # Setup incomplete - will never complete
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        mock_invoke = AsyncMock(return_value=("Incomplete", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
                max_respawns=2,  # Low cap for test
            )

        assert code == 1
        # 1 initial + 2 respawns = 3 calls
        assert mock_invoke.call_count == 3

        # Check incomplete file was written
        incomplete_file = strategy_dir / "cycle-incomplete.md"
        assert incomplete_file.exists()
        content = incomplete_file.read_text()
        assert "respawn_cap_hit" in content

    async def test_disabled_via_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FACTORY_CEO_RESPAWN_DISABLED=1 makes the guard a no-op."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        monkeypatch.setenv("FACTORY_CEO_RESPAWN_DISABLED", "1")

        # Setup incomplete
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        mock_invoke = AsyncMock(return_value=("Output", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            result, code = await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        # Only one call — no respawn even though incomplete
        assert mock_invoke.call_count == 1

    async def test_creates_cycle_state_on_fresh_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh cycle creates cycle.json with the correct mode."""
        from factory.ceo_completion import run_ceo_with_completion_guard, read_cycle_state

        # Setup complete (so no respawns)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        exp_dir = tmp_path / ".factory" / "experiments" / "001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        mock_invoke = AsyncMock(return_value=("Done", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            # Invoke in build mode
            await run_ceo_with_completion_guard(
                tmp_path,
                "Build task",
                mode="design",
                runner_name="claude",
            )

        # Cycle state should be deleted after completion
        assert read_cycle_state(tmp_path) is None

    async def test_deletes_cycle_state_on_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cycle state is deleted when cycle completes successfully."""
        from factory.ceo_completion import (
            run_ceo_with_completion_guard,
            read_cycle_state,
            _cycle_state_path,
        )

        # Setup complete
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        exp_dir = tmp_path / ".factory" / "experiments" / "001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        async def mock_invoke(role, task, path, **kwargs):
            # Verify cycle state exists during invocation
            assert read_cycle_state(path) is not None
            return "Done", 0

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Improve task",
                mode="design",
                runner_name="claude",
            )

        # After completion, cycle state should be gone
        assert not _cycle_state_path(tmp_path).exists()

    async def test_mode_preserved_across_respawns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mode from initial cycle is preserved across all respawns."""
        from factory.ceo_completion import run_ceo_with_completion_guard, read_cycle_state

        # Setup: 2 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n\n#### H2: B\n")
        (tmp_path / ".factory" / "experiments").mkdir(parents=True)

        call_count = 0
        observed_modes = []

        async def mock_invoke(role, task, path, **kwargs):
            nonlocal call_count
            call_count += 1

            # Record mode from cycle state
            state = read_cycle_state(path)
            if state:
                observed_modes.append(state.mode)

            # First call: create 1 verdict
            if call_count == 1:
                exp_dir = path / ".factory" / "experiments" / "001"
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
                return "First run", 0

            # Second call: create 2nd verdict (complete)
            exp_dir = path / ".factory" / "experiments" / "002"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
            return "Second run", 0

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Build task",
                mode="design",  # Start in design mode
                runner_name="claude",
            )

        assert call_count == 2
        # Both invocations should see the same mode
        assert all(m == "design" for m in observed_modes)

    async def test_respawn_increments_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each respawn increments the respawn counter in cycle state."""
        from factory.ceo_completion import run_ceo_with_completion_guard, read_cycle_state

        # Setup: 3 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n\n#### H2: B\n\n#### H3: C\n")
        (tmp_path / ".factory" / "experiments").mkdir(parents=True)

        call_count = 0
        observed_respawns = []

        async def mock_invoke(role, task, path, **kwargs):
            nonlocal call_count
            call_count += 1

            state = read_cycle_state(path)
            if state:
                observed_respawns.append(state.respawns)

            # Each call creates one verdict
            exp_dir = path / ".factory" / "experiments" / f"00{call_count}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
            return f"Run {call_count}", 0

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Improve task",
                mode="design",
                runner_name="claude",
            )

        assert call_count == 3
        # Respawn counter: 0, 1, 2
        assert observed_respawns == [0, 1, 2]

    async def test_respawn_event_includes_cycle_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Respawn events include the cycle_id for correlation."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        # Setup: 2 hypotheses
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n\n#### H2: B\n")
        (tmp_path / ".factory" / "experiments").mkdir(parents=True)

        call_count = 0

        async def mock_invoke(role, task, path, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                exp_dir = path / ".factory" / "experiments" / "001"
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
                return "First", 0

            exp_dir = path / ".factory" / "experiments" / "002"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')
            return "Second", 0

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Task",
                mode="design",
                runner_name="claude",
            )

        # Check respawn event has cycle_id
        events_file = tmp_path / ".factory" / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        respawn_events = [e for e in events if e["type"] == "ceo.respawn"]
        assert len(respawn_events) == 1
        assert "cycle_id" in respawn_events[0]["data"]
        assert "mode" in respawn_events[0]["data"]
        assert respawn_events[0]["data"]["mode"] == "design"


class TestAutoDetectModeWithCycle:
    """Tests for _auto_detect_mode respecting in-flight cycles."""

    def test_returns_cycle_mode_when_inflight(self, tmp_path: Path) -> None:
        """_auto_detect_mode returns cycle mode when cycle.json exists."""
        from factory.cli._mode_handlers import _auto_detect_mode
        from factory.ceo_completion import create_cycle_state, write_cycle_state

        # Create a git repo so state detection doesn't return NO_REPO
        (tmp_path / ".git").mkdir()

        # Write in-flight cycle state for build mode
        state = create_cycle_state("design", "Initial task")
        write_cycle_state(tmp_path, state)

        # Even though project has no factory, should return build (from cycle)
        mode = _auto_detect_mode(tmp_path, has_prompt=False)
        assert mode == "design"

    def test_ignores_cycle_when_force_fresh(self, tmp_path: Path) -> None:
        """_auto_detect_mode ignores cycle.json when force_fresh=True."""
        from factory.cli._mode_handlers import _auto_detect_mode
        from factory.ceo_completion import create_cycle_state, write_cycle_state

        # Create a git repo
        (tmp_path / ".git").mkdir()

        # Write in-flight cycle state for build mode
        state = create_cycle_state("design", "Initial task")
        write_cycle_state(tmp_path, state)

        # With force_fresh, should detect from state (no_factory → design)
        mode = _auto_detect_mode(tmp_path, has_prompt=False, force_fresh=True)
        assert mode == "design"

    def test_detects_normally_when_no_cycle(self, tmp_path: Path) -> None:
        """_auto_detect_mode detects from project state when no cycle.json."""
        from factory.cli._mode_handlers import _auto_detect_mode

        # Create a git repo
        (tmp_path / ".git").mkdir()

        # No cycle state exists
        mode = _auto_detect_mode(tmp_path, has_prompt=False)
        assert mode == "design"  # no_factory state → design

    def test_detects_normally_when_cycle_stale(self, tmp_path: Path) -> None:
        """_auto_detect_mode ignores stale cycle.json."""
        from factory.cli._mode_handlers import _auto_detect_mode
        from factory.ceo_completion import CYCLE_STALENESS_HOURS, _cycle_state_path

        # Create a git repo
        (tmp_path / ".git").mkdir()

        # Write stale cycle state
        state_path = _cycle_state_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        old_time = datetime.now(timezone.utc) - timedelta(hours=CYCLE_STALENESS_HOURS + 1)
        state_data = {
            "cycle_id": "old123",
            "started_at": old_time.isoformat(),
            "mode": "design",
            "initial_prompt": "",
            "respawns": 0,
        }
        state_path.write_text(json.dumps(state_data))

        # Should ignore stale cycle and detect from state
        mode = _auto_detect_mode(tmp_path, has_prompt=False)
        assert mode == "design"  # no_factory state → design


class TestCeoPromptCrossCutting:
    """Tests for CEO prompt cross-cutting content."""

    @pytest.fixture()
    def ceo_prompt(self) -> str:
        """Load the CEO prompt from the factory agents prompts directory."""
        prompt_path = Path(__file__).parent.parent / "factory" / "agents" / "prompts" / "ceo.md"
        return prompt_path.read_text()

    def test_design_mode_routing(self, ceo_prompt: str) -> None:
        """CEO prompt routes to design skill."""
        assert "workflow-design" in ceo_prompt

    def test_mutable_fixed_surfaces_enforced(self, ceo_prompt: str) -> None:
        """CEO prompt mentions scope constraints in Sacred Rules."""
        assert "declared scope" in ceo_prompt

    def test_eval_weight_split(self, ceo_prompt: str) -> None:
        """CEO prompt documents eval weight distribution."""
        assert "hygiene" in ceo_prompt.lower()
        assert "growth" in ceo_prompt.lower()

    def test_hygiene_regression_gate(self, ceo_prompt: str) -> None:
        """CEO prompt requires eval score checks before keeping changes."""
        assert "eval" in ceo_prompt.lower()
        assert "revert" in ceo_prompt.lower()

class TestCeoCompletionBackgroundBypass:
    """Tests for background=True bypassing the respawn loop."""

    async def test_background_bypasses_respawn_loop(self, tmp_path: Path) -> None:
        """run_ceo_with_completion_guard calls invoke_agent directly when background=True."""
        from factory.ceo_completion import run_ceo_with_completion_guard

        (tmp_path / ".factory").mkdir()

        with patch(
            "factory.agents.runner.invoke_agent",
            new_callable=AsyncMock,
            return_value=("bg output", 0),
        ) as mock_invoke:
            stdout, code = await run_ceo_with_completion_guard(
                tmp_path,
                "initial task",
                mode="design",
                background=True,
            )

        assert stdout == "bg output"
        assert code == 0
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs["background"] is True


class TestPrintResumeHint:
    """Tests for print_resume_hint()."""

    def test_prints_hint_when_session_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Resume hint is printed to stderr when session.json exists."""
        from factory.ceo_completion import print_resume_hint, write_ceo_session_id

        write_ceo_session_id(tmp_path, "abc-123", mode="design")
        print_resume_hint(tmp_path)

        captured = capsys.readouterr()
        assert "Session: abc-123" in captured.err
        assert f"Resume with: factory resume {tmp_path}" in captured.err

    def test_no_hint_when_session_cleaned_up(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """No resume hint when session.json was deleted (cycle completed)."""
        from factory.ceo_completion import (
            delete_cycle_state,
            print_resume_hint,
            write_ceo_session_id,
        )

        write_ceo_session_id(tmp_path, "abc-123", mode="design")
        delete_cycle_state(tmp_path)
        print_resume_hint(tmp_path)

        captured = capsys.readouterr()
        assert "Session:" not in captured.err
        assert "Resume with:" not in captured.err

    def test_no_hint_when_no_session_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """No resume hint when session.json never existed."""
        from factory.ceo_completion import print_resume_hint

        print_resume_hint(tmp_path)

        captured = capsys.readouterr()
        assert captured.err == ""


class TestResumeHintInCompletionGuard:
    """Tests for resume hint printing in run_ceo_with_completion_guard."""

    @pytest.fixture(autouse=True)
    def enable_respawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FACTORY_CEO_RESPAWN_DISABLED", raising=False)

    async def test_hint_printed_on_respawn_cap_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Resume hint is printed when respawn cap is exhausted."""
        from factory.ceo_completion import run_ceo_with_completion_guard, write_ceo_session_id

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        write_ceo_session_id(tmp_path, "test-session-id", mode="design")
        mock_invoke = AsyncMock(return_value=("Incomplete", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
                max_respawns=0,
            )

        captured = capsys.readouterr()
        assert "Session: test-session-id" in captured.err
        assert f"Resume with: factory resume {tmp_path}" in captured.err

    async def test_no_hint_on_clean_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """No resume hint when cycle completes successfully."""
        from factory.ceo_completion import run_ceo_with_completion_guard, write_ceo_session_id

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        exp_dir = tmp_path / ".factory" / "experiments" / "001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "verdict.json").write_text('{"verdict": "keep"}')

        write_ceo_session_id(tmp_path, "test-session-id", mode="design")
        mock_invoke = AsyncMock(return_value=("Done", 0))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        captured = capsys.readouterr()
        assert "Session:" not in captured.err

    async def test_hint_printed_on_user_interrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Resume hint is printed when user interrupts with Ctrl+C."""
        from factory.ceo_completion import run_ceo_with_completion_guard, write_ceo_session_id

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("#### H1: A\n")
        (tmp_path / ".factory" / "experiments").mkdir()

        write_ceo_session_id(tmp_path, "interrupt-session", mode="design")
        mock_invoke = AsyncMock(return_value=("Interrupted", 130))

        with patch("factory.agents.runner.invoke_agent", mock_invoke):
            await run_ceo_with_completion_guard(
                tmp_path,
                "Initial task",
                mode="design",
                runner_name="claude",
            )

        captured = capsys.readouterr()
        assert "Session: interrupt-session" in captured.err
        assert f"Resume with: factory resume {tmp_path}" in captured.err


class TestWriteUsageJson:
    """Tests for the public write_usage_json aggregator (CEO skill-engine path)."""

    def test_writes_usage_to_mode_subdir(self, tmp_path: Path) -> None:
        """agent.completed events after `since` are summed into .factory/<mode>/usage.json."""
        from factory.ceo_completion import write_usage_json
        from factory.events import emit_event

        since = datetime.now(timezone.utc) - timedelta(seconds=1)
        emit_event(tmp_path, "agent.completed", agent="researcher", data={
            "input_tokens": 1000, "output_tokens": 500,
            "cache_read_tokens": 200, "total_cost_usd": 0.25,
        })
        emit_event(tmp_path, "agent.completed", agent="builder", data={
            "input_tokens": 3000, "output_tokens": 1500,
            "cache_read_tokens": 600, "total_cost_usd": 0.75,
        })

        write_usage_json(tmp_path, "exploit-pov", since)

        usage_path = tmp_path / ".factory" / "exploit-pov" / "usage.json"
        assert usage_path.exists()
        usage = json.loads(usage_path.read_text())
        assert usage["input_tokens"] == 4000
        assert usage["output_tokens"] == 2000
        assert usage["cache_read_tokens"] == 800
        assert usage["total_cost_usd"] == 1.0

    def test_no_file_when_no_agent_events(self, tmp_path: Path) -> None:
        """No usage.json is written when there are no agent.completed events."""
        from factory.ceo_completion import write_usage_json

        write_usage_json(tmp_path, "design", datetime.now(timezone.utc))

        assert not (tmp_path / ".factory" / "design" / "usage.json").exists()
