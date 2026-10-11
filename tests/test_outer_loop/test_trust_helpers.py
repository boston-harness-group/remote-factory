"""Unit tests for assert_run_trustworthy helper.

Tests the helper against known-good and deliberately broken run directories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_outer_loop.trust_helpers import assert_run_trustworthy


def _build_good_run_dir(base: Path) -> tuple[SimpleNamespace, Path]:
    """Create a run directory that passes all trust checks."""
    ol_dir = base / ".factory" / "outer_loop"
    ol_dir.mkdir(parents=True)

    # best/workflow.json
    best_dir = ol_dir / "best"
    best_dir.mkdir()
    (best_dir / "workflow.json").write_text(json.dumps({"name": "test-wf", "nodes": {}}))
    (best_dir / "run_report.json").write_text(json.dumps({
        "train_score": 0.75,
        "val_score": 0.65,
        "overfit_flag": False,
        "total_candidates_evaluated": 2,
        "generations_completed": 1,
        "convergence_reason": "budget_exhausted",
        "total_cost_usd": 0.01,
    }))

    # archive/generation-000/summary.json
    gen_dir = ol_dir / "archive" / "generation-000"
    gen_dir.mkdir(parents=True)
    (gen_dir / "summary.json").write_text(json.dumps({
        "generation": 0, "best_score": 0.75,
    }))

    # trajectory.jsonl
    (ol_dir / "trajectory.jsonl").write_text(
        json.dumps({"generation": 0, "best_score": 0.75}) + "\n"
    )

    # runs with items.jsonl
    runs_dir = base / ".factory" / "runs" / "test-run"
    runs_dir.mkdir(parents=True)
    (runs_dir / "items.jsonl").write_text(
        json.dumps({"item_id": "a", "status": "ok", "score": 0.8}) + "\n"
    )

    # items/a/ with output + sha256
    item_dir = runs_dir / "items" / "a"
    item_dir.mkdir(parents=True)
    content = b"# Agent output\n"
    (item_dir / "document.md").write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    (item_dir / "sha256.json").write_text(json.dumps({"document.md": sha}))

    result = SimpleNamespace(best_score=0.75)
    return result, base


class TestAssertRunTrustworthyGood:
    """Tests with a known-good run directory."""

    def test_good_dir_passes(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        diag = assert_run_trustworthy(result, run_dir)
        assert diag["checks_failed"] == 0
        assert diag["checks_passed"] > 0

    def test_good_dir_returns_diagnostics(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        diag = assert_run_trustworthy(result, run_dir)
        assert "details" in diag
        assert all(d["passed"] for d in diag["details"])


class TestAssertRunTrustworthyBroken:
    """Tests with deliberately broken run directories."""

    def test_missing_best_workflow(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        (run_dir / ".factory" / "outer_loop" / "best" / "workflow.json").unlink()
        with pytest.raises(AssertionError, match="best_workflow_exists"):
            assert_run_trustworthy(result, run_dir)

    def test_missing_run_report(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        (run_dir / ".factory" / "outer_loop" / "best" / "run_report.json").unlink()
        with pytest.raises(AssertionError, match="run_report_exists"):
            assert_run_trustworthy(result, run_dir)

    def test_score_mismatch(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        # Change result score to not match report
        result.best_score = 0.99
        with pytest.raises(AssertionError, match="run_report_score_matches"):
            assert_run_trustworthy(result, run_dir)

    def test_empty_trajectory(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        (run_dir / ".factory" / "outer_loop" / "trajectory.jsonl").write_text("")
        with pytest.raises(AssertionError, match="trajectory_has_entries"):
            assert_run_trustworthy(result, run_dir)

    def test_sha256_hash_mismatch(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        # Corrupt the file
        item_dir = run_dir / ".factory" / "runs" / "test-run" / "items" / "a"
        (item_dir / "document.md").write_text("corrupted content")
        with pytest.raises(AssertionError, match="sha256"):
            assert_run_trustworthy(result, run_dir)

    def test_missing_item_field(self, tmp_path: Path) -> None:
        result, run_dir = _build_good_run_dir(tmp_path)
        runs_dir = run_dir / ".factory" / "runs" / "test-run"
        # Write item without required 'status' field
        (runs_dir / "items.jsonl").write_text(
            json.dumps({"item_id": "x", "score": 0.5}) + "\n"
        )
        with pytest.raises(AssertionError, match="status"):
            assert_run_trustworthy(result, run_dir)
