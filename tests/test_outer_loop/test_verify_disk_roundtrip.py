"""Integration test: verify data survives the disk round-trip.

Exercises the full pipeline:
  _write_cycle_summary() → cycle_summary.json → read-back →
  CycleRecord with eval_details → OuterLoopReflector._extract_eval_patterns()
  → non-empty failure/success patterns.

Also covers CycleRecordCache persistence of eval_details and instance_results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory.cycle_analyzer import CycleRecord
from factory.inner_loop import InnerLoop
from factory.outer_loop.evaluator import CycleRecordCache
from factory.outer_loop.reflector import OuterLoopReflector


def _load_cycle_summary_from_disk(project_dir: Path, mode: str) -> CycleRecord | None:
    """Read cycle_summary.json and reconstruct a CycleRecord.

    Replaces the removed ``_load_cycle_summary`` CLI helper.
    cycle_summary.json is no longer a score channel; this helper exists
    only so the round-trip tests can verify the written data.
    """
    summary_path = (
        project_dir / ".factory" / "outer_loop" / "runs" / mode / "cycle_summary.json"
    )
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())

    record = CycleRecord(
        cycle_number=0,
        mode=mode,
        started_at=None,
        ended_at=None,
        duration_s=data.get("duration_ms", 0) / 1000,
        score_start=None,
        score_end=data.get("score"),
        score_delta=None,
        kept=data.get("kept", 0),
        reverted=data.get("reverted", 0),
    )

    instance_results = data.get("instance_results")
    if instance_results:
        record.instance_results = instance_results

    # Build eval_details from verify / test_details if present
    eval_details: dict = {}
    if "verify" in data:
        eval_details["verify"] = data["verify"]
    if "test_details" in data:
        eval_details["test_details"] = data["test_details"]
    if eval_details:
        record.eval_details = eval_details

    return record


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / ".factory").mkdir()
    return tmp_path


@pytest.fixture()
def loop(project_dir: Path) -> InnerLoop:
    return InnerLoop(project_dir=project_dir, mode="evolve-roundtrip")


def _sample_instance_results() -> list[dict]:
    return [
        {
            "instance_id": "inst-a",
            "passed": True,
            "score": 1.0,
            "details": {"returncode": 0, "stdout": "ok"},
        },
        {
            "instance_id": "inst-b",
            "passed": False,
            "score": 0.0,
            "details": {"returncode": 1, "stderr": "error"},
        },
        {
            "instance_id": "inst-c",
            "passed": False,
            "score": 0.0,
            "details": {"returncode": 2},
        },
    ]


class TestVerifyDiskRoundtrip:
    """End-to-end: write → read → reflect produces patterns from verify data."""

    def test_write_read_reflect(self, loop: InnerLoop, project_dir: Path) -> None:
        instance_results = _sample_instance_results()

        loop._write_cycle_summary(
            returncode=0,
            event_offset=0,
            duration_ms=5000,
            builder_committed=False,
            experiments=0,
            test_score=0.33,
            instance_results=instance_results,
            kept=1,
            reverted=2,
        )

        summary_path = (
            project_dir / ".factory" / "outer_loop" / "runs"
            / "evolve-roundtrip" / "cycle_summary.json"
        )
        data = json.loads(summary_path.read_text())
        assert "instance_results" in data
        assert "verify" in data
        assert data["verify"]["verify_count"] == 3
        assert data["verify"]["passed_count"] == 1
        assert data["verify"]["failed_count"] == 2
        assert data["kept"] == 1
        assert data["reverted"] == 2

        record = _load_cycle_summary_from_disk(project_dir, "evolve-roundtrip")
        assert record is not None
        assert record.kept == 1
        assert record.reverted == 2
        assert record.instance_results is not None
        assert len(record.instance_results) == 3
        assert record.eval_details is not None
        assert "verify" in record.eval_details
        verify = record.eval_details["verify"]
        assert verify["verify_count"] == 3
        assert verify["passed_count"] == 1
        assert verify["failed_count"] == 2

        winner_record = CycleRecord(
            cycle_number=1,
            mode="winner",
            started_at=None,
            ended_at=None,
            duration_s=5.0,
            score_start=0.0,
            score_end=0.9,
            score_delta=0.9,
            eval_details={
                "verify": {
                    "verify_count": 3,
                    "passed_count": 3,
                    "failed_count": 0,
                    "instance_results": [
                        {"index": i, "passed": True, "score": 1.0}
                        for i in range(3)
                    ],
                },
            },
        )

        reflector = OuterLoopReflector(k=1)
        records = [
            ("winner-mode", 0.9, winner_record),
            ("evolve-roundtrip", 0.33, record),
        ]
        report = reflector.reflect(records, generation=0)

        verify_failures = [
            p for p in report.failure_patterns if "verify" in p.lower()
        ]
        assert len(verify_failures) > 0, (
            f"Expected verify failure patterns, got: {report.failure_patterns}"
        )
        assert any("2/3" in p for p in verify_failures)

        rc_failures = [p for p in report.failure_patterns if "returncode" in p]
        assert len(rc_failures) >= 1

        verify_successes = [
            p for p in report.success_patterns if "verify" in p.lower()
        ]
        assert len(verify_successes) > 0

    def test_summary_without_instance_results(
        self, loop: InnerLoop, project_dir: Path,
    ) -> None:
        loop._write_cycle_summary(
            returncode=0,
            event_offset=0,
            duration_ms=1000,
            builder_committed=True,
            experiments=1,
        )

        record = _load_cycle_summary_from_disk(project_dir, "evolve-roundtrip")
        assert record is not None
        assert record.instance_results is None
        assert record.eval_details is None

    def test_summary_with_test_details_only(
        self, loop: InnerLoop, project_dir: Path,
    ) -> None:
        loop._write_cycle_summary(
            returncode=0,
            event_offset=0,
            duration_ms=1000,
            builder_committed=True,
            experiments=1,
            test_details={"returncode": 1, "failed": 2, "total": 5},
        )

        record = _load_cycle_summary_from_disk(project_dir, "evolve-roundtrip")
        assert record is not None
        assert record.eval_details is not None
        assert "test_details" in record.eval_details
        assert record.eval_details["test_details"]["returncode"] == 1


class TestCycleRecordCacheRoundtrip:
    """CycleRecordCache persists and restores eval_details + instance_results."""

    def test_save_load_preserves_eval_details(self, tmp_path: Path) -> None:
        cache = CycleRecordCache()

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={
                "b": AgentNode(
                    id="b", role=AgentRole.BUILDER, model="opus", timeout=60,
                ),
            },
            edges=[],
            start_node="b",
            terminal=True,
        )

        record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=5.0,
            score_start=None,
            score_end=0.75,
            score_delta=None,
            instance_results=_sample_instance_results(),
            eval_details={
                "verify": {
                    "verify_count": 3,
                    "passed_count": 1,
                    "failed_count": 2,
                    "instance_results": [
                        {"index": 0, "passed": True, "score": 1.0},
                        {"index": 1, "passed": False, "score": 0.0},
                        {"index": 2, "passed": False, "score": 0.0},
                    ],
                },
            },
        )
        cache.put(wf, record)

        cache_path = tmp_path / "eval_cache.jsonl"
        cache.save_cache(cache_path)

        new_cache = CycleRecordCache()
        loaded = new_cache.load_cache(cache_path)
        assert loaded == 1

        restored = new_cache.get(wf)
        assert restored is not None
        assert restored.score_end == 0.75
        assert restored.instance_results is not None
        assert len(restored.instance_results) == 3
        assert restored.eval_details is not None
        assert "verify" in restored.eval_details
        assert restored.eval_details["verify"]["passed_count"] == 1

    def test_save_load_without_eval_details(self, tmp_path: Path) -> None:
        cache = CycleRecordCache()

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="bare-wf",
            nodes={
                "b": AgentNode(
                    id="b", role=AgentRole.BUILDER, model="opus", timeout=60,
                ),
            },
            edges=[],
            start_node="b",
            terminal=True,
        )

        record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=5.0,
            score_start=None,
            score_end=0.5,
            score_delta=None,
        )
        cache.put(wf, record)

        cache_path = tmp_path / "eval_cache.jsonl"
        cache.save_cache(cache_path)

        new_cache = CycleRecordCache()
        new_cache.load_cache(cache_path)
        restored = new_cache.get(wf)
        assert restored is not None
        assert restored.eval_details is None
        assert restored.instance_results is None
