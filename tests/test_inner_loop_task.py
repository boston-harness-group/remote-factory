"""Tests for InnerLoop.step() with task-driven execution path.

All task-attached runs go through compose() → DataNode → data_runtime.run_fork().
Tests mock WorkflowExecutor.execute() to return ExecutionResult with item_results.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from factory.cycle_analyzer import CycleRecord
from factory.inner_loop import InnerLoop
from factory.task import (
    ScoringContract,
    TaskDefinition,
    TaskInstance,
    VerifyResult,
)
from factory.workflow.executor import ExecutionResult
from factory.workflow.primitives import AgentNode, AgentRole, Workflow


def _make_workflow(name: str = "test") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="build {project_path}",
            ),
        },
        edges=[],
        start_node="builder",
    )


def _make_exec_result(
    item_results: list[dict] | None = None,
    success: bool = True,
) -> ExecutionResult:
    """Build an ExecutionResult with item_results."""
    r = ExecutionResult()
    r.success = success
    r.halted = not success
    r.halt_reason = "" if success else "halted"
    r.nodes_executed = 1
    r.duration_ms = 100.0
    r.item_results = item_results or []
    return r


def _make_task_mock(
    instances: list[TaskInstance] | None = None,
    verify_results: list[VerifyResult] | None = None,
) -> MagicMock:
    """Build a mock task object."""
    task = MagicMock()
    task.instances.return_value = instances or [TaskInstance(id="inst-1")]
    task.setup.return_value = None
    task.prompt.return_value = "test prompt"
    if verify_results:
        task.verify.side_effect = verify_results
    else:
        task.verify.return_value = VerifyResult(passed=True, score=0.8)
    task.definition = TaskDefinition(
        name="mock", scoring=ScoringContract(method="exit_code"),
    )
    return task


class TestStepWithoutTask:
    """task=None path is unchanged (backward compat)."""

    def test_step_returns_cycle_record(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
        )
        assert loop.task is None

    def test_step_dispatches_to_subprocess(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        loop = InnerLoop(project_dir=tmp_path, mode="test")
        assert loop.task is None
        assert hasattr(loop, "_step_subprocess")


class TestStepWithTask:
    """task is set path — goes through DataNode/data_runtime."""

    def test_step_returns_cycle_record_with_score(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "inst-1", "score": 0.8, "status": "ok"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        assert isinstance(record, CycleRecord)
        assert record.score_end == pytest.approx(0.8)
        assert record.instance_results is not None
        assert len(record.instance_results) == 1
        assert record.instance_results[0]["item_id"] == "inst-1"
        assert record.instance_results[0]["score"] == 0.8

    def test_step_aggregates_mean(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "a", "score": 1.0, "status": "ok"},
            {"item_id": "b", "score": 0.5, "status": "ok"},
            {"item_id": "c", "score": 0.0, "status": "failed"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        assert record.score_end == pytest.approx(0.5)
        assert record.instance_results is not None
        assert len(record.instance_results) == 3

    def test_step_handles_errored_items(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "fail", "score": 0.0, "status": "errored",
             "error": "setup_failed"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        # Most items errored → score_end is None
        assert record.score_end is None
        assert record.instance_results is not None
        assert record.instance_results[0]["error"] == "setup_failed"

    def test_step_increments_step_count(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "x", "score": 1.0, "status": "ok"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            r1 = loop.step()
            r2 = loop.step()

        assert r1.cycle_number == 1
        assert r2.cycle_number == 2
        assert len(loop.history()) == 2

    def test_step_with_no_items(self, tmp_path: Path):
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        # Empty item_results → most-errored path
        exec_result = _make_exec_result(item_results=[])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        # No items → score_end is None (errored path)
        assert record.score_end is None

    def test_step_with_mixed_status_items(self, tmp_path: Path):
        """ok + failed items. Errored items excluded from aggregation."""
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "ok1", "score": 0.8, "status": "ok"},
            {"item_id": "ok2", "score": 0.6, "status": "ok"},
            {"item_id": "err", "score": 0.0, "status": "errored"},
            {"item_id": "fail", "score": 0.0, "status": "failed"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        # errored excluded, mean of [0.8, 0.6, 0.0] = 0.4667
        assert record.score_end == pytest.approx((0.8 + 0.6 + 0.0) / 3)

    def test_step_executor_halt_records_details(self, tmp_path: Path):
        """When executor halts, the halt_reason is recorded in eval_details."""
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "halt", "score": 0.0, "status": "failed"},
        ], success=False)
        exec_result.halt_reason = "max_items exceeded"

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        assert record.eval_details is not None
        assert "halt_reason" in record.eval_details


class TestStepAggregatesMethods:
    """Test non-default aggregation methods via CycleRecord.from_run."""

    def _step_with_aggregate(
        self, tmp_path: Path, aggregate_method: str, items: list[dict],
    ) -> CycleRecord:
        from factory.models import AggregateMethod, InnerLoopConfig

        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir(exist_ok=True)

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=items)
        config = InnerLoopConfig(aggregate=AggregateMethod(aggregate_method))

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(
                project_dir=tmp_path, mode="test", task=task, workflow=wf,
                inner_loop_config=config,
            )
            record = loop.step()

        return record

    def test_step_aggregates_median(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "median", [
            {"item_id": "a", "score": 0.2, "status": "ok"},
            {"item_id": "b", "score": 0.5, "status": "ok"},
            {"item_id": "c", "score": 0.9, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(0.5)

    def test_step_aggregates_median_even(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "median", [
            {"item_id": "a", "score": 0.0, "status": "ok"},
            {"item_id": "b", "score": 0.4, "status": "ok"},
            {"item_id": "c", "score": 0.6, "status": "ok"},
            {"item_id": "d", "score": 1.0, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(0.5)

    def test_step_aggregates_max(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "max", [
            {"item_id": "a", "score": 0.1, "status": "ok"},
            {"item_id": "b", "score": 0.3, "status": "ok"},
            {"item_id": "c", "score": 0.9, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(0.9)

    def test_step_aggregates_max_single(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "max", [
            {"item_id": "a", "score": 0.42, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(0.42)

    def test_step_aggregates_all_pass_true(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "all_pass", [
            {"item_id": "a", "score": 1.0, "status": "ok"},
            {"item_id": "b", "score": 1.0, "status": "ok"},
            {"item_id": "c", "score": 1.0, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(1.0)

    def test_step_aggregates_all_pass_false(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "all_pass", [
            {"item_id": "a", "score": 1.0, "status": "ok"},
            {"item_id": "b", "score": 0.9, "status": "ok"},
            {"item_id": "c", "score": 1.0, "status": "ok"},
        ])
        assert record.score_end == pytest.approx(0.0)

    def test_step_aggregates_all_pass_empty(self, tmp_path: Path):
        record = self._step_with_aggregate(tmp_path, "all_pass", [])
        # Empty → most-errored → score_end is None
        assert record.score_end is None


class TestCycleRecordInstanceResults:
    """CycleRecord.instance_results field."""

    def test_default_none(self):
        record = CycleRecord(
            cycle_number=0,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=0,
            score_start=None,
            score_end=None,
            score_delta=None,
        )
        assert record.instance_results is None

    def test_set_to_list(self):
        record = CycleRecord(
            cycle_number=0,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=0,
            score_start=None,
            score_end=None,
            score_delta=None,
            instance_results=[{"instance_id": "a", "score": 1.0}],
        )
        assert record.instance_results is not None
        assert len(record.instance_results) == 1


class TestCycleRecordFromRun:
    """Test CycleRecord.from_run aggregation directly."""

    def test_mean_aggregation(self):
        items = [
            {"item_id": "a", "score": 0.8, "status": "ok"},
            {"item_id": "b", "score": 0.6, "status": "ok"},
            {"item_id": "c", "score": 0.0, "status": "failed"},
        ]
        record = CycleRecord.from_run(items, aggregate="mean")
        assert record.score_end == pytest.approx((0.8 + 0.6 + 0.0) / 3)

    def test_errored_excluded(self):
        items = [
            {"item_id": "a", "score": 0.8, "status": "ok"},
            {"item_id": "b", "score": 0.0, "status": "errored"},
        ]
        record = CycleRecord.from_run(items, aggregate="mean")
        assert record.score_end == pytest.approx(0.8)

    def test_all_errored_gives_none(self):
        items = [
            {"item_id": "a", "score": 0.0, "status": "errored"},
            {"item_id": "b", "score": 0.0, "status": "errored"},
        ]
        record = CycleRecord.from_run(items, aggregate="mean")
        assert record.score_end is None

    def test_cost_fields_default_zero(self):
        items = [{"item_id": "a", "score": 1.0, "status": "ok"}]
        record = CycleRecord.from_run(items)
        assert record.total_cost_usd == 0.0
        assert record.cost_by_agent == {}


def _write_agent_events(factory_dir: Path, costs: list[tuple[str, float]]) -> None:
    """Write agent.started + agent.completed events with cost data to events.jsonl."""
    events_path = factory_dir / "events.jsonl"
    lines: list[str] = []
    for role, cost in costs:
        lines.append(json.dumps({
            "type": "agent.started",
            "agent": role,
            "timestamp": "2026-01-01T00:00:00",
            "data": {},
        }))
        lines.append(json.dumps({
            "type": "agent.completed",
            "agent": role,
            "timestamp": "2026-01-01T00:01:00",
            "data": {"total_cost_usd": cost, "output_tokens": 500},
        }))
    with open(events_path, "a") as f:
        f.write("\n".join(lines) + "\n")


class TestStepWithTaskCostFromRecord:
    """Verify that cost data comes through CycleRecord."""

    def test_cost_zero_when_no_events(self, tmp_path: Path):
        """When no cost data provided, cost should be zero."""
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()

        task = _make_task_mock()
        wf = _make_workflow()

        exec_result = _make_exec_result(item_results=[
            {"item_id": "inst-1", "score": 1.0, "status": "ok"},
        ])

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=exec_result,
        ):
            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        assert record.total_cost_usd == pytest.approx(0.0)
        assert record.cost_by_agent == {}
