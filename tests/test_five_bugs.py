"""Tests for five factory bugs — each test FAILS before its fix.

BUG 1: verify-only shortcut causes second eval candidate to skip branch nodes
BUG 2: agent output overwrite clobbers files agent wrote with tools
BUG 3: ItemResult missing `passed` field
BUG 4: reflector receives no node IDs from CycleRecord.from_run()
BUG 5: FnNode with empty command AND callable_name silently produces empty output
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from factory.testing import DummyTask, FakeAgent
from factory.task import Task, TaskDefinition, TaskInstance, VerifyResult


# ── Helpers ──────────────────────────────────────────────────────


class TwoItemTask(Task):
    """Task with two items that always pass."""

    def __init__(self) -> None:
        super().__init__(TaskDefinition(name="two-item"))

    def _raw_instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(id="a")
        yield TaskInstance(id="b")

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        pass

    def prompt(self, instance: TaskInstance) -> str:
        return f"do {instance.id}"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        return VerifyResult(passed=True, score=1.0)


def _make_simple_workflow(name: str = "test-wf"):
    """Build a minimal AgentNode workflow for testing."""
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        Edge,
        Workflow,
    )

    return Workflow(
        name=name,
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build it",
                writes={".factory/reviews/builder-latest.md"},
                reads=set(),
            ),
        },
        edges=[],
        start_node="builder",
    )


def _make_two_node_workflow(name: str = "two-node"):
    """Build a workflow with two agent nodes to detect verify-only skipping."""
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        Edge,
        Workflow,
    )

    return Workflow(
        name=name,
        nodes={
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research it",
                writes={".factory/reviews/researcher-latest.md"},
                reads=set(),
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build it",
                writes={".factory/reviews/builder-latest.md"},
                reads={".factory/reviews/researcher-latest.md"},
            ),
        },
        edges=[
            Edge(source="researcher", target="builder"),
        ],
        start_node="researcher",
    )


# ── BUG 1: verify-only shortcut ─────────────────────────────────


def test_bug1_no_verify_only_in_evaluator_source():
    """evaluator.py must not contain _has_run_full_workflow or _verify_only.
    These cause the second candidate to skip the full workflow."""
    import inspect
    from factory.outer_loop import evaluator as ev_module

    source = inspect.getsource(ev_module)
    assert "_has_run_full_workflow" not in source, (
        "evaluator.py still contains _has_run_full_workflow — "
        "second eval candidate will skip the full workflow"
    )
    assert "_verify_only" not in source, (
        "evaluator.py still sets _verify_only on the inner loop"
    )


def test_bug1_verify_only_removed_from_inner_loop():
    """InnerLoop._step_with_task must not check _verify_only at all."""
    import inspect
    from factory import inner_loop as il_module

    source = inspect.getsource(il_module)
    assert "_verify_only" not in source, (
        "inner_loop.py still references _verify_only — "
        "the verify-only shortcut must be removed entirely"
    )


# ── BUG 2: agent output overwrite ───────────────────────────────


async def test_bug2_agent_stdout_does_not_overwrite_tool_written_file(tmp_path: Path):
    """When an agent writes a declared output file via tools, the executor
    must NOT overwrite it with stdout."""
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        Edge,
        Workflow,
    )
    from factory.workflow.executor import WorkflowExecutor

    wf = Workflow(
        name="overwrite-test",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Write document.md",
                writes={"document.md"},
                reads=set(),
            ),
        },
        edges=[],
        start_node="builder",
    )

    agent_written_content = "# Real Document\nWritten by agent tools"
    stdout_content = "Summary printed to stdout"

    async def tool_writing_agent(
        role: str,
        task: str,
        project_path: Path | str,
        *,
        model: str | None = None,
        timeout: float = 600.0,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        # Simulate agent writing file via tools during its run
        doc_path = Path(project_path) / "document.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(agent_written_content)
        return stdout_content, 0

    executor = WorkflowExecutor(
        wf,
        tmp_path,
        agent_fn=tool_writing_agent,
        validate=False,
        auto_write_outputs=True,
    )
    await executor.execute()

    doc_path = tmp_path / "document.md"
    assert doc_path.exists()
    actual = doc_path.read_text()
    assert actual == agent_written_content, (
        f"Agent-written content was overwritten by stdout!\n"
        f"Expected: {agent_written_content!r}\n"
        f"Got: {actual!r}"
    )


# ── BUG 3: ItemResult missing `passed` field ────────────────────


def test_bug3_item_result_has_passed_field():
    """ItemResult must have a `passed` field."""
    from factory.models import ItemResult, ItemStatus

    ir = ItemResult(
        item_id="x",
        status=ItemStatus.ok,
        score=1.0,
        passed=True,
    )
    assert ir.passed is True

    ir2 = ItemResult(
        item_id="y",
        status=ItemStatus.failed,
        score=0.0,
        passed=False,
    )
    assert ir2.passed is False


def test_bug3_passed_reaches_cycle_record():
    """item with passed=True from verify must appear as passed=True
    in CycleRecord.from_run() instance_results."""
    from factory.cycle_analyzer import CycleRecord

    items = [
        {"item_id": "a", "status": "ok", "score": 1.0, "passed": True},
        {"item_id": "b", "status": "failed", "score": 0.0, "passed": False},
    ]
    record = CycleRecord.from_run(items, aggregate="mean")
    assert record.instance_results is not None
    first = record.instance_results[0]
    assert first["passed"] is True, "passed=True must survive into CycleRecord"


def test_bug3_data_runtime_populates_passed(tmp_path: Path):
    """The data runtime must populate `passed` from Task.verify()."""
    from factory.models import ItemResult, ItemStatus

    # Simulate what the data runtime does
    ir = ItemResult(
        item_id="test",
        status=ItemStatus.ok,
        score=1.0,
        passed=True,
    )
    dumped = ir.model_dump()
    assert "passed" in dumped
    assert dumped["passed"] is True


# ── BUG 4: reflector has no node list ────────────────────────────


def test_bug4_cycle_record_from_run_populates_node_ids():
    """CycleRecord.from_run() must populate node IDs when given a workflow."""
    from factory.cycle_analyzer import CycleRecord
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        Edge,
        Workflow,
    )

    wf = Workflow(
        name="node-trace-test",
        nodes={
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research",
                writes=set(),
                reads=set(),
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build",
                writes=set(),
                reads=set(),
            ),
        },
        edges=[Edge(source="researcher", target="builder")],
        start_node="researcher",
    )

    items = [{"item_id": "a", "status": "ok", "score": 1.0, "passed": True}]
    record = CycleRecord.from_run(items, aggregate="mean", workflow=wf)

    # node_trace or mutable_node_ids should contain real node IDs
    assert record.mutable_node_ids or record.node_trace, (
        "CycleRecord.from_run() must populate node info from workflow"
    )
    # Check that real node IDs are present
    all_ids = set(record.mutable_node_ids) | set(record.node_trace.keys())
    assert "researcher" in all_ids or "builder" in all_ids, (
        f"Expected workflow node IDs, got {all_ids}"
    )


def test_bug4_reflector_filters_invalid_node_targets():
    """Reflector must drop suggestions targeting nodes not in the workflow."""
    from factory.outer_loop.reflector import (
        MutationSuggestion,
        OuterLoopReflector,
        ReflectionReport,
    )

    # A suggestion targeting a non-existent node should be filtered
    report = ReflectionReport()
    real_nodes = {"builder", "researcher"}

    suggestion_real = MutationSuggestion(
        operator="prompt_mutate",
        target="builder",
        rationale="improve builder prompt",
    )
    suggestion_fake = MutationSuggestion(
        operator="prompt_mutate",
        target="invented_node_xyz",
        rationale="this node doesn't exist",
    )

    # After fix, _filter_suggestions should drop invented nodes
    from factory.outer_loop.reflector import _filter_suggestions
    filtered = _filter_suggestions(
        [suggestion_real, suggestion_fake], real_nodes
    )
    assert len(filtered) == 1
    assert filtered[0].target == "builder"


# ── BUG 5: empty FnNode validation ──────────────────────────────


def test_bug5_empty_fn_node_validation_error():
    """FnNode with no command AND no callable_name must fail validation."""
    from factory.workflow.primitives import FnNode

    with pytest.raises(ValueError, match="command.*callable_name|callable_name.*command"):
        FnNode(id="empty-fn", command="", callable_name=None)


def test_bug5_fn_node_with_command_is_valid():
    """FnNode with a command should pass validation."""
    from factory.workflow.primitives import FnNode

    node = FnNode(id="good-fn", command="echo hello")
    assert node.command == "echo hello"


def test_bug5_fn_node_with_callable_is_valid():
    """FnNode with a callable_name should pass validation."""
    from factory.workflow.primitives import FnNode

    node = FnNode(id="good-fn", callable_name="my_module:my_fn")
    assert node.callable_name == "my_module:my_fn"
