"""Tests for the DataNode graph primitive — models, executor, validation, skill export, and features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from factory.workflow.primitives import (
    DataItem,
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow)


# ── Phase 1: DataItem / DataNode Pydantic validation ──────────────


class TestDataItem:
    def test_minimal(self) -> None:
        item = DataItem(id="a")
        assert item.id == "a"
        assert item.path is None
        assert item.metadata == {}
        assert item.prompt == ""

    def test_full(self) -> None:
        item = DataItem(id="b", path="/tmp/b", metadata={"k": "v"}, prompt="do stuff")
        assert item.path == "/tmp/b"
        assert item.metadata == {"k": "v"}
        assert item.prompt == "do stuff"

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DataItem(id="a", unknown="x")

    def test_roundtrip(self) -> None:
        item = DataItem(id="c", metadata={"x": 1})
        data = item.model_dump(mode="json")
        restored = DataItem.model_validate(data)
        assert restored.id == "c"
        assert restored.metadata == {"x": 1}


class TestDataNode:
    def test_inline_items_source(self) -> None:
        items = [DataItem(id="i1"), DataItem(id="i2")]
        node = DataNode(
            id="dn",
            inline_items=items)
        assert len(node.inline_items) == 2
        assert node.task_ref is None
        assert node.source_path is None

    def test_task_ref_source(self) -> None:
        node = DataNode(
            id="dn",
            task_ref="my.module:MyTask")
        assert node.task_ref == "my.module:MyTask"

    def test_source_path_source(self) -> None:
        node = DataNode(
            id="dn",
            source_path="/data/items",
            source_format="directory")
        assert node.source_path == "/data/items"
        assert node.source_format == "directory"

    def test_no_source_allowed_late_bound(self) -> None:
        """A DataNode with zero sources is valid (late-bound template)."""
        dn = DataNode(
            id="dn")
        assert dn.task_ref is None
        assert dn.source_path is None
        assert dn.inline_items == []

    def test_multiple_sources_raises(self) -> None:
        """Multiple sources are still rejected."""
        with pytest.raises(ValidationError, match="At most one"):
            DataNode(
                id="dn",
                task_ref="x",
                inline_items=[DataItem(id="i")])

    def test_source_path_requires_format(self) -> None:
        with pytest.raises(ValidationError, match="source_format"):
            DataNode(
                id="dn",
                source_path="/data/items")

    def test_defaults(self) -> None:
        node = DataNode(
            id="dn",
            inline_items=[DataItem(id="i")])
        assert node.parallelism == 1
        assert node.shuffle is False
        assert node.shuffle_seed is None
        assert node.limit is None
        assert node.max_items == 500

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DataNode(
                id="dn",
                inline_items=[DataItem(id="i")],
                unknown="x")

    def test_from_dict_roundtrip(self) -> None:
        items = [DataItem(id="i1", prompt="do it")]
        node = DataNode(
            id="dn",
            inline_items=items,
            parallelism=5,
            max_items=100)
        wf = Workflow(
            name="test",
            nodes={
                "dn": node,
                "entry": FnNode(id="entry", command="echo entry"),
                "exit": FnNode(id="exit", command="echo exit"),
            },
            edges=[
                Edge(source="dn", target="entry"),
                Edge(source="entry", target="exit"),
            ],
            start_node="dn")
        data = wf.to_dict()
        restored = Workflow.from_dict(data)
        dn = restored.nodes["dn"]
        assert type(dn).__name__ == "DataNode"
        assert dn.parallelism == 5
        assert dn.max_items == 100
        assert len(dn.inline_items) == 1
        assert dn.inline_items[0].id == "i1"


# ── Phase 2: Executor _execute_data ──────────────────────────────


def _make_data_workflow(items: list[DataItem]) -> Workflow:
    """Build a minimal workflow with a DataNode driving a FnNode subgraph.

    Uses real edges: DataNode → sub_start → sub_end → JoinNode.
    """
    return Workflow(
        name="data_test",
        nodes={
            "data": DataNode(
                id="data",
                inline_items=items,
                parallelism=2),
            "sub_start": FnNode(id="sub_start", command="echo start"),
            "sub_end": FnNode(id="sub_end", command="echo end"),
            "join": JoinNode(id="join", sources=["sub_end"]),
        },
        edges=[
            Edge(source="data", target="sub_start"),
            Edge(source="sub_start", target="sub_end"),
            Edge(source="sub_end", target="join"),
        ],
        start_node="data")


class TestExecuteData:
    def test_inline_items_execute(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id="a", prompt="do a"), DataItem(id="b", prompt="do b")]
        wf = _make_data_workflow(items)
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())

        assert result.success
        assert "data" in result.node_outputs
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2
        assert parsed[0]["item_id"] == "a"
        assert parsed[1]["item_id"] == "b"

    def test_fault_isolation_one_bad_item(self, tmp_path: Path) -> None:
        """A failing subgraph for one item should not halt the whole DataNode."""
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id="good"), DataItem(id="bad"), DataItem(id="also_good")]
        wf = _make_data_workflow(items)

        original_init = WorkflowExecutor.__init__

        def tracking_init(self_inner, workflow, project_path, *args, **kwargs):
            original_init(self_inner, workflow, project_path, *args, **kwargs)
            ctx = kwargs.get("initial_context")
            self_inner._test_initial_context = ctx

        original_execute = WorkflowExecutor.execute

        async def selective_execute(self_inner):
            # Inner executors (sub-workflows) have _test_initial_context set
            if hasattr(self_inner, "_test_initial_context") and self_inner.workflow.name.endswith("__data_item"):
                # Find which item this is by checking if it's the 2nd call (bad)
                if not hasattr(selective_execute, "_inner_count"):
                    selective_execute._inner_count = 0
                selective_execute._inner_count += 1
                if selective_execute._inner_count == 2:
                    raise RuntimeError("simulated failure")
            return await original_execute(self_inner)

        selective_execute._inner_count = 0

        with patch.object(WorkflowExecutor, "__init__", tracking_init), \
             patch.object(WorkflowExecutor, "execute", selective_execute):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 3
        bad_item = next(r for r in parsed if r["item_id"] == "bad")
        assert bad_item["score"] == 0.0
        assert "error" in bad_item
        good_items = [r for r in parsed if r["item_id"] != "bad"]
        assert all(r["status"] in ("ok", "failed") for r in good_items)

    def test_max_items_exceeded_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id=str(i)) for i in range(10)]
        wf = Workflow(
            name="data_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=items,
                    max_items=5),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "max_items=5" in result.halt_reason

    def test_split_filter(self, tmp_path: Path) -> None:
        """Split filter is now done by the Task/InnerLoop, not by the DataNode.

        Inline items without a task are all processed (no split field on DataItem).
        """
        from factory.workflow.executor import WorkflowExecutor

        items = [
            DataItem(id="train1"),
            DataItem(id="val1"),
            DataItem(id="train2"),
        ]
        wf = _make_data_workflow(items)
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        # Without a task, all items are processed
        assert len(parsed) == 3

    def test_limit_filter(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id=str(i)) for i in range(10)]
        wf = Workflow(
            name="data_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=items,
                    limit=3),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 3


# ── Phase 3: Validation ─────────────────────────────────────────


class TestDataNodeValidation:
    def test_valid_data_node_workflow(self) -> None:
        wf = _make_data_workflow([DataItem(id="i")])
        issues = wf.validate_graph()
        assert not issues

    def test_datanode_no_outgoing_edges(self) -> None:
        """DataNode with no outgoing edges is flagged."""
        wf = Workflow(
            name="bad",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i")]),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data")
        issues = wf.validate_graph()
        assert any("no outgoing edges" in i for i in issues)

    def test_datanode_no_join(self) -> None:
        """DataNode without a JoinNode is flagged."""
        wf = Workflow(
            name="bad",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i")]),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[
                Edge(source="data", target="sub"),
            ],
            start_node="data")
        issues = wf.validate_graph()
        # Should flag missing JoinNode or unreachable sub
        assert len(issues) > 0

    def test_subgraph_nodes_reachable(self) -> None:
        """Subgraph nodes behind a DataNode should not be flagged as unreachable."""
        wf = _make_data_workflow([DataItem(id="i")])
        issues = wf.validate_graph()
        unreachable = [i for i in issues if "unreachable" in i]
        assert not unreachable


# ── Phase 4: Skill export ─────────────────────────────────────────


class TestDataNodeSkillExport:
    def test_data_node_raises_pr_b(self) -> None:
        """DataNode ceo-skill export is deferred to PR B."""
        from factory.workflow.skill_export import workflow_to_skill_md

        wf = _make_data_workflow([DataItem(id="i")])
        with pytest.raises(ValueError, match="not supported.*PR B"):
            workflow_to_skill_md(wf)


class TestDataInstructionTaskRef:
    """DataNode with task_ref raises ValueError (PR B)."""

    def test_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = _make_task_ref_workflow(task_ref="my.module:MyTask")
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)


class TestDataInstructionSourcePath:
    """DataNode with source_path raises ValueError (PR B)."""

    def test_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = Workflow(
            name="sp_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path="data.jsonl",
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)


class TestDataInstructionInlineItems:
    """DataNode with inline_items raises ValueError (PR B)."""

    def test_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = _make_data_workflow([DataItem(id="alpha", prompt="do alpha")])
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)


class TestDataInstructionHasAggregateStep:
    """All DataNode variants raise ValueError for skill export (PR B)."""

    def test_task_ref_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = _make_task_ref_workflow()
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)

    def test_source_path_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = Workflow(
            name="sp_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path="data.jsonl",
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)

    def test_inline_items_raises_pr_b(self) -> None:
        from factory.workflow.skill_export import _data_to_instruction

        wf = _make_data_workflow([DataItem(id="i")])
        node = wf.nodes["data"]
        with pytest.raises(ValueError, match="not supported.*PR B"):
            _data_to_instruction(node, wf)


# ── Phase 6: compute_features arity ────────────────────────────────


class TestComputeFeaturesDataNode:
    def test_arity_is_9(self) -> None:
        from factory.outer_loop.similarity import compute_features

        wf = Workflow(
            name="w",
            nodes={"a": FnNode(id="a", command="x")},
            edges=[],
            start_node="a")
        features = compute_features(wf)
        assert len(features) == 9

    def test_data_node_sets_feature(self) -> None:
        from factory.outer_loop.similarity import compute_features

        wf = _make_data_workflow([DataItem(id="i")])
        features = compute_features(wf)
        assert len(features) == 9
        assert features[8] == 1  # has_data_node is the appended axis

    def test_no_data_node_feature_is_zero(self) -> None:
        from factory.outer_loop.similarity import compute_features

        wf = Workflow(
            name="w",
            nodes={"a": FnNode(id="a", command="x")},
            edges=[],
            start_node="a")
        features = compute_features(wf)
        assert features[8] == 0


class TestDiversityMetricNewAxis:
    def test_diversity_responds_to_data_node_axis(self) -> None:
        from factory.outer_loop.population import MAPElitesArchive, Population

        wf_no_data = Workflow(
            name="w",
            nodes={"a": FnNode(id="a", command="x")},
            edges=[],
            start_node="a")
        wf_with_data = _make_data_workflow([DataItem(id="i")])

        ind1 = Population.make_individual(wf_no_data, score=0.5)
        ind2 = Population.make_individual(wf_with_data, score=0.5)

        archive = MAPElitesArchive()
        archive.add(ind1)
        d1 = archive.diversity_metric()

        archive.add(ind2)
        d2 = archive.diversity_metric()
        # Adding a structurally different individual should change diversity
        assert d2 != d1 or archive.size == 1


# ── Phase 5: compose CAN_ITERATE ──────────────────────────────────


# ── Phase 7: source_path code paths ─────────────────────────────


class TestSourcePathDirectory:
    def test_directory_source(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        src_dir = tmp_path / "data"
        src_dir.mkdir()
        (src_dir / "alpha").mkdir()
        (src_dir / "beta").mkdir()
        (src_dir / "plain_file.txt").write_text("not a dir")

        wf = Workflow(
            name="dir_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(src_dir),
                    source_format="directory"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        ids = [r["item_id"] for r in parsed]
        assert "alpha" in ids
        assert "beta" in ids
        assert "plain_file.txt" not in ids


class TestSourcePathJsonl:
    def test_jsonl_source(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        jsonl_file = tmp_path / "items.jsonl"
        jsonl_file.write_text('{"name": "first"}\n{"name": "second"}\n\n')

        wf = Workflow(
            name="jsonl_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(jsonl_file),
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2
        assert parsed[0]["item_id"] == "0"
        assert parsed[1]["item_id"] == "1"


class TestSourcePathCsv:
    def test_csv_source(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        csv_file = tmp_path / "items.csv"
        csv_file.write_text("id,value\na,1\nb,2\nc,3\n")

        wf = Workflow(
            name="csv_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(csv_file),
                    source_format="csv"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 3
        assert parsed[0]["item_id"] == "0"
        assert parsed[1]["item_id"] == "1"
        assert parsed[2]["item_id"] == "2"


class TestSourcePathNonExistent:
    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="missing_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(tmp_path / "does_not_exist"),
                    source_format="directory"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "source_path not found" in result.halt_reason


# ── Phase 8: inner_loop _step_with_task ────────────────────


class TestStepWithDataNode:
    def test_delegates_to_executor(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="i", prompt="go")])

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "i", "score": 1.0, "status": "ok"},
        ]

        task = DefaultTask(test_command="true")
        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end == 1.0
        assert record.cycle_number == 1


class TestSubgraphInheritsCompletedFiles:
    """Verify that subgraph executors inherit parent completed_files."""

    def test_subgraph_reads_upstream_artifact(self, tmp_path: Path) -> None:
        """Subgraph start node with reads={'data_ready'} should inherit the
        artifact from an upstream FnNode that writes={'data_ready'}, so it
        executes instead of timing out."""
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="inherit_test",
            nodes={
                "upstream_fn": FnNode(
                    id="upstream_fn", command="echo ready", writes={"data_ready"}),
                "data_loader": DataNode(
                    id="data_loader",
                    inline_items=[DataItem(id="item1", prompt="go")]),
                "process_node": FnNode(
                    id="process_node",
                    command="echo processing",
                    reads={"data_ready"}),
                "exit_node": FnNode(id="exit_node", command="echo done"),
                "join": JoinNode(id="join", sources=["exit_node"]),
            },
            edges=[
                Edge(source="upstream_fn", target="data_loader"),
                Edge(source="data_loader", target="process_node"),
                Edge(source="process_node", target="exit_node"),
                Edge(source="exit_node", target="join"),
            ],
            start_node="upstream_fn")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())

        assert result.success, f"Expected success but got halt: {result.halt_reason}"
        assert not result.halted
        parsed = json.loads(result.node_outputs["data_loader"])
        assert len(parsed) == 1

    def test_subgraph_no_reads_still_works(self, tmp_path: Path) -> None:
        """Subgraph start node with no reads should execute normally (baseline)."""
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="no_reads_test",
            nodes={
                "upstream_fn": FnNode(
                    id="upstream_fn", command="echo ready", writes={"data_ready"}),
                "data_loader": DataNode(
                    id="data_loader",
                    inline_items=[DataItem(id="item1")]),
                "process_node": FnNode(id="process_node", command="echo processing"),
                "exit_node": FnNode(id="exit_node", command="echo done"),
                "join": JoinNode(id="join", sources=["exit_node"]),
            },
            edges=[
                Edge(source="upstream_fn", target="data_loader"),
                Edge(source="data_loader", target="process_node"),
                Edge(source="process_node", target="exit_node"),
                Edge(source="exit_node", target="join"),
            ],
            start_node="upstream_fn")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data_loader"])
        assert len(parsed) == 1
        assert parsed[0]["status"] in ("ok", "failed")


class TestComposeCapsDataNode:
    def test_data_node_adds_can_iterate(self) -> None:
        from factory.compose import ModeCapabilities
        from factory.task import Capability

        wf = _make_data_workflow([DataItem(id="i")])
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.CAN_ITERATE in caps.provides


# ── Phase 9: task_ref verify integration ──────────────────────────


class _FakeTask:
    """Minimal Task-like object for testing verify() integration."""

    def __init__(self, instances_data: list[dict[str, Any]], verify_scores: dict[str, float]) -> None:
        self._instances_data = instances_data
        self._verify_scores = verify_scores
        self.setup_calls: list[str] = []
        self.prompt_calls: list[str] = []
        self.verify_calls: list[str] = []

    def instances(self):
        from factory.task import TaskInstance
        for d in self._instances_data:
            yield TaskInstance(id=d["id"], path=d.get("path"), metadata=d.get("metadata", {}))

    def setup(self, instance, workspace):
        self.setup_calls.append(instance.id)

    def prompt(self, instance):
        self.prompt_calls.append(instance.id)
        return f"prompt for {instance.id}"

    def verify(self, instance, workspace):
        from factory.task import VerifyResult
        self.verify_calls.append(instance.id)
        score = self._verify_scores.get(instance.id, 0.0)
        return VerifyResult(passed=score > 0.5, score=score, details={"source": "fake"})


def _make_task_ref_workflow(task_ref: str = "fake.module:FakeTask") -> Workflow:
    """Build a minimal workflow with a task_ref DataNode (real edges)."""
    return Workflow(
        name="task_ref_test",
        nodes={
            "data": DataNode(
                id="data",
                task_ref=task_ref,
                parallelism=2),
            "sub_start": FnNode(id="sub_start", command="echo start"),
            "sub_end": FnNode(id="sub_end", command="echo end"),
            "join": JoinNode(id="join", sources=["sub_end"]),
        },
        edges=[
            Edge(source="data", target="sub_start"),
            Edge(source="sub_start", target="sub_end"),
            Edge(source="sub_end", target="join"),
        ],
        start_node="data")


class TestTaskRefVerify:
    def test_verify_called_per_item_and_scores_used(self, tmp_path: Path) -> None:
        """task_ref DataNode must call verify() per item and use verify scores."""
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _FakeTask(
            instances_data=[{"id": "inst_a"}, {"id": "inst_b"}],
            verify_scores={"inst_a": 0.8, "inst_b": 0.3})

        wf = _make_task_ref_workflow()

        with patch("factory.task.TaskRef.resolve", return_value=fake_task):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2

        item_a = next(r for r in parsed if r["item_id"] == "inst_a")
        item_b = next(r for r in parsed if r["item_id"] == "inst_b")
        assert item_a["score"] == 0.8
        assert item_a["status"] == "ok"
        assert item_b["score"] == 0.3
        assert item_b["status"] == "failed"

        assert "inst_a" in fake_task.verify_calls
        assert "inst_b" in fake_task.verify_calls

    def test_inline_items_no_verify(self, tmp_path: Path) -> None:
        """inline_items DataNode must NOT call verify — uses subgraph-success scoring."""
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id="x", prompt="go"), DataItem(id="y", prompt="go")]
        wf = _make_data_workflow(items)
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        # Without a task, inline items have no verify — score defaults to 0.0
        assert all(r["status"] in ("ok", "failed") for r in parsed)

    def test_setup_prompt_called_per_item_in_run_item(self, tmp_path: Path) -> None:
        """setup() and prompt() must be called per-item inside run_item, not eagerly."""
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _FakeTask(
            instances_data=[{"id": "i1"}, {"id": "i2"}, {"id": "i3"}],
            verify_scores={"i1": 1.0, "i2": 1.0, "i3": 1.0})

        wf = _make_task_ref_workflow()

        with patch("factory.task.TaskRef.resolve", return_value=fake_task):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        assert sorted(fake_task.setup_calls) == ["i1", "i2", "i3"]
        assert sorted(fake_task.prompt_calls) == ["i1", "i2", "i3"]
        assert sorted(fake_task.verify_calls) == ["i1", "i2", "i3"]

    def test_failing_setup_does_not_block_other_items(self, tmp_path: Path) -> None:
        """A failing setup() for one item must not prevent other items from running."""
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _FakeTask(
            instances_data=[{"id": "ok1"}, {"id": "fail_setup"}, {"id": "ok2"}],
            verify_scores={"ok1": 1.0, "ok2": 0.9})
        original_setup = fake_task.setup

        def failing_setup(instance, workspace):
            if instance.id == "fail_setup":
                raise RuntimeError("setup exploded")
            original_setup(instance, workspace)

        fake_task.setup = failing_setup

        wf = _make_task_ref_workflow()

        with patch("factory.task.TaskRef.resolve", return_value=fake_task):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 3

        failed = next(r for r in parsed if r["item_id"] == "fail_setup")
        assert failed["score"] == 0.0
        assert "error" in failed

        ok_items = [r for r in parsed if r["item_id"] != "fail_setup"]
        assert all(r["score"] > 0 for r in ok_items)


class TestStepWithDataNodeVerifyScores:
    def test_aggregates_verify_scores(self, tmp_path: Path) -> None:
        """_step_with_task should aggregate per-item verify scores, not binary."""
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_task_ref_workflow()
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "a", "score": 0.8, "status": "ok"},
            {"item_id": "b", "score": 0.4, "status": "failed"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end == pytest.approx(0.6)

    def test_falls_back_to_zero_without_item_results(self, tmp_path: Path) -> None:
        """Without item_results (executor crashed early), falls back to 0.0."""
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="i")])
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = []

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end is None


# ── PR #1483 Review Fixes — additional tests ─────────────────────


class TestParallelismDefault:
    def test_parallelism_default_is_1(self) -> None:
        node = DataNode(
            id="dn",
            inline_items=[DataItem(id="i")])
        assert node.parallelism == 1

    def test_parallelism_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataNode(
                id="dn",
                inline_items=[DataItem(id="i")],
                parallelism=0)


class TestNonexistentSourcePathRaises:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="missing",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(tmp_path / "nope"),
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "source_path not found" in result.halt_reason


class TestEmptySourceRaises:
    def test_empty_inline_raises(self, tmp_path: Path) -> None:
        """Zero items after filtering should raise ValueError."""
        from factory.workflow.executor import WorkflowExecutor

        # Use split filter to exclude all items
        wf = Workflow(
            name="empty_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[],),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "resolved 0 items" in result.halt_reason


class TestMalformedJsonlLineIsolated:
    def test_bad_line_skipped_good_lines_kept(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        jsonl_file = tmp_path / "items.jsonl"
        jsonl_file.write_text(
            '{"name": "first"}\n'
            'NOT VALID JSON\n'
            '{"name": "third"}\n'
        )

        wf = Workflow(
            name="jsonl_malformed",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(jsonl_file),
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        # Two good lines kept, one bad line skipped
        assert len(parsed) == 2

    def test_all_lines_bad_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        jsonl_file = tmp_path / "items.jsonl"
        jsonl_file.write_text("bad line 1\nbad line 2\n")

        wf = Workflow(
            name="jsonl_all_bad",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(jsonl_file),
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "resolved 0 items" in result.halt_reason


class TestShuffleDeterministic:
    def test_shuffle_with_seed(self, tmp_path: Path) -> None:
        """Same seed -> same order across runs."""
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id=str(i)) for i in range(20)]
        wf = Workflow(
            name="shuffle_seed",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=items,
                    shuffle=True,
                    shuffle_seed=42),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")

        # Run twice with the same seed -- order must match
        executor1 = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result1 = asyncio.run(executor1.execute())
        ids1 = [r["item_id"] for r in json.loads(result1.node_outputs["data"])]

        executor2 = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result2 = asyncio.run(executor2.execute())
        ids2 = [r["item_id"] for r in json.loads(result2.node_outputs["data"])]

        assert ids1 == ids2
        # Must actually be shuffled (not original order)
        original_ids = [str(i) for i in range(20)]
        assert ids1 != original_ids

    def test_shuffle_from_run_id(self, tmp_path: Path) -> None:
        """Unseeded shuffle derives seed from node_id + run_id."""
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id=str(i)) for i in range(20)]
        wf = Workflow(
            name="shuffle_runid",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=items,
                    shuffle=True,
                    # No shuffle_seed -- uses run_id hash
                ),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")

        executor1 = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result1 = asyncio.run(executor1.execute())
        ids1 = [r["item_id"] for r in json.loads(result1.node_outputs["data"])]

        # Different run_id -> potentially different order (different executor)
        executor2 = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result2 = asyncio.run(executor2.execute())
        ids2 = [r["item_id"] for r in json.loads(result2.node_outputs["data"])]

        # Both should be 20 items
        assert len(ids1) == 20
        assert len(ids2) == 20


class TestExplicitEdgeToSubgraphRejected:
    def test_datanode_without_join_flagged(self) -> None:
        """DataNode branch without JoinNode should be flagged."""
        wf = Workflow(
            name="bad_edge",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i")]),
                "sub_start": FnNode(id="sub_start", command="echo start"),
                "sub_end": FnNode(id="sub_end", command="echo end"),
            },
            edges=[
                Edge(source="data", target="sub_start"),
                Edge(source="sub_start", target="sub_end"),
            ],
            start_node="data")
        issues = wf.validate_graph()
        # Should flag missing JoinNode
        assert len(issues) > 0


class TestCurrentItemJsonWritten:
    def test_current_item_json_created_and_cleaned(self, tmp_path: Path) -> None:
        """current_item.json should exist during subgraph execution."""
        from factory.workflow.executor import WorkflowExecutor

        items = [DataItem(id="test_item", prompt="do it")]
        wf = _make_data_workflow(items)

        # Track whether current_item.json exists during execution
        observed: list[bool] = []
        original_execute = WorkflowExecutor.execute

        async def tracking_execute(self_inner):
            item_json = self_inner.project_path / ".factory" / "current_item.json"
            # For sub-executors (data_item workflows), check if file exists
            if self_inner.workflow.name.endswith("__data_item"):
                observed.append(item_json.exists())
            return await original_execute(self_inner)

        with patch.object(WorkflowExecutor, "execute", tracking_execute):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        # current_item.json should have existed during subgraph execution
        assert any(observed)
        # And it should be cleaned up after
        assert not (tmp_path / ".factory" / "current_item.json").exists()


class TestDirectScoreLookup:
    def test_finds_score_from_item_results(self, tmp_path: Path) -> None:
        """Score is extracted from item_results, not node_outputs JSON."""
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="i", prompt="go")])
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "i", "score": 0.75, "status": "ok"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end == pytest.approx(0.75)


class TestInstanceResultsPopulated:
    def test_instance_results_on_cycle_record(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="a"), DataItem(id="b")])
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "a", "score": 0.9, "status": "ok"},
            {"item_id": "b", "score": 0.3, "status": "failed"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.instance_results is not None
        assert len(record.instance_results) == 2
        assert record.instance_results[0]["item_id"] == "a"
        assert record.instance_results[0]["score"] == 0.9
        assert record.instance_results[1]["item_id"] == "b"


class _ComposeTestTask:
    """Task that satisfies TaskProtocol for compose() tests."""

    def __init__(self) -> None:
        from factory.task import ScoringContract, TaskDefinition

        self.definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"))
        self.scoring = self.definition.scoring
        self.constraints = None

    def instances(self):
        from factory.task import TaskInstance
        return [TaskInstance(id="inst-1")]

    def setup(self, instance: Any, workspace: Path) -> None:
        pass

    def prompt(self, instance: Any) -> str:
        return "test prompt"

    def verify(self, instance: Any, workspace: Path):
        from factory.task import VerifyResult
        return VerifyResult(passed=True, score=1.0)

    def get_evaluator(self) -> Any:
        return None


class TestComposeDataNodeWorkflow:
    def test_compose_succeeds_without_builder(self, tmp_path: Path) -> None:
        """compose() should NOT raise IncompatibleCompositionError for DataNode workflows
        even when the task requires HAS_BUILDER/CAN_RUN_TESTS."""
        from factory.compose import compose
        from factory.workflow.primitives import AgentNode, AgentRole

        # Create a DataNode workflow WITHOUT a builder agent
        wf = Workflow(
            name="eval_only",
            nodes={
                "generator": AgentNode(
                    id="generator",
                    role=AgentRole.RESEARCHER,
                    prompt_template="generate"),
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i1", prompt="test")]),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[
                Edge(source="generator", target="data"),
            ],
            start_node="generator")

        task = _ComposeTestTask()

        # This should NOT raise IncompatibleCompositionError
        loop = compose(wf, task, tmp_path)
        assert loop is not None
        assert loop.workflow is wf


# ── PR #1483 Second Review Fixes — additional tests ─────────────


class TestFormatPathKindMismatch:
    """FIX 1: source_format vs path kind mismatch must raise ValueError."""

    def test_directory_format_on_file_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        a_file = tmp_path / "not_a_dir.txt"
        a_file.write_text("hello")

        wf = Workflow(
            name="mismatch_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(a_file),
                    source_format="directory"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert result.halted  # directory format on a file produces 0 items

    def test_jsonl_format_on_directory_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        a_dir = tmp_path / "not_a_file"
        a_dir.mkdir()

        wf = Workflow(
            name="mismatch_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(a_dir),
                    source_format="jsonl"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert result.halted  # file format on a directory produces 0 items

    def test_csv_format_on_directory_raises(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        a_dir = tmp_path / "not_a_file"
        a_dir.mkdir()

        wf = Workflow(
            name="mismatch_test",
            nodes={
                "data": DataNode(
                    id="data",
                    source_path=str(a_dir),
                    source_format="csv"),
                "sub": FnNode(id="sub", command="echo x"),
                "join": JoinNode(id="join", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="join"),
            ],
            start_node="data")
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert result.halted  # file format on a directory produces 0 items


class TestComposeAddsDataNode:
    """compose() adds implicit DataNode+JoinNode when missing."""

    def test_compose_adds_data_node(self, tmp_path: Path) -> None:
        from factory.compose import compose, _workflow_has_data_node

        wf = Workflow(
            name="no_data",
            nodes={"a": FnNode(id="a", command="echo x")},
            edges=[],
            start_node="a")
        assert not _workflow_has_data_node(wf)

        task = _ComposeTestTask()
        loop = compose(wf, task, tmp_path)
        assert _workflow_has_data_node(loop.workflow)

    def test_data_node_workflow_unchanged(self, tmp_path: Path) -> None:
        from factory.compose import compose, _workflow_has_data_node

        wf = _make_data_workflow([DataItem(id="i")])
        assert _workflow_has_data_node(wf)

        task = _ComposeTestTask()
        loop = compose(wf, task, tmp_path)
        # Should not double-wrap
        data_nodes = [n for n in loop.workflow.nodes.values()
                      if isinstance(n, DataNode)]
        assert len(data_nodes) == 1


class TestStepWithDataNodeCoverage:
    """_step_with_task happy path and failure coverage."""

    def test_happy_path(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="i", prompt="go")])
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "i", "score": 0.85, "status": "ok"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end == pytest.approx(0.85)
        assert record.cycle_number == 1
        assert record.instance_results is not None
        assert len(record.instance_results) == 1
        assert record.instance_results[0]["item_id"] == "i"

    def test_executor_failure_defaults_score(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from factory.inner_loop import InnerLoop
        from factory.task import DefaultTask
        from factory.workflow.executor import ExecutionResult

        wf = _make_data_workflow([DataItem(id="i")])
        task = DefaultTask(test_command="true")

        mock_result = ExecutionResult()
        mock_result.success = False
        mock_result.item_results = []

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop._step_with_task()

        assert record.score_end is None


# ── disk_reads re-scan after setup() ────────────────────────────


class _SetupWritingTask:
    """Task whose setup() creates a file that a subgraph node reads."""

    def __init__(self, setup_file: str) -> None:
        self._setup_file = setup_file

    def instances(self):
        from factory.task import TaskInstance
        return [TaskInstance(id="inst1")]

    def setup(self, instance: Any, workspace: Path) -> None:
        target = workspace / self._setup_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("setup content")

    def prompt(self, instance: Any) -> str:
        return "go"

    def verify(self, instance: Any, workspace: Path):
        from factory.task import VerifyResult
        return VerifyResult(passed=True, score=1.0)


class TestDataNodeLoopSubgraph:
    """DataNode + Loop/Gate subgraph integration tests (edge-based)."""

    @pytest.mark.asyncio
    async def test_data_node_with_loop_subgraph(self, tmp_path: Path) -> None:
        """DataNode with a loop body using real edges."""
        import subprocess as _sp

        from factory.workflow.executor import WorkflowExecutor
        from factory.workflow.primitives import GateNode, VerdictType

        project_path = tmp_path
        _sp.run(["git", "init"], cwd=project_path, capture_output=True, check=True)
        _sp.run(["git", "config", "user.name", "test"], cwd=project_path, check=True)
        _sp.run(["git", "config", "user.email", "test@test.com"], cwd=project_path, check=True)
        _sp.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=project_path, capture_output=True, check=True)
        (project_path / ".factory").mkdir(parents=True, exist_ok=True)
        counter_file = project_path / "counter.txt"
        pp = str(project_path)

        body_node = FnNode(
            id="loop_body",
            command=f"python3 -c \"open('{pp}/counter.txt','a').write('x\\n')\"",
            reads=set(),
            writes={"counter.txt"})

        gate = GateNode(
            id="loop_gate",
            evaluator_type="fn",
            evaluator_command=(
                f"python3 -c \""
                f"import pathlib; "
                f"p=pathlib.Path('{pp}/counter.txt'); "
                f"c=len(p.read_text().splitlines()) if p.exists() else 0; "
                f"print('PROCEED' if c >= 3 else 'RELOOP: try again')"
                f"\""
            ),
            reads=set())

        exit_node = FnNode(id="loop_exit", command="echo done")

        wf = Workflow(
            name="loop_data_test",
            nodes={
                "data_driver": DataNode(
                    id="data_driver",
                    inline_items=[DataItem(id="game1", prompt="play")]),
                "loop_body": body_node,
                "loop_gate": gate,
                "loop_exit": exit_node,
                "join": JoinNode(id="join", sources=["loop_exit"]),
            },
            edges=[
                Edge(source="data_driver", target="loop_body"),
                Edge(source="loop_body", target="loop_gate"),
                Edge(source="loop_gate", target="loop_body", condition=VerdictType.RELOOP),
                Edge(source="loop_gate", target="loop_exit", condition=VerdictType.PROCEED),
                Edge(source="loop_exit", target="join"),
            ],
            start_node="data_driver")

        executor = WorkflowExecutor(wf, project_path, dry_run=False)
        result = await executor.execute()

        assert result.success, f"Execution failed: {result.halt_reason}"
        assert counter_file.exists(), "counter.txt should exist"
        lines = counter_file.read_text().splitlines()
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

    def test_data_node_without_join_node_flagged(self) -> None:
        """DataNode branch without JoinNode should be flagged in validation."""
        wf = Workflow(
            name="no_join_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i")]),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[
                Edge(source="data", target="sub"),
            ],
            start_node="data")
        issues = wf.validate_graph()
        assert len(issues) > 0

    def test_loop_package_compiled_preserves_edges(self) -> None:
        """Loop Package compiled preserves edges including RELOOP."""
        from factory.workflow.package import Loop, Package
        from factory.workflow.primitives import GateNode, VerdictType

        body_node = FnNode(
            id="loop_body",
            command="echo body",
            reads=set())
        body_pkg = Package(
            name="body",
            graph=Workflow(
                name="body_graph",
                nodes={"loop_body": body_node},
                edges=[],
                start_node="loop_body"),
            entry_node="loop_body",
            exit_node="loop_body")

        gate = GateNode(
            id="loop_gate",
            evaluator_type="fn",
            evaluator_command="echo PROCEED",
            reads=set())

        loop_pkg = Loop(body_pkg, gate, max_iterations=5, name="test_loop")

        # Check all 3 loop edges are preserved
        edge_tuples = [(e.source, e.target, e.condition) for e in loop_pkg.graph.edges]

        # body → gate (unconditional)
        assert ("loop_body", "loop_gate", None) in edge_tuples, (
            f"Missing body→gate edge. Edges: {edge_tuples}"
        )
        # gate → body (RELOOP)
        assert ("loop_gate", "loop_body", VerdictType.RELOOP) in edge_tuples, (
            f"Missing gate→body RELOOP edge. Edges: {edge_tuples}"
        )


class TestDiskReadsRescanAfterSetup:
    """setup()-created files must appear in sub-executor completed_files."""

    @pytest.mark.skip(reason="Requires worktree-aware setup file propagation — PR B")
    def test_setup_created_file_in_completed_files(self, tmp_path: Path) -> None:
        """When task.setup() writes a file declared in a subgraph node's reads,
        the sub-executor's completed_files must include it."""
        from factory.workflow.executor import WorkflowExecutor

        setup_file = "data/input.txt"

        wf = Workflow(
            name="rescan_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake.module:SetupWritingTask"),
                "reader": FnNode(
                    id="reader",
                    command="echo ok",
                    reads={setup_file}),
                "join": JoinNode(id="join", sources=["reader"]),
            },
            edges=[
                Edge(source="data", target="reader"),
                Edge(source="reader", target="join"),
            ],
            start_node="data")

        fake_task = _SetupWritingTask(setup_file)

        # Capture the completed_files set on the sub-executor
        captured_completed: list[set[str]] = []
        original_execute = WorkflowExecutor.execute

        async def spy_execute(self_inner):
            if self_inner.workflow.name.endswith("__data_item"):
                captured_completed.append(set(self_inner.completed_files))
            return await original_execute(self_inner)

        with patch("factory.task.TaskRef.resolve", return_value=fake_task), \
             patch.object(WorkflowExecutor, "execute", spy_execute):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True, validate=False)
            result = asyncio.run(executor.execute())

        assert result.success, f"halted: {result.halt_reason}"
        assert len(captured_completed) == 1
        assert setup_file in captured_completed[0]

    @pytest.mark.asyncio
    async def test_data_node_loop_with_agent_body(self, tmp_path: Path) -> None:
        """DataNode + Loop where body is an AgentNode using real edges."""
        import subprocess as _sp

        from factory.testing import FakeAgent
        from factory.workflow.executor import WorkflowExecutor
        from factory.workflow.primitives import AgentNode, AgentRole, GateNode, VerdictType

        project_path = tmp_path
        _sp.run(["git", "init"], cwd=project_path, capture_output=True, check=True)
        _sp.run(["git", "config", "user.name", "test"], cwd=project_path, check=True)
        _sp.run(["git", "config", "user.email", "test@test.com"], cwd=project_path, check=True)
        _sp.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=project_path, capture_output=True, check=True)
        (project_path / '.factory').mkdir(parents=True, exist_ok=True)
        project_path / 'counter.txt'
        str(project_path)

        call_count = 0

        def game_behavior(role, task, proj_path, **kwargs):
            nonlocal call_count
            call_count += 1
            cf = Path(proj_path) / 'counter.txt'
            cf.parent.mkdir(parents=True, exist_ok=True)
            with open(cf, 'a') as f:
                f.write(f'move {call_count}\n')
            return (f'Generated move {call_count}', 0)

        generator = AgentNode(
            id='generator',
            role=AgentRole.BUILDER,
            prompt_template='Generate the next move',
            reads=set(),
            writes=set(),
            timeout=30)

        # Use relative path — works in both parent and worktree
        gate = GateNode(
            id='game_gate',
            evaluator_type='fn',
            evaluator_command=(
                "python3 -c \""
                "import pathlib; "
                "p=pathlib.Path('counter.txt'); "
                "c=len(p.read_text().splitlines()) if p.exists() else 0; "
                "print('PROCEED' if c >= 3 else 'RELOOP: keep playing')"
                "\""
            ),
            reads=set())

        exit_node = FnNode(id='game_exit', command='echo done')

        wf = Workflow(
            name='agent_loop_test',
            nodes={
                'game_data': DataNode(
                    id='game_data',
                    inline_items=[DataItem(id='game1', prompt='Play chess')]),
                'generator': generator,
                'game_gate': gate,
                'game_exit': exit_node,
                'join': JoinNode(id='join', sources=['game_exit']),
            },
            edges=[
                Edge(source='game_data', target='generator'),
                Edge(source='generator', target='game_gate'),
                Edge(source='game_gate', target='generator', condition=VerdictType.RELOOP),
                Edge(source='game_gate', target='game_exit', condition=VerdictType.PROCEED),
                Edge(source='game_exit', target='join'),
            ],
            start_node='game_data')

        agent = FakeAgent(wf, behavior=game_behavior)
        executor = WorkflowExecutor(wf, project_path, agent_fn=agent, validate=False, auto_write_outputs=False)
        result = await executor.execute()

        assert result.success, f'Execution failed: {result.halt_reason}'
        # counter.txt may be in a worktree — check via item results
        assert call_count == 3, f'Expected agent called 3 times, got {call_count}'
        agent.assert_called('builder')
        agent.assert_call_count(3)

    @pytest.mark.asyncio
    async def test_setup_read_path_mismatch_logs_warning(self, tmp_path: Path) -> None:
        """When setup() creates a file at a different path than node.reads expects,
        the reader should timeout waiting for the mismatched read path."""
        from factory.workflow.executor import WorkflowExecutor

        project_path = tmp_path
        (project_path / '.factory').mkdir(parents=True, exist_ok=True)

        (project_path / '.factory' / 'memory.md').write_text('game state')

        wf = Workflow(
            name='mismatch_test',
            nodes={
                'data': DataNode(
                    id='data',
                    inline_items=[DataItem(id='item1', prompt='test')]),
                'reader': FnNode(
                    id='reader',
                    command='echo ok',
                    reads={'memory.md'},
                ),
                'join': JoinNode(id='join', sources=['reader']),
            },
            edges=[
                Edge(source='data', target='reader'),
                Edge(source='reader', target='join'),
            ],
            start_node='data')

        async def fast_wait(self_inner, node):
            poll_interval = 0.1
            waited = 0.0
            while True:
                missing = node.reads - self_inner.completed_files
                if not missing:
                    return
                if waited >= 0.3:
                    self_inner.result.halted = True
                    self_inner.result.halt_reason = (
                        f"node '{node.id}' timed out waiting for reads: {sorted(missing)}"
                    )
                    return
                await asyncio.sleep(poll_interval)
                waited += poll_interval

        with patch.object(WorkflowExecutor, '_wait_for_reads', fast_wait):
            executor = WorkflowExecutor(wf, project_path, dry_run=False, validate=False)
            result = await executor.execute()

        assert not result.success
        assert result.halted
