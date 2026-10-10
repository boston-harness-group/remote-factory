"""Unit tests for DataNode fork/join validation rules (Step 2.6).

Tests:
- DataNode without JoinNode
- Tool engine rejects DataNode graphs
- ceo-skill / ceo-tool DataNode error
- Split subset error
- Mutation property test (random mutations → valid or None)
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow,
)


# ── Validation rule tests ────────────────────────────────────────────


class TestDataNodeValidation:
    """Validation rules for DataNode fork/join structure."""

    def test_datanode_without_join_fails(self) -> None:
        """DataNode without a downstream JoinNode is a validation error."""
        wf = Workflow(
            name="no-join",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
                "work": FnNode(id="work", command="echo work"),
            },
            edges=[Edge(source="data", target="work")],
            start_node="data",
        )
        issues = wf.validate_graph()
        assert any("JoinNode" in i for i in issues), (
            f"Expected JoinNode validation error, got: {issues}"
        )

    def test_datanode_with_join_passes(self) -> None:
        """DataNode with a downstream JoinNode passes validation."""
        wf = Workflow(
            name="with-join",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
                "work": FnNode(id="work", command="echo work"),
                "join": JoinNode(id="join", sources=["work"]),
            },
            edges=[
                Edge(source="data", target="work"),
                Edge(source="work", target="join"),
            ],
            start_node="data",
        )
        issues = wf.validate_graph()
        assert not any("JoinNode" in i for i in issues), (
            f"Unexpected JoinNode error: {issues}"
        )

    def test_datanode_no_outgoing_edges(self) -> None:
        """DataNode with no outgoing edges is a validation error."""
        wf = Workflow(
            name="no-edges",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
            },
            edges=[],
            start_node="data",
        )
        issues = wf.validate_graph()
        assert any("outgoing" in i.lower() or "no outgoing" in i.lower() for i in issues), (
            f"Expected outgoing edge error, got: {issues}"
        )


# ── Tool engine rejection ────────────────────────────────────────────


class TestToolEngineRejectsDataNode:
    """Tool engine must reject DataNode workflows."""

    def test_tool_init_rejects_datanode(self, tmp_path: Path) -> None:
        """tool_init raises ValueError for DataNode workflows."""
        from factory.workflow.tool import tool_init

        # Register a workflow with DataNode
        from factory.workflow.registry import WorkflowRegistry

        wf = Workflow(
            name="data-tool-test",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
                "work": FnNode(id="work", command="echo work"),
                "join": JoinNode(id="join", sources=["work"]),
            },
            edges=[
                Edge(source="data", target="work"),
                Edge(source="work", target="join"),
            ],
            start_node="data",
        )

        # Mock the registry to return our workflow
        with pytest.raises(ValueError, match="Tool engine does not support DataNode"):
            from unittest.mock import patch
            with patch.object(WorkflowRegistry, "get_workflow", return_value=wf):
                tool_init("data-tool-test", tmp_path)


# ── ceo-skill / ceo-tool DataNode error ──────────────────────────────


class TestCeoSkillDataNodeError:
    """ceo-skill + DataNode raises clear error."""

    def test_ceo_skill_datanode_raises(self, tmp_path: Path) -> None:
        """ceo-skill execution_strategy + DataNode raises ValueError."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir()
        wf = Workflow(
            name="ceo-test",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
                "work": FnNode(id="work", command="echo work"),
                "join": JoinNode(id="join", sources=["work"]),
            },
            edges=[
                Edge(source="data", target="work"),
                Edge(source="work", target="join"),
            ],
            start_node="data",
        )
        task = MagicMock()
        task.instances.return_value = []
        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf,
            execution_strategy="ceo-skill",
        )
        with pytest.raises(ValueError, match="not supported"):
            loop.step()

    def test_ceo_tool_datanode_raises(self, tmp_path: Path) -> None:
        """ceo-tool execution_strategy + DataNode raises ValueError."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir()
        wf = Workflow(
            name="ceo-test",
            nodes={
                "data": DataNode(id="data", inline_items=[DataItem(id="i1")]),
                "work": FnNode(id="work", command="echo work"),
                "join": JoinNode(id="join", sources=["work"]),
            },
            edges=[
                Edge(source="data", target="work"),
                Edge(source="work", target="join"),
            ],
            start_node="data",
        )
        task = MagicMock()
        task.instances.return_value = []
        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf,
            execution_strategy="ceo-tool",
        )
        with pytest.raises(ValueError, match="not supported"):
            loop.step()


# ── ceo-skill export raises ──────────────────────────────────────────


class TestSkillExportDataNodeError:
    """Skill export raises for DataNode workflows."""

    def test_data_to_instruction_raises(self) -> None:
        """_data_to_instruction raises ValueError."""
        from factory.workflow.skill_export import _data_to_instruction

        dn = DataNode(id="data", inline_items=[DataItem(id="i1")])
        wf = Workflow(
            name="t",
            nodes={"data": dn},
            edges=[],
            start_node="data",
        )
        with pytest.raises(ValueError, match="not supported"):
            _data_to_instruction(dn, wf)


# ── Split subset error ───────────────────────────────────────────────


class TestSplitSubsetError:
    """Zero items after filtering raises ValueError."""

    def test_zero_items_raises(self) -> None:
        """run_fork with zero items raises ValueError."""
        import asyncio

        from factory.workflow.data_runtime import run_fork

        wf = Workflow(
            name="empty-test",
            nodes={
                "data": DataNode(id="data", inline_items=[]),
                "work": FnNode(id="work", command="echo work"),
                "join": JoinNode(id="join", sources=["work"]),
            },
            edges=[
                Edge(source="data", target="work"),
                Edge(source="work", target="join"),
            ],
            start_node="data",
        )
        dn = wf.nodes["data"]
        assert isinstance(dn, DataNode)
        with pytest.raises(ValueError, match="0 items"):
            asyncio.run(run_fork(
                wf, dn, "data", Path("/tmp"),
            ))


# ── Mutation property test ───────────────────────────────────────────


class TestMutationProperty:
    """Random mutations of a fork workflow always give a valid graph or None."""

    def test_mutations_always_valid_or_none(self) -> None:
        """50 random mutations of a DataNode workflow: validate_and_repair
        returns a valid graph or None."""
        from factory.outer_loop.mutations import (
            _try_mutation,
            MutationType,
            validate_and_repair,
        )

        wf = Workflow(
            name="fork-mut",
            nodes={
                "plan": FnNode(id="plan", command="echo plan"),
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i1", prompt="test")],
                ),
                "work": AgentNode(
                    id="work",
                    role=AgentRole.BUILDER,
                    prompt_template="Build.",
                ),
                "check": FnNode(id="check", command="echo check"),
                "join": JoinNode(id="join", sources=["check"]),
                "summarize": FnNode(id="summarize", command="echo summary"),
            },
            edges=[
                Edge(source="plan", target="data"),
                Edge(source="data", target="work"),
                Edge(source="work", target="check"),
                Edge(source="check", target="join"),
                Edge(source="join", target="summarize"),
            ],
            start_node="plan",
        )
        frozen = frozenset({"data", "join"})
        mutation_types = list(MutationType)

        random.seed(42)
        valid_count = 0
        for _ in range(50):
            mt = random.choice(mutation_types)
            result = _try_mutation(wf, mt, frozen)
            if result is None:
                continue  # Mutation failed gracefully — OK
            child_wf, _record = result
            # validate_and_repair should return valid or None
            repaired = validate_and_repair(child_wf)
            if repaired is None:
                continue  # Repair rejected it — OK
            issues = repaired.validate_graph()
            real_issues = [
                i for i in issues
                if "empty prompt_template" not in i
            ]
            assert not real_issues, (
                f"Mutation {mt} + repair produced invalid workflow: {real_issues}"
            )
            valid_count += 1
        # At least some mutations should succeed
        assert valid_count > 0, "No mutations succeeded at all"


# ── Legacy DataNode load-time conversion ─────────────────────────────


class TestLegacyLoadConversion:
    """Workflow.from_dict strips subgraph_entry/exit and adds edges + JoinNode."""

    def test_node_insert_on_datanode_workflow_validates(self) -> None:
        """node_insert on a DataNode workflow produces a graph that passes validate_workflow()."""
        from factory.workflow.validation import validate_workflow

        # Create a simple DataNode workflow
        wf = Workflow(
            name='test-wf',
            nodes={
                'data': DataNode(id='data'),
                'builder': AgentNode(
                    id='builder', role=AgentRole.BUILDER,
                    prompt_template='do work', writes={'output.txt'}, reads=set(),
                ),
                '_join_data': JoinNode(id='_join_data', sources=['builder']),
            },
            edges=[
                Edge(source='data', target='builder'),
                Edge(source='builder', target='_join_data'),
            ],
            start_node='data',
        )

        # The branch sub-workflow (what data_runtime creates)
        sub = wf.subgraph({'builder'}, name='test-wf__data_item', start_node='builder')

        # Now manually insert a node that reads current_item.json (simulating node_insert)
        reviewer = AgentNode(
            id='reviewer_99', role=AgentRole.CODE_REVIEWER,
            prompt_template='review', reads={'.factory/current_item.json'}, writes=set(),
        )
        sub.nodes['reviewer_99'] = reviewer
        sub.edges.append(Edge(source='builder', target='reviewer_99'))

        issues = validate_workflow(sub)
        assert not issues, f'Validation should pass for data_item subgraph: {issues}'

    def test_from_dict_converts_legacy(self) -> None:
        """Legacy DataNode with subgraph_entry/exit is auto-converted."""
        data = {
            "name": "legacy",
            "nodes": {
                "data": {
                    "_type": "DataNode",
                    "id": "data",
                    "subgraph_entry": "work",
                    "subgraph_exit": "check",
                },
                "work": {"_type": "FnNode", "id": "work", "command": "echo work"},
                "check": {"_type": "FnNode", "id": "check", "command": "echo check"},
            },
            "edges": [
                {"source": "work", "target": "check"},
            ],
            "start_node": "data",
        }
        wf = Workflow.from_dict(data)
        dn = wf.nodes["data"]
        assert isinstance(dn, DataNode)
        # subgraph_entry/exit should not be attributes
        assert not hasattr(dn, "subgraph_entry")
        # Edge from data → work should exist
        edge_pairs = {(e.source, e.target) for e in wf.edges}
        assert ("data", "work") in edge_pairs
        # JoinNode should be auto-created
        join_ids = [nid for nid, n in wf.nodes.items() if isinstance(n, JoinNode)]
        assert len(join_ids) >= 1
