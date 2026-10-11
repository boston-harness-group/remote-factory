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
        import random as _random
        from factory.outer_loop.mutations import insert_node
        from factory.workflow.validation import validate_workflow

        wf = Workflow(
            name='test-wf',
            nodes={
                'data': DataNode(id='data'),
                'builder': AgentNode(id='builder', role=AgentRole.BUILDER,
                    prompt_template='do work', writes={'output.txt'}, reads=set()),
                '_join_data': JoinNode(id='_join_data', sources=['builder']),
            },
            edges=[Edge(source='data', target='builder'), Edge(source='builder', target='_join_data')],
            start_node='data',
            runtime_inputs=frozenset({'.factory/current_item.json'}),
        )
        _random.seed(42)
        reviewer = AgentNode(
            id='reviewer_42', role=AgentRole.CODE_REVIEWER,
            prompt_template='review the work', reads={'.factory/current_item.json'}, writes=set(),
        )
        result = insert_node(wf, reviewer, 'builder', frozen_nodes={'data', '_join_data'})
        assert result is not None, 'insert_node returned None'
        new_wf, rec = result
        issues = validate_workflow(new_wf)
        assert not issues, f'validate_workflow failed after insert_node: {issues}'

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


# ── Bug 5 (moved from test_five_bugs): FnNode validation ────────


def test_empty_fn_node_validation_error():
    """FnNode with no command AND no callable_name must fail validation."""
    from factory.workflow.primitives import FnNode

    with pytest.raises(ValueError, match="command.*callable_name|callable_name.*command"):
        FnNode(id="empty-fn", command="", callable_name=None)


def test_fn_node_with_command_is_valid():
    """FnNode with a command should pass validation."""
    from factory.workflow.primitives import FnNode

    node = FnNode(id="good-fn", command="echo hello")
    assert node.command == "echo hello"


def test_fn_node_with_callable_is_valid():
    """FnNode with a callable_name should pass validation."""
    from factory.workflow.primitives import FnNode

    node = FnNode(id="good-fn", callable_name="my_module:my_fn")
    assert node.callable_name == "my_module:my_fn"


def test_node_insert_inherits_model_and_timeout():
    """node_insert copies model and timeout from nearest AgentNode."""
    import random

    from factory.outer_loop.mutations import MutationType, _try_mutation
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        DataNode,
        Edge,
        JoinNode,
        Workflow,
    )

    wf = Workflow(
        name="model-inherit-test",
        nodes={
            "data": DataNode(id="data"),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                timeout=300,
                prompt_template="build it",
                writes={"out.txt"},
                reads=set(),
            ),
            "_join_data": JoinNode(id="_join_data", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="_join_data"),
        ],
        start_node="data",
    )

    # Try multiple seeds to get a successful node_insert
    result = None
    for seed in range(100):
        random.seed(seed)
        result = _try_mutation(
            wf, MutationType.NODE_INSERT, frozenset({"data", "_join_data"})
        )
        if result is not None:
            break

    assert result is not None, "node_insert never succeeded"
    new_wf, _rec = result
    # Find the inserted node(s)
    new_node_ids = set(new_wf.nodes.keys()) - set(wf.nodes.keys())
    assert len(new_node_ids) >= 1
    for nid in new_node_ids:
        node = new_wf.nodes[nid]
        if isinstance(node, AgentNode):
            assert node.model == "claude-haiku-4-5-20251001", (
                f"Inserted node {nid} has model={node.model!r}, expected haiku"
            )
            assert node.timeout == 300, (
                f"Inserted node {nid} has timeout={node.timeout}, expected 300"
            )


def test_infer_agent_params_from_target():
    """_infer_agent_params returns model/timeout from the target node."""
    from factory.outer_loop.mutations import _infer_agent_params
    from factory.workflow.primitives import AgentNode, AgentRole, Edge, Workflow

    wf = Workflow(
        name="infer-test",
        nodes={
            "a": AgentNode(
                id="a",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                timeout=300,
            ),
            "b": AgentNode(
                id="b",
                role=AgentRole.CODE_REVIEWER,
                model="claude-sonnet-4-20250514",
                timeout=600,
            ),
        },
        edges=[Edge(source="a", target="b")],
        start_node="a",
    )

    # Target is node "a" -> should get a's model and timeout
    params = _infer_agent_params(wf, "a")
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["timeout"] == 300

    # Target is node "b" -> should get b's model and timeout
    params = _infer_agent_params(wf, "b")
    assert params["model"] == "claude-sonnet-4-20250514"
    assert params["timeout"] == 600


def test_infer_agent_params_fallback():
    """_infer_agent_params falls back to first AgentNode with model when target is not an AgentNode."""
    from factory.outer_loop.mutations import _infer_agent_params
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        DataNode,
        Edge,
        JoinNode,
        Workflow,
    )

    wf = Workflow(
        name="fallback-test",
        nodes={
            "data": DataNode(id="data"),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                timeout=300,
            ),
            "_join": JoinNode(id="_join", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="_join"),
        ],
        start_node="data",
    )

    # Target is a DataNode -> should fall back to builder's model/timeout
    params = _infer_agent_params(wf, "data")
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["timeout"] == 300


def test_infer_agent_params_empty_model():
    """_infer_agent_params returns empty dict when no node has a model set."""
    from factory.outer_loop.mutations import _infer_agent_params
    from factory.workflow.primitives import AgentNode, AgentRole, Workflow

    wf = Workflow(
        name="empty-model-test",
        nodes={
            "a": AgentNode(id="a", role=AgentRole.BUILDER, model=""),
        },
        edges=[],
        start_node="a",
    )

    params = _infer_agent_params(wf, "a")
    assert "model" not in params


def test_infer_agent_params_nonexistent_node():
    """_infer_agent_params handles a nonexistent near_node_id gracefully."""
    from factory.outer_loop.mutations import _infer_agent_params
    from factory.workflow.primitives import AgentNode, AgentRole, Workflow

    wf = Workflow(
        name="nonexistent-test",
        nodes={
            "a": AgentNode(
                id="a",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                timeout=300,
            ),
        },
        edges=[],
        start_node="a",
    )

    # Non-existent node -> falls back to scanning
    params = _infer_agent_params(wf, "does-not-exist")
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["timeout"] == 300
