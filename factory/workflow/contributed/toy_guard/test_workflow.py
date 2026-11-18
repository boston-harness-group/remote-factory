"""Structural checks for the toy-guard contributed workflow."""

from __future__ import annotations

from factory.workflow.contributed.toy_guard.workflow import (
    GUARD_MAX_ITERATIONS,
    GUARDS,
    TOY_AGENT_ID,
    meta,
    workflow,
)
from factory.workflow.definitions import register_all
from factory.workflow.guardrails import GUARDRAILS_HEADER, guard_dir, guard_slug
from factory.workflow.primitives import (
    AgentNode,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    VerdictType,
)
from factory.workflow.validation import validate_workflow

SEGMENT = guard_dir(TOY_AGENT_ID).name
CHECKER_IDS = [f"guard_{SEGMENT}_{guard_slug(guard.rule)}" for guard in GUARDS]
RESET_ID = f"guard_{SEGMENT}_reset"
FORK_ID = f"guard_{SEGMENT}_fork"
JOIN_ID = f"guard_{SEGMENT}_join"
GATE_ID = f"guard_{SEGMENT}_gate"


class TestToyGuardMeta:
    def test_meta_has_name(self) -> None:
        assert meta["name"] == "toy-guard"

    def test_meta_has_description(self) -> None:
        assert "guard" in meta["description"].lower()


class TestToyGuardGraph:
    def test_node_count(self) -> None:
        wf = workflow()
        assert len(wf.nodes) == 8
        expected = {"draft", "publish", RESET_ID, FORK_ID, JOIN_ID, GATE_ID, *CHECKER_IDS}
        assert sorted(wf.nodes) == sorted(expected)

    def test_edge_count(self) -> None:
        wf = workflow()
        assert len(wf.edges) == 6

    def test_graph_validates(self) -> None:
        wf = workflow()
        assert validate_workflow(wf) == []

    def test_stage_present(self) -> None:
        wf = workflow()
        assert isinstance(wf.nodes[RESET_ID], FnNode)
        assert isinstance(wf.nodes[FORK_ID], ForkNode)
        assert isinstance(wf.nodes[JOIN_ID], JoinNode)
        assert isinstance(wf.nodes[GATE_ID], GateNode)
        for checker_id in CHECKER_IDS:
            assert isinstance(wf.nodes[checker_id], AgentNode)

    def test_fork_and_join_targets(self) -> None:
        wf = workflow()
        fork = wf.nodes[FORK_ID]
        join = wf.nodes[JOIN_ID]
        assert isinstance(fork, ForkNode)
        assert isinstance(join, JoinNode)
        assert fork.targets == CHECKER_IDS
        assert join.sources == CHECKER_IDS

    def test_exactly_one_guarded_agent(self) -> None:
        wf = workflow()
        guarded = [
            node_id
            for node_id, node in wf.nodes.items()
            if isinstance(node, AgentNode) and node.guards
        ]
        assert guarded == [TOY_AGENT_ID]
        draft = wf.nodes[TOY_AGENT_ID]
        assert isinstance(draft, AgentNode)
        assert [guard.rule for guard in draft.guards] == [guard.rule for guard in GUARDS]

    def test_gate_carries_guard_budget(self) -> None:
        wf = workflow()
        gate = wf.nodes[GATE_ID]
        assert isinstance(gate, GateNode)
        assert gate.evaluator_type == "fn"
        assert gate.max_iterations == GUARD_MAX_ITERATIONS == 2

    def test_guard_block_present_exactly_once(self) -> None:
        wf = workflow()
        draft = wf.nodes[TOY_AGENT_ID]
        assert isinstance(draft, AgentNode)
        assert draft.prompt_template.count(GUARDRAILS_HEADER) == 1
        for guard in GUARDS:
            assert guard.rule in draft.prompt_template

    def test_proceed_edge_to_publish(self) -> None:
        wf = workflow()
        edges = [
            edge
            for edge in wf.edges
            if edge.source == GATE_ID
            and edge.target == "publish"
            and edge.condition == VerdictType.PROCEED
        ]
        assert len(edges) == 1

    def test_reloop_edge_to_draft(self) -> None:
        wf = workflow()
        edges = [
            edge
            for edge in wf.edges
            if edge.source == GATE_ID
            and edge.target == TOY_AGENT_ID
            and edge.condition == VerdictType.RELOOP
        ]
        assert len(edges) == 1


class TestToyGuardRegistration:
    def test_registered_in_register_all(self) -> None:
        workflows = register_all()
        assert "toy-guard" in workflows

    def test_registered_workflow_validates(self) -> None:
        wf = register_all()["toy-guard"]
        assert validate_workflow(wf) == []
        assert GATE_ID in wf.nodes
