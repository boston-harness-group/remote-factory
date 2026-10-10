"""Tests for gate/verdict routing: PROCEED, RELOOP, HALT paths.

Models the builder -> gate_qa -> (RELOOP -> builder | PROCEED -> next | HALT -> End)
pattern from the factory's definitions.py gate routing.
"""

from typing import Callable

from pydantic_graph import BaseNode, End, EndMarker, GraphBuilder, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.nodes.gates import GateBaseNode
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult, VerdictType


# ── Test workflow nodes ─────────────────────────────────────────


class MockNextNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Terminal node reached after gate PROCEED."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        ctx.state.node_outputs["MockNextNode"] = "executed"
        ctx.state.events.append({"node": "MockNextNode", "action": "next"})
        return End(HaltResult(reason="proceed_complete"))


class MockBuilderNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Simulated builder that feeds into the QA gate."""

    def __init__(
        self,
        gate_verdict_fn: Callable[[FactoryState], VerdictType] | None = None,
        gate_max_iterations: int = 3,
    ) -> None:
        self.gate_verdict_fn = gate_verdict_fn
        self.gate_max_iterations = gate_max_iterations

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> "QAGateNode":
        feedback = ctx.state.node_feedback.get("MockBuilderNode", "")
        output = f"built (feedback: {feedback})" if feedback else "built"
        ctx.state.node_outputs["MockBuilderNode"] = output
        ctx.state.events.append(
            {
                "node": "MockBuilderNode",
                "action": "build",
                "feedback_received": feedback,
            }
        )
        return QAGateNode(
            gate_id="qa_gate",
            verdict_fn=self.gate_verdict_fn,
            max_iterations=self.gate_max_iterations,
        )


class QAGateNode(GateBaseNode):
    """Concrete gate modeling the QA verdict pattern."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> MockBuilderNode | MockNextNode | End[HaltResult]:
        verdict = self.evaluate_verdict(ctx.state)

        if verdict == VerdictType.HALT:
            self.record_verdict_event(ctx, verdict)
            return End(HaltResult(reason="gate_halted"))

        if verdict == VerdictType.RELOOP:
            target_id = "MockBuilderNode"
            halt = self.check_and_increment_iteration(ctx, target_id)
            if halt is not None:
                self.record_verdict_event(ctx, VerdictType.HALT, target_id)
                return halt
            iteration = ctx.state.iteration_counts[(self.gate_id, target_id)]
            self.inject_feedback(
                ctx, target_id, f"iteration {iteration}: needs improvement"
            )
            self.record_verdict_event(ctx, verdict, target_id)
            return MockBuilderNode(
                gate_verdict_fn=self.verdict_fn,
                gate_max_iterations=self.max_iterations,
            )

        self.record_verdict_event(ctx, verdict)
        return MockNextNode()


# ── Graph builder ───────────────────────────────────────────────


def _build_gate_graph(
    verdict_fn: Callable[[FactoryState], VerdictType] | None = None,
    max_iterations: int = 3,
) -> tuple[
    GraphBuilder[FactoryState, FactoryDeps, MockBuilderNode, HaltResult],
    MockBuilderNode,
]:
    builder: GraphBuilder[FactoryState, FactoryDeps, MockBuilderNode, HaltResult] = (
        GraphBuilder(
            name="gate-routing-test",
            state_type=FactoryState,
            deps_type=FactoryDeps,
            input_type=MockBuilderNode,
            output_type=HaltResult,
        )
    )
    builder.add_edge(builder.start_node, MockBuilderNode)
    builder.add(builder.node(MockBuilderNode))
    builder.add(builder.node(QAGateNode))
    builder.add(builder.node(MockNextNode))

    start_node = MockBuilderNode(
        gate_verdict_fn=verdict_fn,
        gate_max_iterations=max_iterations,
    )
    return builder, start_node


# ── Tests ───────────────────────────────────────────────────────


async def test_proceed_path(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """PROCEED: gate returns MockNextNode, execution continues to End."""
    graph_builder, start = _build_gate_graph(
        verdict_fn=lambda _: VerdictType.PROCEED
    )
    graph = graph_builder.build()

    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "proceed_complete"

    node_order = [e["node"] for e in factory_state.events]
    assert node_order == ["MockBuilderNode", "qa_gate", "MockNextNode"]

    verdict_events = [
        e for e in factory_state.events if e.get("action") == "gate_verdict"
    ]
    assert len(verdict_events) == 1
    assert verdict_events[0]["verdict"] == "proceed"


async def test_halt_path(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """HALT: gate returns End(HaltResult) immediately."""
    graph_builder, start = _build_gate_graph(
        verdict_fn=lambda _: VerdictType.HALT
    )
    graph = graph_builder.build()

    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "gate_halted"

    node_order = [e["node"] for e in factory_state.events]
    assert node_order == ["MockBuilderNode", "qa_gate"]


async def test_reloop_then_max_iterations_halt(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """RELOOP: builder re-executes with feedback; max_iterations halts."""
    max_iter = 2
    graph_builder, start = _build_gate_graph(
        verdict_fn=lambda _: VerdictType.RELOOP,
        max_iterations=max_iter,
    )
    graph = graph_builder.build()

    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert "max_iterations" in result.reason
    assert f"({max_iter})" in result.reason

    builder_events = [
        e for e in factory_state.events if e["node"] == "MockBuilderNode"
    ]
    assert len(builder_events) == max_iter + 1

    key = ("qa_gate", "MockBuilderNode")
    assert factory_state.iteration_counts[key] == max_iter + 1


async def test_reloop_iteration_count_increments(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Iteration count increments on each RELOOP pass."""
    call_count = 0

    def reloop_twice(state: FactoryState) -> VerdictType:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return VerdictType.RELOOP
        return VerdictType.PROCEED

    graph_builder, start = _build_gate_graph(verdict_fn=reloop_twice)
    graph = graph_builder.build()

    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "proceed_complete"

    key = ("qa_gate", "MockBuilderNode")
    assert factory_state.iteration_counts[key] == 2

    builder_events = [
        e for e in factory_state.events if e["node"] == "MockBuilderNode"
    ]
    assert len(builder_events) == 3


async def test_feedback_accessible_on_reentry(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """node_feedback is accessible to the reloop target on re-entry."""
    call_count = 0

    def reloop_once(state: FactoryState) -> VerdictType:
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return VerdictType.RELOOP
        return VerdictType.PROCEED

    graph_builder, start = _build_gate_graph(verdict_fn=reloop_once)
    graph = graph_builder.build()

    await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    builder_events = [
        e for e in factory_state.events if e["node"] == "MockBuilderNode"
    ]
    assert len(builder_events) == 2

    assert builder_events[0]["feedback_received"] == ""
    assert "iteration 1" in builder_events[1]["feedback_received"]

    assert "MockBuilderNode" in factory_state.node_feedback
    assert "needs improvement" in factory_state.node_feedback["MockBuilderNode"]


async def test_feedback_updates_on_each_reloop(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Feedback is updated on each RELOOP iteration."""
    call_count = 0

    def reloop_twice(state: FactoryState) -> VerdictType:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return VerdictType.RELOOP
        return VerdictType.PROCEED

    graph_builder, start = _build_gate_graph(verdict_fn=reloop_twice)
    graph = graph_builder.build()

    await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    builder_events = [
        e for e in factory_state.events if e["node"] == "MockBuilderNode"
    ]
    assert builder_events[0]["feedback_received"] == ""
    assert "iteration 1" in builder_events[1]["feedback_received"]
    assert "iteration 2" in builder_events[2]["feedback_received"]


async def test_mermaid_shows_branching_topology() -> None:
    """Mermaid diagram shows all 3 verdict edges from the gate."""
    graph_builder, _ = _build_gate_graph()
    graph = graph_builder.build()
    mermaid = graph.render()

    assert "MockBuilderNode" in mermaid
    assert "QAGateNode" in mermaid
    assert "MockNextNode" in mermaid

    assert "MockBuilderNode --> QAGateNode" in mermaid
    assert "decision --> MockBuilderNode" in mermaid
    assert "decision --> MockNextNode" in mermaid
    assert "MockNextNode --> [*]" in mermaid
    assert "decision --> [*]" in mermaid
    assert "QAGateNode --> decision" in mermaid


async def test_graph_iter_yields_events_proceed(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Graph.iter() yields node events plus EndMarker for PROCEED path."""
    graph_builder, start = _build_gate_graph(
        verdict_fn=lambda _: VerdictType.PROCEED
    )
    graph = graph_builder.build()

    events: list[object] = []
    async with graph.iter(
        state=factory_state, deps=factory_deps, inputs=start
    ) as run:
        async for event in run:
            events.append(event)

    assert isinstance(events[-1], EndMarker)
    assert len(events) == 4


async def test_graph_iter_yields_events_reloop(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Graph.iter() yields extra events for RELOOP iterations."""
    call_count = 0

    def reloop_once(state: FactoryState) -> VerdictType:
        nonlocal call_count
        call_count += 1
        return VerdictType.RELOOP if call_count <= 1 else VerdictType.PROCEED

    graph_builder, start = _build_gate_graph(verdict_fn=reloop_once)
    graph = graph_builder.build()

    events: list[object] = []
    async with graph.iter(
        state=factory_state, deps=factory_deps, inputs=start
    ) as run:
        async for event in run:
            events.append(event)

    assert isinstance(events[-1], EndMarker)
    assert len(events) == 6


async def test_default_verdict_is_proceed(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """GateBaseNode with no verdict_fn defaults to PROCEED."""
    graph_builder, _ = _build_gate_graph(verdict_fn=None)
    start = MockBuilderNode(gate_verdict_fn=None)
    graph = graph_builder.build()

    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "proceed_complete"
