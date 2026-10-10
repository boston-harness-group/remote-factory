"""Tests for parallel fork/join via asyncio.gather inside ForkJoinNode.

Test workflow: MockBuilderNode → DeepQAForkJoinNode(3 QA agents) → QAResultGateNode → End

Verifies:
    a. All 3 children execute concurrently (wall-clock ≈ max(child_times), not sum)
    b. All child outputs appear in ctx.state.node_outputs
    c. Mermaid rendering (ForkJoinNode as single node)
    d. Event ordering — fork_join_complete after all child_completed events
"""

import asyncio
import time

from pydantic_graph import BaseNode, End, EndMarker, GraphBuilder, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.nodes.parallel import ChildAgent, ForkJoinNode
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


# ── Simulated QA child agents ─────────────────────────────────


async def _health_checker(
    ctx: GraphRunContext[FactoryState, FactoryDeps],
) -> str:
    await asyncio.sleep(0.05)
    return "health_check: all passing"


async def _code_reviewer(
    ctx: GraphRunContext[FactoryState, FactoryDeps],
) -> str:
    await asyncio.sleep(0.08)
    return "code_review: 7/7 categories PASS"


async def _adversarial_tester(
    ctx: GraphRunContext[FactoryState, FactoryDeps],
) -> str:
    await asyncio.sleep(0.06)
    return "adversarial_qa: feature verified"


QA_CHILDREN = [
    ChildAgent(name="health_checker", fn=_health_checker),
    ChildAgent(name="code_reviewer", fn=_code_reviewer),
    ChildAgent(name="adversarial_tester", fn=_adversarial_tester),
]


# ── Test workflow nodes ────────────────────────────────────────


class QAResultGateNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Simple gate that checks QA results and ends."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        qa_keys = {"health_checker", "code_reviewer", "adversarial_tester"}
        present = qa_keys & ctx.state.node_outputs.keys()
        ctx.state.node_outputs["QAResultGateNode"] = f"reviewed {len(present)}/3"
        ctx.state.events.append(
            {"node": "QAResultGateNode", "action": "gate_check", "qa_count": len(present)}
        )
        return End(HaltResult(reason="qa_complete"))


class DeepQAForkJoinNode(ForkJoinNode):
    """Concrete fork/join modeling the deep-QA parallel subgraph."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> QAResultGateNode:
        await self.execute_children(ctx)
        return QAResultGateNode()


class MockBuilderNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Simulated builder that feeds into the QA fork/join."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> DeepQAForkJoinNode:
        ctx.state.node_outputs["MockBuilderNode"] = "build complete"
        ctx.state.events.append({"node": "MockBuilderNode", "action": "build"})
        return DeepQAForkJoinNode(children=QA_CHILDREN, node_id="deep_qa")


# ── Graph builder ──────────────────────────────────────────────


def _build_parallel_graph() -> (
    GraphBuilder[FactoryState, FactoryDeps, MockBuilderNode, HaltResult]
):
    builder: GraphBuilder[FactoryState, FactoryDeps, MockBuilderNode, HaltResult] = (
        GraphBuilder(
            name="parallel-fork-join-test",
            state_type=FactoryState,
            deps_type=FactoryDeps,
            input_type=MockBuilderNode,
            output_type=HaltResult,
        )
    )
    builder.add_edge(builder.start_node, MockBuilderNode)
    builder.add(builder.node(MockBuilderNode))
    builder.add(builder.node(DeepQAForkJoinNode))
    builder.add(builder.node(QAResultGateNode))
    return builder


# ── Tests ──────────────────────────────────────────────────────


async def test_all_children_execute_concurrently(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Wall-clock time ≈ max(child_times), not sum — proves asyncio.gather concurrency."""
    graph = _build_parallel_graph().build()

    start = time.monotonic()
    result = await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )
    elapsed = time.monotonic() - start

    assert isinstance(result, HaltResult)
    assert result.reason == "qa_complete"

    # Children sleep 0.05 + 0.08 + 0.06 = 0.19s total.
    # If concurrent, wall-clock ≈ max(0.08) = ~0.08s.
    # Allow generous margin for CI but ensure it's well under sequential sum.
    assert elapsed < 0.15, (
        f"Wall-clock {elapsed:.3f}s too close to sequential sum 0.19s — "
        f"children may not be running concurrently"
    )


async def test_all_child_outputs_in_state(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """All 3 child outputs appear in ctx.state.node_outputs."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    assert factory_state.node_outputs["health_checker"] == "health_check: all passing"
    assert factory_state.node_outputs["code_reviewer"] == "code_review: 7/7 categories PASS"
    assert factory_state.node_outputs["adversarial_tester"] == "adversarial_qa: feature verified"


async def test_builder_output_preserved(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Builder output remains in state after fork/join completes."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    assert factory_state.node_outputs["MockBuilderNode"] == "build complete"
    assert factory_state.node_outputs["QAResultGateNode"] == "reviewed 3/3"


async def test_gate_sees_all_qa_results(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """QAResultGateNode receives all 3 QA outputs from the fork/join."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    gate_events = [
        e for e in factory_state.events if e["node"] == "QAResultGateNode"
    ]
    assert len(gate_events) == 1
    assert gate_events[0]["qa_count"] == 3


async def test_event_ordering(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Events: builder → child_completed × 3 → fork_join_complete → gate_check."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    actions = [e["action"] for e in factory_state.events]

    assert actions[0] == "build"

    child_completed_indices = [
        i for i, a in enumerate(actions) if a == "child_completed"
    ]
    assert len(child_completed_indices) == 3

    fork_join_idx = actions.index("fork_join_complete")
    assert all(ci < fork_join_idx for ci in child_completed_indices)

    assert actions[-1] == "gate_check"


async def test_fork_join_complete_event_lists_children(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """fork_join_complete event contains all child names and timing."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    fj_events = [
        e for e in factory_state.events if e["action"] == "fork_join_complete"
    ]
    assert len(fj_events) == 1

    fj = fj_events[0]
    assert set(fj["children"]) == {"health_checker", "code_reviewer", "adversarial_tester"}
    assert fj["total_duration_ms"] > 0


async def test_per_child_timing_recorded(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Each child_completed event has duration_ms > 0."""
    graph = _build_parallel_graph().build()

    await graph.run(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    )

    child_events = [
        e for e in factory_state.events if e["action"] == "child_completed"
    ]
    assert len(child_events) == 3

    for event in child_events:
        assert event["duration_ms"] > 0
        assert event["child"] in {"health_checker", "code_reviewer", "adversarial_tester"}


async def test_mermaid_shows_fork_join_as_single_node() -> None:
    """Mermaid renders ForkJoinNode as one node (not visual fan-out)."""
    graph = _build_parallel_graph().build()
    mermaid = graph.render()

    assert "MockBuilderNode" in mermaid
    assert "DeepQAForkJoinNode" in mermaid
    assert "QAResultGateNode" in mermaid

    assert "MockBuilderNode --> DeepQAForkJoinNode" in mermaid
    assert "DeepQAForkJoinNode --> QAResultGateNode" in mermaid
    assert "QAResultGateNode --> [*]" in mermaid


async def test_graph_iter_yields_correct_event_count(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Graph.iter() yields one event per node step plus EndMarker."""
    graph = _build_parallel_graph().build()

    events: list[object] = []
    async with graph.iter(
        state=factory_state, deps=factory_deps, inputs=MockBuilderNode()
    ) as run:
        async for event in run:
            events.append(event)

    assert isinstance(events[-1], EndMarker)
    # 3 nodes (MockBuilder, DeepQAForkJoin, QAResultGate) + EndMarker = 4
    assert len(events) == 4


async def test_child_failure_does_not_crash_fork_join(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """A failing child records the error without crashing the fork/join."""

    async def _failing_agent(
        ctx: GraphRunContext[FactoryState, FactoryDeps],
    ) -> str:
        raise RuntimeError("agent crashed")

    children = [
        ChildAgent(name="good_agent", fn=_health_checker),
        ChildAgent(name="bad_agent", fn=_failing_agent),
    ]

    class FailTestForkJoinNode(ForkJoinNode):
        async def run(
            self, ctx: GraphRunContext[FactoryState, FactoryDeps]
        ) -> End[HaltResult]:
            await self.execute_children(ctx)
            return End(HaltResult(reason="done_with_failures"))

    builder: GraphBuilder[FactoryState, FactoryDeps, FailTestForkJoinNode, HaltResult] = (
        GraphBuilder(
            name="fail-test",
            state_type=FactoryState,
            deps_type=FactoryDeps,
            input_type=FailTestForkJoinNode,
            output_type=HaltResult,
        )
    )
    builder.add_edge(builder.start_node, FailTestForkJoinNode)
    builder.add(builder.node(FailTestForkJoinNode))
    graph = builder.build()

    result = await graph.run(
        state=factory_state,
        deps=factory_deps,
        inputs=FailTestForkJoinNode(children=children),
    )

    assert isinstance(result, HaltResult)
    assert result.reason == "done_with_failures"

    assert factory_state.node_outputs["good_agent"] == "health_check: all passing"
    assert "bad_agent" not in factory_state.node_outputs

    failed_events = [
        e for e in factory_state.events if e["action"] == "child_failed"
    ]
    assert len(failed_events) == 1
    assert failed_events[0]["child"] == "bad_agent"
    assert "agent crashed" in failed_events[0]["error"]
