"""Smoke test: import pydantic-graph, build a trivial 2-node graph, run it, assert End reached."""

from pydantic_graph import BaseNode, End, EndMarker, GraphBuilder, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


class StepOne(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> "StepTwo":
        ctx.state.node_outputs["StepOne"] = "executed"
        ctx.state.events.append({"node": "StepOne", "action": "run"})
        return StepTwo()


class StepTwo(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        ctx.state.node_outputs["StepTwo"] = "executed"
        ctx.state.events.append({"node": "StepTwo", "action": "run"})
        return End(HaltResult(reason="complete"))


def _build_graph() -> GraphBuilder[FactoryState, FactoryDeps, StepOne, HaltResult]:
    builder: GraphBuilder[FactoryState, FactoryDeps, StepOne, HaltResult] = GraphBuilder(
        name="smoke-test",
        state_type=FactoryState,
        deps_type=FactoryDeps,
        input_type=StepOne,
        output_type=HaltResult,
    )
    builder.add_edge(builder.start_node, StepOne)
    builder.add(builder.node(StepOne))
    builder.add(builder.node(StepTwo))
    return builder


async def test_two_node_graph_runs_to_end(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    graph = _build_graph().build()
    result = await graph.run(state=factory_state, deps=factory_deps, inputs=StepOne())

    assert isinstance(result, HaltResult)
    assert result.reason == "complete"
    assert factory_state.node_outputs["StepOne"] == "executed"
    assert factory_state.node_outputs["StepTwo"] == "executed"
    assert len(factory_state.events) == 2


async def test_graph_iter_yields_events(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    graph = _build_graph().build()
    events: list[object] = []
    async with graph.iter(state=factory_state, deps=factory_deps, inputs=StepOne()) as run:
        async for event in run:
            events.append(event)

    assert len(events) == 3
    assert isinstance(events[-1], EndMarker)


async def test_mermaid_rendering() -> None:
    graph = _build_graph().build()
    mermaid = graph.render()
    assert "StepOne" in mermaid
    assert "StepTwo" in mermaid
    assert "[*]" in mermaid
