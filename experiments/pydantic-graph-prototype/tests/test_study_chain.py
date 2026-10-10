"""Tests for the study chain: execution order, state mutations, Mermaid output."""

from pathlib import Path

from pydantic_graph import EndMarker

from pg_factory.deps import FactoryDeps
from pg_factory.graphs.study import build_study_graph
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


async def test_study_chain_execution_order(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Nodes execute in the correct linear order."""
    graph = build_study_graph().build()
    from pg_factory.nodes.study import GraphUpdateNode

    result = await graph.run(
        state=factory_state, deps=factory_deps, inputs=GraphUpdateNode()
    )

    assert isinstance(result, HaltResult)
    assert result.reason == "study_complete"

    expected_order = [
        "GraphUpdateNode",
        "StudyNode",
        "GraphExplorerNode",
        "ConcatStudyNode",
    ]
    actual_order = [e["node"] for e in factory_state.events]
    assert actual_order == expected_order


async def test_study_chain_state_mutations(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Each node records its output to ctx.state.node_outputs."""
    graph = build_study_graph().build()
    from pg_factory.nodes.study import GraphUpdateNode

    await graph.run(state=factory_state, deps=factory_deps, inputs=GraphUpdateNode())

    assert "GraphUpdateNode" in factory_state.node_outputs
    assert "StudyNode" in factory_state.node_outputs
    assert "GraphExplorerNode" in factory_state.node_outputs
    assert "ConcatStudyNode" in factory_state.node_outputs


async def test_study_chain_dry_run_creates_files(
    factory_state: FactoryState,
    tmp_path: Path,
) -> None:
    """In dry_run mode, nodes create mock output files."""
    deps = FactoryDeps(project_path=tmp_path, dry_run=True)
    graph = build_study_graph().build()
    from pg_factory.nodes.study import GraphUpdateNode

    await graph.run(state=factory_state, deps=deps, inputs=GraphUpdateNode())

    strategy_dir = tmp_path / ".factory" / "strategy"
    assert (strategy_dir / "observations.md").exists()
    assert (strategy_dir / "graph-context.md").exists()
    assert (strategy_dir / "study-combined.md").exists()

    combined = (strategy_dir / "study-combined.md").read_text()
    assert "Observations" in combined
    assert "Graph Context" in combined


async def test_study_chain_iter_yields_events(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Graph.iter() yields one event per node step plus EndMarker."""
    graph = build_study_graph().build()
    from pg_factory.nodes.study import GraphUpdateNode

    events: list[object] = []
    async with graph.iter(
        state=factory_state, deps=factory_deps, inputs=GraphUpdateNode()
    ) as run:
        async for event in run:
            events.append(event)

    assert len(events) == 5
    assert isinstance(events[-1], EndMarker)


async def test_study_chain_mermaid_contains_all_nodes() -> None:
    """Mermaid output contains all 4 node names."""
    graph = build_study_graph().build()
    mermaid = graph.render()

    assert "GraphUpdateNode" in mermaid
    assert "StudyNode" in mermaid
    assert "GraphExplorerNode" in mermaid
    assert "ConcatStudyNode" in mermaid


async def test_study_chain_mermaid_topology() -> None:
    """Mermaid output shows the correct linear edge topology."""
    graph = build_study_graph().build()
    mermaid = graph.render()

    assert "GraphUpdateNode --> StudyNode" in mermaid
    assert "StudyNode --> GraphExplorerNode" in mermaid
    assert "GraphExplorerNode --> ConcatStudyNode" in mermaid
    assert "ConcatStudyNode --> [*]" in mermaid


async def test_study_chain_events_have_action_field(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Each event recorded by nodes has a distinct action field."""
    graph = build_study_graph().build()
    from pg_factory.nodes.study import GraphUpdateNode

    await graph.run(state=factory_state, deps=factory_deps, inputs=GraphUpdateNode())

    actions = [e["action"] for e in factory_state.events]
    assert actions == ["graph_update", "study", "explore", "concat"]
