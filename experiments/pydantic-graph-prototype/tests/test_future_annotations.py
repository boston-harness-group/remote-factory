"""Test that `from __future__ import annotations` works with pydantic-graph's
runtime type-hint introspection for edge inference.

This is identified as the #1 risk to the prototype (research-similar).
If this fails, the entire migration approach changes.
"""

from pydantic_graph import GraphBuilder

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult
from tests._future_annotations_nodes import AlphaNode, BetaNode, GammaNode


def _build_future_graph() -> GraphBuilder[FactoryState, FactoryDeps, AlphaNode, HaltResult]:
    builder: GraphBuilder[FactoryState, FactoryDeps, AlphaNode, HaltResult] = GraphBuilder(
        name="future-annotations-test",
        state_type=FactoryState,
        deps_type=FactoryDeps,
        input_type=AlphaNode,
        output_type=HaltResult,
    )
    builder.add_edge(builder.start_node, AlphaNode)
    builder.add(builder.node(AlphaNode))
    builder.add(builder.node(BetaNode))
    builder.add(builder.node(GammaNode))
    return builder


async def test_future_annotations_graph_builds() -> None:
    """Graph() should correctly infer edges even when annotations are stringified."""
    graph = _build_future_graph().build()
    assert graph is not None


async def test_future_annotations_graph_runs(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """Full execution should work: Alpha -> Beta -> Gamma -> End."""
    graph = _build_future_graph().build()
    result = await graph.run(state=factory_state, deps=factory_deps, inputs=AlphaNode())

    assert isinstance(result, HaltResult)
    assert result.reason == "chain_complete"
    assert factory_state.node_outputs["AlphaNode"] == "alpha_done"
    assert factory_state.node_outputs["BetaNode"] == "beta_done"
    assert factory_state.node_outputs["GammaNode"] == "gamma_done"


async def test_future_annotations_mermaid_has_all_nodes() -> None:
    """Mermaid rendering should include all 3 node names despite stringified annotations."""
    graph = _build_future_graph().build()
    mermaid = graph.render()
    assert "AlphaNode" in mermaid
    assert "BetaNode" in mermaid
    assert "GammaNode" in mermaid


async def test_future_annotations_edge_topology() -> None:
    """Verify the inferred edge topology matches: Alpha -> Beta -> Gamma -> End."""
    graph = _build_future_graph().build()
    mermaid = graph.render()
    assert "AlphaNode --> BetaNode" in mermaid
    assert "BetaNode --> GammaNode" in mermaid
    assert "GammaNode --> [*]" in mermaid
