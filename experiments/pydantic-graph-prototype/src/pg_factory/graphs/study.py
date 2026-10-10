"""Study chain graph assembly.

Assembles the 4-node study chain into a pydantic-graph Graph:
    GraphUpdateNode → StudyNode → GraphExplorerNode → ConcatStudyNode → End
"""

from pydantic_graph import GraphBuilder

from pg_factory.deps import FactoryDeps
from pg_factory.nodes.study import (
    ConcatStudyNode,
    GraphExplorerNode,
    GraphUpdateNode,
    StudyNode,
)
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


def build_study_graph() -> GraphBuilder[FactoryState, FactoryDeps, GraphUpdateNode, HaltResult]:
    builder: GraphBuilder[FactoryState, FactoryDeps, GraphUpdateNode, HaltResult] = GraphBuilder(
        name="study-chain",
        state_type=FactoryState,
        deps_type=FactoryDeps,
        input_type=GraphUpdateNode,
        output_type=HaltResult,
    )
    builder.add_edge(builder.start_node, GraphUpdateNode)
    builder.add(builder.node(GraphUpdateNode))
    builder.add(builder.node(StudyNode))
    builder.add(builder.node(GraphExplorerNode))
    builder.add(builder.node(ConcatStudyNode))
    return builder
