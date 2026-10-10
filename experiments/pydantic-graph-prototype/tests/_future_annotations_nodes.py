"""Module that uses `from __future__ import annotations` with pydantic-graph BaseNode subclasses.

This is the critical risk test: pydantic-graph uses runtime type-hint introspection
to infer graph edges from return types. `from __future__ import annotations` turns all
annotations into strings (PEP 563), which could break this introspection.

This module must be a separate file (not inline in the test) because `from __future__`
only applies at the module level.
"""

from __future__ import annotations

from pydantic_graph import BaseNode, End, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


class AlphaNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> BetaNode:
        ctx.state.node_outputs["AlphaNode"] = "alpha_done"
        return BetaNode()


class BetaNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> GammaNode:
        ctx.state.node_outputs["BetaNode"] = "beta_done"
        return GammaNode()


class GammaNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        ctx.state.node_outputs["GammaNode"] = "gamma_done"
        return End(HaltResult(reason="chain_complete"))
