"""Parallel fork/join node — wraps asyncio.gather inside a single BaseNode.

Ports the deep-QA fork/join pattern from executor.py's _execute_fork method:
    fork_qa → [health_checker, code_reviewer, adversarial_tester] → join_qa

Structural difference from the original:
    executor.py uses ForkNode + JoinNode as separate graph primitives with the
    executor dispatching branches via asyncio.gather. In pydantic-graph, parallel
    execution is encapsulated inside a single ForkJoinNode's run() method — the
    graph sees one node, while internally asyncio.gather runs all children
    concurrently. This trades Mermaid visual fan-out for composition simplicity:
    the Mermaid diagram shows ForkJoinNode as a single node rather than a
    fork → branches → join fan-out. A custom Mermaid post-processor could expand
    it, but for the prototype the single-node representation is acceptable since
    the internal parallelism is observable through state.events timing data.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_graph import BaseNode, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult


@dataclass
class ChildAgent:
    """A simulated agent invocation to be run as one branch of a fork/join."""

    name: str
    fn: Callable[[GraphRunContext[FactoryState, FactoryDeps]], Awaitable[str]]


class ForkJoinNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Base node that runs multiple child agents concurrently via asyncio.gather.

    Subclasses implement run() with a concrete return type to declare the
    next node in the graph. The run() body calls execute_children() then
    returns the successor.
    """

    def __init__(self, children: list[ChildAgent], node_id: str = "") -> None:
        self.children = children
        self.node_id = node_id or self.__class__.__name__

    async def execute_children(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> dict[str, dict[str, Any]]:
        """Run all children concurrently via asyncio.gather.

        Records per-child outputs to ctx.state.node_outputs and timing
        events to ctx.state.events. Returns per-child result dicts.
        """
        results: dict[str, dict[str, Any]] = {}

        async def run_child(child: ChildAgent) -> None:
            start = time.monotonic()
            try:
                output = await child.fn(ctx)
                elapsed_ms = (time.monotonic() - start) * 1000
                ctx.state.node_outputs[child.name] = output
                ctx.state.events.append(
                    {
                        "node": self.node_id,
                        "action": "child_completed",
                        "child": child.name,
                        "duration_ms": round(elapsed_ms, 2),
                    }
                )
                results[child.name] = {
                    "output": output,
                    "duration_ms": elapsed_ms,
                    "success": True,
                }
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                ctx.state.events.append(
                    {
                        "node": self.node_id,
                        "action": "child_failed",
                        "child": child.name,
                        "error": str(exc),
                        "duration_ms": round(elapsed_ms, 2),
                    }
                )
                results[child.name] = {
                    "output": "",
                    "duration_ms": elapsed_ms,
                    "success": False,
                    "error": str(exc),
                }

        fork_start = time.monotonic()
        await asyncio.gather(*(run_child(c) for c in self.children))
        total_ms = (time.monotonic() - fork_start) * 1000

        ctx.state.events.append(
            {
                "node": self.node_id,
                "action": "fork_join_complete",
                "children": [c.name for c in self.children],
                "total_duration_ms": round(total_ms, 2),
            }
        )

        return results
