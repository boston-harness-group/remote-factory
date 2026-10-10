"""Gate/verdict routing nodes — port of GateNode verdict routing from primitives.py + executor.py.

GateBaseNode provides the core gate pattern:
    - Configurable verdict function determines PROCEED / RELOOP / HALT
    - Iteration counting: tracks (gate_id, target_id) counts in FactoryState
    - Feedback injection: writes to node_feedback so reloop targets can read context
    - Max-iteration halt: returns End(HaltResult) when count exceeds max_iterations

Structural difference from the original:
    executor.py's _execute_gate uses runtime edge matching — it looks up the next
    node by string ID from the workflow's edge list based on VerdictType. In
    pydantic-graph, the gate's run() return type union declares all possible
    successors at definition time, and the graph enforces valid routing via the
    type system. Invalid edges become type errors, not runtime crashes.
"""

from typing import Callable

from pydantic_graph import BaseNode, End, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult, VerdictType


class GateBaseNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Abstract base for gate nodes with iteration counting and feedback injection.

    Subclasses define run() with a concrete return union of their
    proceed/reloop target node types plus End[HaltResult].
    """

    def __init__(
        self,
        gate_id: str = "",
        max_iterations: int = 3,
        verdict_fn: Callable[[FactoryState], VerdictType] | None = None,
    ) -> None:
        self.gate_id = gate_id
        self.max_iterations = max_iterations
        self.verdict_fn = verdict_fn

    def evaluate_verdict(self, state: FactoryState) -> VerdictType:
        if self.verdict_fn:
            return self.verdict_fn(state)
        return VerdictType.PROCEED

    def check_and_increment_iteration(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps], target_id: str
    ) -> End[HaltResult] | None:
        """Increment iteration count; return End if max_iterations exceeded."""
        key = (self.gate_id, target_id)
        count = ctx.state.iteration_counts.get(key, 0) + 1
        ctx.state.iteration_counts[key] = count
        if count > self.max_iterations:
            return End(
                HaltResult(
                    reason=f"max_iterations ({self.max_iterations}) exceeded "
                    f"for {self.gate_id} -> {target_id}"
                )
            )
        return None

    def inject_feedback(
        self,
        ctx: GraphRunContext[FactoryState, FactoryDeps],
        target_id: str,
        feedback: str,
    ) -> None:
        ctx.state.node_feedback[target_id] = feedback

    def record_verdict_event(
        self,
        ctx: GraphRunContext[FactoryState, FactoryDeps],
        verdict: VerdictType,
        target: str | None = None,
    ) -> None:
        ctx.state.events.append(
            {
                "node": self.gate_id or self.__class__.__name__,
                "action": "gate_verdict",
                "verdict": verdict.value,
                "target": target,
            }
        )
