from dataclasses import dataclass
from enum import Enum


class VerdictType(Enum):
    """Maps from the current engine's VerdictType enum."""

    PROCEED = "proceed"
    RELOOP = "reloop"
    HALT = "halt"


@dataclass(frozen=True)
class HaltResult:
    """Terminal result type for End[HaltResult] returns in pydantic-graph.

    When a node or gate determines execution should stop, it returns
    End(HaltResult(reason="...")) to terminate the graph run.
    """

    reason: str
    verdict: VerdictType = VerdictType.HALT
