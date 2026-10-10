from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactoryState:
    """Mutable state threaded through a pydantic-graph workflow via GraphRunContext.state.

    Maps from the current executor's mutable fields:
    - iteration_counts: tracks gate reloop counts per (gate_id, target_id) pair
    - node_feedback: carries feedback from gate verdicts to reloop target nodes
    - node_outputs: stores each node's output keyed by node class name
    - completed_files: tracks which logical file artifacts have been produced
    - events: append-only event log for observability
    """

    iteration_counts: dict[tuple[str, str], int] = field(
        default_factory=lambda: dict[tuple[str, str], int]()
    )
    node_feedback: dict[str, str] = field(
        default_factory=lambda: dict[str, str]()
    )
    node_outputs: dict[str, str] = field(
        default_factory=lambda: dict[str, str]()
    )
    completed_files: set[str] = field(default_factory=lambda: set[str]())
    events: list[dict[str, Any]] = field(
        default_factory=lambda: list[dict[str, Any]]()
    )
