from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _noop_emitter(event: dict[str, Any]) -> None:
    pass


@dataclass
class FactoryDeps:
    """Immutable dependencies injected into graph nodes via GraphRunContext.deps.

    Maps from the current executor's constructor args:
    - project_path: root path of the project being operated on
    - dry_run: when True, nodes simulate execution without side effects
    - event_emitter: callback for structured event emission
    """

    project_path: Path = field(default_factory=lambda: Path("."))
    dry_run: bool = False
    event_emitter: Callable[[dict[str, Any]], None] = field(default=_noop_emitter)
