"""Which runtimes this machine actually uses.

`ls` shows one table covering both targets, which is right for someone who uses both and wrong for
everyone else: reaching a cluster costs a network round trip, and an unreachable one costs a
multi-second timeout followed by an error about a target the user never asked for. Somebody who
answered "local" at setup should not be told their cluster is down.

So the cluster is only consulted when there is a reason to think it is wanted: the user set it up,
has run something on it, or asked for it now with `--target k8s`. The record is a plain list of
target names, written when a target is set up or provisioned.
"""

from __future__ import annotations

import json

import structlog

from factory.contained.workspace import contained_home

log = structlog.get_logger()

TARGETS = ("local", "k8s", "openshell")


def _record_path():
    return contained_home() / "targets.json"


def record_target(target: str) -> None:
    """Note that this machine uses `target`. Idempotent, and never fatal."""
    if target not in TARGETS:
        return
    used = set(used_targets())
    if target in used:
        return
    used.add(target)
    path = _record_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(used)))
    except OSError as exc:
        # A machine whose home directory is read-only still has to be able to run; the only cost of
        # failing here is that `ls` asks about one target more than it needs to.
        log.debug("contained_usage_not_recorded", error=str(exc))


def used_targets() -> list[str]:
    """The targets this machine has set up or provisioned, oldest record first."""
    try:
        data = json.loads(_record_path().read_text())
    except (OSError, ValueError):
        return []
    return [t for t in data if t in TARGETS] if isinstance(data, list) else []


def uses(target: str) -> bool:
    return target in used_targets()
