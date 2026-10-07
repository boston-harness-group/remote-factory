"""WiringTask — Task subclass with 6 instances for E2E outer-loop wiring tests.

Instances (splits assigned via holdout_ids, NOT preset on TaskInstance):
  i1: train (not in holdout_ids), setup OK, verify score=0.85
  i2: train (not in holdout_ids), setup OK, verify score=0.72
  i3: train (not in holdout_ids), setup FAILS (RuntimeError)
  i4: train (not in holdout_ids), setup OK, verify score=0.0 (failed)
  i5: val   (in holdout_ids),     setup OK, verify score=0.90
  i6: val   (in holdout_ids),     setup OK, verify score=0.95

Split assignment is done by the base Task._assign_splits() method using
holdout_ids from InstancesConfig.  This tests the real TOML parsing path
where holdout_ids drives the train/val partition (no preset split field).

Usage:
    from tests.fixtures.wiring_task import WiringTask
    task = WiringTask()
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from factory.task import (
    InstancesConfig,
    Task,
    TaskDefinition,
    TaskInstance,
    VerifyResult,
)

# Instances WITHOUT preset splits — _assign_splits() uses holdout_ids
_INSTANCE_IDS = ["i1", "i2", "i3", "i4", "i5", "i6"]

_SCORES: dict[str, float] = {
    "i1": 0.85,
    "i2": 0.72,
    "i3": 0.0,   # never reached (setup fails)
    "i4": 0.0,   # verify returns 0.0 — failed
    "i5": 0.90,
    "i6": 0.95,
}

_SETUP_FAIL_IDS = {"i3"}


class WiringTask(Task):
    """Deterministic task for outer-loop E2E wiring tests.

    Instances have NO preset split field.  The base class ``_assign_splits()``
    reads ``holdout_ids`` from ``InstancesConfig`` and assigns:

    - IDs in holdout_ids → split="val"
    - all others         → split="train"

    This exercises the real holdout_ids code-path that TOML-parsed task
    definitions use.
    """

    def __init__(self, project_dir: str | Path | None = None) -> None:
        defn = TaskDefinition(
            name="wiring-task",
            instances_config=InstancesConfig(
                holdout_ids=["i5", "i6"],
            ),
        )
        super().__init__(definition=defn)
        self._project_dir = Path(project_dir) if project_dir else None

    # ── Override _raw_instances (not instances) ──────────────────

    def _raw_instances(self) -> Iterator[TaskInstance]:
        """Yield instances WITHOUT split — _assign_splits() handles that."""
        for iid in _INSTANCE_IDS:
            yield TaskInstance(id=iid)

    # ── setup / prompt / verify hooks ────────────────────────────

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        if instance.id in _SETUP_FAIL_IDS:
            raise RuntimeError("rigged failure")

    def prompt(self, instance: TaskInstance) -> str:
        return f"Solve {instance.id}"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        score = _SCORES.get(instance.id, 0.0)
        return VerifyResult(passed=score > 0.0, score=score)

    def get_evaluator(self) -> None:
        return None
