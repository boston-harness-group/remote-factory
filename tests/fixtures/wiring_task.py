"""WiringTask — Task subclass with 6 instances for E2E outer-loop wiring tests.

Instances:
  i1: split=train, setup OK, verify score=0.85
  i2: split=train, setup OK, verify score=0.72
  i3: split=train, setup FAILS (RuntimeError)
  i4: split=train, setup OK, verify score=0.0 (failed)
  i5: split=val,   setup OK, verify score=0.90
  i6: split=val,   setup OK, verify score=0.95

Usage:
    from tests.fixtures.wiring_task import WiringTask
    task = WiringTask("/path/to/project")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

from factory.task import (
    InstancesConfig,
    Task,
    TaskDefinition,
    TaskInstance,
    VerifyResult,
)

_INSTANCES = [
    TaskInstance(id="i1", split="train"),
    TaskInstance(id="i2", split="train"),
    TaskInstance(id="i3", split="train"),
    TaskInstance(id="i4", split="train"),
    TaskInstance(id="i5", split="val"),
    TaskInstance(id="i6", split="val"),
]

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
    """Deterministic task for outer-loop E2E wiring tests."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        defn = TaskDefinition(
            name="wiring-task",
            instances_config=InstancesConfig(
                holdout_ids=["i5", "i6"],
            ),
        )
        super().__init__(definition=defn)
        self._project_dir = Path(project_dir) if project_dir else None

    # ── Four hooks ───────────────────────────────────────────────

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[TaskInstance]:
        for inst in _INSTANCES:
            if split == "all" or inst.split == split:
                yield inst

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
