"""CompressOuterLoop — multi-cycle optimizer wrapping CompressInnerLoop."""

from __future__ import annotations

from typing import Any

import structlog

from factory.compress.inner_loop import CompressInnerLoop
from factory.compress.mutator import WorkflowMutator
from factory.cycle_analyzer import CycleRecord
from factory.models import OuterLoopConfig
from factory.strategy import detect_research_plateau

log = structlog.get_logger()


class CompressOuterLoop:
    """Outer loop that drives CompressInnerLoop cycles with plateau detection and mutation.

    Public API:
        converged()           — check if optimization has converged
        generate_directives() — produce steering directives for the inner loop
        step()                — run one outer-loop iteration
        best_overall_technique() — best technique across all cycles
        run()                 — run the full outer loop until convergence
    """

    def __init__(
        self,
        inner: CompressInnerLoop,
        config: OuterLoopConfig,
        *,
        target_score: float = 0.9,
        plateau_threshold: int = 3,
        mutator: WorkflowMutator | None = None,
    ) -> None:
        self.inner = inner
        self.config = config
        self.target_score = target_score
        self.plateau_threshold = plateau_threshold
        self.mutator = mutator
        self._cycle_count = 0
        self._plateau_count = 0
        self._plateau_detected = False
        self.reason: str | None = None

    def converged(self) -> bool:
        """Check if the outer loop should stop."""
        max_cycles = self.config.max_outer_cycles
        if max_cycles is not None and self._cycle_count >= max_cycles:
            self.reason = "max_cycles_reached"
            return True

        history = self.inner.history()
        if not history:
            return False

        latest = history[-1]
        if latest.score_end is not None and latest.score_end >= self.target_score:
            self.reason = "target_reached"
            return True

        summaries = [
            {"metric_value": r.score_end}
            for r in history
            if r.score_end is not None
        ]
        self._plateau_detected = detect_research_plateau(
            summaries, threshold=self.plateau_threshold
        )
        if self._plateau_detected:
            self._plateau_count += 1
            if self.mutator and self._plateau_count < 3:
                log.info(
                    "plateau_detected_will_mutate",
                    plateau_count=self._plateau_count,
                )
                return False
            self.reason = "plateau"
            return True

        return False

    def generate_directives(self, plateau_detected: bool = False) -> dict[str, Any]:
        """Produce steering directives for the next inner-loop cycle."""
        directives: dict[str, Any] = {
            "outer_cycle": self._cycle_count + 1,
            "target_score": self.target_score,
        }

        history = self.inner.history()
        if history:
            latest = history[-1]
            if latest.score_end is not None:
                directives["current_score"] = latest.score_end
                directives["gap"] = self.target_score - latest.score_end

        best = self.inner.best_technique()
        if best:
            directives["best_technique"] = best

        trajectory = self.inner.compression_trajectory()
        if trajectory:
            directives["trajectory_summary"] = trajectory[-3:]

        if plateau_detected:
            directives["plateau_detected"] = True
            directives["guidance"] = "Try a fundamentally different compression approach"

        return directives

    def step(self) -> CycleRecord:
        """Run one outer-loop iteration: generate directives, run inner loop, check plateau."""
        plateau_detected = self._plateau_detected

        if plateau_detected and self.mutator and self._plateau_count >= 2 and self.inner.workflow:
            log.info("triggering_workflow_mutation", plateau_count=self._plateau_count)
            self.inner.workflow = self.mutator.mutate(
                self.inner.workflow,
                self.inner.history(),
            )
            self._plateau_count = 0

        directives = self.generate_directives(plateau_detected=plateau_detected)
        record = self.inner.step(directives=directives)
        self._cycle_count += 1

        log.info(
            "outer_loop_step",
            cycle=self._cycle_count,
            score=record.score_end,
            delta=record.score_delta,
        )
        return record

    def best_overall_technique(self) -> str | None:
        """Return the best technique across all inner-loop cycles."""
        return self.inner.best_technique()

    def run(self) -> list[CycleRecord]:
        """Run the full outer loop until convergence."""
        records: list[CycleRecord] = []

        while not self.converged():
            record = self.step()
            records.append(record)

        log.info(
            "outer_loop_finished",
            total_cycles=self._cycle_count,
            reason=self.reason,
        )

        return records

    def _summarize(self) -> dict[str, Any]:
        """Summarize the outer loop run."""
        history = self.inner.history()
        return {
            "total_cycles": self._cycle_count,
            "reason": self.reason,
            "best_technique": self.best_overall_technique(),
            "final_score": history[-1].score_end if history else None,
            "total_cost": self.inner.total_cost(),
            "trajectory": self.inner.score_trajectory(),
        }
