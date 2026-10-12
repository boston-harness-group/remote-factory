"""Overfit / cheating detection for evolved workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from factory.outer_loop.models import AuditResult

if TYPE_CHECKING:
    from factory.outer_loop.evaluator import SwarmEvaluator
    from factory.workflow.primitives import Workflow

log = structlog.get_logger()

OVERFIT_THRESHOLD = 0.15


class OverfitDetector:
    """Detects overfitting by comparing training vs holdout scores."""

    def __init__(self, threshold: float = OVERFIT_THRESHOLD) -> None:
        self._threshold = threshold

    def audit(
        self,
        best_workflow: Workflow,
        training_instances: list[str],
        holdout_instances: list[str],
        evaluator: SwarmEvaluator | None = None,
        project_dir: str = "",
        training_score: float | None = None,
        holdout_score: float | None = None,
    ) -> AuditResult:
        """Run the best workflow on both training and holdout instances.

        Flags overfit if (training - holdout) / training > threshold.

        When training_score / holdout_score are provided, skips the
        corresponding evaluation and uses the provided scores directly.
        """
        if training_score is None:
            assert evaluator is not None, "evaluator required when training_score not provided"
            train_result = evaluator.evaluate(best_workflow, project_dir, training_instances)
            training_score = train_result.score

        if holdout_score is None:
            assert evaluator is not None, "evaluator required when holdout_score not provided"
            holdout_result = evaluator.evaluate(best_workflow, project_dir, holdout_instances)
            holdout_score = holdout_result.score

        if training_score > 0:
            delta = (training_score - holdout_score) / training_score
        else:
            delta = 0.0

        overfit_flag = delta > self._threshold

        if overfit_flag:
            log.warning(
                "overfit_detected",
                training_score=training_score,
                holdout_score=holdout_score,
                delta=delta,
                threshold=self._threshold,
            )
        else:
            log.info(
                "overfit_audit_passed",
                training_score=training_score,
                holdout_score=holdout_score,
                delta=delta,
            )

        details = (
            f"training={training_score:.4f} holdout={holdout_score:.4f} "
            f"delta={delta:.4f} threshold={self._threshold}"
        )

        return AuditResult(
            training_score=training_score,
            holdout_score=holdout_score,
            delta=delta,
            overfit_flag=overfit_flag,
            details=details,
        )
