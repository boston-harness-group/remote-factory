"""Tests for OverfitDetector."""

from __future__ import annotations

from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.overfit import OverfitDetector
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    Workflow,
)


def _make_config() -> SwarmConfig:
    return SwarmConfig(
        benchmark="test",
        budget=50,
        training_instances=["t1", "t2"],
        holdout_instances=["h1"],
    )


def _make_workflow() -> Workflow:
    return Workflow(
        name="test",
        nodes={
            "a": FnNode(id="a", command="echo a"),
            "b": AgentNode(
                id="b",
                role=AgentRole.BUILDER,
                prompt_template="Build the project at {project_path}.",
            ),
        },
        edges=[Edge(source="a", target="b")],
        start_node="a",
    )


class TestOverfitDetector:
    def test_no_overfit(self) -> None:
        config = _make_config()
        scores = {"t1": 0.8, "t2": 0.8, "h1": 0.75}

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            avg = sum(scores.get(i, 0.0) for i in instances) / max(len(instances), 1)
            return EvalResult(score=avg, benchmark_score=avg, hygiene_score=0.8)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        detector = OverfitDetector(threshold=0.15)

        wf = _make_workflow()
        result = detector.audit(wf, ["t1", "t2"], ["h1"], evaluator, "/tmp")

        assert not result.overfit_flag
        assert result.training_score > 0
        assert result.holdout_score > 0
        assert result.delta < 0.15

    def test_overfit_detected(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            if "h1" in instances:
                return EvalResult(score=0.5, benchmark_score=0.5, hygiene_score=0.5)
            return EvalResult(score=0.9, benchmark_score=0.9, hygiene_score=0.9)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        detector = OverfitDetector(threshold=0.15)

        wf = _make_workflow()
        result = detector.audit(wf, ["t1", "t2"], ["h1"], evaluator, "/tmp")

        assert result.overfit_flag
        assert result.delta > 0.15

    def test_equal_scores(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.7, benchmark_score=0.7, hygiene_score=0.7)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        detector = OverfitDetector()

        wf = _make_workflow()
        result = detector.audit(wf, ["t1"], ["h1"], evaluator, "/tmp")

        assert not result.overfit_flag
        assert result.delta == 0.0

    def test_zero_training_score(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.0, benchmark_score=0.0)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        detector = OverfitDetector()

        wf = _make_workflow()
        result = detector.audit(wf, ["t1"], ["h1"], evaluator, "/tmp")

        assert not result.overfit_flag
        assert result.delta == 0.0

    def test_custom_threshold(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            if "h1" in instances:
                # Composite: 0.6*0.9 + 0.2*1.0 + 0.1 + 0.1 = 0.94
                return EvalResult(score=0.0, benchmark_score=0.9, hygiene_score=1.0)
            # Composite: 0.6*1.0 + 0.2*1.0 + 0.1 + 0.1 = 1.0
            return EvalResult(score=0.0, benchmark_score=1.0, hygiene_score=1.0)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        # Delta = (1.0 - 0.94) / 1.0 = 0.06 → passes at 0.15, fails at 0.05
        detector_strict = OverfitDetector(threshold=0.05)
        detector_loose = OverfitDetector(threshold=0.15)

        wf = _make_workflow()
        strict_result = detector_strict.audit(wf, ["t1"], ["h1"], evaluator, "/tmp")
        loose_result = detector_loose.audit(wf, ["t1"], ["h1"], evaluator, "/tmp")

        assert strict_result.overfit_flag
        assert not loose_result.overfit_flag

    def test_details_populated(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.8, benchmark_score=0.8)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        detector = OverfitDetector()
        wf = _make_workflow()
        result = detector.audit(wf, ["t1"], ["h1"], evaluator, "/tmp")

        assert "training=" in result.details
        assert "holdout=" in result.details
        assert "delta=" in result.details
