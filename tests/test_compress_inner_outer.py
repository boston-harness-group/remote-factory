"""Tests for factory/compress/ — evaluator, inner loop, outer loop, and mutator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.compress.evaluator import CompressEvaluator
from factory.compress.inner_loop import CompressInnerLoop
from factory.compress.mutator import WorkflowMutator
from factory.compress.outer_loop import CompressOuterLoop
from factory.cycle_analyzer import CycleRecord, ExperimentRecord
from factory.inner_loop import Evaluator, InnerLoop
from factory.models import OuterLoopConfig
from factory.workflow.primitives import AgentNode, AgentRole, Edge, GateNode, Workflow


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_artifact(directory: Path, name: str, data: dict) -> Path:
    p = directory / name
    p.write_text(json.dumps(data))
    return p


def _make_workflow() -> Workflow:
    return Workflow(
        name="test-compress",
        nodes={
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research compression techniques",
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build the solution",
            ),
            "gate_review": GateNode(
                id="gate_review",
                evaluator_type="fn",
            ),
        },
        edges=[
            Edge(source="researcher", target="builder"),
            Edge(source="builder", target="gate_review"),
        ],
        start_node="researcher",
    )


def _make_cycle_record(
    *,
    cycle_number: int = 1,
    score_end: float | None = 0.5,
    score_delta: float | None = 0.1,
    hypothesis: str = "test technique",
    verdict: str = "keep",
) -> CycleRecord:
    return CycleRecord(
        cycle_number=cycle_number,
        mode="compress",
        started_at=None,
        ended_at=None,
        duration_s=10.0,
        score_start=None,
        score_end=score_end,
        score_delta=score_delta,
        experiments=[
            ExperimentRecord(
                exp_id=cycle_number,
                hypothesis=hypothesis,
                verdict=verdict,
                score_before=None,
                score_after=score_end,
                score_delta=score_delta,
                cost_usd=1.0,
                duration_s=10.0,
            )
        ],
    )


# ── CompressEvaluator tests ───────────────────────────────────


class TestCompressEvaluator:
    def test_parse_valid_artifact(self, artifact_dir: Path) -> None:
        path = _write_artifact(artifact_dir, "result.json", {
            "compression_ratio": 0.8,
            "quality_retention": 0.95,
            "inference_latency": 0.1,
            "technique": "pruning",
        })
        evaluator = CompressEvaluator()
        result = evaluator.parse(path)
        assert result.valid
        assert result.score > 0
        expected = 0.8 * 0.4 + 0.95 * 0.5 - (1.0 - max(0.0, 1.0 - 0.1)) * 0.1
        assert abs(result.score - expected) < 1e-9

    def test_parse_missing_file(self, artifact_dir: Path) -> None:
        result = CompressEvaluator().parse(artifact_dir / "nonexistent.json")
        assert not result.valid
        assert result.score == 0.0

    def test_parse_malformed_json(self, artifact_dir: Path) -> None:
        path = artifact_dir / "bad.json"
        path.write_text("{not valid json")
        result = CompressEvaluator().parse(path)
        assert not result.valid
        assert result.score == 0.0

    def test_parse_missing_required_fields(self, artifact_dir: Path) -> None:
        path = _write_artifact(artifact_dir, "partial.json", {
            "compression_ratio": 0.8,
        })
        result = CompressEvaluator().parse(path)
        assert not result.valid
        assert result.score == 0.0

    def test_parse_many_selects_best(self, artifact_dir: Path) -> None:
        p1 = _write_artifact(artifact_dir, "r1.json", {
            "compression_ratio": 0.5,
            "quality_retention": 0.5,
            "inference_latency": 0.1,
        })
        p2 = _write_artifact(artifact_dir, "r2.json", {
            "compression_ratio": 0.9,
            "quality_retention": 0.95,
            "inference_latency": 0.05,
        })
        evaluator = CompressEvaluator()
        result = evaluator.parse_many([p1, p2])
        assert result.valid
        assert result.score == evaluator.parse(p2).score

    def test_get_info(self) -> None:
        info = CompressEvaluator().get_info()
        assert info["benchmark"] == "compression"
        assert "compression_ratio" in info["metrics"]
        assert "weights" in info

    def test_combined_score_formula(self, artifact_dir: Path) -> None:
        path = _write_artifact(artifact_dir, "score.json", {
            "compression_ratio": 1.0,
            "quality_retention": 1.0,
            "inference_latency": 0.0,
        })
        evaluator = CompressEvaluator(w_compression=0.4, w_quality=0.5, w_latency=0.1)
        result = evaluator.parse(path)
        expected = 1.0 * 0.4 + 1.0 * 0.5 - (1.0 - 1.0) * 0.1
        assert abs(result.score - expected) < 1e-9
        assert result.score == 0.9

    def test_implements_evaluator_protocol(self) -> None:
        assert isinstance(CompressEvaluator(), Evaluator)


# ── CompressInnerLoop tests ───────────────────────────────────


class TestCompressInnerLoop:
    def test_initialization_defaults(self, tmp_path: Path) -> None:
        loop = CompressInnerLoop(tmp_path)
        assert loop.mode == "compress"
        assert "gate_review" in loop.frozen_nodes
        assert "gate_precheck" in loop.frozen_nodes

    def test_subclasses_inner_loop(self, tmp_path: Path) -> None:
        loop = CompressInnerLoop(tmp_path)
        assert isinstance(loop, InnerLoop)

    def test_technique_history_empty(self, tmp_path: Path) -> None:
        loop = CompressInnerLoop(tmp_path)
        assert loop.technique_history() == []

    def test_best_technique_from_history(self, tmp_path: Path) -> None:
        loop = CompressInnerLoop(tmp_path)
        eval_dir = tmp_path / ".factory" / "experiments" / "001"
        eval_dir.mkdir(parents=True)
        artifact = eval_dir / "eval_result.json"
        artifact.write_text(json.dumps({
            "compression_ratio": 0.9,
            "quality_retention": 0.95,
            "technique": "quantization",
        }))

        record = CycleRecord(
            cycle_number=1,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=None,
            score_end=0.85,
            score_delta=0.1,
            experiments=[
                ExperimentRecord(
                    exp_id=1,
                    hypothesis="quantize model",
                    verdict="keep",
                    score_before=0.75,
                    score_after=0.85,
                    score_delta=0.1,
                    cost_usd=1.0,
                    duration_s=10.0,
                    eval_artifacts=[str(artifact)],
                )
            ],
        )
        loop._history.append(record)
        assert loop.best_technique() == "quantization"

    def test_compression_trajectory(self, tmp_path: Path) -> None:
        loop = CompressInnerLoop(tmp_path)
        loop._history.append(CycleRecord(
            cycle_number=1,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=5.0,
            score_start=None,
            score_end=0.5,
            score_delta=None,
        ))
        traj = loop.compression_trajectory()
        assert len(traj) == 1
        assert traj[0]["cycle"] == 1
        assert traj[0]["score"] == 0.5


# ── CompressOuterLoop tests ───────────────────────────────────


class TestCompressOuterLoop:
    def test_directive_generation_no_plateau(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        config = OuterLoopConfig(max_outer_cycles=10)
        outer = CompressOuterLoop(inner, config)
        directives = outer.generate_directives(plateau_detected=False)
        assert "outer_cycle" in directives
        assert "target_score" in directives
        assert "plateau_detected" not in directives

    def test_directive_generation_with_plateau(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        config = OuterLoopConfig(max_outer_cycles=10)
        outer = CompressOuterLoop(inner, config)
        directives = outer.generate_directives(plateau_detected=True)
        assert directives["plateau_detected"] is True
        assert "guidance" in directives

    def test_directive_generation_includes_history(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        inner._history.append(CycleRecord(
            cycle_number=1,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=None,
            score_end=0.7,
            score_delta=None,
        ))
        config = OuterLoopConfig(max_outer_cycles=10)
        outer = CompressOuterLoop(inner, config, target_score=0.9)
        directives = outer.generate_directives()
        assert directives["current_score"] == 0.7
        assert abs(directives["gap"] - 0.2) < 1e-9

    def test_convergence_on_target_reached(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        inner._history.append(CycleRecord(
            cycle_number=1,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=None,
            score_end=0.95,
            score_delta=None,
        ))
        config = OuterLoopConfig(max_outer_cycles=10)
        outer = CompressOuterLoop(inner, config, target_score=0.9)
        assert outer.converged()
        assert outer.reason == "target_reached"

    def test_convergence_on_max_cycles(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        config = OuterLoopConfig(max_outer_cycles=2)
        outer = CompressOuterLoop(inner, config)
        outer._cycle_count = 2
        assert outer.converged()
        assert outer.reason == "max_cycles_reached"

    def test_public_api_names(self) -> None:
        """Verify public API methods have no underscore prefix."""
        assert not CompressOuterLoop.converged.__name__.startswith("_")
        assert not CompressOuterLoop.generate_directives.__name__.startswith("_")
        assert not CompressOuterLoop.step.__name__.startswith("_")
        assert not CompressOuterLoop.best_overall_technique.__name__.startswith("_")

    def test_step_runs_one_cycle(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        config = OuterLoopConfig(max_outer_cycles=5)
        outer = CompressOuterLoop(inner, config)

        mock_record = CycleRecord(
            cycle_number=1,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=None,
            score_end=0.6,
            score_delta=None,
        )

        with patch.object(inner, "step", return_value=mock_record) as mock_step:
            record = outer.step()
            mock_step.assert_called_once()
            assert record.score_end == 0.6
            assert outer._cycle_count == 1

    def test_run_plateau_counting_with_mutator(self, tmp_path: Path) -> None:
        """Verify _plateau_count increments once per iteration in run(), not twice."""
        workflow = _make_workflow()
        inner = CompressInnerLoop(tmp_path, workflow=workflow, frozen_nodes=frozenset())
        mutator = WorkflowMutator(frozen_nodes=frozenset({"gate_review"}))
        config = OuterLoopConfig(max_outer_cycles=10)
        outer = CompressOuterLoop(
            inner, config, mutator=mutator, plateau_threshold=2, target_score=0.99,
        )

        inner._history.extend([
            _make_cycle_record(cycle_number=1, score_end=0.5, score_delta=0.1, hypothesis="good"),
            _make_cycle_record(cycle_number=2, score_end=0.5, score_delta=-0.1, hypothesis="bad"),
            _make_cycle_record(cycle_number=3, score_end=0.5, score_delta=0.0),
        ])

        step_call_count = 0
        original_prompt = inner.workflow.nodes["researcher"].prompt_template

        def fake_step(*, directives: dict | None = None) -> CycleRecord:
            nonlocal step_call_count
            step_call_count += 1
            rec = _make_cycle_record(
                cycle_number=3 + step_call_count, score_end=0.5, score_delta=0.0,
            )
            inner._history.append(rec)
            return rec

        with patch.object(inner, "step", side_effect=fake_step):
            # First iteration: plateau detected → _plateau_count goes to 1 (not 2).
            # Mutator should NOT fire (count < 2).
            assert not outer.converged()
            outer.step()
            assert outer._plateau_count == 1
            assert inner.workflow.nodes["researcher"].prompt_template == original_prompt

            # Second iteration: plateau again → _plateau_count goes to 2.
            # Mutator SHOULD fire (count >= 2), then reset to 0.
            assert not outer.converged()
            outer.step()
            assert outer._plateau_count == 0
            assert inner.workflow.nodes["researcher"].prompt_template != original_prompt

    def test_best_overall_technique_delegates(self, tmp_path: Path) -> None:
        inner = CompressInnerLoop(tmp_path)
        config = OuterLoopConfig(max_outer_cycles=5)
        outer = CompressOuterLoop(inner, config)

        with patch.object(inner, "best_technique", return_value="pruning") as mock_bt:
            assert outer.best_overall_technique() == "pruning"
            mock_bt.assert_called_once()


# ── WorkflowMutator tests ────────────────────────────────────


class TestWorkflowMutator:
    def test_classify_techniques_with_history(self) -> None:
        history = [
            _make_cycle_record(cycle_number=1, score_delta=0.15, hypothesis="pruning"),
            _make_cycle_record(cycle_number=2, score_delta=-0.05, hypothesis="distillation"),
            _make_cycle_record(cycle_number=3, score_delta=0.005, hypothesis="quantization"),
        ]
        mutator = WorkflowMutator()
        perf = mutator.classify_techniques(history)
        assert "pruning" in perf["successful"]
        assert "distillation" in perf["failed"]
        assert "quantization" in perf["plateau"]

    def test_build_prompt_amendments(self) -> None:
        mutator = WorkflowMutator()
        perf = {
            "successful": ["pruning"],
            "failed": ["distillation"],
            "plateau": [],
        }
        amendments = mutator.build_prompt_amendments(perf)
        assert "constraints" in amendments
        assert "distillation" in amendments["constraints"]
        assert "priorities" in amendments
        assert "pruning" in amendments["priorities"]

    def test_build_prompt_amendments_empty(self) -> None:
        mutator = WorkflowMutator()
        amendments = mutator.build_prompt_amendments({"successful": [], "failed": [], "plateau": []})
        assert amendments == {}

    def test_mutate_produces_modified_copy(self) -> None:
        workflow = _make_workflow()
        original_prompt = workflow.nodes["researcher"].prompt_template
        history = [
            _make_cycle_record(score_delta=0.2, hypothesis="pruning"),
            _make_cycle_record(score_delta=-0.1, hypothesis="bad approach"),
        ]
        mutator = WorkflowMutator(frozen_nodes=frozenset({"gate_review"}))
        mutated = mutator.mutate(workflow, history)

        assert mutated is not workflow
        assert mutated.nodes["researcher"].prompt_template != original_prompt
        assert workflow.nodes["researcher"].prompt_template == original_prompt

    def test_frozen_node_raises_valueerror(self) -> None:
        workflow = _make_workflow()
        mutator = WorkflowMutator(frozen_nodes=frozenset({"gate_review"}))
        history = [_make_cycle_record()]

        with pytest.raises(ValueError, match="Cannot mutate frozen node"):
            mutator.mutate(workflow, history, focus_nodes=["gate_review"])

    def test_mutate_skips_non_agent_nodes(self) -> None:
        workflow = _make_workflow()
        history = [_make_cycle_record(score_delta=0.2, hypothesis="good")]
        mutator = WorkflowMutator()
        mutated = mutator.mutate(workflow, history)
        gate_node = mutated.nodes["gate_review"]
        assert not hasattr(gate_node, "prompt_template")

    def test_outer_loop_integration_with_mutator(self, tmp_path: Path) -> None:
        workflow = _make_workflow()
        inner = CompressInnerLoop(tmp_path, workflow=workflow, frozen_nodes=frozenset())
        mutator = WorkflowMutator(frozen_nodes=frozenset({"gate_review"}))
        config = OuterLoopConfig(max_outer_cycles=5)
        outer = CompressOuterLoop(inner, config, mutator=mutator, plateau_threshold=1)
        outer._plateau_count = 1

        inner._history.append(_make_cycle_record(
            cycle_number=1, score_end=0.5, score_delta=0.1, hypothesis="good_technique",
        ))
        inner._history.append(_make_cycle_record(
            cycle_number=2, score_end=0.5, score_delta=-0.05, hypothesis="bad_technique",
        ))
        inner._history.append(_make_cycle_record(
            cycle_number=3, score_end=0.5, score_delta=0.0, hypothesis="neutral_technique",
        ))
        inner._history.append(_make_cycle_record(
            cycle_number=4, score_end=0.5, score_delta=0.0, hypothesis="stale_technique",
        ))

        original_prompt = inner.workflow.nodes["researcher"].prompt_template

        mock_record = CycleRecord(
            cycle_number=5,
            mode="compress",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=None,
            score_end=0.55,
            score_delta=0.05,
        )
        with patch.object(inner, "step", return_value=mock_record):
            assert not outer.converged()
            outer.step()

        assert inner.workflow.nodes["researcher"].prompt_template != original_prompt
