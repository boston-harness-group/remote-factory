"""Tests for SwarmEvaluator and FitnessCache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.evaluator import CycleRecordCache, FitnessCache, SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.subset import FixedSubsetSelector
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    Workflow,
)


def _make_config(**overrides: object) -> SwarmConfig:
    defaults: dict[str, object] = {
        "benchmark": "test",
        "budget": 50,
        "training_instances": ["t1", "t2"],
        "holdout_instances": ["h1"],
    }
    defaults.update(overrides)
    return SwarmConfig(**defaults)  # type: ignore[arg-type]


def _make_simple_workflow(name: str = "test_wf") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "study": FnNode(id="study", command="echo study", writes={".factory/obs.md"}),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build the project at {project_path}.",
                reads={".factory/obs.md"},
            ),
            "gate": GateNode(id="gate", evaluator_type="fn"),
        },
        edges=[
            Edge(source="study", target="builder"),
            Edge(source="builder", target="gate"),
        ],
        start_node="study",
    )


class TestFitnessCache:
    def test_miss_then_hit(self) -> None:
        cache = FitnessCache()
        wf = _make_simple_workflow()
        instances = ["t1", "t2"]

        assert cache.get(wf, instances) is None
        cache.put(wf, instances, 0.85, 1.5)
        result = cache.get(wf, instances)
        assert result is not None
        score, cost, ts = result
        assert score == 0.85
        assert cost == 1.5
        assert ts > 0

    def test_different_instances_separate_keys(self) -> None:
        cache = FitnessCache()
        wf = _make_simple_workflow()
        cache.put(wf, ["t1"], 0.7, 1.0)
        cache.put(wf, ["t1", "t2"], 0.85, 2.0)

        r1 = cache.get(wf, ["t1"])
        r2 = cache.get(wf, ["t1", "t2"])
        assert r1 is not None and r2 is not None
        assert r1[0] == 0.7
        assert r2[0] == 0.85

    def test_size(self) -> None:
        cache = FitnessCache()
        wf = _make_simple_workflow()
        assert cache.size == 0
        cache.put(wf, ["t1"], 0.5, 0.0)
        assert cache.size == 1


class TestSwarmEvaluator:
    def test_evaluate_with_fn(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(
                score=0.0, benchmark_score=0.8, hygiene_score=0.9,
                cost_usd=1.0, complexity=5.0,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])

        assert result.score > 0
        assert result.benchmark_score == 0.8

    def test_evaluate_uses_cache(self) -> None:
        config = _make_config()
        call_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal call_count
            call_count += 1
            return EvalResult(score=0.0, benchmark_score=0.7, hygiene_score=0.8)

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        wf = _make_simple_workflow()
        evaluator.evaluate(wf, "/tmp/test", ["t1"])
        evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert call_count == 1

    def test_mandatory_component_rejection(self) -> None:
        config = _make_config(mandatory_node_roles=["health_checker"])
        evaluator = SwarmEvaluator(config)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score == 0.0
        assert result.details.get("rejected") == "mandatory_component_missing"

    def test_mandatory_component_passes(self) -> None:
        config = _make_config(mandatory_node_roles=["builder"])

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.0, benchmark_score=0.5, hygiene_score=0.5)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score > 0

    def test_frozen_node_rejection(self) -> None:
        config = _make_config(frozen_node_ids=["missing_node"])
        evaluator = SwarmEvaluator(config)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score == 0.0
        assert result.details.get("rejected") == "frozen_node_violated"

    def test_frozen_node_passes(self) -> None:
        config = _make_config(frozen_node_ids=["study"])

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.0, benchmark_score=0.6, hygiene_score=0.7)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score > 0

    def test_evaluate_batch(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.0, benchmark_score=0.5, hygiene_score=0.5)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf1 = _make_simple_workflow("wf1")
        wf2 = _make_simple_workflow("wf2")
        results = evaluator.evaluate_batch([wf1, wf2], "/tmp/test", ["t1"])
        assert len(results) == 2
        assert all(r.score > 0 for r in results)

    def test_multi_metric_composition(self) -> None:
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(
                score=0.0, benchmark_score=1.0, hygiene_score=1.0,
                cost_usd=0.0, complexity=0.0,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        # 0.6*1.0 + 0.2*1.0 + 0.1*(1-0) + 0.1*(1-0) = 1.0
        assert result.score == 1.0

    def test_no_evaluator_fn(self) -> None:
        config = _make_config()
        evaluator = SwarmEvaluator(config)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.details.get("note") == "no_evaluator_fn_configured"

    @patch("factory.outer_loop.evaluator.SwarmEvaluator._cleanup_worktree")
    @patch("factory.outer_loop.evaluator.SwarmEvaluator._create_worktree")
    def test_task_path_wires_subset_selector(
        self, mock_create_wt: MagicMock, mock_cleanup_wt: MagicMock, tmp_path: Path
    ) -> None:
        """When _evaluate_via_inner_loop uses the task path with instances, it sets _subset_selector."""
        wt_path = tmp_path / "wt"
        wt_path.mkdir()
        mock_create_wt.return_value = wt_path

        captured_loop: list[object] = []

        mock_task = MagicMock()
        config = _make_config()
        config.set_task(mock_task)

        mock_loop = MagicMock()
        mock_loop.step.return_value = CycleRecord(
            cycle_number=1, mode="test",
            started_at="2026-01-01T00:00:00", ended_at="2026-01-01T00:01:00",
            duration_s=60.0, score_start=0.0, score_end=0.7, score_delta=0.7,
            kept=1, reverted=0, total_cost_usd=0.5,
        )
        mock_loop.mode = "task-eval"
        mock_loop.instance_results = None

        def fake_compose(workflow: object, task: object, project_dir: object) -> MagicMock:
            captured_loop.append(mock_loop)
            return mock_loop

        instances = ["inst_a", "inst_b"]

        factory_fn = MagicMock()
        evaluator = SwarmEvaluator(config, inner_loop_factory=factory_fn)

        with patch("factory.compose.compose", fake_compose):
            evaluator._evaluate_via_inner_loop(
                _make_simple_workflow(), str(tmp_path), instances
            )

        assert len(captured_loop) == 1
        factory_fn.assert_not_called()
        assert hasattr(mock_loop, "_subset_selector")
        sel = mock_loop._subset_selector
        assert isinstance(sel, FixedSubsetSelector)
        assert sel._training_instances == instances

    @patch("factory.outer_loop.evaluator.SwarmEvaluator._cleanup_worktree")
    @patch("factory.outer_loop.evaluator.SwarmEvaluator._create_worktree")
    def test_task_path_no_subset_selector_when_empty_instances(
        self, mock_create_wt: MagicMock, mock_cleanup_wt: MagicMock, tmp_path: Path
    ) -> None:
        """When instances is empty, _subset_selector should NOT be set."""
        wt_path = tmp_path / "wt"
        wt_path.mkdir()
        mock_create_wt.return_value = wt_path

        mock_task = MagicMock()
        config = _make_config()
        config.set_task(mock_task)

        mock_loop = MagicMock()
        mock_loop.step.return_value = CycleRecord(
            cycle_number=1, mode="test",
            started_at="2026-01-01T00:00:00", ended_at="2026-01-01T00:01:00",
            duration_s=60.0, score_start=0.0, score_end=0.7, score_delta=0.7,
            kept=1, reverted=0, total_cost_usd=0.5,
        )
        mock_loop.mode = "task-eval"
        mock_loop.instance_results = None

        def fake_compose(workflow: object, task: object, project_dir: object) -> MagicMock:
            return mock_loop

        factory_fn = MagicMock()
        evaluator = SwarmEvaluator(config, inner_loop_factory=factory_fn)

        with patch("factory.compose.compose", fake_compose):
            evaluator._evaluate_via_inner_loop(
                _make_simple_workflow(), str(tmp_path), []
            )

        factory_fn.assert_not_called()
        assert not isinstance(getattr(mock_loop, "_subset_selector", None), FixedSubsetSelector)

    def test_loads_cache_from_disk(self, tmp_path: Path) -> None:
        config = _make_config()
        wf = _make_simple_workflow()
        wf_hash = CycleRecordCache.workflow_hash(wf)

        cache_path = tmp_path / ".factory" / "outer_loop" / "eval_cache.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"workflow_hash": wf_hash, "score": 0.9, "cost": 1.5, "kept": 2, "reverted": 0}
        cache_path.write_text(json.dumps(entry) + "\n")

        evaluator = SwarmEvaluator(config, project_dir=tmp_path)
        assert evaluator.cycle_cache.size == 1

        cached = evaluator.cycle_cache.get(wf)
        assert cached is not None
        assert cached.score_end == 0.9


class TestCycleRecordCache:
    def _make_record(self, score: float = 0.8, cost: float = 1.0) -> CycleRecord:
        return CycleRecord(
            cycle_number=1,
            mode="test",
            started_at="2026-01-01T00:00:00",
            ended_at="2026-01-01T00:10:00",
            duration_s=600.0,
            score_start=0.0,
            score_end=score,
            score_delta=score,
            kept=3,
            reverted=1,
            total_cost_usd=cost,
        )

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        cache = CycleRecordCache()
        wf = _make_simple_workflow()
        record = self._make_record(0.85, 2.0)
        cache.put(wf, record)

        path = tmp_path / "cache.jsonl"
        cache.save_cache(path)
        assert path.exists()

        cache2 = CycleRecordCache()
        loaded = cache2.load_cache(path)
        assert loaded == 1
        assert cache2.size == 1

        restored = cache2.get(wf)
        assert restored is not None
        assert restored.score_end == 0.85
        assert restored.total_cost_usd == 2.0

    def test_save_is_append_only(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        wf1 = _make_simple_workflow("wf1")
        wf2 = _make_simple_workflow("wf2")

        cache1 = CycleRecordCache()
        cache1.put(wf1, self._make_record(0.7))
        cache1.save_cache(path)

        cache2 = CycleRecordCache()
        cache2.put(wf2, self._make_record(0.9))
        cache2.save_cache(path)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_save_deduplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        wf = _make_simple_workflow()

        cache = CycleRecordCache()
        cache.put(wf, self._make_record())
        cache.save_cache(path)
        cache.save_cache(path)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_load_skips_corrupt_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        valid = json.dumps({"workflow_hash": "abc123", "score": 0.5, "cost": 1.0})
        path.write_text(f"not-json\n{valid}\n\n")

        cache = CycleRecordCache()
        loaded = cache.load_cache(path)
        assert loaded == 1

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        cache = CycleRecordCache()
        loaded = cache.load_cache(tmp_path / "missing.jsonl")
        assert loaded == 0
        assert cache.size == 0

    def test_checkpoint_cache(self, tmp_path: Path) -> None:
        config = _make_config()
        evaluator = SwarmEvaluator(config, project_dir=tmp_path)
        wf = _make_simple_workflow()
        record = self._make_record(0.75)
        evaluator.cycle_cache.put(wf, record)

        evaluator.checkpoint_cache()

        cache_path = tmp_path / ".factory" / "outer_loop" / "eval_cache.jsonl"
        assert cache_path.exists()
        lines = cache_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["score"] == 0.75


class TestGraphValidationPreCheck:
    """Tests for CHECK 4: graph validation in evaluate()."""

    def test_invalid_workflow_rejected(self) -> None:
        """Workflow failing validate_graph() is rejected with score 0."""
        config = _make_config()
        evaluator = SwarmEvaluator(config)
        # Create workflow with empty prompt — fails validation
        wf = Workflow(
            name="bad",
            nodes={
                "a": AgentNode(id="a", role=AgentRole.BUILDER),
            },
            edges=[],
            start_node="a",
        )
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score == 0.0
        assert result.details.get("rejected") == "graph_validation_failed"
        assert "validation_errors" in result.details

    def test_valid_workflow_passes_through(self) -> None:
        """Valid workflow is not rejected by graph validation."""
        config = _make_config()

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(score=0.0, benchmark_score=0.7, hygiene_score=0.8)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score > 0
        assert result.details.get("rejected") is None

    def test_graph_validation_before_evaluator_fn(self) -> None:
        """Graph validation rejects before the evaluator function is ever called."""
        config = _make_config()
        call_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal call_count
            call_count += 1
            return EvalResult(score=0.0, benchmark_score=0.5)

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        wf = Workflow(
            name="bad",
            nodes={
                "a": AgentNode(id="a", role=AgentRole.BUILDER),
            },
            edges=[],
            start_node="a",
        )
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])
        assert result.score == 0.0
        assert call_count == 0, "Evaluator fn should not be called for invalid workflow"
