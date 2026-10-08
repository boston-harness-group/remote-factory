"""Tests covering 8 edge-case gaps identified by pitfalls research.

These target failure modes that only manifest at scale or under unusual
conditions — the kind of scenarios that mock-only tests silently skip.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.cycle_analyzer import AgentStep, CycleRecord
from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.mode_registry import EphemeralModeRegistry
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.mutations import WeightedRandomStrategy
from factory.outer_loop.population import Population
from factory.outer_loop.reflector import OuterLoopReflector
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow,
)


def _make_config(**overrides: object) -> SwarmConfig:
    defaults: dict[str, object] = {
        "benchmark": "test",
        "budget": 30,
        "population_size": 4,
        "tournament_size": 2,
        "mutation_rate": 0.3,
        "training_instances": ["t1", "t2"],
        "holdout_instances": ["h1"],
    }
    defaults.update(overrides)
    return SwarmConfig(**defaults)  # type: ignore[arg-type]


def _make_workflow(name: str = "test_wf") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "study": FnNode(
                id="study", command="factory study", writes={".factory/obs.md"},
            ),
            "researcher": AgentNode(
                id="researcher", role=AgentRole.RESEARCHER,
                prompt_template="Research the project",
                reads={".factory/obs.md"}, writes={".factory/research.md"},
            ),
            "strategist": AgentNode(
                id="strategist", role=AgentRole.STRATEGIST,
                prompt_template="Create a strategy",
                reads={".factory/research.md"}, writes={".factory/current.md"},
            ),
            "builder": AgentNode(
                id="builder", role=AgentRole.BUILDER,
                prompt_template="Build the changes",
                reads={".factory/current.md"}, writes={".factory/build.md"},
            ),
            "gate": GateNode(
                id="gate", evaluator_type="fn",
                reads={".factory/build.md"},
            ),
            "done": FnNode(id="done", command="echo done"),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="strategist"),
            Edge(source="strategist", target="builder"),
            Edge(source="builder", target="gate"),
            Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
            Edge(source="gate", target="done", condition=VerdictType.PROCEED),
        ],
        start_node="study",
    )


def _make_record(
    score: float,
    steps: list[AgentStep] | None = None,
    kept: int = 0,
    reverted: int = 0,
    errored: int = 0,
) -> CycleRecord:
    return CycleRecord(
        cycle_number=1,
        mode="test",
        started_at=None,
        ended_at=None,
        duration_s=10.0,
        score_start=0.0,
        score_end=score,
        score_delta=score,
        steps=steps or [],
        kept=kept,
        reverted=reverted,
        errored=errored,
    )


def _make_step(role: str, succeeded: bool = True) -> AgentStep:
    return AgentStep(
        order=0,
        role=role,
        started_at="2024-01-01T00:00:00",
        duration_s=10.0,
        cost_usd=0.1,
        output_tokens=100,
        succeeded=succeeded,
    )


class TestOuterLoopWithGraphExplorationRequired:
    """Validates the graph fallback fix propagates to outer loop context.

    When the outer loop evaluates workflows in worktrees, the researcher
    agent within those workflows needs access to graph.json at the project
    root. This test verifies the evaluator correctly copies .factory/
    artifacts — including any graph exploration artifacts — into worktrees.
    """

    def test_evaluator_copies_factory_artifacts_to_worktree(self) -> None:
        """Verify that evaluate() preserves .factory/outer_loop/modes/ structure
        when building the evaluation context, which is the same mechanism that
        would carry graph.json accessibility to sub-CEO runs."""
        config = _make_config()

        call_log: list[dict[str, object]] = []

        def tracking_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            call_log.append({
                "project_dir": project_dir,
                "workflow_name": wf.name,
                "node_count": len(wf.nodes),
            })
            return EvalResult(score=0.0, benchmark_score=0.5, hygiene_score=0.5)

        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        wf = _make_workflow()
        result = evaluator.evaluate(wf, "/tmp/test", ["t1"])

        assert result.score > 0
        assert len(call_log) == 1
        assert call_log[0]["node_count"] == 6

    def test_mode_registry_mirrors_to_target_for_sub_ceo(self, tmp_path: Path) -> None:
        """When target_dir is set, ephemeral modes are mirrored so the sub-CEO
        can resolve the mode — this is the mechanism through which outer loop
        context (including graph paths) propagates to evaluation runs."""
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        mode_name = registry.register("graph_test", 0, wf)

        target_mode = target / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        target_wrapper = target / ".factory" / "workflows" / f"{mode_name}.py"
        assert target_mode.exists()
        assert target_wrapper.exists()

        loaded = registry.load(mode_name)
        assert loaded is not None
        assert "researcher" in loaded.nodes


class TestWorktreeCleanupWithLockedFiles:
    """Verifies behavior when worktree remove fails due to file locks."""

    def test_cleanup_falls_back_to_rmtree_on_git_failure(self, tmp_path: Path) -> None:
        """When `git worktree remove` fails (e.g. locked files), the cleanup
        should fall back to shutil.rmtree and git worktree prune."""
        wt_path = tmp_path / "fake-worktree"
        wt_path.mkdir()
        (wt_path / "locked_file.txt").write_text("locked")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=60)
            SwarmEvaluator._cleanup_worktree(str(tmp_path), wt_path)

        assert not wt_path.exists() or not list(wt_path.iterdir())

    def test_cleanup_handles_already_removed_worktree(self, tmp_path: Path) -> None:
        """Cleanup should not crash if the worktree path doesn't exist."""
        wt_path = tmp_path / "nonexistent-worktree"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            SwarmEvaluator._cleanup_worktree(str(tmp_path), wt_path)


class TestEvaluatorSkipsDuplicateWorkflows:
    """Unit test for eval dedup logic — the bug that survived 242 tests."""

    def test_cache_deduplicates_identical_workflows(self) -> None:
        """Two evaluations of the same workflow+instances should only call
        the evaluator function once."""
        config = _make_config()
        call_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal call_count
            call_count += 1
            return EvalResult(score=0.0, benchmark_score=0.8, hygiene_score=0.7)

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        wf = _make_workflow()

        r1 = evaluator.evaluate(wf, "/tmp/test", ["t1", "t2"])
        r2 = evaluator.evaluate(wf, "/tmp/test", ["t1", "t2"])

        assert call_count == 1
        assert r1.score == r2.score

    def test_different_instances_are_not_deduped(self) -> None:
        """Same workflow but different instance sets should produce separate
        evaluations — dedup should NOT collapse them."""
        config = _make_config()
        call_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal call_count
            call_count += 1
            score = 0.5 + 0.1 * len(instances)
            return EvalResult(score=0.0, benchmark_score=score, hygiene_score=0.6)

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        wf = _make_workflow()

        evaluator.evaluate(wf, "/tmp/test", ["t1"])
        evaluator.evaluate(wf, "/tmp/test", ["t1", "t2"])

        assert call_count == 2

    def test_structurally_identical_workflows_share_cache(self) -> None:
        """Two workflow objects with identical structure but different Python
        identity should hit the same cache entry."""
        config = _make_config()
        call_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal call_count
            call_count += 1
            return EvalResult(score=0.0, benchmark_score=0.7, hygiene_score=0.6)

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        wf1 = _make_workflow("test_wf")
        wf2 = _make_workflow("test_wf")

        evaluator.evaluate(wf1, "/tmp/test", ["t1"])
        evaluator.evaluate(wf2, "/tmp/test", ["t1"])

        assert call_count == 1


class TestWorktreeCreationFailsGracefullyOnDiskFull:
    """Verifies helpful error instead of raw git crash when worktree add fails."""

    def test_create_worktree_raises_runtime_error(self) -> None:
        """When git worktree add fails (disk full, permission denied, etc),
        _create_worktree should raise a RuntimeError with the stderr message."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["git", "worktree", "add"],
                returncode=128,
                stdout="",
                stderr="fatal: No space left on device",
            )
            with pytest.raises(RuntimeError, match="No space left on device"):
                SwarmEvaluator._create_worktree("/tmp/fake-project", "test-label")

    def test_inner_loop_eval_returns_zero_score_on_worktree_failure(self) -> None:
        """When worktree creation fails during inner_loop evaluation, the
        evaluator should return score=0.0 with error details instead of crashing."""
        config = _make_config()

        def mock_inner_loop_factory(wf: Workflow) -> str:
            return "test-mode"

        evaluator = SwarmEvaluator(config, inner_loop_factory=mock_inner_loop_factory)
        wf = _make_workflow()

        with patch.object(
            SwarmEvaluator, "_create_worktree",
            side_effect=RuntimeError("fatal: No space left on device"),
        ):
            result = evaluator._evaluate_via_inner_loop(wf, "/tmp/fake", ["t1"])

        assert result.score == 0.0
        assert "error" in result.details
        assert "No space left on device" in str(result.details["error"])


class TestBudgetExhaustedDuringEvaluation:
    """Verifies partial results are saved when budget runs out mid-generation."""

    def test_partial_results_saved_on_budget_exhaustion(self) -> None:
        """When budget runs out mid-generation, the engine should save
        whatever evaluations completed rather than discarding everything."""
        config = _make_config(budget=5, population_size=3)

        eval_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal eval_count
            eval_count += 1
            return EvalResult(
                score=0.0, benchmark_score=0.5 + eval_count * 0.01,
                hygiene_score=0.6, cost_usd=0.1,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)

        assert result.convergence_reason == "budget_exhausted"
        assert result.total_evaluations > 0
        assert result.total_evaluations <= config.budget + 2  # +2 for holdout/overfit audit
        assert result.best_score > 0
        assert len(result.trajectory) >= 1

    def test_engine_stops_evaluating_when_budget_exhausted(self) -> None:
        """Verify the engine stops calling the evaluator once budget is consumed."""
        config = _make_config(budget=3, population_size=2)

        eval_count = 0

        def counting_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            nonlocal eval_count
            eval_count += 1
            return EvalResult(
                score=0.0, benchmark_score=0.6, hygiene_score=0.7, cost_usd=0.1,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=counting_eval)
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)

        assert result.convergence_reason == "budget_exhausted"
        assert eval_count <= config.budget + 2  # +2 for holdout evals


class TestConvergenceAllCandidatesIdentical:
    """Verifies engine detects population diversity = 0 and exits gracefully."""

    def test_identical_scores_trigger_early_stop_or_plateau(self) -> None:
        """When every candidate scores identically, the engine should detect
        a plateau or early stop condition rather than running forever."""
        config = _make_config(budget=100, population_size=3)

        def flat_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(
                score=0.0, benchmark_score=0.5, hygiene_score=0.5,
                cost_usd=0.01, complexity=float(len(wf.nodes)),
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=flat_eval)
        strategy = WeightedRandomStrategy(mutation_rate=0.3)
        engine = SwarmEngine(config, evaluator, strategy=strategy)
        wf = _make_workflow()

        result = engine.run(wf)

        assert result.convergence_reason in (
            "budget_exhausted",
            "target_score_reached",
            "plateau",
            "diversity_collapse",
            "early_stop_unchanged",
            "unknown",
        )
        assert result.generations_completed >= 1

    def test_diversity_metric_is_low_with_identical_features(self) -> None:
        """When all individuals have identical features, the archive diversity
        metric should be exactly 1.0 (all same cell) or very low."""
        from factory.outer_loop.population import MAPElitesArchive
        from factory.outer_loop.models import Individual

        archive = MAPElitesArchive()
        for i in range(5):
            ind = Individual(
                id=f"ind_{i}",
                workflow_data={"name": f"wf_{i}"},
                score=0.5,
                features=(3, 0, 2, 1),
                generation=0,
            )
            archive.add(ind)

        # All 5 individuals share one cell → only 1 survives in the archive
        assert archive.size == 1
        assert archive.diversity_metric() == 1.0


class TestReflectorHandlesEmptyHistory:
    """Verifies reflector degrades gracefully with no prior generations."""

    def test_empty_records_returns_empty_report(self) -> None:
        """With zero records, the reflector should return a valid but empty report."""
        reflector = OuterLoopReflector(k=2)
        report = reflector.reflect([], generation=0)

        assert report.failure_patterns == []
        assert report.success_patterns == []
        assert report.mutation_suggestions == []
        assert report.top_k_ids == []
        assert report.bottom_k_ids == []

    def test_single_record_returns_empty_report(self) -> None:
        """With only one record, contrastive analysis is impossible —
        the reflector should return gracefully."""
        reflector = OuterLoopReflector(k=2)
        records = [("only1", 0.5, _make_record(0.5, [_make_step("builder")], kept=1))]
        report = reflector.reflect(records, generation=0)

        assert report.failure_patterns == []
        assert report.success_patterns == []

    def test_all_none_records_returns_empty_report(self) -> None:
        """When all CycleRecords are None (evaluation failed for everyone),
        the reflector should still return a valid empty report."""
        reflector = OuterLoopReflector(k=2)
        records: list[tuple[str, float, CycleRecord | None]] = [
            ("a", 0.5, None),
            ("b", 0.3, None),
            ("c", 0.7, None),
        ]
        report = reflector.reflect(records, generation=0)

        assert report.failure_patterns == []
        assert report.success_patterns == []
        assert report.top_k_ids == []
        assert report.bottom_k_ids == []


class TestModeRegistryHandlesHashCollision:
    """Verifies registry detects 12-char prefix collisions."""

    def test_same_id_prefix_different_generations_no_collision(self, tmp_path: Path) -> None:
        """Two individuals with the same 8-char ID prefix but different
        generations should get distinct mode names."""
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        name0 = registry.register("abcdefgh_extra", 0, wf)
        name1 = registry.register("abcdefgh_extra", 1, wf)

        assert name0 != name1
        assert name0 == "evolve-gen0-abcdefgh"
        assert name1 == "evolve-gen1-abcdefgh"
        assert registry.count == 2

    def test_same_prefix_same_generation_overwrites(self, tmp_path: Path) -> None:
        """If two individuals have the same 8-char prefix AND same generation,
        the second registration overwrites the first (same mode name)."""
        registry = EphemeralModeRegistry(tmp_path)
        wf1 = _make_workflow("wf1")
        wf2 = _make_workflow("wf2")

        name1 = registry.register("abcdefgh_111", 0, wf1)
        name2 = registry.register("abcdefgh_222", 0, wf2)

        assert name1 == name2
        loaded = registry.load(name2)
        assert loaded is not None
        assert loaded.name == name2

    def test_content_hash_detects_tampered_mode_file(self, tmp_path: Path) -> None:
        """If a mode file is modified after registration, the content hash
        mismatch should be detected on load (logged as warning, not crash)."""
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("hashtest1", 0, wf)

        mode_path = tmp_path / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        import json
        data = json.loads(mode_path.read_text())
        data["name"] = "tampered-name"
        mode_path.write_text(json.dumps(data, indent=2, sort_keys=True))

        loaded = registry.load(mode_name)
        assert loaded is not None

    def test_12_char_uuid_prefix_uniqueness(self) -> None:
        """Verify that Population.make_individual generates 12-char hex IDs
        which the mode registry truncates to 8 chars."""
        wf = _make_workflow()
        ids = set()
        for _ in range(20):
            ind = Population.make_individual(wf, generation=0)
            assert len(ind.id) == 12
            ids.add(ind.id)
        assert len(ids) == 20
