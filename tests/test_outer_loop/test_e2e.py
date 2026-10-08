"""End-to-end integration test for the outer loop evolutionary search.

Creates a simple seed workflow, uses a mock evaluator that rewards more agent
nodes (so evolution discovers this), runs 3 generations with population=4,
and verifies the evolutionary loop actually improves over the seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.filesystem import (
    export_best_workflow,
    init_filesystem,
    load_checkpoint,
    save_best,
    save_checkpoint,
    save_generation,
    save_map_elites,
)
from factory.outer_loop.models import (
    EvalResult,
    OuterLoopState,
    SwarmConfig,
)
from factory.outer_loop.mutations import WeightedRandomStrategy
from factory.outer_loop.population import Population
from factory.outer_loop.similarity import NoveltyFilter
from factory.task import TaskInstance, VerifyResult
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow,
)


class _StubTask:
    """Minimal task stub that yields instances matching test training/holdout IDs."""

    def __init__(self, instance_ids: list[str]) -> None:
        self._ids = instance_ids

    def instances(self, split: str = "all") -> list[TaskInstance]:
        return [TaskInstance(id=i) for i in self._ids]

    def setup(self, instance: object, workspace: object) -> None:
        pass

    def prompt(self, instance: object) -> str:
        return "stub"

    def verify(self, instance: object, workspace: object) -> VerifyResult:
        return VerifyResult(passed=True, score=1.0)


def _seed_workflow() -> Workflow:
    """A simple 3-node seed workflow."""
    return Workflow(
        name="seed",
        nodes={
            "study": FnNode(
                id="study",
                command="factory study {project_path}",
                writes={".factory/obs.md"},
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                reads={".factory/obs.md"},
                writes={".factory/build.md"},
            ),
            "gate": GateNode(
                id="gate",
                evaluator_type="fn",
                reads={".factory/build.md"},
            ),
        },
        edges=[
            Edge(source="study", target="builder"),
            Edge(source="builder", target="gate"),
            Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
        ],
        start_node="study",
    )


def _make_feature_evaluator() -> SwarmEvaluator:
    """Evaluator that rewards more agent nodes — evolution should discover this."""
    def eval_fn(
        wf: Workflow, project_dir: str, instances: list[str],
    ) -> EvalResult:
        agent_count = sum(
            1 for n in wf.nodes.values() if isinstance(n, AgentNode)
        )
        node_count = len(wf.nodes)
        score = min(0.3 + agent_count * 0.1 + node_count * 0.02, 0.95)
        return EvalResult(
            score=0.0,
            benchmark_score=score,
            hygiene_score=0.6,
            cost_usd=0.01,
            complexity=float(node_count),
        )

    config = SwarmConfig(
        benchmark="test-e2e",
        budget=60,
        population_size=4,
        tournament_size=2,
        mutation_rate=0.5,
        training_instances=["t1", "t2", "t3"],
        
    )
    return SwarmEvaluator(config, evaluator_fn=eval_fn)


def _make_holdout_evaluator(training_score: float = 0.8) -> SwarmEvaluator:
    """Evaluator with distinct training vs holdout behavior for overfit testing."""
    def eval_fn(
        wf: Workflow, project_dir: str, instances: list[str],
    ) -> EvalResult:
        if any(i.startswith("h") for i in instances):
            score = training_score * 0.7
        else:
            score = training_score
        return EvalResult(
            score=0.0,
            benchmark_score=score,
            hygiene_score=0.6,
            cost_usd=0.01,
            complexity=float(len(wf.nodes)),
        )

    config = SwarmConfig(
        benchmark="test-overfit",
        budget=30,
        population_size=4,
        training_instances=["t1", "t2"],
        
    )
    return SwarmEvaluator(config, evaluator_fn=eval_fn)


class TestE2EEvolution:
    def test_evolution_improves_over_seed(self) -> None:
        """The best evolved workflow should score higher than the seed."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=60,
            population_size=4,
            tournament_size=2,
            mutation_rate=0.5,
            training_instances=["t1", "t2", "t3"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "t3", "h1"]))

        seed_score = evaluator.evaluate(seed_wf, "", ["t1", "t2", "t3"]).score

        strategy = WeightedRandomStrategy(mutation_rate=0.5)
        novelty = NoveltyFilter(min_edit_distance=1)
        engine = SwarmEngine(
            config, evaluator,
            strategy=strategy,
            novelty_filter=novelty,
        )

        result = engine.run(seed_wf)

        assert result.best_score > seed_score, (
            f"Best evolved score {result.best_score} should exceed "
            f"seed score {seed_score}"
        )

    def test_archive_populated(self) -> None:
        """MAP-Elites archive should have entries after evolution."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert result.archive_size > 0

    def test_trajectory_recorded(self) -> None:
        """Generation trajectory should be recorded."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert len(result.trajectory) >= 1
        assert result.generations_completed >= 1

    def test_hyperparameter_history_complete(self) -> None:
        """Every generation should have a HyperparameterRecord."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert len(result.hyperparameter_history) == result.generations_completed
        for hp in result.hyperparameter_history:
            assert hp.mutation_rate > 0
            assert hp.population_size > 0

    def test_best_workflow_is_valid(self) -> None:
        """The best workflow should be a valid Workflow."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert result.best_workflow_data != {}
        reconstructed = Workflow.from_dict(result.best_workflow_data)  # type: ignore[arg-type]
        assert len(reconstructed.nodes) > 0
        assert len(reconstructed.edges) > 0

    def test_pareto_front_non_empty(self) -> None:
        """Pareto front should contain at least one individual."""
        seed_wf = _seed_workflow()
        evaluator = _make_feature_evaluator()
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert len(result.pareto_front) > 0


class TestE2EOverfitDetection:
    def test_overfit_flagged(self) -> None:
        """When holdout score drops >15%, overfit should be flagged."""
        seed_wf = _seed_workflow()
        evaluator = _make_holdout_evaluator(training_score=0.8)
        config = SwarmConfig(
            benchmark="test-overfit",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        engine = SwarmEngine(config, evaluator)
        result = engine.run(seed_wf)

        assert result.overfit_flag is True
        assert result.val_score > 0


class TestE2EFilesystem:
    def test_init_and_checkpoint(self, tmp_path: Path) -> None:
        """Filesystem init creates directories and checkpoint round-trips."""
        config = SwarmConfig(
            benchmark="test-fs",
            budget=10,
            training_instances=["t1"],
            
        )
        root = init_filesystem(tmp_path, config)

        assert (root / "config.json").exists()
        assert (root / "state.json").exists()
        assert (root / "fitness_cache.json").exists()
        assert (root / "trajectory.jsonl").exists()
        assert (root / "archive").is_dir()
        assert (root / "map-elites").is_dir()
        assert (root / "best").is_dir()

        loaded_state = load_checkpoint(tmp_path)
        assert loaded_state is not None
        assert loaded_state.budget_remaining == 10

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        """Checkpoint save/load round-trip preserves state."""
        config = SwarmConfig(
            benchmark="test-ckpt",
            budget=50,
            training_instances=["t1"],
            
        )
        init_filesystem(tmp_path, config)

        state = OuterLoopState(
            generation=3,
            total_evaluations=25,
            best_score=0.72,
            budget_remaining=25,
            score_trajectory=[0.5, 0.6, 0.65, 0.72],
        )
        save_checkpoint(tmp_path, state)

        loaded = load_checkpoint(tmp_path)
        assert loaded is not None
        assert loaded.generation == 3
        assert loaded.total_evaluations == 25
        assert loaded.best_score == 0.72
        assert loaded.budget_remaining == 25
        assert len(loaded.score_trajectory) == 4

    def test_export_best_workflow(self, tmp_path: Path) -> None:
        """Export produces a portable Python file."""
        seed_wf = _seed_workflow()
        wf_data = seed_wf.to_dict()

        path = export_best_workflow(tmp_path, wf_data, "test-bench")

        assert path.exists()
        content = path.read_text()
        assert "meta" in content
        assert "test-bench-evolved" in content
        assert "def workflow()" in content

    def test_save_generation_creates_artifacts(self, tmp_path: Path) -> None:
        """save_generation creates generation directory with artifacts."""
        from factory.outer_loop.models import GenerationSummary, HyperparameterRecord
        from factory.outer_loop.population import Population

        config = SwarmConfig(
            benchmark="test-gen",
            budget=10,
            training_instances=["t1"],
            
        )
        init_filesystem(tmp_path, config)

        seed_wf = _seed_workflow()
        pop = Population()
        ind = Population.make_individual(seed_wf, generation=0)
        ind = ind.model_copy(update={"score": 0.5})
        pop.add(ind)

        hp = HyperparameterRecord(
            generation=0,
            mutation_rate=0.3,
            population_size=1,
            tournament_size=2,
            designer_ratio=0.3,
            best_score=0.5,
            mean_score=0.5,
        )
        summary = GenerationSummary(
            generation=0,
            population_size=1,
            best_score=0.5,
            mean_score=0.5,
            diversity=0.0,
            hyperparameters=hp,
        )
        save_generation(tmp_path, 0, summary, pop)

        gen_dir = tmp_path / ".factory" / "outer_loop" / "archive" / "generation-000"
        assert gen_dir.exists()
        assert (gen_dir / "summary.json").exists()
        assert (gen_dir / "hyperparameters.json").exists()
        assert (gen_dir / "variant-00" / "workflow.json").exists()
        assert (gen_dir / "variant-00" / "scores.json").exists()

        traj = tmp_path / ".factory" / "outer_loop" / "trajectory.jsonl"
        lines = traj.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["generation"] == 0
        assert entry["best_score"] == 0.5


class TestE2EFullPipeline:
    def test_full_pipeline_with_filesystem(self, tmp_path: Path) -> None:
        """Full pipeline: init → evolve → save → export."""
        seed_wf = _seed_workflow()
        config = SwarmConfig(
            benchmark="test-full",
            budget=30,
            population_size=4,
            training_instances=["t1", "t2"],
            
        )
        config.set_task(_StubTask(["t1", "t2", "h1"]))
        evaluator = _make_feature_evaluator()

        init_filesystem(tmp_path, config)

        engine = SwarmEngine(
            config, evaluator,
            novelty_filter=NoveltyFilter(min_edit_distance=1),
        )
        result = engine.run(seed_wf)

        state = OuterLoopState(
            generation=result.generations_completed,
            total_evaluations=result.total_evaluations,
            best_score=result.best_score,
            budget_remaining=config.budget - result.total_evaluations,
            convergence_reason=result.convergence_reason,
            score_trajectory=[s.best_score for s in result.trajectory],
            hyperparameter_history=result.hyperparameter_history,
        )
        save_checkpoint(tmp_path, state)
        save_best(tmp_path, result)
        save_map_elites(tmp_path, engine.archive)

        for i, summary in enumerate(result.trajectory):
            pop = Population()
            ind = Population.make_individual(seed_wf, generation=i)
            ind = ind.model_copy(update={"score": summary.best_score})
            pop.add(ind)
            save_generation(tmp_path, i, summary, pop)

        export_path = export_best_workflow(
            tmp_path, result.best_workflow_data, "test-full",
        )

        assert export_path.exists()
        assert (tmp_path / ".factory" / "outer_loop" / "state.json").exists()
        assert (tmp_path / ".factory" / "outer_loop" / "best" / "workflow.json").exists()
        assert (tmp_path / ".factory" / "outer_loop" / "map-elites" / "grid.json").exists()

        loaded = load_checkpoint(tmp_path)
        assert loaded is not None
        assert loaded.generation == result.generations_completed
        assert loaded.best_score == result.best_score
