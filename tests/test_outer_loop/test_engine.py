"""Tests for SwarmEngine and BudgetTracker."""

from __future__ import annotations

import pytest

from factory.outer_loop.engine import BudgetTracker, SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.mutations import WeightedRandomStrategy
from factory.outer_loop.population import Population
from factory.outer_loop.similarity import NoveltyFilter
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
    }
    defaults.update(overrides)
    return SwarmConfig(**defaults)  # type: ignore[arg-type]


def _make_workflow() -> Workflow:
    return Workflow(
        name="test_evo",
        nodes={
            "study": FnNode(
                id="study", command="factory study", writes={".factory/obs.md"},
            ),
            "researcher": AgentNode(
                id="researcher", role=AgentRole.RESEARCHER,
                prompt_template="Research the project at {project_path}.",
                reads={".factory/obs.md"}, writes={".factory/research.md"},
            ),
            "strategist": AgentNode(
                id="strategist", role=AgentRole.STRATEGIST,
                prompt_template="Strategize improvements for {project_path}.",
                reads={".factory/research.md"}, writes={".factory/current.md"},
            ),
            "builder": AgentNode(
                id="builder", role=AgentRole.BUILDER,
                prompt_template="Build improvements for {project_path}.",
                reads={".factory/current.md"}, writes={".factory/build.md"},
            ),
            "gate": GateNode(
                id="gate", evaluator_type="fn",
                reads={".factory/build.md"},
            ),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="strategist"),
            Edge(source="strategist", target="builder"),
            Edge(source="builder", target="gate"),
            Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
        ],
        start_node="study",
    )


def _make_deterministic_evaluator(
    base_score: float = 0.5, increment: float = 0.02,
) -> SwarmEvaluator:
    """Returns an evaluator that gives incrementally higher scores to different workflows."""
    counter: dict[str, int] = {"n": 0}

    def eval_fn(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
        counter["n"] += 1
        score = min(base_score + counter["n"] * increment, 1.0)
        return EvalResult(
            score=0.0, benchmark_score=score, hygiene_score=0.7,
            cost_usd=0.1, complexity=len(wf.nodes),
        )

    config = _make_config()
    return SwarmEvaluator(config, evaluator_fn=eval_fn)


class TestBudgetTracker:
    def test_initial_state(self) -> None:
        bt = BudgetTracker(100)
        assert bt.remaining == 100
        assert bt.consumed == 0
        assert not bt.exhausted
        assert bt.total_cost_usd == 0.0

    def test_consume(self) -> None:
        bt = BudgetTracker(10)
        bt.consume(3, cost_usd=1.5)
        assert bt.consumed == 3
        assert bt.remaining == 7
        assert bt.total_cost_usd == 1.5

    def test_exhausted(self) -> None:
        bt = BudgetTracker(5)
        bt.consume(5)
        assert bt.exhausted
        assert bt.remaining == 0

    def test_over_consume(self) -> None:
        bt = BudgetTracker(3)
        bt.consume(5)
        assert bt.exhausted
        assert bt.remaining == 0

    def test_elapsed(self) -> None:
        bt = BudgetTracker(10)
        assert bt.elapsed_seconds >= 0


class TestSwarmEngineSeed:
    def test_seed_creates_population(self) -> None:
        config = _make_config(population_size=4)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        pop = engine.seed(wf)
        assert pop.size >= 1
        assert pop.size <= 4

    def test_seed_slot_zero_is_original(self) -> None:
        config = _make_config(population_size=3, designer_count=0)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        pop = engine.seed(wf)
        individuals = pop.individuals
        original = [i for i in individuals if i.parent_id is None]
        assert len(original) == 1

    def test_seed_diversity(self) -> None:
        config = _make_config(population_size=4)
        evaluator = _make_deterministic_evaluator()
        novelty = NoveltyFilter(min_edit_distance=1)
        engine = SwarmEngine(config, evaluator, novelty_filter=novelty)
        wf = _make_workflow()

        pop = engine.seed(wf)
        ids = {i.id for i in pop.individuals}
        assert len(ids) == pop.size

    def test_seed_uses_registry_workflow_when_seed_workflow_set(self) -> None:
        """When config.seed_workflow names a registered workflow, seed() uses it."""
        from unittest.mock import patch

        registry_wf = Workflow(
            name="registry-seed",
            nodes={
                "builder": AgentNode(
                    id="builder", role=AgentRole.BUILDER,
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )

        config = _make_config(population_size=2, designer_count=0, seed_workflow="improve")
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)

        with patch(
            "factory.outer_loop.engine.WorkflowRegistry.get_workflow",
            return_value=registry_wf,
        ) as mock_get:
            fallback_wf = _make_workflow()
            pop = engine.seed(fallback_wf, config)
            mock_get.assert_called_once_with("improve")

        seed_ind = [i for i in pop.individuals if i.parent_id is None][0]
        seed_data = Workflow.from_dict(seed_ind.workflow_data)  # type: ignore[arg-type]
        assert seed_data.name == "registry-seed"

    def test_seed_falls_back_when_seed_workflow_not_found(self) -> None:
        """When seed_workflow is set but not found in registry, falls back to base_workflow."""
        from unittest.mock import patch

        config = _make_config(population_size=2, designer_count=0, seed_workflow="nonexistent")
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)

        with patch(
            "factory.outer_loop.engine.WorkflowRegistry.get_workflow",
            return_value=None,
        ):
            fallback_wf = _make_workflow()
            pop = engine.seed(fallback_wf, config)

        seed_ind = [i for i in pop.individuals if i.parent_id is None][0]
        seed_data = Workflow.from_dict(seed_ind.workflow_data)  # type: ignore[arg-type]
        assert seed_data.name == "test_evo"

    def test_seed_ignores_empty_seed_workflow(self) -> None:
        """When seed_workflow is empty, uses the passed-in base_workflow."""
        from unittest.mock import patch

        config = _make_config(population_size=2, designer_count=0, seed_workflow="")
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)

        with patch(
            "factory.outer_loop.engine.WorkflowRegistry.get_workflow",
        ) as mock_get:
            fallback_wf = _make_workflow()
            pop = engine.seed(fallback_wf, config)
            mock_get.assert_not_called()

        seed_ind = [i for i in pop.individuals if i.parent_id is None][0]
        seed_data = Workflow.from_dict(seed_ind.workflow_data)  # type: ignore[arg-type]
        assert seed_data.name == "test_evo"


class TestSwarmEngineEvolve:
    def test_evolve_generation_returns_summary(self) -> None:
        config = _make_config(budget=50, population_size=3)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()
        pop = engine.seed(wf)

        summary = engine.evolve_generation(pop, generation=1)

        assert summary.generation == 1
        assert summary.population_size > 0
        assert summary.best_score >= 0
        assert summary.hyperparameters is not None
        assert summary.hyperparameters.generation == 1

    def test_evolve_updates_archive(self) -> None:
        config = _make_config(budget=50, population_size=3)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()
        pop = engine.seed(wf)

        engine.evolve_generation(pop, generation=1)
        assert engine.archive.size > 0

    def test_hyperparameter_record_logged(self) -> None:
        config = _make_config(budget=50, population_size=3)
        evaluator = _make_deterministic_evaluator()
        strategy = WeightedRandomStrategy(mutation_rate=0.4, designer_ratio=0.2)
        engine = SwarmEngine(config, evaluator, strategy=strategy)
        wf = _make_workflow()
        pop = engine.seed(wf)

        summary = engine.evolve_generation(pop, generation=0)

        assert summary.hyperparameters is not None
        hp = summary.hyperparameters
        assert hp.mutation_rate == 0.4
        assert hp.designer_ratio == 0.2
        assert hp.population_size > 0


class TestEvolveGenerationKnobValues:
    def test_evolve_generation_builds_knob_values_by_id(self) -> None:
        """evolve_generation builds knob_values_by_id from population and passes to reflect."""
        from unittest.mock import patch

        wf_with_knobs = Workflow(
            name="knobbed",
            nodes={
                "builder": AgentNode(
                    id="builder", role=AgentRole.BUILDER,
                    writes={".factory/build.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
            knob_values={"temperature": 0.8, "strategy": "explore"},
        )

        config = _make_config(budget=50, population_size=2)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        pop = engine.seed(wf_with_knobs)

        captured_kwargs: list[dict] = []
        from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport

        def spy_reflect(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return ReflectionReport()

        with patch.object(OuterLoopReflector, "reflect", side_effect=spy_reflect):
            engine.evolve_generation(pop, generation=1)

        assert len(captured_kwargs) >= 1
        kvbi = captured_kwargs[0].get("knob_values_by_id")
        assert kvbi is not None
        assert len(kvbi) > 0
        has_original_knobs = any(
            knobs.get("temperature") == 0.8 for knobs in kvbi.values()
        )
        assert has_original_knobs


class TestSwarmEngineRun:
    def test_run_terminates_on_budget(self) -> None:
        config = _make_config(budget=30, population_size=2)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
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
        assert result.total_evaluations > 0
        assert result.generations_completed >= 1
        assert len(result.trajectory) > 0

    def test_run_terminates_on_target_score(self) -> None:
        config = _make_config(budget=100, population_size=2, target_score=0.6)

        def high_score_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(
                score=0.0, benchmark_score=0.9, hygiene_score=0.9,
                cost_usd=0.01, complexity=3.0,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=high_score_eval)
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)
        assert result.convergence_reason == "target_score_reached"
        assert result.best_score >= 0.6

    def test_run_holdout_audit(self) -> None:
        config = _make_config(budget=15, population_size=2)

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            if "h1" in instances:
                return EvalResult(score=0.0, benchmark_score=0.6, hygiene_score=0.6)
            return EvalResult(score=0.0, benchmark_score=0.7, hygiene_score=0.7)

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)
        assert result.val_score > 0
        assert isinstance(result.overfit_flag, bool)

    def test_run_hyperparameter_history(self) -> None:
        config = _make_config(budget=15, population_size=2)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)
        assert len(result.hyperparameter_history) == result.generations_completed

    def test_run_pareto_front(self) -> None:
        config = _make_config(budget=15, population_size=2)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)
        assert result.archive_size > 0
        assert len(result.pareto_front) > 0

    def test_run_result_fields(self) -> None:
        config = _make_config(budget=10, population_size=2)
        evaluator = _make_deterministic_evaluator()
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)
        assert result.best_workflow_data != {}
        assert result.total_cost_usd >= 0
        assert result.convergence_reason != ""


class TestSwarmEnginePlateau:
    def test_plateau_detection(self) -> None:
        config = _make_config(budget=100, population_size=2)

        def flat_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            return EvalResult(
                score=0.0, benchmark_score=0.5, hygiene_score=0.5,
                cost_usd=0.01, complexity=3.0,
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=flat_eval)
        strategy = WeightedRandomStrategy(mutation_rate=0.3)
        engine = SwarmEngine(config, evaluator, strategy=strategy)
        wf = _make_workflow()

        result = engine.run(wf)
        # With flat scores, should converge via plateau, early stop, or budget
        assert result.convergence_reason in (
            "budget_exhausted",
            "target_score_reached",
            "plateau",
            "diversity_collapse",
            "early_stop_unchanged",
            "unknown",
        )

    def test_plateau_increases_mutation_rate(self) -> None:
        strategy = WeightedRandomStrategy(mutation_rate=0.3)
        assert strategy.get_mutation_rate(0) == 0.3
        strategy.on_plateau()
        assert strategy.get_mutation_rate(0) == pytest.approx(0.5)

    def test_improvement_resets_mutation_rate(self) -> None:
        strategy = WeightedRandomStrategy(mutation_rate=0.3)
        strategy.on_plateau()
        assert strategy.get_mutation_rate(0) == pytest.approx(0.5)
        strategy.on_improvement()
        assert strategy.get_mutation_rate(0) == 0.3


class TestSwarmEngineIntegration:
    def test_3_generations_with_mock(self) -> None:
        """Integration test: 3 generations, pop=4, mock fitness, verify trajectory."""
        config = _make_config(budget=50, population_size=4, target_score=None)

        eval_counter: dict[str, int] = {"n": 0}

        def mock_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            eval_counter["n"] += 1
            score = min(0.3 + eval_counter["n"] * 0.01, 1.0)
            return EvalResult(
                score=0.0, benchmark_score=score, hygiene_score=0.6,
                cost_usd=0.05, complexity=float(len(wf.nodes)),
            )

        evaluator = SwarmEvaluator(config, evaluator_fn=mock_eval)
        engine = SwarmEngine(config, evaluator)
        wf = _make_workflow()

        result = engine.run(wf)

        assert result.generations_completed >= 1
        assert result.total_evaluations > 0
        assert len(result.trajectory) >= 1
        assert result.best_score > 0
        assert len(result.hyperparameter_history) == result.generations_completed

        for hp in result.hyperparameter_history:
            assert hp.mutation_rate > 0
            assert hp.population_size > 0


class TestSkipReEvaluation:
    """Regression tests for issue #1536: already-scored individuals must not be re-evaluated."""

    def test_population_not_reevaluated_across_generations(self) -> None:
        """Pre-scored individuals must NOT be re-evaluated or consume budget."""
        eval_calls: list[str] = []

        def tracking_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            eval_calls.append(wf.name)
            return EvalResult(
                score=0.0, benchmark_score=0.6, hygiene_score=0.7,
                cost_usd=0.05, complexity=float(len(wf.nodes)),
            )

        config = _make_config(budget=50, population_size=2, designer_count=0)
        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        engine = SwarmEngine(config, evaluator)

        # Build a population with pre-scored individuals
        pop = Population()
        wf = _make_workflow()
        ind1 = Population.make_individual(wf, generation=0, score=0.5)
        ind2 = Population.make_individual(wf, generation=0, score=0.3)
        pop.add(ind1)
        pop.add(ind2)

        # Add to archive so offspring can be generated
        engine.archive.add(ind1)
        engine.archive.add(ind2)

        budget_before = engine.budget.remaining
        eval_calls.clear()

        engine.evolve_generation(pop, generation=1)

        # The pre-scored individuals should NOT have triggered evaluator calls
        # Only offspring should have been evaluated
        budget_consumed = budget_before - engine.budget.remaining
        # Budget should NOT include 2 extra for re-evaluating ind1 and ind2
        # At most pop_size offspring + holdout
        assert budget_consumed <= config.population_size + 1  # +1 for potential holdout

    def test_zero_score_not_reevaluated(self) -> None:
        """Individual with score=0.0 (legitimate zero) must NOT be re-evaluated."""
        eval_calls: list[str] = []

        def tracking_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            eval_calls.append("called")
            return EvalResult(
                score=0.0, benchmark_score=0.5, hygiene_score=0.5,
                cost_usd=0.05, complexity=3.0,
            )

        config = _make_config(budget=50, population_size=2, designer_count=0)
        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        engine = SwarmEngine(config, evaluator)

        # Build population with a zero-scored individual
        pop = Population()
        wf = _make_workflow()
        ind_zero = Population.make_individual(wf, generation=0, score=0.0)
        pop.add(ind_zero)
        engine.archive.add(ind_zero)

        eval_calls.clear()
        budget_before = engine.budget.remaining

        engine.evolve_generation(pop, generation=1)

        # The zero-scored individual should NOT have been re-evaluated
        # Only offspring evaluations should consume budget
        budget_consumed = budget_before - engine.budget.remaining
        # If the zero-scored ind were re-evaluated, budget_consumed would be at least 1 more
        # than the number of offspring
        offspring_count = pop.size - 1  # subtract the original zero-scored
        assert budget_consumed <= offspring_count + 1  # +1 for potential holdout

    def test_none_score_gets_evaluated(self) -> None:
        """Individual with score=None (unevaluated) MUST be evaluated."""
        eval_calls: list[str] = []

        def tracking_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
            eval_calls.append("called")
            return EvalResult(
                score=0.0, benchmark_score=0.75, hygiene_score=0.7,
                cost_usd=0.05, complexity=3.0,
            )

        config = _make_config(budget=50, population_size=2, designer_count=0, training_instances=[])
        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        engine = SwarmEngine(config, evaluator)

        # Build population with an unevaluated individual (score=None)
        pop = Population()
        wf = _make_workflow()
        ind_none = Population.make_individual(wf, generation=0)  # score=None by default
        assert ind_none.score is None
        pop.add(ind_none)

        eval_calls.clear()

        engine.evolve_generation(pop, generation=0)

        # The None-scored individual MUST be evaluated
        assert len(eval_calls) >= 1
