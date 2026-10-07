"""Core evolutionary search controller for workflow optimization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.outer_loop.designer import DesignerAgent
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.mode_registry import EphemeralModeRegistry
from factory.outer_loop.models import (
    GenerationSummary,
    HyperparameterRecord,
    MutationRecord,
    OuterLoopResult,
    SwarmConfig,
)
from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport
from factory.outer_loop.mutations import (
    MutationStrategy,
    WeightedRandomStrategy,
    apply_random_mutation,
)
from factory.outer_loop.overfit import OverfitDetector
from factory.outer_loop.population import MAPElitesArchive, Population
from factory.outer_loop.similarity import NoveltyFilter
from factory.outer_loop.subset import FixedSubsetSelector, SubsetSelector
from factory.workflow.primitives import Workflow
from factory.workflow.registry import WorkflowRegistry

if TYPE_CHECKING:
    pass

log = structlog.get_logger()

PLATEAU_WINDOW = 3


def _auto_frozen_nodes(workflow: Workflow) -> set[str]:
    """Return node IDs that should always be frozen during mutation.

    Includes ONLY DataNode IDs — their subgraphs are the evolution surface
    and must remain mutable for the outer loop to improve them.
    """
    from factory.workflow.primitives import DataNode

    frozen: set[str] = set()
    for nid, node in workflow.nodes.items():
        if isinstance(node, DataNode):
            frozen.add(nid)
    return frozen


class BudgetTracker:
    """Tracks evaluation budget consumption, cost, and wall-clock time."""

    def __init__(self, total_budget: int) -> None:
        self._total = total_budget
        self._consumed = 0
        self._cost_usd = 0.0
        self._start_time = time.monotonic()
        self._warned_80 = False
        self._warned_95 = False

    @property
    def remaining(self) -> int:
        return max(0, self._total - self._consumed)

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def total_cost_usd(self) -> float:
        return self._cost_usd

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def exhausted(self) -> bool:
        return self._consumed >= self._total

    def consume(self, count: int = 1, cost_usd: float = 0.0) -> None:
        self._consumed += count
        self._cost_usd += cost_usd
        pct = self._consumed / self._total if self._total > 0 else 1.0
        if pct >= 0.95 and not self._warned_95:
            log.warning("budget_95_percent", consumed=self._consumed, total=self._total)
            self._warned_95 = True
        elif pct >= 0.80 and not self._warned_80:
            log.warning("budget_80_percent", consumed=self._consumed, total=self._total)
            self._warned_80 = True


class SwarmEngine:
    """Orchestrates the evolutionary search loop."""

    def __init__(
        self,
        config: SwarmConfig,
        evaluator: SwarmEvaluator,
        strategy: MutationStrategy | None = None,
        subset_selector: SubsetSelector | None = None,
        overfit_detector: OverfitDetector | None = None,
        novelty_filter: NoveltyFilter | None = None,
        designer: DesignerAgent | None = None,
        mode_registry: EphemeralModeRegistry | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._strategy: MutationStrategy = strategy or WeightedRandomStrategy(
            mutation_rate=config.mutation_rate,
        )
        self._subset: SubsetSelector = subset_selector or FixedSubsetSelector(
            config.training_instances,
        )
        self._overfit = overfit_detector or OverfitDetector()
        self._novelty = novelty_filter or NoveltyFilter(min_edit_distance=3)
        self._designer = designer or DesignerAgent()
        self._budget = BudgetTracker(config.budget)
        self._archive = MAPElitesArchive()
        self._score_trajectory: list[float] = []
        self._mode_registry = mode_registry
        self._project_dir = project_dir
        self._reflector = OuterLoopReflector(project_dir=project_dir, llm_reflect=True)
        self._last_reflection: ReflectionReport | None = None
        self._initial_diversity: float = 0.0
        self._top_ids_history: list[frozenset[str]] = []

    @property
    def archive(self) -> MAPElitesArchive:
        return self._archive

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    def seed(
        self,
        base_workflow: Workflow,
        config: SwarmConfig | None = None,
    ) -> Population:
        """Create the initial population from a base workflow.

        If config.seed_workflow is set, looks up the workflow from
        WorkflowRegistry and uses it instead of the passed-in base_workflow.
        Falls back to base_workflow when seed_workflow is empty or not found.

        Slot 0: unmodified seed.
        Slots 1..N-designer_count: random mutations of seed.
        Last designer_count slots: from-scratch designs via DesignerAgent.
        """
        cfg = config or self._config
        pop = Population()

        if cfg.seed_workflow:
            registry_wf = WorkflowRegistry.get_workflow(cfg.seed_workflow)
            if registry_wf is not None:
                log.info("seed_workflow_from_registry", name=cfg.seed_workflow)
                base_workflow = registry_wf
            else:
                log.warning("seed_workflow_not_found", name=cfg.seed_workflow)

        seed_ind = Population.make_individual(base_workflow, generation=0)
        pop.add(seed_ind)
        self._novelty.add(base_workflow)
        if self._mode_registry:
            self._mode_registry.register(seed_ind.id, 0, base_workflow)

        designer_count = cfg.designer_count
        mutation_slots = max(0, cfg.population_size - 1 - designer_count)

        attempts = 0
        max_attempts = mutation_slots * 10
        while pop.size < 1 + mutation_slots and attempts < max_attempts:
            attempts += 1
            result = apply_random_mutation(
                base_workflow,
                self._strategy,
                generation=0,
                frozen_nodes=set(cfg.frozen_node_ids) | _auto_frozen_nodes(base_workflow),
            )
            if result is None:
                continue
            mutated_wf, mutation_rec = result
            if not self._novelty.is_novel(mutated_wf):
                continue
            self._novelty.add(mutated_wf)
            ind = Population.make_individual(
                mutated_wf,
                generation=0,
                parent_id=seed_ind.id,
                mutation_record=mutation_rec,
            )
            pop.add(ind)
            if self._mode_registry:
                self._mode_registry.register(ind.id, 0, mutated_wf)

        if designer_count > 0:
            self._add_designer_variants(pop, cfg, designer_count, base_workflow)

        log.info(
            "population_seeded",
            size=pop.size,
            target=cfg.population_size,
            designer_variants=min(designer_count, pop.size),
        )
        return pop

    def _add_designer_variants(
        self,
        pop: Population,
        cfg: SwarmConfig,
        designer_count: int,
        seed_workflow: Workflow | None = None,
    ) -> None:
        """Add from-scratch designed workflows to the population."""
        benchmark_spec = cfg.benchmark
        designs: list[Workflow] = []
        frozen_ids = set(cfg.frozen_node_ids) if cfg.frozen_node_ids else set()
        if seed_workflow is not None:
            frozen_ids |= _auto_frozen_nodes(seed_workflow)
        frozen = frozen_ids if frozen_ids else None

        exec_strategy = cfg.execution_strategy

        if designer_count >= 1:
            try:
                minimal = self._designer.design_minimal(
                    benchmark_spec,
                    seed_workflow=seed_workflow,
                    frozen_node_ids=frozen,
                    execution_strategy=exec_strategy,
                )
                designs.append(minimal)
            except Exception:
                log.warning("designer_minimal_failed", exc_info=True)

        if designer_count >= 2:
            try:
                thorough = self._designer.design_thorough(
                    benchmark_spec,
                    seed_workflow=seed_workflow,
                    frozen_node_ids=frozen,
                    execution_strategy=exec_strategy,
                )
                designs.append(thorough)
            except Exception:
                log.warning("designer_thorough_failed", exc_info=True)

        for i in range(2, designer_count):
            try:
                custom = self._designer.design_custom(
                    benchmark_spec,
                    {"max_nodes": 4 + i, "parallel": i % 2 == 0},
                    seed_workflow=seed_workflow,
                    frozen_node_ids=frozen,
                    execution_strategy=exec_strategy,
                )
                designs.append(custom)
            except Exception:
                log.warning("designer_custom_failed", index=i, exc_info=True)

        for wf in designs:
            if pop.size >= cfg.population_size:
                break
            if self._novelty.is_novel(wf):
                self._novelty.add(wf)
                ind = Population.make_individual(wf, generation=0)
                pop.add(ind)
                if self._mode_registry:
                    self._mode_registry.register(ind.id, 0, wf)

    def evolve_generation(
        self,
        population: Population,
        generation: int,
        project_dir: str = "",
    ) -> GenerationSummary:
        """Run one generation of evolution."""
        # Use Task.instances(split="train") when Task is available (firewall)
        task = self._config.get_task()
        if task is not None:
            try:
                instances = [inst.id for inst in task.instances(split="train")]
                if self._config.training_instances:
                    allowed = set(self._config.training_instances)
                    instances = [i for i in instances if i in allowed]
            except TypeError:
                # Backward compat: Task subclass overrides instances() without split param
                log.warning(
                    "task_instances_no_split",
                    task=type(task).__name__,
                    msg="Task.instances() does not accept split param, falling back to config",
                )
                if self._config.training_instances:
                    instances = self._subset.select(
                        self._config.training_instances, generation, self._budget.remaining
                    )
                else:
                    # No split configured anywhere — use all instances (pre-split behavior)
                    instances = [inst.id for inst in task.instances()]
        else:
            # Legacy path: no Task with split support, use SubsetSelector
            instances = self._subset.select(
                self._config.training_instances, generation, self._budget.remaining
            )

        # Evaluate current population (skip already-scored individuals)
        for ind in population.individuals:
            if self._budget.exhausted:
                break
            if ind.score is not None:
                log.debug("skip_evaluated", individual_id=ind.id, score=ind.score)
                continue
            wf = Workflow.from_dict(ind.workflow_data)  # type: ignore[arg-type]
            ev = self._evaluator.evaluate(wf, project_dir, instances, individual_id=ind.id)
            self._budget.consume(1, cost_usd=ev.cost_usd)
            updated = ind.model_copy(update={"score": ev.score, "cost_usd": ev.cost_usd})
            population.remove(ind.id)
            population.add(updated)
            self._archive.add(updated)

        # Reflect on this generation's results
        if generation > 0 or len(population.individuals) >= 2:
            records = []
            kvbi: dict[str, dict[str, object]] = {}
            for ind in population.individuals:
                cycle_rec = self._evaluator.get_cycle_record(ind.id)
                records.append((ind.id, ind.score if ind.score is not None else 0.0, cycle_rec))
                ind_wf = Workflow.from_dict(ind.workflow_data)  # type: ignore[arg-type]
                if ind_wf.knob_values:
                    kvbi[ind.id] = dict(ind_wf.knob_values)
            self._last_reflection = self._reflector.reflect(
                records, generation, knob_values_by_id=kvbi,
            )

        # Select parents and create offspring
        mutations_applied: list[MutationRecord] = []
        novel_count = 0
        rejected_dupes = 0
        offspring: list[tuple[Workflow, MutationRecord, str]] = []

        mutation_rate = self._strategy.get_mutation_rate(generation)
        for _ in range(self._config.population_size):
            parent = self._archive.sample_parent(
                self._config.tournament_size,
                rank_weighted=self._config.rank_weighted_selection,
            )
            if parent is None:
                continue
            parent_wf = Workflow.from_dict(parent.workflow_data)  # type: ignore[arg-type]
            mutation_result = apply_random_mutation(
                parent_wf,
                self._strategy,
                generation,
                frozen_nodes=set(self._config.frozen_node_ids) | _auto_frozen_nodes(parent_wf),
                reflection_report=self._last_reflection,
            )
            if mutation_result is None:
                continue
            child_wf, mutation_rec = mutation_result
            if self._novelty.is_novel(child_wf):
                self._novelty.add(child_wf)
                offspring.append((child_wf, mutation_rec, parent.id))
                mutations_applied.append(mutation_rec)
                novel_count += 1
            else:
                rejected_dupes += 1

        # Evaluate offspring and add to population
        for child_wf, mutation_rec, parent_id in offspring:
            if self._budget.exhausted:
                break
            ind = Population.make_individual(
                child_wf,
                generation=generation,
                parent_id=parent_id,
                mutation_record=mutation_rec,
            )
            if self._mode_registry:
                self._mode_registry.register(ind.id, generation, child_wf)
            eval_result = self._evaluator.evaluate(child_wf, project_dir, instances, individual_id=ind.id)
            self._budget.consume(1, cost_usd=eval_result.cost_usd)
            updated = ind.model_copy(update={"score": eval_result.score, "cost_usd": eval_result.cost_usd})
            population.add(updated)
            self._archive.add(updated)

        # Cleanup non-surviving ephemeral mode files
        if self._mode_registry:
            survivor_names = set()
            for ind in population.individuals:
                for g in range(generation + 1):
                    survivor_names.add(f"evolve-gen{g}-{ind.id[:8]}")
            self._mode_registry.cleanup_generation(survivor_names)

        # Track best score and diversity
        best = population.best()
        best_score = best.score if best and best.score is not None else 0.0
        mean_score = population.mean_score()
        diversity = self._archive.diversity_metric()
        self._score_trajectory.append(best_score)

        if generation == 0:
            self._initial_diversity = diversity if diversity > 0 else 1.0

        top_3 = sorted(
            population.individuals,
            key=lambda i: i.score if i.score is not None else -1.0,
            reverse=True,
        )[:3]
        self._top_ids_history.append(frozenset(i.id for i in top_3))

        self._log_event(generation, best_score, mean_score, diversity, self._archive.size)
        self._log_costs(generation, population)

        hp_record = HyperparameterRecord(
            generation=generation,
            mutation_rate=mutation_rate,
            population_size=population.size,
            tournament_size=self._config.tournament_size,
            designer_ratio=self._strategy.get_designer_ratio(generation),
            operator_weights=(
                self._strategy.get_operator_weights()
                if hasattr(self._strategy, "get_operator_weights")
                else {}
            ),
            best_score=best_score,
            mean_score=mean_score,
            diversity=diversity,
            novel_count=novel_count,
        )

        return GenerationSummary(
            generation=generation,
            population_size=population.size,
            best_score=best_score,
            mean_score=mean_score,
            diversity=diversity,
            mutations_applied=mutations_applied,
            novel_count=novel_count,
            rejected_duplicates=rejected_dupes,
            hyperparameters=hp_record,
        )

    def _detect_plateau(self) -> bool:
        """Detect plateau: N consecutive generations with improvement < threshold."""
        window = self._config.plateau_window
        threshold = self._config.plateau_threshold
        if len(self._score_trajectory) < window + 1:
            return False
        recent = self._score_trajectory[-(window + 1):]
        baseline = recent[0]
        return all(abs(s - baseline) < threshold for s in recent[1:])

    def _detect_diversity_collapse(self) -> bool:
        """Detect diversity collapse: archive diversity below floor."""
        if not self._initial_diversity:
            return False
        current = self._archive.diversity_metric()
        return current < self._config.diversity_floor * self._initial_diversity

    def _detect_early_stop(self) -> bool:
        """Detect early stop: top 3 individuals unchanged for N generations."""
        n = self._config.early_stop_unchanged
        if len(self._top_ids_history) < n:
            return False
        recent = self._top_ids_history[-n:]
        return all(s == recent[0] for s in recent[1:])

    def _log_event(
        self, generation: int, best_score: float, mean_score: float,
        diversity: float, archive_size: int,
    ) -> None:
        if not self._project_dir:
            return
        import json
        events_path = self._project_dir / ".factory" / "outer_loop" / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "generation": generation,
            "best_score": best_score,
            "mean_score": mean_score,
            "diversity": diversity,
            "archive_size": archive_size,
        }
        with events_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_costs(self, generation: int, population: Population) -> None:
        if not self._project_dir:
            return
        import json
        costs_path = self._project_dir / ".factory" / "outer_loop" / "costs.jsonl"
        costs_path.parent.mkdir(parents=True, exist_ok=True)
        for ind in population.individuals:
            entry = {
                "generation": generation,
                "individual_id": ind.id,
                "score": ind.score,
                "cost_usd": ind.cost_usd,
            }
            with costs_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    def run(
        self,
        base_workflow: Workflow,
        project_dir: str = "",
    ) -> OuterLoopResult:
        """Run the full evolutionary search loop."""
        population = self.seed(base_workflow)
        generation = 0
        summaries: list[GenerationSummary] = []
        hp_history: list[HyperparameterRecord] = []

        while not self._should_terminate(generation):
            log.info("generation_start", generation=generation, budget_remaining=self._budget.remaining)
            summary = self.evolve_generation(population, generation, project_dir)
            summaries.append(summary)
            if summary.hyperparameters:
                hp_history.append(summary.hyperparameters)

            # Plateau detection with adaptive response
            if self._detect_plateau():
                if hasattr(self._strategy, "on_plateau"):
                    self._strategy.on_plateau()  # type: ignore[union-attr]
                    log.info("plateau_detected_adapting", generation=generation)
            elif len(self._score_trajectory) >= 2 and self._score_trajectory[-1] > self._score_trajectory[-2]:
                if hasattr(self._strategy, "on_improvement"):
                    self._strategy.on_improvement()  # type: ignore[union-attr]

            generation += 1

        convergence_reason = self._get_convergence_reason(generation)
        log.info("evolution_complete", reason=convergence_reason, generations=generation)

        # End-of-run holdout evaluation (firewall: only runs after evolution)
        best = self._archive.best()
        audit_result = None
        holdout_score_val = 0.0
        task = self._config.get_task()

        if best:
            # Determine holdout instances: Task splits take priority, then SwarmConfig
            holdout_instances: list[str] = []
            train_instances: list[str] = []
            if task is not None:
                try:
                    holdout_instances = [inst.id for inst in task.instances(split="val")]
                except TypeError:
                    log.warning(
                        "task_instances_no_split",
                        task=type(task).__name__,
                        msg="Task.instances() does not accept split param, falling back to config (val)",
                    )
                    holdout_instances = list(self._config.holdout_instances)
                try:
                    train_instances = [inst.id for inst in task.instances(split="train")]
                except TypeError:
                    if self._config.training_instances:
                        train_instances = list(self._config.training_instances)
                    else:
                        # No split configured anywhere — use all instances (pre-split behavior)
                        train_instances = [inst.id for inst in task.instances()]

            # Fall back to SwarmConfig holdout_instances if Task has no splits
            if not holdout_instances and self._config.holdout_instances:
                holdout_instances = list(self._config.holdout_instances)
                train_instances = list(self._config.training_instances)

            if holdout_instances:
                best_wf = Workflow.from_dict(best.workflow_data)  # type: ignore[arg-type]
                # Evaluate WITHOUT individual_id → no CycleRecord stored
                holdout_result = self._evaluator.evaluate(
                    best_wf, project_dir, holdout_instances,
                )
                holdout_score_val = holdout_result.score
                best = best.model_copy(update={"val_score": holdout_score_val})

                # Audit with pre-computed training score to skip redundant re-eval
                audit_result = self._overfit.audit(
                    best_wf,
                    train_instances,
                    holdout_instances,
                    self._evaluator,
                    project_dir,
                    training_score=best.score,
                )
            else:
                log.info("holdout_no_instances", msg="No holdout instances declared — skipping holdout evaluation")

        pareto = self._archive.pareto_front()

        return OuterLoopResult(
            best_workflow_data=best.workflow_data if best else {},
            best_score=best.score if best and best.score is not None else 0.0,
            val_score=holdout_score_val,
            overfit_flag=audit_result.overfit_flag if audit_result else False,
            trajectory=summaries,
            total_cost_usd=self._budget.total_cost_usd,
            convergence_reason=convergence_reason,
            generations_completed=generation,
            total_evaluations=self._budget.consumed,
            archive_size=self._archive.size,
            total_candidates_evaluated=self._budget.consumed,
            pareto_front=pareto,
            hyperparameter_history=hp_history,
        )

    def _should_terminate(self, generation: int) -> bool:
        if self._budget.exhausted:
            return True
        if self._config.target_score is not None and self._score_trajectory:
            if self._score_trajectory[-1] >= self._config.target_score:
                return True
        if self._detect_plateau():
            window = self._config.plateau_window
            if len(self._score_trajectory) >= window + 2:
                recent = self._score_trajectory[-(window + 2):]
                threshold = self._config.plateau_threshold
                if all(abs(s - recent[0]) < threshold for s in recent[1:]):
                    return True
        if self._detect_diversity_collapse():
            return True
        if self._detect_early_stop():
            return True
        return False

    def _get_convergence_reason(self, generation: int) -> str:
        if self._budget.exhausted:
            return "budget_exhausted"
        if self._config.target_score is not None and self._score_trajectory:
            if self._score_trajectory[-1] >= self._config.target_score:
                return "target_score_reached"
        if self._detect_plateau():
            return "plateau"
        if self._detect_diversity_collapse():
            return "diversity_collapse"
        if self._detect_early_stop():
            return "early_stop_unchanged"
        return "unknown"
