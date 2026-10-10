"""CLI subcommands for the outer loop evolutionary search."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.cli._helpers import _resolve_inactivity_timeout

if TYPE_CHECKING:
    from factory.outer_loop.evaluator import CycleRecord
    from factory.outer_loop.mode_registry import EphemeralModeRegistry
    from factory.outer_loop.models import Individual
    from factory.workflow.primitives import Workflow

_log = structlog.get_logger()


def _resolve_seed_workflow(ref: str) -> Workflow:
    """Dynamically import a callable that returns a Package or Workflow.

    The ref format is 'module.path:callable_name'. The callable must return
    either a Package (which is compiled to a Workflow) or a Workflow directly.
    """
    import importlib

    if ":" not in ref:
        raise ValueError(
            f"Invalid seed-workflow ref {ref!r}. "
            f"Expected 'module.path:callable' format (e.g. 'my_pkg.pipeline:build_pipeline')."
        )
    module_path, callable_name = ref.rsplit(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import module {module_path!r} from seed-workflow ref {ref!r}. "
            f"Ensure the package is installed."
        ) from exc
    fn = getattr(mod, callable_name, None)
    if fn is None:
        raise ImportError(
            f"Module {module_path!r} has no attribute {callable_name!r} "
            f"(from seed-workflow ref {ref!r})."
        )
    if not callable(fn):
        raise TypeError(
            f"{ref!r} resolved to {fn!r}, which is not callable."
        )
    result = fn()

    from factory.workflow.package import Package
    from factory.workflow.primitives import Workflow as WF

    if isinstance(result, Package):
        return result.compile()
    if isinstance(result, WF):
        return result
    raise TypeError(
        f"{ref!r} returned {type(result).__name__}, expected Package or Workflow."
    )


def _check_disk_space(project_path: Path, population_size: int) -> bool:
    """Check that enough disk space is available for the outer loop.

    Requires population_size * 0.2 + 10 GB free.
    Returns True if sufficient, False otherwise (prints error to stderr).
    """
    required_gb = population_size * 0.2 + 10
    free_bytes = shutil.disk_usage(project_path).free
    available_gb = free_bytes / (1024**3)

    if available_gb < required_gb:
        print(
            f"Insufficient disk: need {required_gb:.1f}GB, have {available_gb:.1f}GB",
            file=sys.stderr,
        )
        _log.error(
            "disk_space_insufficient",
            required_gb=required_gb,
            available_gb=round(available_gb, 1),
            population_size=population_size,
        )
        return False
    _log.info(
        "disk_space_ok",
        required_gb=required_gb,
        available_gb=round(available_gb, 1),
    )
    return True


def _make_inner_loop_factory(
    registry: EphemeralModeRegistry,
) -> Callable[[Workflow], str]:
    """Build a callable that finds the existing registered mode for a workflow.

    Looks up by structural hash instead of creating eval-copy modes.
    This bridges SwarmEvaluator → compose() / InnerLoop: without it,
    _inner_loop_factory is None and evaluation returns a dummy score=0.0.
    """
    _hash_to_mode: dict[str, str] = {}

    def _factory(workflow: Workflow) -> str:
        from factory.outer_loop.similarity import structural_hash

        wf_hash = structural_hash(workflow)
        if wf_hash in _hash_to_mode:
            return _hash_to_mode[wf_hash]

        for mode_name in registry.list_modes():
            existing_wf = registry.load(mode_name)
            if existing_wf is not None:
                existing_hash = structural_hash(existing_wf)
                _hash_to_mode[existing_hash] = mode_name
                if existing_hash == wf_hash:
                    return mode_name

        ind_id = wf_hash[:12]
        name = registry.register(ind_id, 0, workflow)
        _hash_to_mode[wf_hash] = name
        return name

    return _factory


def cmd_outer_loop(args: argparse.Namespace) -> int:
    """Dispatch outer-loop subcommands."""
    sub = getattr(args, "outer_loop_command", None)
    if not sub:
        print("Usage: factory outer-loop {calibrate,evaluate,reflect,evolve,status,promote}", file=sys.stderr)
        return 1

    handlers = {
        "calibrate": _cmd_calibrate,
        "evaluate": _cmd_evaluate,
        "reflect": _cmd_reflect,
        "evolve": _cmd_evolve,
        "status": _cmd_status,
        "promote": _cmd_promote,
        "prep-instances": _cmd_prep_instances,
        "list-benchmarks": _cmd_list_benchmarks,
    }
    handler = handlers.get(sub)
    if handler is None:
        print(f"Unknown outer-loop subcommand: {sub}", file=sys.stderr)
        return 1
    return handler(args)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Seed the initial population from a base workflow."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    print(f"Calibrating outer loop for {project_path}")

    from factory.outer_loop.engine import SwarmEngine
    from factory.outer_loop.evaluator import SwarmEvaluator
    from factory.outer_loop.filesystem import init_filesystem, load_config, save_checkpoint
    from factory.outer_loop.mode_registry import EphemeralModeRegistry
    from factory.outer_loop.models import OuterLoopState, SwarmConfig

    population_size = getattr(args, "population_size", 4)
    if not _check_disk_space(project_path, population_size):
        return 1

    config = load_config(project_path)
    if config is None:
        benchmark = getattr(args, "benchmark", "featurebench")
        budget = getattr(args, "budget", 50)
        population_size = getattr(args, "population_size", 4)
        designer_count = 0 if benchmark == "featurebench" else 2
        target_proj = getattr(args, "project_dir", None)
        test_cmd = getattr(args, "test_command", "")
        test_fmt = getattr(args, "test_format", "")

        bench_config = None
        try:
            from factory.outer_loop.benchmark_config import load_benchmark_config
            bench_config = load_benchmark_config(benchmark, project_path)
            _log.info("benchmark_config_loaded", benchmark=benchmark)
        except FileNotFoundError:
            _log.info("benchmark_config_not_found", benchmark=benchmark)

        resolved_test_format = test_fmt or (bench_config.test_format if bench_config else "pytest")
        resolved_test_command = test_cmd or (bench_config.test_command if bench_config else "")
        resolved_seed_workflow = bench_config.seed_workflow if bench_config else ""
        resolved_instance_format = bench_config.instance_format if bench_config else "directory"
        resolved_prep_command = bench_config.prep_command if bench_config else ""

        task_module = getattr(args, "task_module", "")
        seed_workflow_module = getattr(args, "seed_workflow", "")
        config = SwarmConfig(
            benchmark=benchmark,
            budget=budget,
            population_size=population_size,
            designer_count=designer_count,
            training_instances=getattr(args, "training_instances", []),
            holdout_instances=getattr(args, "holdout_instances", []),
            target_project=str(Path(target_proj).resolve()) if target_proj else "",
            test_command=resolved_test_command,
            test_format=resolved_test_format,
            seed_workflow=resolved_seed_workflow,
            instance_format=resolved_instance_format,
            prep_command=resolved_prep_command,
            task_module=task_module,
            seed_workflow_module=seed_workflow_module,
        )
        if task_module:
            _log.info("task_module_resolved", ref=task_module)

    root = init_filesystem(project_path, config)

    benchmark = config.benchmark
    if benchmark == "featurebench":
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        base_workflow = Workflow(
            name="featurebench-seed",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    model="opus",
                    timeout=int(_resolve_inactivity_timeout()),
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )
    else:
        try:
            from factory.workflow.contributed.featurebench.workflow import (
                workflow as featurebench_workflow,
            )

            base_workflow = featurebench_workflow()
        except ImportError:
            print(f"Error: could not load contributed workflow for benchmark '{benchmark}'.", file=sys.stderr)
            return 1

    # --seed-workflow is required — resolve the callable to get the base workflow.
    try:
        base_workflow = _resolve_seed_workflow(config.seed_workflow_module)
        _log.info(
            "seed_workflow_from_module",
            ref=config.seed_workflow_module,
            knobs=list(base_workflow.knob_values.keys()) if base_workflow.knob_values else [],
        )
    except (ValueError, ImportError, TypeError) as exc:
        print(f"Error: --seed-workflow: {exc}", file=sys.stderr)
        return 1

    target_dir = Path(config.target_project) if config.target_project else None
    registry = EphemeralModeRegistry(project_path, target_dir=target_dir)
    registry.prune_stale_modes()
    evaluator = SwarmEvaluator(
        config, inner_loop_factory=_make_inner_loop_factory(registry), project_dir=project_path,
    )

    from factory.outer_loop.similarity import NoveltyFilter

    min_ged = 1 if len(base_workflow.nodes) <= 2 else 3
    engine = SwarmEngine(
        config=config,
        evaluator=evaluator,
        novelty_filter=NoveltyFilter(min_edit_distance=min_ged),
        mode_registry=registry,
        project_dir=project_path,
    )

    population = engine.seed(base_workflow, config)

    pop_dir = root / "population"
    population.save(pop_dir)

    state = OuterLoopState(
        budget_remaining=config.budget,
        generation=0,
    )
    save_checkpoint(project_path, state)

    modes = registry.list_modes()
    print(f"Outer loop initialized at {root}")
    print(f"Seeded {population.size} individuals ({len(modes)} ephemeral modes):")
    for mode_name in modes:
        print(f"  - {mode_name}")
    print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


def _mode_suffix(mode_name: str, generation: int) -> str:
    """Strip the ``evolve-gen{N}-`` prefix from a mode name.

    The registry names modes ``evolve-gen{N}-{individual_id[:8]}``, so the
    remainder identifies the individual it was registered from.
    """
    prefix = f"evolve-gen{generation}-"
    return mode_name[len(prefix):] if mode_name.startswith(prefix) else mode_name


def _make_offspring_individual(
    mode_name: str,
    generation: int,
    workflow: Workflow,
    *,
    parent_id: str | None,
    mutation_record: object = None,
) -> Individual:
    """Build the ``Individual`` that mirrors a freshly registered offspring mode.

    The id is derived from the mode name rather than the requested child id so
    it always matches what :func:`_mode_suffix` extracts during score
    propagation — ``register`` truncates ids to 8 characters.
    """
    from factory.outer_loop.models import Individual as Ind
    from factory.outer_loop.models import MutationRecord
    from factory.outer_loop.similarity import compute_features

    try:
        features = compute_features(workflow)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("offspring_feature_computation_failed", mode=mode_name, error=str(exc))
        features = ()

    return Ind(
        id=_mode_suffix(mode_name, generation),
        workflow_data=workflow.to_dict(),
        score=0.0,
        features=features,
        generation=generation,
        parent_id=parent_id,
        mutation_record=mutation_record if isinstance(mutation_record, MutationRecord) else None,
    )


def _propagate_scores_to_population(
    project_path: Path,
    generation: int,
    results: dict[str, dict[str, float]],
) -> int:
    """Write evaluation scores back into ``population/population.json``.

    Without this the population keeps ``score=0.0`` for every individual and
    the next generation's parent selection has no fitness signal to act on.
    Returns the number of individuals updated.
    """
    from factory.outer_loop.population import Population

    pop_dir = project_path / ".factory" / "outer_loop" / "population"
    if not (pop_dir / "population.json").exists():
        _log.debug("population_not_found_skipping_score_propagation", path=str(pop_dir))
        return 0

    population = Population.load(pop_dir)
    if population.size == 0:
        return 0

    updated = 0
    for mode_name, res in results.items():
        suffix = _mode_suffix(mode_name, generation)
        ind = population.get(suffix)
        if ind is None:
            ind = next(
                (i for i in population.individuals if i.id.startswith(suffix)),
                None,
            )
        if ind is None:
            _log.debug("population_individual_not_found", mode=mode_name, suffix=suffix)
            continue
        updated_ind = ind.model_copy(update={
            "score": res.get("score", 0.0),
            "cost_usd": res.get("cost_usd", 0.0),
        })
        population.remove(ind.id)
        population.add(updated_ind)
        updated += 1

    if updated:
        population.save(pop_dir)
        _log.info("population_scores_updated", generation=generation, count=updated)
    return updated


def _load_mode_scores(project_path: Path, generation: int) -> dict[str, float]:
    """Load ``results/gen{N}.json`` as a ``{mode_name: score}`` mapping.

    Returns an empty mapping when the generation has not been evaluated yet.
    """
    results_path = (
        project_path / ".factory" / "outer_loop" / "results" / f"gen{generation}.json"
    )
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("mode_scores_load_failed", generation=generation, error=str(exc))
        return {}
    if not isinstance(data, dict):
        return {}

    scores: dict[str, float] = {}
    for mode_name, entry in data.items():
        if isinstance(entry, dict):
            value = entry.get("score", 0.0)
        else:
            value = entry
        try:
            scores[mode_name] = float(value)
        except (TypeError, ValueError):
            scores[mode_name] = 0.0
    return scores


def _select_parent(
    modes: list[str],
    mode_scores: dict[str, float],
    tournament_size: int = 2,
) -> str | None:
    """Tournament selection over mode names, ranked by evaluation score.

    Picks ``tournament_size`` candidates uniformly at random and returns the
    highest-scoring one. Unevaluated modes score 0.0, so an all-unevaluated
    generation degrades to uniform random selection.
    """
    import random

    if not modes:
        return None
    k = min(tournament_size, len(modes))
    contenders = random.sample(modes, k)
    return max(contenders, key=lambda m: mode_scores.get(m, 0.0))


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the current generation's population."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    generation = getattr(args, "generation", 0)
    from factory.outer_loop.evaluator import SwarmEvaluator
    from factory.outer_loop.filesystem import load_checkpoint, load_config, save_checkpoint
    from factory.outer_loop.mode_registry import EphemeralModeRegistry
    from factory.outer_loop.models import OuterLoopState

    config = load_config(project_path)
    if config is None:
        print("Error: no outer loop config found. Run 'factory outer-loop calibrate' first.", file=sys.stderr)
        return 1

    cli_task_module = getattr(args, "task_module", "")
    if cli_task_module:
        config = config.model_copy(update={"task_module": cli_task_module})
        _log.info("task_module_override", ref=cli_task_module)

    eval_project_dir = getattr(args, "project_dir", None)
    if eval_project_dir is not None:
        eval_project_dir = str(Path(eval_project_dir).resolve())
    elif config.target_project:
        eval_project_dir = config.target_project
    else:
        eval_project_dir = str(project_path)
    print(f"Evaluating generation {generation} at {project_path} (target: {eval_project_dir})")

    target_dir = Path(eval_project_dir) if eval_project_dir != str(project_path) else None
    registry = EphemeralModeRegistry(project_path, target_dir=target_dir)
    all_modes = registry.list_modes()
    gen_prefix = f"evolve-gen{generation}-"
    eval_prefix = f"evolve-gen{generation}-eval-"
    modes = [m for m in all_modes if m.startswith(gen_prefix) and not m.startswith(eval_prefix)]
    if not modes:
        print("Error: no ephemeral modes found. Run 'factory outer-loop calibrate' first.", file=sys.stderr)
        return 1

    evaluator = SwarmEvaluator(config, inner_loop_factory=_make_inner_loop_factory(registry))
    results: dict[str, dict[str, float]] = {}
    for mode_name in modes:
        wf = registry.load(mode_name)
        if wf is None:
            continue
        ev = evaluator.evaluate(wf, eval_project_dir, config.training_instances)
        results[mode_name] = {"score": ev.score, "cost_usd": ev.cost_usd}
        print(f"  {mode_name}: score={ev.score:.4f} cost=${ev.cost_usd:.4f}")

        runs_dir = project_path / ".factory" / "outer_loop" / "runs" / mode_name
        runs_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {
            "mode": mode_name,
            "score": ev.score,
            "cost_usd": ev.cost_usd,
            "benchmark_score": ev.benchmark_score,
        }
        if ev.details:
            summary.update(ev.details)
        (runs_dir / "cycle_summary.json").write_text(json.dumps(summary, indent=2))

    results_dir = project_path / ".factory" / "outer_loop" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"gen{generation}.json"
    results_path.write_text(json.dumps(results, indent=2))

    updated = _propagate_scores_to_population(project_path, generation, results)
    if updated:
        print(f"Updated {updated} population individuals with evaluation scores")

    state = load_checkpoint(project_path) or OuterLoopState(budget_remaining=config.budget)
    gen_best = max((r["score"] for r in results.values()), default=0.0)
    new_best = max(state.best_score, gen_best)
    state = state.model_copy(update={
        "generation": generation,
        "total_evaluations": state.total_evaluations + len(results),
        "best_score": new_best,
    })
    save_checkpoint(project_path, state)

    print(f"Evaluated {len(results)} candidates. Results saved to {results_path}")
    return 0


def _load_cycle_summary(project_path: Path, mode_name: str) -> CycleRecord | None:
    """Load a CycleRecord from a persisted cycle_summary.json."""
    from factory.cycle_analyzer import CycleRecord as CR

    summary_path = project_path / ".factory" / "outer_loop" / "runs" / mode_name / "cycle_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
        duration_ms = data.get("duration_ms", 0)

        instance_results = data.get("instance_results")

        eval_details: dict[str, object] | None = None
        verify = data.get("verify")
        test_details = data.get("test_details")
        if verify is not None or test_details is not None:
            eval_details = {}
            if verify is not None:
                eval_details["verify"] = verify
            if test_details is not None:
                eval_details["test_details"] = test_details
            rejected = data.get("rejected")
            if rejected is not None:
                eval_details["rejected"] = rejected
            error = data.get("error")
            if isinstance(error, str):
                eval_details["error"] = error

        return CR(
            cycle_number=0,
            mode=mode_name,
            started_at=None,
            ended_at=None,
            duration_s=duration_ms / 1000.0 if duration_ms else 0.0,
            score_start=None,
            score_end=data.get("score"),
            score_delta=None,
            kept=data.get("kept", 0),
            reverted=data.get("reverted", 0),
            errored=data.get("agents_failed", 0),
            total_cost_usd=data.get("cost_usd", 0.0),
            instance_results=instance_results,
            eval_details=eval_details,
        )
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def _cmd_reflect(args: argparse.Namespace) -> int:
    """Run contrastive reflection on the current generation."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    generation = getattr(args, "generation", 0)
    print(f"Reflecting on generation {generation} at {project_path}")

    from factory.outer_loop.filesystem import load_config
    from factory.outer_loop.mode_registry import EphemeralModeRegistry
    from factory.outer_loop.reflector import OuterLoopReflector

    config = load_config(project_path)
    if config is None:
        print("Error: no outer loop config found.", file=sys.stderr)
        return 1

    eval_project_dir: str
    if config.target_project:
        eval_project_dir = config.target_project
    else:
        eval_project_dir = str(project_path)

    target_dir = Path(eval_project_dir) if eval_project_dir != str(project_path) else None
    registry = EphemeralModeRegistry(project_path, target_dir=target_dir)
    reflector = OuterLoopReflector(project_dir=project_path, llm_reflect=True)

    results_path = project_path / ".factory" / "outer_loop" / "results" / f"gen{generation}.json"
    saved_results: dict[str, dict[str, float]] = {}
    if results_path.exists():
        try:
            saved_results = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    records: list[tuple[str, float, CycleRecord | None]] = []
    needs_eval: list[tuple[str, Workflow]] = []

    for mode_name in registry.list_modes():
        saved = saved_results.get(mode_name)
        if saved is not None:
            score = float(saved.get("score", 0.0))
            cycle_rec = _load_cycle_summary(project_path, mode_name)
            records.append((mode_name, score, cycle_rec))
            continue
        cycle_rec = _load_cycle_summary(project_path, mode_name)
        if cycle_rec is not None and cycle_rec.score_end is not None:
            records.append((mode_name, cycle_rec.score_end, cycle_rec))
            continue
        wf = registry.load(mode_name)
        if wf is not None:
            needs_eval.append((mode_name, wf))

    if needs_eval:
        from factory.outer_loop.evaluator import SwarmEvaluator

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=_make_inner_loop_factory(registry),
        )
        for mode_name, wf in needs_eval:
            ev = evaluator.evaluate(wf, eval_project_dir, config.training_instances)
            cycle_rec = evaluator.get_cycle_record(mode_name)
            records.append((mode_name, ev.score, cycle_rec))

    if len(records) < 2:
        print("Not enough candidates for reflection (need >= 2).", file=sys.stderr)
        return 1

    kvbi: dict[str, dict[str, object]] = {}
    for mode_name_r, _, _ in records:
        wf = registry.load(mode_name_r)
        if wf is not None and wf.knob_values:
            kvbi[mode_name_r] = dict(wf.knob_values)

    report = reflector.reflect(records, generation, knob_values_by_id=kvbi)
    print(f"Reflection complete: {len(report.failure_patterns)} failures, "
          f"{len(report.success_patterns)} successes, "
          f"{len(report.mutation_suggestions)} suggestions")
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Produce the next generation via mutation and selection."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    generation = getattr(args, "generation", 0)
    print(f"Evolving generation {generation} at {project_path}")

    from factory.outer_loop.filesystem import load_config
    from factory.outer_loop.mode_registry import EphemeralModeRegistry
    from factory.outer_loop.mutations import WeightedRandomStrategy, apply_random_mutation
    from factory.outer_loop.population import Population
    from factory.outer_loop.reflector import ReflectionReport

    config = load_config(project_path)
    if config is None:
        print("Error: no outer loop config found.", file=sys.stderr)
        return 1

    if not _check_disk_space(project_path, config.population_size):
        return 1

    pop_dir = project_path / ".factory" / "outer_loop" / "population"
    population = Population.load(pop_dir) if (pop_dir / "population.json").exists() else None

    target_dir = Path(config.target_project) if config.target_project else None
    registry = EphemeralModeRegistry(project_path, target_dir=target_dir)
    registry.prune_stale_modes()
    modes = registry.list_modes()
    if not modes:
        print("Error: no ephemeral modes to evolve.", file=sys.stderr)
        return 1

    reflection_report: ReflectionReport | None = None
    reflection_path = (
        project_path / ".factory" / "outer_loop" / "reflections" / f"gen{generation}.json"
    )
    if reflection_path.exists():
        try:
            data = json.loads(reflection_path.read_text())
            from factory.outer_loop.reflector import MutationSuggestion

            raw_typed = data.pop("typed_suggestions", [])
            filtered = {k: v for k, v in data.items() if k != "generation"}
            reflection_report = ReflectionReport(**filtered)
            if isinstance(raw_typed, list):
                for item in raw_typed:
                    if isinstance(item, dict):
                        reflection_report.typed_suggestions.append(
                            MutationSuggestion(
                                operator=item.get("operator", ""),
                                target=item.get("target", ""),
                                rationale=item.get("rationale", ""),
                                value=item.get("value"),
                            )
                        )
            _log.info("reflection_report_loaded", generation=generation,
                      typed_suggestions=len(reflection_report.typed_suggestions))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            _log.warning("reflection_report_load_failed", error=str(exc))

    mode_scores = _load_mode_scores(project_path, generation)
    gen_prefix = f"evolve-gen{generation}-"
    eval_prefix = f"evolve-gen{generation}-eval-"
    parent_modes = [
        m for m in modes if m.startswith(gen_prefix) and not m.startswith(eval_prefix)
    ]
    if not parent_modes:
        _log.warning("no_modes_for_generation_using_all", generation=generation)
        parent_modes = modes
    if not mode_scores:
        _log.warning("no_evaluation_results_selection_is_uniform", generation=generation)

    strategy = WeightedRandomStrategy(mutation_rate=config.mutation_rate)
    offspring_count = 0

    for _ in range(config.population_size):
        parent_mode = _select_parent(parent_modes, mode_scores)
        if parent_mode is None:
            break
        wf = registry.load(parent_mode)
        if wf is None:
            continue
        result = apply_random_mutation(
            wf, strategy, generation + 1,
            frozen_nodes=set(config.frozen_node_ids),
            reflection_report=reflection_report,
        )
        if result is not None:
            child_wf, mutation_rec = result
            child_id = f"gen{generation + 1}_{offspring_count}"
            child_mode = registry.register(child_id, generation + 1, child_wf)
            if population is not None:
                population.add(
                    _make_offspring_individual(
                        child_mode, generation + 1, child_wf,
                        parent_id=_mode_suffix(parent_mode, generation),
                        mutation_record=mutation_rec,
                    )
                )
            offspring_count += 1
            print(
                f"  Created offspring {child_id} via {mutation_rec.operator.value} "
                f"(parent {parent_mode} score={mode_scores.get(parent_mode, 0.0):.4f})"
            )

    if population is not None and offspring_count:
        population.save(pop_dir)
        _log.info("population_offspring_added", generation=generation + 1, count=offspring_count)

    print(f"Evolution complete: {offspring_count} offspring created for generation {generation + 1}")
    return 0


def _scan_eval_worktrees(project_path: Path) -> dict[str, Path]:
    """Scan .eval-worktrees/ for active worktree directories.

    Returns {label: worktree_path} where label is extracted from `wt-{label}-{uuid8hex}`.
    """
    wt_base = project_path.parent / ".eval-worktrees"
    result: dict[str, Path] = {}
    try:
        entries = list(wt_base.iterdir())
    except (FileNotFoundError, PermissionError):
        return result
    for entry in entries:
        try:
            if not entry.is_dir() or not entry.name.startswith("wt-"):
                continue
        except (PermissionError, OSError):
            continue
        name = entry.name
        if re.fullmatch(r"wt-.+-[0-9a-f]{8}", name):
            label = name[3:-(8 + 1)]  # strip "wt-" prefix and "-{8hex}" suffix
            result[label] = entry
    return result


def _get_last_agent_phase(wt_path: Path) -> str | None:
    """Read the last agent.started event from a worktree's events.jsonl."""
    events_path = wt_path / ".factory" / "events.jsonl"
    try:
        text = events_path.read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    last_role: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "agent.started":
            last_role = event.get("agent")
    return last_role


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}m{s:02d}s"
    h, remainder = divmod(total, 3600)
    m = remainder // 60
    return f"{h}h{m:02d}m"


def _cmd_status(args: argparse.Namespace) -> int:
    """Show outer loop progress and metrics."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    check_converge = getattr(args, "check_converge", False)

    from factory.outer_loop.filesystem import load_checkpoint, load_config
    from factory.outer_loop.mode_registry import EphemeralModeRegistry

    config = load_config(project_path)
    state = load_checkpoint(project_path)
    registry = EphemeralModeRegistry(project_path)

    print("=== Outer Loop Status ===")
    if config:
        print(f"Benchmark: {config.benchmark}")
        print(f"Population size: {config.population_size}")
        print(f"Budget: {config.budget}")
    else:
        print("No outer loop config found.")

    if state:
        print(f"Generation: {state.generation}")
        gen = state.generation
    else:
        gen = 0

    modes = registry.list_modes()
    gen_prefix = f"evolve-gen{gen}-"
    eval_prefix = f"evolve-gen{gen}-eval-"
    gen_modes = [m for m in modes if m.startswith(gen_prefix) and not m.startswith(eval_prefix)]

    runs_dir = project_path / ".factory" / "outer_loop" / "runs"
    worktrees = _scan_eval_worktrees(project_path)

    completed: list[str] = []
    in_progress: list[tuple[str, str | None, float, str]] = []
    pending: list[str] = []
    now = time.time()

    for mode_name in gen_modes:
        summary_path = runs_dir / mode_name / "cycle_summary.json"
        if summary_path.exists():
            completed.append(mode_name)
            continue
        matched_wt: str | None = None
        for label, wt_path in worktrees.items():
            if mode_name.startswith(label) or label.startswith(mode_name[:12]):
                matched_wt = label
                break
        if matched_wt is not None:
            wt_path = worktrees[matched_wt]
            del worktrees[matched_wt]
            phase = _get_last_agent_phase(wt_path)
            try:
                elapsed = now - wt_path.stat().st_mtime
            except OSError:
                elapsed = 0.0
            wt_dir_name = wt_path.name
            in_progress.append((mode_name, phase, elapsed, wt_dir_name))
        else:
            pending.append(mode_name)

    n_in_progress = len(in_progress)
    total_evals = state.total_evaluations if state else 0
    if n_in_progress:
        print(f"Total evaluations: {total_evals} ({n_in_progress} in progress)")
    else:
        print(f"Total evaluations: {total_evals}")

    if state:
        print(f"Best score: {state.best_score:.4f}")
        print(f"Budget remaining: {state.budget_remaining}")
        if state.convergence_reason:
            print(f"Convergence: {state.convergence_reason}")
        if state.score_trajectory:
            print(f"Score trajectory: {[f'{s:.3f}' for s in state.score_trajectory[-5:]]}")
    else:
        print("No checkpoint found — outer loop not started.")

    print(f"Ephemeral modes: {len(modes)}")

    if in_progress:
        print()
        print("In progress:")
        for mode_name, phase, elapsed, wt_dir in in_progress:
            phase_str = f"[{phase}]" if phase else "[starting]"
            print(f"  {mode_name}  {phase_str}  {_format_elapsed(elapsed)}  {wt_dir}")

    if completed:
        print()
        print(f"Completed: {len(completed)}")

    if pending:
        print()
        print(f"Pending: {len(pending)}")

    traj_path = project_path / ".factory" / "outer_loop" / "trajectory.jsonl"
    if traj_path.exists():
        lines = traj_path.read_text().strip().splitlines()
        print(f"Trajectory entries: {len(lines)}")

    events_path = project_path / ".factory" / "outer_loop" / "events.jsonl"
    if events_path.exists():
        lines = events_path.read_text().strip().splitlines()
        print(f"Event log entries: {len(lines)}")

    if check_converge:
        if state and state.convergence_reason:
            print("CONVERGED")
            return 0
        else:
            print("NOT CONVERGED")
            return 1

    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    """Promote the best evolved workflow to a permanent mode."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()
    mode_name = getattr(args, "mode_name", None)
    permanent_name = getattr(args, "permanent_name", "evolved")

    if not mode_name:
        print("Error: --mode-name required", file=sys.stderr)
        return 1

    from factory.outer_loop.mode_registry import EphemeralModeRegistry

    registry = EphemeralModeRegistry(project_path)
    dest = registry.promote(mode_name, permanent_name)
    if dest:
        print(f"Promoted {mode_name} → {dest}")
        return 0
    else:
        print(f"Failed to promote {mode_name}", file=sys.stderr)
        return 1


def _cmd_prep_instances(args: argparse.Namespace) -> int:
    """Prepare benchmark instances from config."""
    benchmark = getattr(args, "benchmark", "featurebench")
    instances = getattr(args, "instances", [])
    output_dir = Path(getattr(args, "output_dir", ".")).resolve()
    project_path = Path(getattr(args, "project_path", ".")).resolve()

    if not instances:
        print("Error: --instances required", file=sys.stderr)
        return 1

    from factory.outer_loop.benchmark_config import load_benchmark_config
    from factory.outer_loop.instance_prep import prepare_instances

    try:
        config = load_benchmark_config(benchmark, project_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    prepared = prepare_instances(config, instances, output_dir)
    print(f"Prepared {len(prepared)}/{len(instances)} instances:")
    for p in prepared:
        print(f"  ✓ {p.name}")
    failed = set(instances) - {p.name for p in prepared}
    for name in sorted(failed):
        print(f"  ✗ {name}")
    return 0 if len(prepared) == len(instances) else 1


def _cmd_list_benchmarks(args: argparse.Namespace) -> int:
    """List all available benchmark configurations."""
    project_path = Path(getattr(args, "project_path", ".")).resolve()

    from factory.outer_loop.benchmark_config import list_benchmarks

    configs = list_benchmarks(project_path)
    if not configs:
        print("No benchmark configurations found.")
        return 0

    print(f"Available benchmarks ({len(configs)}):")
    for cfg in configs:
        print(f"  {cfg.name:20s}  format={cfg.test_format:12s}  instances={cfg.instance_format}")
        if cfg.description:
            print(f"    {cfg.description}")
    return 0


def add_outer_loop_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the outer-loop subcommand group to the CLI parser."""
    outer = subparsers.add_parser(
        "outer-loop",
        help="Evolutionary workflow search",
    )

    outer_sub = outer.add_subparsers(dest="outer_loop_command")

    cal = outer_sub.add_parser("calibrate", help="Seed initial population")
    cal.add_argument("project_path", nargs="?", default=".")
    cal.add_argument("--benchmark", default="featurebench")
    cal.add_argument("--budget", type=int, default=50)
    cal.add_argument("--population-size", type=int, default=4)
    cal.add_argument("--training-instances", nargs="*", default=[])
    cal.add_argument("--holdout-instances", nargs="*", default=[])
    cal.add_argument(
        "--project-dir",
        default=None,
        help="Target project dir for sub-CEO evaluation (defaults to project_path)",
    )
    cal.add_argument(
        "--test-command",
        default="",
        help="Test command for scoring (e.g. 'pytest tests/test_outputs.py -v')",
    )
    cal.add_argument(
        "--test-format",
        default="",
        help="Test output format: pytest, exit_code, json, exact_match (auto-detected from benchmark config if omitted)",
    )
    cal.add_argument(
        "--task-module",
        default="",
        help="Task class ref as 'module.path:ClassName' (e.g. chess_evolve.task:ChessEvolveTask)",
    )
    cal.add_argument(
        "--seed-workflow",
        required=True,
        help="Seed workflow callable as 'module.path:callable' — returns a Package or Workflow with OptKnobs (required)",
    )

    ev = outer_sub.add_parser("evaluate", help="Evaluate current generation")
    ev.add_argument("project_path", nargs="?", default=".")
    ev.add_argument("--generation", type=int, default=0)
    ev.add_argument(
        "--project-dir",
        default=None,
        help="Target project dir for sub-CEO evaluation (defaults to project_path)",
    )
    ev.add_argument(
        "--task-module",
        default="",
        help="Task class ref as 'module.path:ClassName' (overrides config value)",
    )

    ref = outer_sub.add_parser("reflect", help="Run reflection on generation")
    ref.add_argument("project_path", nargs="?", default=".")
    ref.add_argument("--generation", type=int, default=0)

    evo = outer_sub.add_parser("evolve", help="Produce next generation")
    evo.add_argument("project_path", nargs="?", default=".")
    evo.add_argument("--generation", type=int, default=0)

    st = outer_sub.add_parser("status", help="Show progress and metrics")
    st.add_argument("project_path", nargs="?", default=".")
    st.add_argument("--check-converge", action="store_true")

    pr = outer_sub.add_parser("promote", help="Promote best workflow")
    pr.add_argument("project_path", nargs="?", default=".")
    pr.add_argument("--mode-name", required=True)
    pr.add_argument("--permanent-name", default="evolved")

    prep = outer_sub.add_parser("prep-instances", help="Prepare benchmark instances")
    prep.add_argument("benchmark", help="Benchmark name (e.g. featurebench, swebench)")
    prep.add_argument("--instances", nargs="+", required=True, help="Instance IDs to prepare")
    prep.add_argument("--output-dir", default=".", help="Output directory for prepared instances")
    prep.add_argument("--project-path", default=".", help="Project path for config lookup")

    lb = outer_sub.add_parser("list-benchmarks", help="List available benchmarks")
    lb.add_argument("project_path", nargs="?", default=".")
