"""Experiment filesystem setup and checkpoint/resume for the outer loop."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from factory.outer_loop.models import (
    GenerationSummary,
    OuterLoopResult,
    OuterLoopState,
    SwarmConfig,
)
from factory.outer_loop.population import MAPElitesArchive, Population
from factory.workflow.primitives import Workflow

log = structlog.get_logger()


def init_filesystem(project_path: Path, config: SwarmConfig) -> Path:
    """Create the .factory/outer_loop/ directory structure.

    Returns the outer_loop root directory.
    """
    root = project_path / ".factory" / "outer_loop"
    root.mkdir(parents=True, exist_ok=True)

    (root / "archive").mkdir(exist_ok=True)
    (root / "map-elites").mkdir(exist_ok=True)
    (root / "best").mkdir(exist_ok=True)

    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2)
    )

    state = OuterLoopState(budget_remaining=config.budget)
    state_path = root / "state.json"
    state_path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2)
    )

    cache_path = root / "fitness_cache.json"
    if not cache_path.exists():
        cache_path.write_text("{}")

    trajectory_path = root / "trajectory.jsonl"
    if not trajectory_path.exists():
        trajectory_path.touch()

    log.info("outer_loop_filesystem_initialized", root=str(root))
    return root


def save_generation(
    project_path: Path,
    generation: int,
    summary: GenerationSummary,
    population: Population,
) -> None:
    """Save generation artifacts to .factory/outer_loop/archive/generation-NNN/."""
    root = project_path / ".factory" / "outer_loop"
    gen_dir = root / "archive" / f"generation-{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    summary_path = gen_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2)
    )

    if summary.hyperparameters:
        hp_path = gen_dir / "hyperparameters.json"
        hp_path.write_text(
            json.dumps(summary.hyperparameters.model_dump(mode="json"), indent=2)
        )

    for i, ind in enumerate(population.individuals):
        var_dir = gen_dir / f"variant-{i:02d}"
        var_dir.mkdir(exist_ok=True)
        (var_dir / "workflow.json").write_text(
            json.dumps(ind.workflow_data, indent=2, default=str)
        )
        if ind.mutation_record:
            (var_dir / "mutation.json").write_text(
                json.dumps(ind.mutation_record.model_dump(mode="json"), indent=2)
            )
        (var_dir / "scores.json").write_text(
            json.dumps({"score": ind.score, "cost_usd": ind.cost_usd}, indent=2)
        )

    traj_path = root / "trajectory.jsonl"
    with traj_path.open("a") as f:
        entry = {
            "generation": generation,
            "best_score": summary.best_score,
            "mean_score": summary.mean_score,
            "diversity": summary.diversity,
            "novel_count": summary.novel_count,
        }
        f.write(json.dumps(entry) + "\n")


def save_checkpoint(
    project_path: Path,
    state: OuterLoopState,
) -> None:
    """Write OuterLoopState to .factory/outer_loop/state.json."""
    state_path = project_path / ".factory" / "outer_loop" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2)
    )
    log.info("outer_loop_checkpoint_saved", generation=state.generation)


def load_checkpoint(project_path: Path) -> OuterLoopState | None:
    """Load OuterLoopState from .factory/outer_loop/state.json if it exists."""
    state_path = project_path / ".factory" / "outer_loop" / "state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
        return OuterLoopState.model_validate(data, strict=False)
    except Exception:
        log.warning("outer_loop_checkpoint_load_failed", exc_info=True)
        return None


def load_config(project_path: Path) -> SwarmConfig | None:
    """Load SwarmConfig from .factory/outer_loop/config.json if it exists."""
    config_path = project_path / ".factory" / "outer_loop" / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
        return SwarmConfig.model_validate(data, strict=False)
    except Exception:
        log.warning("outer_loop_config_load_failed", exc_info=True)
        return None


def save_map_elites(project_path: Path, archive: MAPElitesArchive) -> None:
    """Persist the MAP-Elites grid to .factory/outer_loop/map-elites/grid.json."""
    grid_path = project_path / ".factory" / "outer_loop" / "map-elites" / "grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)

    grid_data: dict[str, object] = {}
    for key, ind in archive._grid.items():
        grid_data[str(key)] = ind.model_dump(mode="json")

    grid_path.write_text(json.dumps(grid_data, indent=2, default=str))


def save_best(
    project_path: Path,
    result: OuterLoopResult,
) -> None:
    """Write the best workflow and audit results to .factory/outer_loop/best/."""
    best_dir = project_path / ".factory" / "outer_loop" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    (best_dir / "workflow.json").write_text(
        json.dumps(result.best_workflow_data, indent=2, default=str)
    )

    if (result.val_score is not None and result.val_score > 0) or result.overfit_flag:
        audit = {
            "val_score": result.val_score,
            "overfit_flag": result.overfit_flag,
            "best_score": result.best_score,
        }
        (best_dir / "holdout_audit.json").write_text(
            json.dumps(audit, indent=2)
        )

    # Write run_report.json with train/val summary
    run_report = {
        "train_score": result.best_score,
        "val_score": result.val_score,
        "overfit_flag": result.overfit_flag,
        "total_candidates_evaluated": result.total_candidates_evaluated,
        "generations_completed": result.generations_completed,
        "convergence_reason": result.convergence_reason,
        "total_cost_usd": result.total_cost_usd,
    }
    (best_dir / "run_report.json").write_text(
        json.dumps(run_report, indent=2)
    )


def export_best_workflow(
    project_path: Path,
    best_workflow_data: dict[str, object],
    benchmark_name: str,
) -> Path:
    """Export the best workflow as a portable .factory/workflows/<benchmark>-evolved.py.

    Returns the path to the exported file.
    """
    workflows_dir = project_path / ".factory" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    export_path = workflows_dir / f"{benchmark_name}-evolved.py"

    wf = Workflow.from_dict(best_workflow_data)  # type: ignore[arg-type]

    wf_json = json.dumps(wf.to_dict(), indent=4, default=str)
    content = (
        f'"""Auto-evolved workflow for {benchmark_name}."""\n'
        f"\n"
        f"from factory.workflow.primitives import (\n"
        f"    AgentNode,\n"
        f"    AgentRole,\n"
        f"    Edge,\n"
        f"    FnNode,\n"
        f"    GateNode,\n"
        f"    Study,\n"
        f"    VerdictType,\n"
        f"    Workflow,\n"
        f")\n"
        f"\n"
        f"\n"
        f"meta = {{\n"
        f'    "name": "{benchmark_name}-evolved",\n'
        f'    "description": "Evolved workflow for {benchmark_name} benchmark",\n'
        f"}}\n"
        f"\n"
        f"\n"
        f"def workflow() -> Workflow:\n"
        f'    """Evolved workflow for {benchmark_name}."""\n'
        f"    return Workflow.from_dict({wf_json})\n"
    )

    export_path.write_text(content)

    also_best = project_path / ".factory" / "outer_loop" / "best" / "workflow.py"
    also_best.parent.mkdir(parents=True, exist_ok=True)
    also_best.write_text(export_path.read_text())

    log.info("best_workflow_exported", path=str(export_path))
    return export_path
