"""EvalRunner — compute mandatory dimensions and merge with project-specific evals.

The factory's eval system has mandatory dimensions that apply to every project:
  - 6 hygiene dimensions (tests, lint, type_check, coverage, config_parser, architecture)
  - 5 growth dimensions (capability_surface, experiment_diversity, observability,
    research_grounding, factory_effectiveness)

Projects can ADD dimensions via eval/score.py and via project_eval in factory.md.
The mandatory dimensions are computed by the factory itself, not by per-project scripts.

Weight distribution (configurable via factory.md ## Eval Weights):
  - No project eval: 50% hygiene + 50% growth (default)
  - With project eval: configurable, default 30% hygiene + 20% growth + 50% project
"""

import asyncio
import json
import os
from pathlib import Path

from factory.eval.growth import compute_growth_results
from factory.eval.hygiene import compute_hygiene_results
from factory.eval.scorer import compute_composite
from factory.models import (
    CompositeScore,
    EvalResult,
    EvalWeights,
    ProjectEvalDimension,
    TierWeights,
)


def _effective_weights(
    eval_weights: EvalWeights,
    has_custom_project: bool,
) -> tuple[float, float, float]:
    """Determine effective hygiene/growth/project weight split."""
    if not has_custom_project:
        return 0.50, 0.50, 0.0

    if eval_weights.project > 0:
        total = eval_weights.hygiene + eval_weights.growth + eval_weights.project
        return (
            eval_weights.hygiene / total,
            eval_weights.growth / total,
            eval_weights.project / total,
        )

    # Has project eval but user didn't set weights → auto-distribute
    return 0.30, 0.20, 0.50


def _normalize_tier(
    results: list[EvalResult],
    target_weight: float,
    weight_overrides: dict[str, float] | None = None,
) -> list[EvalResult]:
    """Normalize a list of EvalResults so their weights sum to target_weight.

    If weight_overrides is provided, matching dimension weights are replaced
    before normalization (sparse override).
    """
    if not results or target_weight <= 0:
        return []
    if weight_overrides:
        results = [
            EvalResult(
                name=r.name,
                score=r.score,
                weight=weight_overrides.get(r.name, r.weight),
                passed=r.passed,
                details=r.details,
            )
            for r in results
        ]
    weight_sum = sum(r.weight for r in results)
    if weight_sum <= 0:
        return results
    return [
        EvalResult(
            name=r.name,
            score=r.score,
            weight=(r.weight / weight_sum) * target_weight,
            passed=r.passed,
            details=r.details,
        )
        for r in results
    ]


def _merge_all(
    hygiene_results: list[EvalResult],
    project_results: list[EvalResult],
    growth_results: list[EvalResult],
    custom_project_results: list[EvalResult] | None = None,
    eval_weights: EvalWeights | None = None,
    hygiene_weight_overrides: dict[str, float] | None = None,
    growth_weight_overrides: dict[str, float] | None = None,
) -> list[EvalResult]:
    """Merge mandatory hygiene + project additions + mandatory growth + custom project eval.

    Weight distribution (three tiers):
      - Hygiene (mandatory 6 + eval/score.py additions): configurable (default 50%)
      - Growth (mandatory 5+): configurable (default 50%)
      - Project eval (user-defined in factory.md): configurable (default 0%, auto 50% when present)

    When custom_project_results is non-empty, weights shift to accommodate project eval.
    Within-tier weight overrides are applied before normalization.
    """
    mandatory_names = {r.name for r in hygiene_results} | {r.name for r in growth_results}
    additional = [r for r in project_results if r.name not in mandatory_names]
    all_hygiene = list(hygiene_results) + additional

    custom = custom_project_results or []
    weights = eval_weights or EvalWeights()
    h_w, g_w, p_w = _effective_weights(weights, bool(custom))

    normalized_hygiene = _normalize_tier(all_hygiene, h_w, hygiene_weight_overrides)
    normalized_growth = _normalize_tier(growth_results, g_w, growth_weight_overrides)
    normalized_project = _normalize_tier(custom, p_w)

    return normalized_hygiene + normalized_growth + normalized_project


async def _run_project_eval(
    eval_command: str,
    project_path: Path,
    timeout: float = 300.0,
) -> list[EvalResult]:
    """Run the project's eval/score.py (if it exists) and return additional results.

    Returns an empty list if the command fails or returns no results.
    These are project-specific ADDITIONS to the mandatory 11 dimensions.
    """
    parts = eval_command.split()
    if not parts:
        return []

    # Clean environment
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    from factory.worktree import is_factory_venv
    if is_factory_venv(project_path):
        venv_path = project_path / ".venv"
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = str(venv_path / "bin") + os.pathsep + env.get("PATH", "")

    try:
        proc = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        await proc.wait()  # type: ignore[union-attr]
        return []
    except FileNotFoundError:
        return []

    if proc.returncode != 0:
        return []

    stdout = stdout_bytes.decode()
    try:
        data = json.loads(stdout)
        return [EvalResult(**r) for r in data["results"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


async def _run_single_project_dimension(
    dim: ProjectEvalDimension,
    project_path: Path,
) -> EvalResult:
    """Run a single user-defined project eval dimension."""
    parts = dim.command.split()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    from factory.worktree import is_factory_venv
    if is_factory_venv(project_path):
        venv_path = project_path / ".venv"
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = str(venv_path / "bin") + os.pathsep + env.get("PATH", "")

    try:
        proc = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=dim.timeout)
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        await proc.wait()  # type: ignore[union-attr]
        return EvalResult(
            name=dim.name,
            score=0.0,
            weight=dim.weight,
            passed=False,
            details=f"Timed out after {dim.timeout}s",
        )
    except FileNotFoundError:
        return EvalResult(
            name=dim.name,
            score=0.0,
            weight=dim.weight,
            passed=False,
            details=f"Command not found: {parts[0]}",
        )

    stdout = stdout_bytes.decode()
    stderr = stderr_bytes.decode()

    if dim.parse == "json":
        try:
            data = json.loads(stdout)
            raw_score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, raw_score))
            return EvalResult(
                name=dim.name,
                score=score,
                weight=dim.weight,
                passed=score >= 0.5,
                details=str(data.get("details", stdout[:500])),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return EvalResult(
                name=dim.name,
                score=0.0,
                weight=dim.weight,
                passed=False,
                details=f"Invalid JSON: {stdout[:200]}",
            )

    # exit_code parse mode
    passed = proc.returncode == 0
    return EvalResult(
        name=dim.name,
        score=1.0 if passed else 0.0,
        weight=dim.weight,
        passed=passed,
        details=(stdout or stderr).strip()[-500:],
    )


async def _run_custom_project_eval(
    dimensions: list[ProjectEvalDimension],
    project_path: Path,
) -> list[EvalResult]:
    """Run all user-defined project eval dimensions from factory.md."""
    results = []
    for dim in dimensions:
        result = await _run_single_project_dimension(dim, project_path)
        results.append(result)
    return results


async def run_eval(
    eval_command: str,
    project_path: Path,
    threshold: float,
    timeout: float = 300.0,
    project_eval: list[ProjectEvalDimension] | None = None,
    eval_weights: EvalWeights | None = None,
    skip_project_eval: bool = False,
    hygiene_weights: TierWeights | None = None,
    growth_weights: TierWeights | None = None,
    eval_spec: list[str] | None = None,
    test_timeout: int = 600,
    test_paths: list[str] | None = None,
) -> CompositeScore:
    """Compute mandatory dimensions + project-specific additions + custom project eval.

    1. Compute 6 mandatory hygiene dimensions (auto-detect project tooling)
    2. Run project's eval/score.py for additional dimensions (optional)
    3. Compute 6 mandatory growth dimensions
    4. Run custom project eval dimensions from factory.md (if configured)
    4b. Auto-promote executable eval_spec items to project eval
    5. Merge all with configurable weight split (with optional within-tier overrides)
    6. Return composite score
    """
    # Step 1: Mandatory hygiene (always runs)
    hygiene_dicts = compute_hygiene_results(
        project_path, test_timeout=test_timeout, test_paths=test_paths,
    )
    hygiene_results = [EvalResult(**r) for r in hygiene_dicts]

    # Step 2: Project-specific additions (optional, additive only)
    project_results = await _run_project_eval(eval_command, project_path, timeout)

    # Step 3: Mandatory growth (always runs)
    growth_dicts = compute_growth_results(project_path)
    growth_results = [EvalResult(**r) for r in growth_dicts]

    # Step 4: Custom project eval (user-defined in factory.md)
    custom_results: list[EvalResult] = []
    if project_eval and not skip_project_eval:
        custom_results = await _run_custom_project_eval(project_eval, project_path)

    # Step 4b: Auto-promote executable eval_spec items to project eval
    if eval_spec and not skip_project_eval:
        from factory.discovery.eval_spec import generate_project_eval_from_spec

        auto_promoted = generate_project_eval_from_spec(eval_spec, project_path)
        if auto_promoted:
            auto_results = await _run_custom_project_eval(auto_promoted, project_path)
            custom_results.extend(auto_results)

    # Convert TierWeights to sparse override dicts
    h_overrides = (
        {k: v for k, v in hygiene_weights.model_dump().items() if v is not None}
        if hygiene_weights
        else None
    )
    g_overrides = (
        {k: v for k, v in growth_weights.model_dump().items() if v is not None}
        if growth_weights
        else None
    )

    # Step 5: Merge all dimensions with weight split
    merged = _merge_all(
        hygiene_results,
        project_results,
        growth_results,
        custom_project_results=custom_results,
        eval_weights=eval_weights,
        hygiene_weight_overrides=h_overrides or None,
        growth_weight_overrides=g_overrides or None,
    )

    # Step 6: Compute composite
    score = compute_composite(merged, guard_violations=[], threshold=threshold)

    # Step 7: Save results to .factory/last_eval.json for dashboard consumption
    last_eval_path = project_path / ".factory" / "last_eval.json"
    if last_eval_path.parent.exists():
        try:
            last_eval_path.write_text(json.dumps(score.model_dump(), indent=2))
        except OSError:
            pass

    return score
