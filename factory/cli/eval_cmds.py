"""CLI eval_cmds commands."""
from __future__ import annotations

import argparse
import json
import subprocess
import structlog
import sys
from pathlib import Path

from factory.cli._helpers import _emit_cli_event, _read_target_branch, _run

log = structlog.get_logger()


def _compute_targeted_test_paths(project_path: Path) -> list[str] | None:
    """Compute changed files and resolve to targeted test paths via the import graph."""
    target_branch = _read_target_branch(project_path)
    try:
        result = subprocess.run(
            ["git", "merge-base", "HEAD", target_branch],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.info("targeted.skip", reason="no_merge_base")
            return None
        baseline = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    try:
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", f"{baseline}..HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if diff_result.returncode != 0:
            return None
        changed_files = [f for f in diff_result.stdout.strip().splitlines() if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if not changed_files:
        return None

    from factory.graph import find_dependent_tests

    dependent = find_dependent_tests(project_path, changed_files)
    if dependent is None:
        log.info("targeted.fallback", reason="find_dependent_tests_returned_none")
        return None

    log.info("targeted.active", test_count=len(dependent), changed_files=len(changed_files))
    return sorted(dependent)


def cmd_eval(args: argparse.Namespace) -> int:
    from factory.eval.runner import run_eval
    from factory.store import ExperimentStore

    project_path = Path(args.path)
    store = ExperimentStore(project_path)
    config = _run(store.read_config())
    skip_project_eval = getattr(args, "skip_project_eval", False)

    test_paths: list[str] | None = None
    targeted = getattr(args, "targeted", False)
    if targeted:
        test_paths = _compute_targeted_test_paths(project_path)
        if test_paths is not None:
            log.info("eval.mode", mode="targeted", test_paths=len(test_paths))
        else:
            log.info("eval.mode", mode="full", reason="targeted_fallback")

    _emit_cli_event(project_path, "eval.started", {"command": config.eval_command})
    score = _run(run_eval(
        config.eval_command, project_path, config.eval_threshold,
        project_eval=config.project_eval or None,
        eval_weights=config.eval_weights,
        skip_project_eval=skip_project_eval,
        test_timeout=config.test_timeout,
        test_paths=test_paths,
    ))
    _emit_cli_event(project_path, "eval.completed", {
        "composite": score.total,
        "passed": score.passed,
        "dimensions": len(score.results),
    })
    print(json.dumps(score.model_dump(), indent=2, default=str))
    return 0 if score.passed else 1


def cmd_guard(args: argparse.Namespace) -> int:
    from factory.eval.guards import check_all

    project_path = Path(args.path)

    # Optionally load scope and fixed surfaces from factory config
    scope = None
    fixed_surfaces = None
    if args.check_scope or args.check_surfaces:
        from factory.store import ExperimentStore
        store = ExperimentStore(project_path)
        config = _run(store.read_config())
        if args.check_scope:
            scope = config.scope
        if args.check_surfaces:
            fixed_surfaces = config.fixed_surfaces

    violations = check_all(
        project_path, args.baseline, allowed_scope=scope, fixed_surfaces=fixed_surfaces,
    )
    _emit_cli_event(project_path, "guard.completed", {
        "violations": len(violations),
        "clean": len(violations) == 0,
    })
    if violations:
        for v in violations:
            print(f"VIOLATION: {v}")
        return 1
    print("clean")
    return 0


def cmd_precheck(args: argparse.Namespace) -> int:
    """Run hard precheck gate before keep/revert decision."""
    from factory.precheck import run_precheck
    from factory.store import ExperimentStore

    project_path = Path(args.path).resolve()
    store = ExperimentStore(project_path)
    config = _run(store.read_config())

    # Load history as dicts for anti-pattern matching
    records = _run(store.load_history())
    history = [
        {
            "id": r.id,
            "hypothesis": r.hypothesis,
            "verdict": r.verdict,
            "delta": r.delta,
        }
        for r in records
    ]

    result = run_precheck(
        score_before=args.score_before,
        score_after=args.score_after,
        threshold=config.eval_threshold,
        hypothesis=args.hypothesis or "",
        history=history,
        project_path=project_path,
        baseline_sha=args.baseline,
        allowed_scope=config.scope if args.baseline else None,
        similarity_threshold=args.similarity_threshold,
        fixed_surfaces=config.fixed_surfaces if config.fixed_surfaces else None,
    )

    # Output as JSON for machine consumption
    output = {
        "passed": result.passed,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in result.checks
        ],
        "blocking_failures": result.blocking_failures,
    }
    print(json.dumps(output, indent=2))

    _emit_cli_event(project_path, "precheck.completed", {
        "passed": result.passed,
        "failures": result.blocking_failures,
    })

    return 0 if result.passed else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """Fetch stored eval baseline for a commit from the eval-data branch."""
    from factory.baseline import fetch_baseline

    project_path = Path(args.path).resolve()

    commit = getattr(args, "commit", None)
    if not commit:
        result = subprocess.run(
            ["git", "merge-base", "HEAD", _read_target_branch(project_path)],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Error: could not determine merge-base commit.", file=sys.stderr)
            return 1
        commit = result.stdout.strip()

    baseline = fetch_baseline(project_path, commit_sha=commit)
    if baseline is None:
        print(f"No baseline found for commit {commit[:12]}", file=sys.stderr)
        return 1

    print(json.dumps(baseline, indent=2, default=str))
    return 0


def cmd_adversarial_state(args: argparse.Namespace) -> int:
    """Inspect or reset adversarial eval loop state."""
    from factory.adversarial import (
        format_adversarial_state,
        load_adversarial_state,
        reset_adversarial_state,
    )

    project_path = Path(args.path).resolve()

    if args.reset:
        reset_adversarial_state(project_path)
        print("Adversarial state reset.")
        return 0

    state = load_adversarial_state(project_path)
    print(format_adversarial_state(state))
    return 0

