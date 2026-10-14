"""CLI store commands."""
from __future__ import annotations

import argparse
import json
import structlog
import sys
from datetime import datetime
from pathlib import Path

from factory.cli._helpers import _detect_pr_number, _emit_cli_event, _run

log = structlog.get_logger()

def cmd_begin(args: argparse.Namespace) -> int:
    from factory.store import ExperimentStore

    project_path = Path(args.path)
    store = ExperimentStore(project_path)
    exp_id = _run(store.begin(args.hypothesis))
    _emit_cli_event(project_path, "experiment.begin", {
        "exp_id": exp_id,
        "hypothesis": args.hypothesis[:200],
    })
    print(exp_id)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    from factory.precheck import run_precheck
    from factory.store import ExperimentStore
    from factory.models import ExperimentRecord, FactoryConfig

    project_path = Path(args.path)
    store = ExperimentStore(project_path)
    score_before = getattr(args, "score_before", None)
    score_after = getattr(args, "score_after", None)
    verdict = args.verdict
    notes = args.notes or ""

    force = getattr(args, "force", False)

    if verdict == "keep" and not force:
        config_path = project_path / ".factory" / "config.json"
        if config_path.exists():
            config = FactoryConfig(**json.loads(config_path.read_text()))
            history = _run(store.load_history())
            history_dicts = [r.model_dump() for r in history]

            precheck_result = run_precheck(
                score_before=score_before,
                score_after=score_after,
                threshold=config.eval_threshold,
                hypothesis=args.hypothesis or "",
                history=history_dicts,
                project_path=project_path,
                hard_constraints=config.hard_constraints,
                exp_id=args.id,
            )

            if not precheck_result.passed:
                verdict = "revert"
                failure_detail = "; ".join(precheck_result.blocking_failures)
                notes = f"[OVERRIDDEN by finalize gate] precheck failed: {failure_detail}. {notes}"
                _emit_cli_event(project_path, "verdict.overridden", {
                    "exp_id": args.id,
                    "original_verdict": "keep",
                    "new_verdict": "revert",
                    "reason": failure_detail,
                })
                print(f"Finalize gate: precheck FAILED — overriding keep to revert ({failure_detail})")

    if verdict == "keep" and force:
        _emit_cli_event(project_path, "verdict.force_kept", {
            "exp_id": args.id,
        })
        print("Finalize gate: precheck SKIPPED (--force)")

    pr_number = args.pr
    if pr_number is None:
        pr_number = _detect_pr_number(project_path)

    cost = args.cost
    if cost is None:
        from factory.events import load_events, sum_agent_usage
        exp_events = load_events(project_path)
        exp_start = None
        for ev in reversed(exp_events):
            if ev.get("type") == "experiment.begin":
                ts_str = ev.get("timestamp")
                if ts_str:
                    exp_start = datetime.fromisoformat(ts_str)
                    break
        cost = sum_agent_usage(project_path, since=exp_start).get("total_cost_usd") or None

    record = ExperimentRecord(
        id=args.id,
        timestamp=datetime.now(),
        hypothesis=args.hypothesis or "",
        change_summary=args.summary or "",
        issue_number=args.issue,
        pr_number=pr_number,
        score_before=score_before,
        score_after=score_after,
        delta=None,
        verdict=verdict,
        cost_usd=cost,
        notes=notes,
    )
    _run(store.finalize(args.id, record))
    delta = None
    if score_before is not None and score_after is not None:
        delta = round(score_after - score_before, 6)
    _emit_cli_event(project_path, "experiment.finalize", {
        "exp_id": args.id,
        "verdict": verdict,
        "hypothesis": (args.hypothesis or "")[:200],
        "pr_number": pr_number,
        "issue_number": args.issue,
        "score_before": score_before,
        "score_after": score_after,
        "delta": delta,
        "cost_usd": cost,
    })
    print(f"Finalized experiment {args.id} — verdict={verdict}")
    return 0


def cmd_message(args: argparse.Namespace) -> int:
    """Queue a message for the CEO agent."""
    from factory.messages import write_message

    project_path = Path(args.path).resolve()
    if not project_path.exists():
        print(f"Error: project path does not exist: {project_path}", file=sys.stderr)
        return 1
    if not (project_path / ".factory").exists():
        print(f"Error: not a factory project (no .factory/ directory): {project_path}", file=sys.stderr)
        return 1
    if not args.text or not args.text.strip():
        print("Error: message text must not be empty.", file=sys.stderr)
        return 1
    try:
        msg = write_message(project_path, args.text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Message queued (id={msg.id}). The CEO will see it at the start of the next cycle.")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from factory.store import ExperimentStore
    from factory.strategy import format_tiered_history

    store = ExperimentStore(Path(args.path))
    records = _run(store.load_history())
    if not records:
        print("No experiments recorded.")
        return 0

    record_dicts = [
        {
            "id": r.id,
            "hypothesis": r.hypothesis,
            "verdict": r.verdict,
            "delta": r.delta,
            "change_summary": r.change_summary,
            "cost_usd": r.cost_usd,
        }
        for r in records
    ]
    print(format_tiered_history(record_dicts))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from factory.state import detect_state
    from factory.store import ExperimentStore

    project_path = Path(args.path).resolve()
    state = detect_state(project_path)
    print(f"Project: {project_path}")
    print(f"State: {state.value}")

    if state.value == "has_factory":
        store = ExperimentStore(project_path)
        try:
            config = _run(store.read_config())
        except FileNotFoundError:
            config = None

        # Try to read latest eval score
        profile = _run(store.read_eval_profile())
        if profile:
            dims = ", ".join(d.name for d in profile.dimensions)
            print(f"Eval dimensions: {dims}")

        records = _run(store.load_history())
        if records:
            kept = sum(1 for r in records if r.verdict == "keep")
            reverted = sum(1 for r in records if r.verdict == "revert")
            total = len(records)
            print(f"Experiments: {total} total ({kept} kept, {reverted} reverted)")
            last = records[-1]
            print(f'Last experiment: #{last.id} — "{last.hypothesis}" ({last.verdict})')
            scores = [r.score_after for r in records if r.score_after is not None]
            if scores:
                print(f"Latest score: {scores[-1]:.3f}")
        else:
            print("Experiments: none")

        if config:
            print(f"Goal: {config.goal}")

    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Generate an end-of-session summary report."""
    from factory.summary import format_summary, generate_summary, save_summary

    project_path = Path(args.path).resolve()
    _emit_cli_event(project_path, "summary.started", {})
    summary = _run(generate_summary(project_path))
    output = format_summary(summary)
    _run(save_summary(project_path, summary))
    _emit_cli_event(project_path, "summary.completed", {
        "kept": len(summary.experiments_kept),
        "reverted": len(summary.experiments_reverted),
        "errored": len(summary.experiments_errored),
        "backlog": len(summary.backlog_remaining),
    })
    print(output)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a complete project snapshot as JSON to stdout."""
    from factory.store import ExperimentStore

    project_path = Path(args.path).resolve()
    factory_dir = project_path / ".factory"

    if not factory_dir.is_dir():
        print(f"Error: {factory_dir} does not exist. Run 'factory init' first.", file=sys.stderr)
        return 1

    store = ExperimentStore(project_path)

    # Read config
    try:
        config = _run(store.read_config())
        config_data = config.model_dump()
    except FileNotFoundError:
        config_data = None

    # Read eval profile
    eval_profile = _run(store.read_eval_profile())
    eval_profile_data = eval_profile.model_dump() if eval_profile else None

    # Read experiment history
    records = _run(store.load_history())
    experiments_data = [r.model_dump() for r in records]

    # Read strategy
    strategy = _run(store.read_strategy())

    # Assemble snapshot
    snapshot = {
        "config": config_data,
        "eval_profile": eval_profile_data,
        "experiments": experiments_data,
        "strategy": strategy,
        "meta": {
            "project_path": str(project_path),
            "timestamp": datetime.now().isoformat(),
            "factory_version": "0.1.0",
        },
    }

    json.dump(snapshot, sys.stdout, indent=2, default=str)
    print()  # trailing newline
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two experiments side-by-side."""
    from factory.analysis import compare_experiments, format_comparison
    from factory.store import ExperimentStore

    project_path = Path(args.path).resolve()
    store = ExperimentStore(project_path)
    comparison = compare_experiments(store, args.id_a, args.id_b)
    print(format_comparison(comparison))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Explain a single experiment with FEEC category and dimension breakdown."""
    from factory.analysis import explain_experiment, format_explanation
    from factory.store import ExperimentStore

    project_path = Path(args.path).resolve()
    store = ExperimentStore(project_path)
    explanation = explain_experiment(store, args.id)
    print(format_explanation(explanation))
    return 0

