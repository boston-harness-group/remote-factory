"""Shared assertion helper for outer-loop trust verification.

``assert_run_trustworthy(result, run_dir)`` reads artifacts from
``.factory/outer_loop/`` and ``.factory/runs/`` and raises ``AssertionError``
if any trust invariant is violated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assert_run_trustworthy(
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    """Verify outer-loop run artifacts are consistent and trustworthy.

    Checks:
    1. best/workflow.json exists and is valid JSON
    2. best/run_report.json exists with matching scores
    3. At least one generation-NNN/ directory exists with summary.json
    4. trajectory.jsonl has at least one entry
    5. If items.jsonl exists in any run, every item has required fields
    6. If sha256.json exists, file hashes match actual files on disk

    Returns a diagnostics dict with check results.
    """
    ol_dir = run_dir / ".factory" / "outer_loop"
    diag: dict[str, Any] = {"checks_passed": 0, "checks_failed": 0, "details": []}

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            diag["checks_passed"] += 1
            diag["details"].append({"check": name, "passed": True})
        else:
            diag["checks_failed"] += 1
            diag["details"].append({"check": name, "passed": False, "msg": msg})
            raise AssertionError(f"Trust check '{name}' failed: {msg}")

    # 1. best/workflow.json
    best_wf_path = ol_dir / "best" / "workflow.json"
    _check(
        "best_workflow_exists",
        best_wf_path.exists(),
        f"Missing {best_wf_path}",
    )
    if best_wf_path.exists():
        try:
            wf_data = json.loads(best_wf_path.read_text())
            _check("best_workflow_valid_json", isinstance(wf_data, dict), "Not a JSON dict")
        except json.JSONDecodeError as e:
            _check("best_workflow_valid_json", False, str(e))

    # 2. best/run_report.json
    report_path = ol_dir / "best" / "run_report.json"
    _check("run_report_exists", report_path.exists(), f"Missing {report_path}")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _check(
            "run_report_train_score",
            "train_score" in report,
            "Missing train_score in run_report",
        )
        if hasattr(result, "best_score"):
            _check(
                "run_report_score_matches",
                abs(report.get("train_score", 0) - result.best_score) < 0.001,
                f"Report train_score={report.get('train_score')} != "
                f"result.best_score={result.best_score}",
            )

    # 3. At least one generation directory
    archive_dir = ol_dir / "archive"
    if archive_dir.exists():
        gen_dirs = sorted(archive_dir.glob("generation-*"))
        _check("has_generation_dir", len(gen_dirs) > 0, "No generation dirs found")
        if gen_dirs:
            summary_path = gen_dirs[0] / "summary.json"
            _check(
                "first_gen_has_summary",
                summary_path.exists(),
                f"Missing {summary_path}",
            )
    else:
        _check("has_archive_dir", False, f"Missing {archive_dir}")

    # 4. trajectory.jsonl
    traj_path = ol_dir / "trajectory.jsonl"
    if traj_path.exists():
        lines = [l.strip() for l in traj_path.read_text().splitlines() if l.strip()]
        _check("trajectory_has_entries", len(lines) > 0, "trajectory.jsonl is empty")
    else:
        _check("trajectory_exists", False, f"Missing {traj_path}")

    # 5. items.jsonl validation in any run
    runs_dir = run_dir / ".factory" / "runs"
    if runs_dir.exists():
        for items_path in runs_dir.rglob("items.jsonl"):
            for line_no, line in enumerate(items_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                for field in ("item_id", "status", "score"):
                    _check(
                        f"items_{items_path.parent.name}_line{line_no}_{field}",
                        field in item,
                        f"Missing '{field}' in {items_path}:{line_no}",
                    )

    # 6. sha256 verification
    if runs_dir.exists():
        for sha_path in runs_dir.rglob("sha256.json"):
            hashes = json.loads(sha_path.read_text())
            item_dir = sha_path.parent
            for rel_path, expected_hash in hashes.items():
                file_path = item_dir / rel_path
                if file_path.exists():
                    import hashlib
                    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    _check(
                        f"sha256_{item_dir.name}_{rel_path}",
                        actual == expected_hash,
                        f"Hash mismatch for {rel_path}: "
                        f"expected={expected_hash[:16]}... actual={actual[:16]}...",
                    )

    return diag
