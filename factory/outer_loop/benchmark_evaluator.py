"""Benchmark evaluator — implements the Evaluator protocol with partial credit scoring.

Parses pytest-json-report output for per-test pass/fail to produce a fraction
score (e.g. 5/8 = 0.625) instead of binary 0/1. This is the gradient signal
that enables evolutionary search to optimize incrementally.

The canonical implementation now lives in factory.outer_loop.evaluators.pytest_evaluator.
This module keeps BenchmarkEvaluator as a backward-compatible class and
parse_pytest_stdout() as a standalone utility.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from factory.inner_loop import EvalResult

log = structlog.get_logger()


class BenchmarkEvaluator:
    """Parses benchmark pytest output for partial credit scoring.

    Looks for pytest-json-report output files (report.json) or factory
    eval artifacts. Computes score as fraction of tests passing.
    """

    def __init__(self, benchmark: str = "featurebench") -> None:
        self.benchmark = benchmark

    def parse(self, artifact_path: Path) -> EvalResult:
        try:
            data = json.loads(Path(artifact_path).read_text())
        except (json.JSONDecodeError, OSError):
            return EvalResult(score=0.0, valid=False)

        score, metrics = self._extract_partial_credit(data)
        return EvalResult(
            score=score,
            metrics=metrics,
            valid=True,
            artifacts=[str(artifact_path)],
        )

    def parse_many(self, artifact_paths: list[Path]) -> EvalResult:
        best = EvalResult(score=0.0, valid=False)
        for p in artifact_paths:
            result = self.parse(p)
            if result.score > best.score:
                best = result
        return best

    def get_info(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "scoring": "partial_credit",
            "metrics": ["tests_passed", "tests_total", "pass_rate"],
        }

    def _extract_partial_credit(self, data: dict) -> tuple[float, dict[str, float]]:
        """Extract partial credit from pytest-json-report or factory eval output."""
        if "tests" in data:
            return self._parse_pytest_json_report(data)

        if "results" in data:
            return self._parse_factory_eval(data)

        if "summary" in data:
            summary = data["summary"]
            passed = summary.get("passed", 0)
            total = summary.get("total", 0)
            if total > 0:
                score = passed / total
                return score, {
                    "tests_passed": float(passed),
                    "tests_total": float(total),
                    "pass_rate": score,
                }

        score = float(data.get("score", data.get("combined_score", 0.0)))
        return score, {"raw_score": score}

    def _parse_pytest_json_report(self, data: dict) -> tuple[float, dict[str, float]]:
        """Parse pytest-json-report format: {"tests": [{"outcome": "passed"}, ...]}"""
        tests = data.get("tests", [])
        if not tests:
            return 0.0, {"tests_passed": 0.0, "tests_total": 0.0, "pass_rate": 0.0}

        passed = sum(1 for t in tests if t.get("outcome") == "passed")
        total = len(tests)
        score = passed / total if total > 0 else 0.0

        return score, {
            "tests_passed": float(passed),
            "tests_total": float(total),
            "pass_rate": score,
        }

    def _parse_factory_eval(self, data: dict) -> tuple[float, dict[str, float]]:
        """Parse factory eval format: {"results": [{"score": 0.8, ...}]}"""
        results = data.get("results", [])
        if not results:
            return 0.0, {}

        scores = [float(r.get("score", 0.0)) for r in results if "score" in r]
        if not scores:
            return 0.0, {}

        avg = sum(scores) / len(scores)
        return avg, {
            "avg_score": avg,
            "max_score": max(scores),
            "min_score": min(scores),
            "num_results": float(len(scores)),
        }


def parse_pytest_stdout(stdout: str) -> dict[str, float]:
    """Parse pytest stdout for pass/fail counts when no JSON report is available.

    Looks for the summary line: "X passed, Y failed, Z errors" or similar.
    Returns metrics dict with tests_passed, tests_total, pass_rate.
    """
    import re

    metrics: dict[str, float] = {"tests_passed": 0.0, "tests_total": 0.0, "pass_rate": 0.0}

    patterns = [
        (r"(\d+)\s+passed", "passed"),
        (r"(\d+)\s+failed", "failed"),
        (r"(\d+)\s+error", "errors"),
        (r"(\d+)\s+skipped", "skipped"),
    ]

    counts: dict[str, int] = {}
    for pattern, key in patterns:
        match = re.search(pattern, stdout)
        if match:
            counts[key] = int(match.group(1))

    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errors = counts.get("errors", 0)
    total = passed + failed + errors

    if total > 0:
        metrics["tests_passed"] = float(passed)
        metrics["tests_total"] = float(total)
        metrics["pass_rate"] = passed / total

    return metrics
