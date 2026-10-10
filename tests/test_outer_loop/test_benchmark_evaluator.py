"""Tests for BenchmarkEvaluator and partial credit scoring."""

from __future__ import annotations

import json
from pathlib import Path

from factory.outer_loop.benchmark_evaluator import (
    BenchmarkEvaluator,
    parse_pytest_stdout,
)


class TestBenchmarkEvaluator:
    def test_parse_pytest_json_report(self, tmp_path: Path) -> None:
        report = {
            "tests": [
                {"nodeid": "test_a", "outcome": "passed"},
                {"nodeid": "test_b", "outcome": "passed"},
                {"nodeid": "test_c", "outcome": "failed"},
                {"nodeid": "test_d", "outcome": "passed"},
            ]
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.valid
        assert result.score == 0.75
        assert result.metrics["tests_passed"] == 3.0
        assert result.metrics["tests_total"] == 4.0
        assert result.metrics["pass_rate"] == 0.75

    def test_parse_all_passing(self, tmp_path: Path) -> None:
        report = {"tests": [{"outcome": "passed"}, {"outcome": "passed"}]}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.score == 1.0

    def test_parse_all_failing(self, tmp_path: Path) -> None:
        report = {"tests": [{"outcome": "failed"}, {"outcome": "failed"}]}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.score == 0.0

    def test_parse_empty_tests(self, tmp_path: Path) -> None:
        report = {"tests": []}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.score == 0.0

    def test_parse_summary_format(self, tmp_path: Path) -> None:
        report = {"summary": {"passed": 5, "total": 8}}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.score == 5 / 8

    def test_parse_factory_eval_format(self, tmp_path: Path) -> None:
        report = {"results": [{"score": 0.8}, {"score": 0.6}]}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert result.score == 0.7

    def test_parse_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")

        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(path)

        assert not result.valid
        assert result.score == 0.0

    def test_parse_missing_file(self) -> None:
        evaluator = BenchmarkEvaluator()
        result = evaluator.parse(Path("/nonexistent/report.json"))

        assert not result.valid
        assert result.score == 0.0

    def test_parse_many(self, tmp_path: Path) -> None:
        for i, scores in enumerate([(3, 4), (5, 8), (1, 2)]):
            passed, total = scores
            report = {"summary": {"passed": passed, "total": total}}
            (tmp_path / f"report_{i}.json").write_text(json.dumps(report))

        evaluator = BenchmarkEvaluator()
        paths = [tmp_path / f"report_{i}.json" for i in range(3)]
        result = evaluator.parse_many(paths)

        assert result.score == 0.75

    def test_get_info(self) -> None:
        evaluator = BenchmarkEvaluator()
        info = evaluator.get_info()
        assert info["benchmark"] == "featurebench"
        assert info["scoring"] == "partial_credit"


class TestParsePytestStdout:
    def test_basic_output(self) -> None:
        stdout = "====== 5 passed, 3 failed in 10.5s ======"
        metrics = parse_pytest_stdout(stdout)
        assert metrics["tests_passed"] == 5.0
        assert metrics["tests_total"] == 8.0
        assert metrics["pass_rate"] == 5 / 8

    def test_all_passed(self) -> None:
        stdout = "====== 10 passed in 5.0s ======"
        metrics = parse_pytest_stdout(stdout)
        assert metrics["tests_passed"] == 10.0
        assert metrics["tests_total"] == 10.0
        assert metrics["pass_rate"] == 1.0

    def test_with_errors(self) -> None:
        stdout = "====== 3 passed, 2 failed, 1 error in 8.0s ======"
        metrics = parse_pytest_stdout(stdout)
        assert metrics["tests_passed"] == 3.0
        assert metrics["tests_total"] == 6.0
        assert metrics["pass_rate"] == 0.5

    def test_empty_output(self) -> None:
        metrics = parse_pytest_stdout("")
        assert metrics["pass_rate"] == 0.0
