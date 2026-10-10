"""Backward-compat shim. Canonical: factory.outer_loop.benchmark_evaluator."""
from factory.outer_loop.benchmark_evaluator import BenchmarkEvaluator, parse_pytest_stdout

FeatureBenchEvaluator = BenchmarkEvaluator

__all__ = ["FeatureBenchEvaluator", "parse_pytest_stdout"]
