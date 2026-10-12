"""End-to-end tests for multi-benchmark support.

Tests 3 benchmarks:
1. FeatureBench — backward compatibility (pytest format)
2. SWE-bench — exit_code format
3. Custom benchmark — user-defined test_command and test_format (json)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from factory.inner_loop import InnerLoop
from factory.outer_loop.benchmark_config import (
    BenchmarkConfig,
    list_benchmarks,
    load_benchmark_config,
)
from factory.outer_loop.evaluators import get_evaluator, list_formats
from factory.outer_loop.evaluators.exact_match import ExactMatchEvaluator
from factory.outer_loop.evaluators.exit_code import ExitCodeEvaluator
from factory.outer_loop.evaluators.json_evaluator import JSONEvaluator
from factory.outer_loop.evaluators.pytest_evaluator import PytestEvaluator
from factory.outer_loop.instance_prep import _needs_shell, prepare_instances, validate_instance
from factory.outer_loop.models import SwarmConfig


# ---------------------------------------------------------------------------
# Phase 1: Evaluator registry and format parsers
# ---------------------------------------------------------------------------


class TestEvaluatorRegistry:
    def test_list_formats_returns_all(self):
        formats = list_formats()
        assert "pytest" in formats
        assert "exit_code" in formats
        assert "json" in formats
        assert "exact_match" in formats

    def test_get_evaluator_pytest(self):
        ev = get_evaluator("pytest")
        assert isinstance(ev, PytestEvaluator)

    def test_get_evaluator_exit_code(self):
        ev = get_evaluator("exit_code")
        assert isinstance(ev, ExitCodeEvaluator)

    def test_get_evaluator_json(self):
        ev = get_evaluator("json", metric_path="pass_rate")
        assert isinstance(ev, JSONEvaluator)

    def test_get_evaluator_exact_match(self):
        ev = get_evaluator("exact_match", answer_extraction=r"\\boxed{(\d+)}")
        assert isinstance(ev, ExactMatchEvaluator)

    def test_get_evaluator_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown test_format"):
            get_evaluator("nonexistent")


class TestPytestEvaluator:
    def test_parse_pytest_json_report(self, tmp_path: Path):
        artifact = tmp_path / "report.json"
        artifact.write_text(json.dumps({
            "tests": [
                {"outcome": "passed"},
                {"outcome": "passed"},
                {"outcome": "failed"},
            ]
        }))
        ev = PytestEvaluator()
        result = ev.parse(artifact)
        assert result.valid
        assert abs(result.score - 2 / 3) < 0.01

    def test_parse_malformed(self, tmp_path: Path):
        artifact = tmp_path / "bad.json"
        artifact.write_text("not json at all")
        ev = PytestEvaluator()
        result = ev.parse(artifact)
        assert not result.valid
        assert result.score == 0.0

    def test_parse_missing_file(self, tmp_path: Path):
        ev = PytestEvaluator()
        result = ev.parse(tmp_path / "nonexistent.json")
        assert not result.valid


class TestExitCodeEvaluator:
    def test_parse_success(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"returncode": 0}))
        ev = ExitCodeEvaluator()
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 1.0

    def test_parse_failure(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"returncode": 1}))
        ev = ExitCodeEvaluator()
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.0

    def test_parse_missing_returncode(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"output": "something"}))
        ev = ExitCodeEvaluator()
        result = ev.parse(artifact)
        assert not result.valid

    def test_parse_many_mixed(self, tmp_path: Path):
        artifacts = []
        for i, rc in enumerate([0, 1, 0]):
            p = tmp_path / f"result_{i}.json"
            p.write_text(json.dumps({"returncode": rc}))
            artifacts.append(p)
        ev = ExitCodeEvaluator()
        result = ev.parse_many(artifacts)
        assert result.valid
        assert abs(result.score - 2 / 3) < 0.01


class TestJSONEvaluator:
    def test_parse_flat_metric(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"pass_rate": 0.85, "total": 20}))
        ev = JSONEvaluator(metric_path="pass_rate")
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.85

    def test_parse_nested_metric(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"stats": {"resolve_rate": 0.72}}))
        ev = JSONEvaluator(metric_path="stats.resolve_rate")
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.72

    def test_parse_missing_metric(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"other": 1.0}))
        ev = JSONEvaluator(metric_path="nonexistent")
        result = ev.parse(artifact)
        assert not result.valid


class TestExactMatchEvaluator:
    def test_exact_match_no_extraction(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"output": "42", "expected": "42"}))
        ev = ExactMatchEvaluator()
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 1.0

    def test_exact_match_mismatch(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({"output": "41", "expected": "42"}))
        ev = ExactMatchEvaluator()
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.0

    def test_exact_match_with_regex(self, tmp_path: Path):
        artifact = tmp_path / "result.json"
        artifact.write_text(json.dumps({
            "output": "The answer is \\boxed{42} as shown.",
            "expected": "42",
        }))
        ev = ExactMatchEvaluator(answer_extraction=r"\\boxed\{(\d+)\}")
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 1.0

    def test_parse_many_accuracy(self, tmp_path: Path):
        artifacts = []
        for i, (out, exp) in enumerate([("42", "42"), ("41", "42"), ("100", "100")]):
            p = tmp_path / f"result_{i}.json"
            p.write_text(json.dumps({"output": out, "expected": exp}))
            artifacts.append(p)
        ev = ExactMatchEvaluator()
        result = ev.parse_many(artifacts)
        assert result.valid
        assert abs(result.score - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# Phase 2: Benchmark config TOML registry
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_load_featurebench(self):
        config = load_benchmark_config("featurebench")
        assert config.name == "featurebench"
        assert config.test_format == "pytest"
        assert config.instance_format == "directory"

    def test_load_swebench(self):
        config = load_benchmark_config("swebench")
        assert config.name == "swebench"
        assert config.test_format == "exit_code"
        assert config.instance_format == "git-repo"
        assert config.prep_command != ""

    def test_load_aime(self):
        config = load_benchmark_config("aime")
        assert config.name == "aime"
        assert config.test_format == "exact_match"
        assert config.instance_format == "question-answer"
        assert config.answer_extraction != ""

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_benchmark_config("nonexistent_benchmark_xyz")

    def test_list_benchmarks_includes_builtins(self):
        configs = list_benchmarks()
        names = {c.name for c in configs}
        assert "featurebench" in names
        assert "swebench" in names
        assert "aime" in names

    def test_project_local_override(self, tmp_path: Path):
        bench_dir = tmp_path / ".factory" / "benchmarks"
        bench_dir.mkdir(parents=True)
        (bench_dir / "featurebench.toml").write_text(
            '[meta]\nname = "featurebench"\ndescription = "overridden"\n'
            '[test]\nformat = "exit_code"\n'
        )
        config = load_benchmark_config("featurebench", tmp_path)
        assert config.test_format == "exit_code"
        assert config.description == "overridden"

    def test_custom_benchmark_toml(self, tmp_path: Path):
        bench_dir = tmp_path / ".factory" / "benchmarks"
        bench_dir.mkdir(parents=True)
        (bench_dir / "my_custom.toml").write_text(
            '[meta]\nname = "my_custom"\ndescription = "Custom benchmark"\n'
            '[test]\nformat = "json"\ncommand = "python run_eval.py"\n'
            'metric_path = "accuracy"\ntimeout = 120\n'
            '[instances]\nformat = "directory"\n'
            '[scoring]\nmethod = "metric_extraction"\n'
        )
        config = load_benchmark_config("my_custom", tmp_path)
        assert config.name == "my_custom"
        assert config.test_format == "json"
        assert config.test_command == "python run_eval.py"
        assert config.metric_path == "accuracy"


# ---------------------------------------------------------------------------
# Phase 3: Wiring — SwarmConfig with new fields
# ---------------------------------------------------------------------------


class TestSwarmConfigMultiBenchmark:
    def test_default_values_backward_compat(self):
        config = SwarmConfig(benchmark="featurebench", budget=10)
        assert config.test_format == "pytest"
        assert config.seed_workflow == ""
        assert config.instance_format == "directory"
        assert config.prep_command == ""

    def test_custom_values(self):
        config = SwarmConfig(
            benchmark="swebench",
            budget=20,
            test_format="exit_code",
            seed_workflow="improve",
            instance_format="git-repo",
            prep_command="git clone {repo_url}",
        )
        assert config.test_format == "exit_code"
        assert config.instance_format == "git-repo"
        assert config.prep_command == "git clone {repo_url}"

    def test_serialization_roundtrip(self):
        config = SwarmConfig(
            benchmark="aime",
            budget=5,
            test_format="exact_match",
            instance_format="question-answer",
        )
        data = config.model_dump(mode="json")
        restored = SwarmConfig.model_validate(data)
        assert restored.test_format == "exact_match"
        assert restored.instance_format == "question-answer"

    def test_checkpoint_migration(self):
        """Old checkpoint JSON without new fields should still parse.

        Note: holdout_instances was removed from SwarmConfig — holdout is
        now controlled by the Task (holdout_ids on TaskDefinition).
        """
        old_data = {
            "benchmark": "featurebench",
            "budget": 50,
            "population_size": 4,
            "tournament_size": 3,
            "mutation_rate": 0.3,
            "frozen_node_ids": [],
            "mandatory_node_roles": [],
            "feature_axes": ["depth", "fork_degree", "agent_count", "gate_count"],
            "mutation_strategy": "weighted_random",
            "designer_count": 2,
            "training_instances": [],
            "plateau_window": 3,
            "plateau_threshold": 0.01,
            "diversity_floor": 0.2,
            "target_project": "",
            "test_command": "",
            "early_stop_unchanged": 3,
        }
        config = SwarmConfig.model_validate(old_data)
        assert config.test_format == "pytest"
        assert config.seed_workflow == ""


class TestInnerLoopTestFormat:
    def test_pytest_format_default(self, tmp_path: Path):
        loop = InnerLoop(project_dir=tmp_path, test_command="echo hello")
        assert loop.test_format == "pytest"

    def test_exit_code_format(self, tmp_path: Path):
        loop = InnerLoop(project_dir=tmp_path, test_command="true", test_format="exit_code")
        assert loop.test_format == "exit_code"


# ---------------------------------------------------------------------------
# Phase 4: Instance preparation
# ---------------------------------------------------------------------------


class TestInstancePrep:
    def test_validate_directory(self, tmp_path: Path):
        instance_dir = tmp_path / "inst1"
        instance_dir.mkdir()
        assert validate_instance(instance_dir, "directory") is True

    def test_validate_nonexistent(self, tmp_path: Path):
        assert validate_instance(tmp_path / "nope", "directory") is False

    def test_validate_question_answer(self, tmp_path: Path):
        instance_dir = tmp_path / "qa1"
        instance_dir.mkdir()
        (instance_dir / "question.txt").write_text("What is 2+2?")
        (instance_dir / "answer.txt").write_text("4")
        assert validate_instance(instance_dir, "question-answer") is True

    def test_validate_question_answer_missing(self, tmp_path: Path):
        instance_dir = tmp_path / "qa2"
        instance_dir.mkdir()
        (instance_dir / "question.txt").write_text("What is 2+2?")
        assert validate_instance(instance_dir, "question-answer") is False

    def test_prepare_directory_instances(self, tmp_path: Path):
        config = BenchmarkConfig(
            name="test",
            instance_format="directory",
            prep_command="mkdir -p {instance_dir}/src",
        )
        prepared = prepare_instances(config, ["inst1", "inst2"], tmp_path / "output")
        assert len(prepared) == 2
        assert (prepared[0] / "src").is_dir()

    def test_prepare_with_shell_operators(self, tmp_path: Path):
        config = BenchmarkConfig(
            name="test_shell",
            instance_format="directory",
            prep_command="mkdir -p {instance_dir}/src && touch {instance_dir}/src/ready.txt",
        )
        prepared = prepare_instances(config, ["s1"], tmp_path / "output")
        assert len(prepared) == 1
        assert (prepared[0] / "src" / "ready.txt").exists()

    def test_needs_shell_detection(self) -> None:
        assert _needs_shell("cmd1 && cmd2") is True
        assert _needs_shell("cmd1 || cmd2") is True
        assert _needs_shell("cmd1 ; cmd2") is True
        assert _needs_shell("cmd1 | cmd2") is True
        assert _needs_shell("simple-command --flag value") is False
        assert _needs_shell("mkdir -p /some/path") is False

    def test_swebench_prep_command_uses_supported_vars(self) -> None:
        config = load_benchmark_config("swebench")
        assert "{instance_id}" in config.prep_command
        assert "{instance_dir}" in config.prep_command
        assert "{repo_url}" not in config.prep_command
        assert "{commit}" not in config.prep_command

    def test_prepare_question_answer_instances(self, tmp_path: Path):
        script = tmp_path / "setup.sh"
        script.write_text(
            '#!/bin/bash\n'
            'echo "What is 1+1?" > "$1/question.txt"\n'
            'echo "2" > "$1/answer.txt"\n'
        )
        script.chmod(0o755)
        config = BenchmarkConfig(
            name="test_qa",
            instance_format="question-answer",
            prep_command=f"{script} {{instance_dir}}",
        )
        prepared = prepare_instances(config, ["q1"], tmp_path / "output")
        assert len(prepared) == 1
        assert (prepared[0] / "question.txt").read_text().strip() == "What is 1+1?"


# ---------------------------------------------------------------------------
# E2E: Full flow tests for 3 benchmarks
# ---------------------------------------------------------------------------


class TestE2EFeatureBenchBackwardCompat:
    """E2E test 1: FeatureBench backward compatibility."""

    def test_featurebench_config_matches_hardcoded(self):
        config = load_benchmark_config("featurebench")
        assert config.test_format == "pytest"
        assert config.instance_format == "directory"
        assert config.seed_workflow == ""

    def test_featurebench_evaluator_is_pytest(self):
        ev = get_evaluator("pytest")
        assert isinstance(ev, PytestEvaluator)
        info = ev.get_info()
        assert info["test_format"] == "pytest"

    def test_featurebench_swarm_config_defaults(self):
        config = SwarmConfig(benchmark="featurebench", budget=10)
        assert config.test_format == "pytest"
        assert config.instance_format == "directory"

    def test_featurebench_full_parse_flow(self, tmp_path: Path):
        """Full flow: create pytest artifacts → parse → get score."""
        artifact = tmp_path / "eval_report.json"
        artifact.write_text(json.dumps({
            "tests": [
                {"outcome": "passed", "nodeid": "test_a"},
                {"outcome": "passed", "nodeid": "test_b"},
                {"outcome": "failed", "nodeid": "test_c"},
                {"outcome": "passed", "nodeid": "test_d"},
            ]
        }))
        ev = get_evaluator("pytest")
        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.75

    def test_backward_compat_import_alias(self):
        from factory.outer_loop.featurebench_evaluator import FeatureBenchEvaluator
        ev = FeatureBenchEvaluator()
        info = ev.get_info()
        assert "benchmark" in info


class TestE2ESWEBench:
    """E2E test 2: SWE-bench with exit_code format."""

    def test_swebench_config_loads(self):
        config = load_benchmark_config("swebench")
        assert config.test_format == "exit_code"
        assert config.instance_format == "git-repo"
        assert "{instance_id}" in config.prep_command
        assert "{instance_dir}" in config.prep_command

    def test_swebench_evaluator(self):
        ev = get_evaluator("exit_code")
        assert isinstance(ev, ExitCodeEvaluator)
        info = ev.get_info()
        assert info["test_format"] == "exit_code"
        assert info["scoring"] == "binary"

    def test_swebench_swarm_config(self):
        config = SwarmConfig(
            benchmark="swebench",
            budget=10,
            test_format="exit_code",
            instance_format="git-repo",
        )
        assert config.test_format == "exit_code"

    def test_swebench_full_parse_flow(self, tmp_path: Path):
        """Full flow: mock subprocess returncode → exit_code parse → binary score."""
        for rc, expected in [(0, 1.0), (1, 0.0), (2, 0.0)]:
            artifact = tmp_path / f"result_{rc}.json"
            artifact.write_text(json.dumps({"returncode": rc}))
            ev = get_evaluator("exit_code")
            result = ev.parse(artifact)
            assert result.valid
            assert result.score == expected

    def test_swebench_inner_loop_exit_code_parsing(self, tmp_path: Path):
        """Test InnerLoop._parse_test_output with exit_code format."""
        loop = InnerLoop(
            project_dir=tmp_path,
            test_command="true",
            test_format="exit_code",
        )
        mock_result = subprocess.CompletedProcess(
            args=["true"], returncode=0, stdout="", stderr=""
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 1.0
        assert details["test_format"] == "exit_code"

        mock_fail = subprocess.CompletedProcess(
            args=["false"], returncode=1, stdout="", stderr=""
        )
        score, details = loop._parse_test_output(mock_fail)
        assert score == 0.0


class TestInnerLoopExactMatch:
    """Tests for InnerLoop._parse_test_output with exact_match format."""

    def test_exact_match_reads_expected_answer_file(self, tmp_path: Path):
        (tmp_path / "expected_answer.txt").write_text("42\n")
        loop = InnerLoop(project_dir=tmp_path, test_command="echo 42", test_format="exact_match")
        mock_result = subprocess.CompletedProcess(
            args=["echo", "42"], returncode=0, stdout="42\n", stderr=""
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 1.0
        assert details["test_format"] == "exact_match"

    def test_exact_match_falls_back_to_expected_txt(self, tmp_path: Path):
        (tmp_path / "expected.txt").write_text("hello\n")
        loop = InnerLoop(project_dir=tmp_path, test_command="echo hello", test_format="exact_match")
        mock_result = subprocess.CompletedProcess(
            args=["echo", "hello"], returncode=0, stdout="hello\n", stderr=""
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 1.0

    def test_exact_match_mismatch(self, tmp_path: Path):
        (tmp_path / "expected_answer.txt").write_text("42\n")
        loop = InnerLoop(project_dir=tmp_path, test_command="echo wrong", test_format="exact_match")
        mock_result = subprocess.CompletedProcess(
            args=["echo", "wrong"], returncode=0, stdout="wrong\n", stderr=""
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 0.0

    def test_exact_match_missing_file(self, tmp_path: Path):
        loop = InnerLoop(project_dir=tmp_path, test_command="echo 42", test_format="exact_match")
        mock_result = subprocess.CompletedProcess(
            args=["echo", "42"], returncode=0, stdout="42\n", stderr=""
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 0.0
        assert details["error"] == "expected_answer_file_missing"


class TestE2ECustomBenchmark:
    """E2E test 3: Custom user-defined benchmark with JSON format.

    Demonstrates the full flow: TOML config → prep → evaluate → score.
    This uses a user-defined benchmark that isn't built-in.
    """

    @pytest.fixture()
    def custom_benchmark_project(self, tmp_path: Path) -> Path:
        """Set up a custom benchmark with TOML config and test script."""
        project = tmp_path / "my_project"
        project.mkdir()
        factory_dir = project / ".factory"
        factory_dir.mkdir()

        bench_dir = factory_dir / "benchmarks"
        bench_dir.mkdir()
        (bench_dir / "my_ml_eval.toml").write_text(
            '[meta]\n'
            'name = "my_ml_eval"\n'
            'description = "Custom ML evaluation benchmark"\n\n'
            '[test]\n'
            'format = "json"\n'
            'command = "python eval_runner.py"\n'
            'metric_path = "accuracy"\n'
            'timeout = 120\n\n'
            '[instances]\n'
            'format = "directory"\n'
            'prep_command = "mkdir -p {instance_dir}/data"\n\n'
            '[scoring]\n'
            'method = "metric_extraction"\n'
        )

        eval_script = project / "eval_runner.py"
        eval_script.write_text(
            'import json\n'
            'print(json.dumps({"accuracy": 0.92, "loss": 0.08, "epochs": 10}))\n'
        )

        return project

    def test_custom_config_loads(self, custom_benchmark_project: Path):
        config = load_benchmark_config("my_ml_eval", custom_benchmark_project)
        assert config.name == "my_ml_eval"
        assert config.test_format == "json"
        assert config.test_command == "python eval_runner.py"
        assert config.metric_path == "accuracy"
        assert config.instance_format == "directory"

    def test_custom_evaluator_creation(self, custom_benchmark_project: Path):
        config = load_benchmark_config("my_ml_eval", custom_benchmark_project)
        ev = get_evaluator(config.test_format, metric_path=config.metric_path)
        assert isinstance(ev, JSONEvaluator)
        assert ev.metric_path == "accuracy"

    def test_custom_instance_prep(self, custom_benchmark_project: Path):
        config = load_benchmark_config("my_ml_eval", custom_benchmark_project)
        output = custom_benchmark_project / "instances"
        prepared = prepare_instances(config, ["exp1", "exp2", "exp3"], output)
        assert len(prepared) == 3
        for p in prepared:
            assert (p / "data").is_dir()

    def test_custom_full_flow(self, custom_benchmark_project: Path):
        """Full E2E: config → evaluator → parse artifacts → score."""
        config = load_benchmark_config("my_ml_eval", custom_benchmark_project)

        ev = get_evaluator(config.test_format, metric_path=config.metric_path)

        artifact = custom_benchmark_project / "result.json"
        artifact.write_text(json.dumps({
            "accuracy": 0.92,
            "loss": 0.08,
            "epochs": 10,
        }))

        result = ev.parse(artifact)
        assert result.valid
        assert result.score == 0.92
        assert "accuracy" in result.metrics

    def test_custom_swarm_config_integration(self, custom_benchmark_project: Path):
        """SwarmConfig populated from custom benchmark config."""
        bench = load_benchmark_config("my_ml_eval", custom_benchmark_project)
        swarm = SwarmConfig(
            benchmark="my_ml_eval",
            budget=10,
            test_format=bench.test_format,
            seed_workflow=bench.seed_workflow,
            instance_format=bench.instance_format,
            prep_command=bench.prep_command,
            test_command=bench.test_command,
        )
        assert swarm.test_format == "json"
        assert swarm.test_command == "python eval_runner.py"
        assert swarm.instance_format == "directory"

    def test_custom_inner_loop_json_parsing(self, custom_benchmark_project: Path):
        """InnerLoop._parse_test_output with json format and custom metric_path."""
        loop = InnerLoop(
            project_dir=custom_benchmark_project,
            test_command="python eval_runner.py",
            test_format="json",
            metric_path="accuracy",
        )
        mock_result = subprocess.CompletedProcess(
            args=["python", "eval_runner.py"],
            returncode=0,
            stdout=json.dumps({"accuracy": 0.92, "loss": 0.08}),
            stderr="",
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 0.92
        assert details["test_format"] == "json"

    def test_custom_inner_loop_json_with_score_key(self, custom_benchmark_project: Path):
        """InnerLoop._parse_test_output extracts 'score' or 'pass_rate' from JSON."""
        loop = InnerLoop(
            project_dir=custom_benchmark_project,
            test_command="echo",
            test_format="json",
        )
        mock_result = subprocess.CompletedProcess(
            args=["echo"],
            returncode=0,
            stdout=json.dumps({"score": 0.85, "details": "ok"}),
            stderr="",
        )
        score, details = loop._parse_test_output(mock_result)
        assert score == 0.85

    def test_custom_list_includes_user_benchmark(self, custom_benchmark_project: Path):
        """list_benchmarks() discovers user-defined benchmarks."""
        configs = list_benchmarks(custom_benchmark_project)
        names = {c.name for c in configs}
        assert "my_ml_eval" in names
        assert "featurebench" in names


# ---------------------------------------------------------------------------
# Cross-benchmark integration tests
# ---------------------------------------------------------------------------


class TestCrossBenchmarkIntegration:
    def test_all_built_in_configs_are_parseable(self):
        configs = list_benchmarks()
        for config in configs:
            ev = get_evaluator(config.test_format)
            info = ev.get_info()
            assert "test_format" in info or "benchmark" in info

    def test_get_info_all_formats(self):
        for fmt in list_formats():
            ev = get_evaluator(fmt)
            info = ev.get_info()
            assert isinstance(info, dict)

    def test_swarm_config_accepts_all_formats(self):
        for fmt in list_formats():
            config = SwarmConfig(
                benchmark="test",
                budget=5,
                test_format=fmt,
            )
            assert config.test_format == fmt
