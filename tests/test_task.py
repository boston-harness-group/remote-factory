"""Tests for factory/task.py — TaskDefinition, ScoringContract, four hooks, backward compat."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.task import (
    Capability,
    InstancesConfig,
    PromptConfig,
    ScoringContract,
    Task,
    TaskConstraints,
    TaskDefinition,
    TaskInstance,
    VerifyResult,
    _RunResult,
    _build_verify_details,
    _needs_shell,
)


# ── TaskInstance tests ───────────────────────────────────────────


class TestTaskInstance:
    def test_create_minimal(self):
        inst = TaskInstance(id="test-01")
        assert inst.id == "test-01"
        assert inst.path is None
        assert inst.metadata == {}

    def test_create_with_path(self, tmp_path: Path):
        inst = TaskInstance(id="t1", path=tmp_path)
        assert inst.path == tmp_path

    def test_create_with_metadata(self):
        inst = TaskInstance(id="t1", metadata={"key": "value"})
        assert inst.metadata["key"] == "value"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            TaskInstance(id="t1", unknown="bad")  # type: ignore[call-arg]


# ── VerifyResult tests ───────────────────────────────────────────


class TestVerifyResult:
    def test_create(self):
        vr = VerifyResult(passed=True, score=0.85)
        assert vr.passed is True
        assert vr.score == 0.85
        assert vr.details == {}

    def test_with_details(self):
        vr = VerifyResult(passed=False, score=0.0, details={"error": "timeout"})
        assert vr.details["error"] == "timeout"


# ── ScoringContract tests ───────────────────────────────────────


class TestScoringContract:
    def test_default_exit_code(self):
        s = ScoringContract()
        assert s.method == "exit_code"
        assert s.metric_path == "score"

    def test_exit_code_scoring(self):
        s = ScoringContract(method="exit_code")
        assert s.method == "exit_code"

    def test_json_scoring(self):
        s = ScoringContract(method="json", metric_path="stats.accuracy")
        assert s.method == "json"
        assert s.metric_path == "stats.accuracy"

    def test_in_task_definition(self):
        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="json", metric_path="result.score"),
        )
        assert defn.scoring.method == "json"
        assert defn.scoring.metric_path == "result.score"


# ── TaskConstraints tests ───────────────────────────────────────


class TestTaskConstraints:
    def test_defaults(self):
        tc = TaskConstraints()
        assert tc.timeout == 600
        assert tc.max_retries == 1
        assert tc.required_capabilities is None

    def test_custom(self):
        tc = TaskConstraints(
            timeout=3600,
            max_retries=3,
            required_capabilities=[Capability.CAN_RUN_TESTS],
        )
        assert tc.timeout == 3600
        assert tc.max_retries == 3
        assert tc.required_capabilities == [Capability.CAN_RUN_TESTS]


# ── TaskDefinition tests ────────────────────────────────────────


class TestTaskDefinition:
    def test_create_minimal(self):
        defn = TaskDefinition(name="test-task")
        assert defn.name == "test-task"
        assert defn.scoring.method == "exit_code"

    def test_from_toml(self, tmp_path: Path):
        toml_content = """
[task]
name = "my-task"
description = "A test task"

[instances]
format = "directory"
source = "data/"

[setup]
command = "pip install -e ."

[prompt]
text = "Do the thing."

[verify]
command = "pytest -xvs"

[scoring]
method = "exit_code"

[constraints]
timeout = 1800
max_retries = 2
"""
        toml_file = tmp_path / "my-task.toml"
        toml_file.write_text(toml_content)

        defn = TaskDefinition.from_toml(toml_file)
        assert defn.name == "my-task"
        assert defn.description == "A test task"
        assert defn.scoring.method == "exit_code"
        assert defn.constraints.timeout == 1800
        assert defn.constraints.max_retries == 2
        assert defn.instances_config.format == "directory"
        assert defn.instances_config.source == "data/"
        assert defn.setup_config.command == "pip install -e ."
        assert defn.prompt_config.text == "Do the thing."
        assert defn.verify_config.command == "pytest -xvs"

    def test_from_toml_pytest_maps_to_exit_code(self, tmp_path: Path):
        """Legacy 'pytest' method maps to 'exit_code' for backward compat."""
        toml_content = """
[task]
name = "legacy-task"

[scoring]
method = "pytest"
"""
        f = tmp_path / "legacy.toml"
        f.write_text(toml_content)
        defn = TaskDefinition.from_toml(f)
        assert defn.scoring.method == "exit_code"

    def test_from_toml_exact_match_maps_to_exit_code(self, tmp_path: Path):
        """Legacy 'exact_match' method maps to 'exit_code' for backward compat."""
        toml_content = """
[task]
name = "math-task"

[scoring]
method = "exact_match"
"""
        f = tmp_path / "math.toml"
        f.write_text(toml_content)
        defn = TaskDefinition.from_toml(f)
        assert defn.scoring.method == "exit_code"

    def test_from_toml_exit_code(self, tmp_path: Path):
        toml_content = """
[task]
name = "swe-task"

[scoring]
method = "exit_code"
"""
        f = tmp_path / "swe.toml"
        f.write_text(toml_content)
        defn = TaskDefinition.from_toml(f)
        assert defn.scoring.method == "exit_code"

    def test_from_toml_json_scoring(self, tmp_path: Path):
        toml_content = """
[task]
name = "json-task"

[scoring]
method = "json"
metric_path = "results.accuracy"
"""
        f = tmp_path / "json.toml"
        f.write_text(toml_content)
        defn = TaskDefinition.from_toml(f)
        assert defn.scoring.method == "json"
        assert defn.scoring.metric_path == "results.accuracy"

    def test_toml_roundtrip(self):
        """Serialise via model_dump and deserialise via model_validate."""
        defn = TaskDefinition(
            name="roundtrip",
            description="Test roundtrip",
            scoring=ScoringContract(method="json", metric_path="a.b"),
            constraints=TaskConstraints(timeout=999, max_retries=5),
        )
        data = defn.model_dump(mode="json")
        restored = TaskDefinition.model_validate(data)
        assert restored.name == "roundtrip"
        assert restored.scoring.method == "json"
        assert restored.scoring.metric_path == "a.b"
        assert restored.constraints.timeout == 999

    def test_to_task(self):
        defn = TaskDefinition(
            name="my-task",
            prompt_config=PromptConfig(text="Do stuff."),
        )
        task = defn.to_task()
        assert task.name == "my-task"
        inst = TaskInstance(id="default")
        assert task.prompt(inst) == "Do stuff."


# ── Task base class tests ───────────────────────────────────────


class TestTask:
    def test_default_instances(self):
        """Default instances() yields a single 'default' instance."""
        task = Task()
        insts = list(task.instances())
        assert len(insts) == 1
        assert insts[0].id == "default"

    def test_default_setup_noop(self, tmp_path: Path):
        """Default setup() is a no-op when no command is set."""
        task = Task()
        inst = TaskInstance(id="t1")
        task.setup(inst, tmp_path)

    def test_default_prompt(self):
        """Default prompt() returns generic text."""
        task = Task()
        inst = TaskInstance(id="t1")
        p = task.prompt(inst)
        assert len(p) > 0
        assert "tests" in p.lower() or "feature" in p.lower()

    def test_custom_prompt(self):
        defn = TaskDefinition(
            name="custom",
            prompt_config=PromptConfig(text="Custom prompt."),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="t1")
        assert task.prompt(inst) == "Custom prompt."

    def test_verify_no_command(self, tmp_path: Path):
        """verify() returns score=0.0 when no command is set."""
        task = Task()
        inst = TaskInstance(id="t1")
        result = task.verify(inst, tmp_path)
        assert isinstance(result, VerifyResult)
        assert result.score == 0.0
        assert result.passed is False

    def test_directory_instances(self, tmp_path: Path):
        """instances() scans directories when configured."""
        (tmp_path / "inst_a").mkdir()
        (tmp_path / "inst_b").mkdir()
        (tmp_path / "not_a_dir.txt").write_text("x")

        defn = TaskDefinition(
            name="dir-task",
            instances_config=InstancesConfig(format="directory", source=str(tmp_path)),
        )
        task = Task(definition=defn)
        insts = list(task.instances())
        ids = [i.id for i in insts]
        assert "inst_a" in ids
        assert "inst_b" in ids
        assert "not_a_dir.txt" not in ids

    def test_name_derivation(self):
        """Task._derive_name() converts CamelCase to kebab-case."""

        class ChessEvolve(Task):
            pass

        t = ChessEvolve()
        assert t.name == "chess-evolve"

    def test_from_legacy(self):
        """from_legacy() constructs Task from flat fields."""
        task = Task.from_legacy(
            name="legacy-test",
            test_command="pytest -v",
            test_format="pytest",
        )
        assert task.name == "legacy-test"
        assert task.scoring.method == "exit_code"
        assert task.definition.verify_config.command == "pytest -v"

    def test_from_legacy_exit_code(self):
        task = Task.from_legacy(
            name="swe",
            test_command="python run.py",
            test_format="exit_code",
        )
        assert task.scoring.method == "exit_code"

    def test_from_legacy_json(self):
        task = Task.from_legacy(
            name="json-test",
            test_command="python eval.py",
            test_format="json",
            metric_path="result.score",
        )
        assert task.scoring.method == "json"

    def test_from_legacy_exact_match_maps_to_exit_code(self):
        task = Task.from_legacy(
            name="math",
            test_command="python solve.py",
            test_format="exact_match",
        )
        assert task.scoring.method == "exit_code"


# ── get_evaluator tests ─────────────────────────────────────────


class TestGetEvaluator:
    def test_exit_code_evaluator(self):
        defn = TaskDefinition(name="test", scoring=ScoringContract(method="exit_code"))
        evaluator = defn.get_evaluator()
        from factory.outer_loop.evaluators.exit_code import ExitCodeEvaluator

        assert isinstance(evaluator, ExitCodeEvaluator)

    def test_json_evaluator(self):
        defn = TaskDefinition(name="test", scoring=ScoringContract(method="json", metric_path="a.b"))
        evaluator = defn.get_evaluator()
        from factory.outer_loop.evaluators.json_evaluator import JSONEvaluator

        assert isinstance(evaluator, JSONEvaluator)


# ── Backward compatibility tests ────────────────────────────────


class TestBackwardCompat:
    def test_inner_loop_without_task(self):
        """InnerLoop works without task parameter (backward compat)."""
        from factory.inner_loop import InnerLoop

        loop = InnerLoop(
            project_dir=Path("/tmp/fake"),
            mode="test",
            test_command="echo ok",
        )
        assert loop.task is None
        assert loop.instance is None
        assert loop.test_command == "echo ok"
        assert loop.test_format == "pytest"  # default when None

    def test_inner_loop_with_task(self, tmp_path: Path):
        """InnerLoop accepts task parameter and derives flat fields."""
        from factory.inner_loop import InnerLoop

        task = Task.from_legacy(
            name="compat-test",
            test_command="pytest -v",
            test_format="exit_code",
        )
        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
        )
        assert loop.task is task
        assert loop.test_command == "pytest -v"

    def test_swarm_config_get_task_from_flat(self):
        """SwarmConfig.get_task() constructs Task from flat fields."""
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            test_command="pytest -v",
            test_format="pytest",
        )
        task = config.get_task()
        assert task.name == "featurebench"
        assert task.scoring.method == "exit_code"

    def test_swarm_config_get_task_explicit(self):
        """SwarmConfig.get_task() returns explicit task when set."""
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(benchmark="test", budget=10)
        explicit_task = Task.from_legacy(name="explicit", test_format="exit_code")
        config.set_task(explicit_task)
        assert config.get_task() is explicit_task

    def test_benchmark_config_to_task(self):
        """BenchmarkConfig.to_task() produces a valid Task."""
        from factory.outer_loop.benchmark_config import BenchmarkConfig

        bc = BenchmarkConfig(
            name="featurebench",
            description="Feature impl bench",
            test_format="pytest",
            test_command="pytest -xvs",
        )
        task = bc.to_task()
        assert task.name == "featurebench"
        assert task.scoring.method == "exit_code"

    def test_research_target_to_task(self):
        """ResearchTarget.to_task() produces a Task with JSON scoring."""
        from factory.models import ResearchTarget

        rt = ResearchTarget(
            objective="Optimize speed",
            metric="speed_ms",
            target=100.0,
            run_command="python bench.py",
            result_path="metrics.speed",
        )
        task = rt.to_task()
        assert task.scoring.method == "json"
        assert task.name == "research-speed_ms"


# ── TaskRef tests ──────────────────────────────────────────────


class TestTaskRef:
    def test_task_ref_resolve_valid(self):
        """TaskRef resolves a valid Task subclass."""
        from factory.task import TaskRef

        ref = TaskRef(ref="factory.task:Task")
        task = ref.resolve()
        assert isinstance(task, Task)

    def test_task_ref_resolve_bad_module(self):
        """TaskRef raises ImportError on non-existent module."""
        from factory.task import TaskRef

        ref = TaskRef(ref="nonexistent.module:SomeTask")
        with pytest.raises(ImportError, match="Could not import module"):
            ref.resolve()

    def test_task_ref_resolve_bad_class(self):
        """TaskRef raises ImportError on valid module but missing class."""
        from factory.task import TaskRef

        ref = TaskRef(ref="factory.task:NonExistentClass")
        with pytest.raises(ImportError, match="has no attribute"):
            ref.resolve()

    def test_task_ref_resolve_no_colon(self):
        """TaskRef raises ValueError on malformed ref without ':'."""
        from factory.task import TaskRef

        ref = TaskRef(ref="factory.task.Task")
        with pytest.raises(ValueError, match="Expected 'module.path:ClassName'"):
            ref.resolve()

    def test_task_ref_resolve_not_task_subclass(self):
        """TaskRef raises TypeError when resolved class is not a Task subclass."""
        from factory.task import TaskRef

        ref = TaskRef(ref="factory.task:EvaluatorRef")
        with pytest.raises(TypeError, match="not a Task subclass"):
            ref.resolve()

    def test_swarm_config_get_task_from_task_module(self):
        """SwarmConfig.get_task() resolves task_module when set."""
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(
            benchmark="test",
            budget=10,
            task_module="factory.task:Task",
        )
        task = config.get_task()
        assert isinstance(task, Task)

    def test_swarm_config_get_task_precedence(self):
        """set_task() wins over task_module."""
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(
            benchmark="test",
            budget=10,
            task_module="factory.task:Task",
        )
        explicit = Task.from_legacy(name="explicit")
        config.set_task(explicit)
        assert config.get_task() is explicit

    def test_swarm_config_task_module_serializes(self):
        """task_module survives model_dump → SwarmConfig round-trip."""
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(
            benchmark="test",
            budget=10,
            task_module="my_project.task:MyTask",
        )
        data = config.model_dump(mode="json")
        restored = SwarmConfig(**data)
        assert restored.task_module == "my_project.task:MyTask"


# ── Task independence test ───────────────────────────────────────


class TestTaskIndependence:
    def test_no_forbidden_imports(self):
        """factory/task.py has zero module-level imports from forbidden modules."""
        import ast

        task_path = Path(__file__).parent.parent / "factory" / "task.py"
        tree = ast.parse(task_path.read_text())

        forbidden_prefixes = [
            "factory.workflow",
            "factory.outer_loop",
            "factory.agents",
            "factory.compose",
        ]

        violations = []

        # Check top-level statements only (not function-scoped imports)
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module:
                for prefix in forbidden_prefixes:
                    if stmt.module.startswith(prefix):
                        violations.append(f"line {stmt.lineno}: from {stmt.module} import ...")
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    for prefix in forbidden_prefixes:
                        if alias.name.startswith(prefix):
                            violations.append(f"line {stmt.lineno}: import {alias.name}")

        assert violations == [], "Forbidden module-level imports in task.py:\n" + "\n".join(violations)


# ── Review fix tests ────────────────────────────────────────────


class TestShellRename:
    """Task.run() was renamed to Task.shell() — shell-command utility."""

    def test_shell_method_exists(self):
        task = Task()
        assert hasattr(task, "shell")
        assert callable(task.shell)

    def test_shell_runs_command(self, tmp_path: Path):
        from factory.task import _RunResult

        task = Task()
        result = task.shell("echo hello", cwd=tmp_path)
        assert isinstance(result, _RunResult)
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_shell_called_by_setup(self, tmp_path: Path):
        """setup() uses shell() internally."""
        marker = tmp_path / "setup_ran"
        defn = TaskDefinition(
            name="test",
            setup_config=__import__("factory.task", fromlist=["SetupConfig"]).SetupConfig(
                command=f"touch {marker}",
            ),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="t1")
        task.setup(inst, tmp_path)
        assert marker.exists()


class TestNeedsShell:
    """Item 1: _needs_shell detects shell operators."""

    def test_simple_command(self):
        assert _needs_shell("pytest -xvs") is False

    def test_and_operator(self):
        assert _needs_shell("cd /tmp && pytest") is True

    def test_or_operator(self):
        assert _needs_shell("test -f x || echo no") is True

    def test_semicolon(self):
        assert _needs_shell("echo a; echo b") is True

    def test_pipe(self):
        assert _needs_shell("grep foo | wc -l") is True


class TestUnknownScoringRaises:
    """Item 5: from_toml raises ValueError on unknown scoring method."""

    def test_unknown_method(self, tmp_path: Path):
        toml_content = """
[task]
name = "bad-task"

[scoring]
method = "bogus"
"""
        f = tmp_path / "bad.toml"
        f.write_text(toml_content)
        with pytest.raises(ValueError, match="Unknown scoring method"):
            TaskDefinition.from_toml(f)

    def test_binary_alias_removed(self, tmp_path: Path):
        toml_content = """
[task]
name = "bin-task"

[scoring]
method = "binary"
"""
        f = tmp_path / "bin.toml"
        f.write_text(toml_content)
        with pytest.raises(ValueError, match="Unknown scoring method"):
            TaskDefinition.from_toml(f)


class TestSwarmConfigCaching:
    """Item 6: SwarmConfig.get_task() returns the same object on repeated calls."""

    def test_caches_result(self):
        from factory.outer_loop.models import SwarmConfig

        config = SwarmConfig(
            benchmark="test-bench",
            budget=10,
            test_command="pytest",
        )
        t1 = config.get_task()
        t2 = config.get_task()
        assert t1 is t2


class TestRegistryProjectPath:
    """Item 7: TaskRegistry rediscovers when project_path changes."""

    def test_different_project_rediscovers(self, tmp_path: Path):
        from factory.task_registry import TaskRegistry

        proj_a = tmp_path / "a"
        proj_b = tmp_path / "b"
        for p in (proj_a, proj_b):
            task_dir = p / ".factory" / "tasks"
            task_dir.mkdir(parents=True)
            (task_dir / "local.toml").write_text(f"""
[task]
name = "local-{p.name}"
description = "From project {p.name}"

[scoring]
method = "exit_code"
""")

        TaskRegistry.reset()
        TaskRegistry.discover(proj_a)
        assert "local-a" in TaskRegistry._entries

        TaskRegistry.discover(proj_b)
        assert "local-b" in TaskRegistry._entries


class TestTestFormatSentinel:
    """Item 9: InnerLoop test_format sentinel — only infer when None."""

    def test_explicit_pytest_preserved_with_exit_code_task(self, tmp_path: Path):
        from factory.inner_loop import InnerLoop

        task = Task.from_legacy(
            name="exit-test",
            test_command="python run.py",
            test_format="exit_code",
        )
        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            test_format="pytest",
        )
        assert loop.test_format == "pytest"

    def test_none_infers_from_task(self, tmp_path: Path):
        from factory.inner_loop import InnerLoop

        task = Task.from_legacy(
            name="exit-test",
            test_command="python run.py",
            test_format="exit_code",
        )
        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
        )
        assert loop.test_format == "exit_code"


class TestVersionField:
    """Item 10a: TaskDefinition has a version field."""

    def test_default_empty(self):
        defn = TaskDefinition(name="t")
        assert defn.version == ""

    def test_from_toml_parses_version(self, tmp_path: Path):
        toml_content = """
[task]
name = "versioned"
version = "1.2.3"
"""
        f = tmp_path / "v.toml"
        f.write_text(toml_content)
        defn = TaskDefinition.from_toml(f)
        assert defn.version == "1.2.3"


class TestCapabilityTyping:
    """Item 10b: required_capabilities uses Capability enum."""

    def test_pydantic_validates_capabilities(self):
        tc = TaskConstraints(
            required_capabilities=[Capability.CAN_RUN_TESTS, Capability.HAS_BUILDER],
        )
        assert all(isinstance(c, Capability) for c in tc.required_capabilities)

    def test_string_coerced_to_capability(self):
        tc = TaskConstraints(required_capabilities=["can_run_tests"])
        assert tc.required_capabilities[0] == Capability.CAN_RUN_TESTS

    def test_invalid_capability_raises(self):
        with pytest.raises(Exception):
            TaskConstraints(required_capabilities=["nonexistent"])


class TestEvaluatorRefAlignment:
    """EvaluatorRef shorthands resolve to correct evaluator classes."""

    def test_exit_code_shorthand(self):
        from factory.outer_loop.evaluators.exit_code import ExitCodeEvaluator
        from factory.task import EvaluatorRef

        evaluator = EvaluatorRef(ref="exit_code").resolve()
        assert isinstance(evaluator, ExitCodeEvaluator)

    def test_json_shorthand(self):
        from factory.outer_loop.evaluators.json_evaluator import JSONEvaluator
        from factory.task import EvaluatorRef

        evaluator = EvaluatorRef(ref="json").resolve()
        assert isinstance(evaluator, JSONEvaluator)


class TestTaskCreateNameDerivation:
    """Item 13: _cmd_task_create handles URLs with trailing slashes."""

    def test_url_trailing_slash(self, tmp_path: Path):
        import argparse

        from factory.cli.task import _cmd_task_create

        args = argparse.Namespace(
            source="https://github.com/user/my-repo/",
            project=str(tmp_path),
        )
        _cmd_task_create(args)
        assert (tmp_path / ".factory" / "tasks" / "my-repo.toml").exists()

    def test_url_with_query_params(self, tmp_path: Path):
        import argparse

        from factory.cli.task import _cmd_task_create

        args = argparse.Namespace(
            source="https://github.com/user/my-repo?tab=code",
            project=str(tmp_path),
        )
        _cmd_task_create(args)
        assert (tmp_path / ".factory" / "tasks" / "my-repo.toml").exists()


# ── _build_verify_details tests ─────────────────────────────────


class TestBuildVerifyDetails:
    def test_scoring_contract_always_present(self):
        result = _RunResult(returncode=0, stdout="ok", stderr="")
        details = _build_verify_details("exit_code", result, True)
        assert details["scoring_contract"] == "exit_code"

    def test_returncode_always_present(self):
        result = _RunResult(returncode=42, stdout="", stderr="")
        details = _build_verify_details("exit_code", result, False)
        assert details["returncode"] == 42

    def test_passed_true_includes_stdout_stderr(self):
        """stdout/stderr are always included for debugging, regardless of pass/fail."""
        result = _RunResult(returncode=0, stdout="output", stderr="err")
        details = _build_verify_details("exit_code", result, True)
        assert details["stdout"] == "output"
        assert details["stderr"] == "err"

    def test_passed_false_includes_stdout_stderr(self):
        result = _RunResult(returncode=1, stdout="fail output", stderr="fail err")
        details = _build_verify_details("exit_code", result, False)
        assert details["stdout"] == "fail output"
        assert details["stderr"] == "fail err"

    def test_truncation_at_2000_chars(self):
        long_out = "x" * 3000
        long_err = "y" * 3000
        result = _RunResult(returncode=1, stdout=long_out, stderr=long_err)
        details = _build_verify_details("exit_code", result, False)
        assert len(details["stdout"]) == 2000
        assert len(details["stderr"]) == 2000

    def test_extra_kwargs_merged(self):
        result = _RunResult(returncode=0, stdout="", stderr="")
        details = _build_verify_details(
            "json", result, True,
            metric_path="score", raw_value=0.95,
        )
        assert details["metric_path"] == "score"
        assert details["raw_value"] == 0.95

    def test_all_scoring_names(self):
        result = _RunResult(returncode=0, stdout="", stderr="")
        for name in ("exit_code", "json", "unknown"):
            details = _build_verify_details(name, result, True)
            assert details["scoring_contract"] == name
            assert "returncode" in details


class TestVerifyDetailsConsistency:
    """Verify that each scoring branch in Task.verify() populates details consistently."""

    def _make_task(self, scoring, verify_cmd="echo ok"):
        from factory.task import VerifyConfig
        defn = TaskDefinition(
            name="test",
            scoring=scoring,
            verify_config=VerifyConfig(command=verify_cmd),
        )
        return Task(definition=defn)

    def test_exit_code_scoring_has_scoring_contract(self, tmp_path: Path):
        task = self._make_task(ScoringContract(method="exit_code"))
        result = task.verify(TaskInstance(id="t1"), tmp_path)
        assert result.details["scoring_contract"] == "exit_code"
        assert "returncode" in result.details

    def test_exit_code_failure_includes_stdout_stderr(self, tmp_path: Path):
        task = self._make_task(ScoringContract(method="exit_code"), verify_cmd="false")
        result = task.verify(TaskInstance(id="t1"), tmp_path)
        assert result.details["scoring_contract"] == "exit_code"
        assert "stdout" in result.details
        assert "stderr" in result.details

    def test_json_scoring_has_scoring_contract(self, tmp_path: Path):
        task = self._make_task(
            ScoringContract(method="json", metric_path="score"),
            verify_cmd='echo \'{"score": 0.8}\'',
        )
        result = task.verify(TaskInstance(id="t1"), tmp_path)
        assert result.details["scoring_contract"] == "json"
        assert result.details["metric_path"] == "score"
        assert result.details["raw_value"] == 0.8
        assert "returncode" in result.details

    def test_json_scoring_failure_has_scoring_contract(self, tmp_path: Path):
        task = self._make_task(
            ScoringContract(method="json", metric_path="score"),
            verify_cmd="echo not-json",
        )
        result = task.verify(TaskInstance(id="t1"), tmp_path)
        assert result.details["scoring_contract"] == "json"
        assert result.details["error"] == "json_parse_failed"
        assert "returncode" in result.details

    def test_exit_code_with_json_output_uses_json(self, tmp_path: Path):
        """When exit_code method receives JSON with passed/score, it uses those values."""
        task = self._make_task(
            ScoringContract(method="exit_code"),
            verify_cmd='echo \'{"passed": true, "score": 0.75}\'',
        )
        result = task.verify(TaskInstance(id="t1"), tmp_path)
        assert result.passed is True
        assert result.score == 0.75
