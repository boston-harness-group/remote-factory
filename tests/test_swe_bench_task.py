"""Tests for SWEBenchTask — registration, four hooks, default run(), compose integration."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from factory.compose import compose  # noqa: E402
from factory.inner_loop import InnerLoop  # noqa: E402
from factory.task import Task, TaskInstance, VerifyResult  # noqa: E402
from swe_bench_task import SWEBenchTask  # noqa: E402


# ── Registration ─────────────────────────────────────────────────


class TestRegistration:
    def test_task_registry_discovers_swe_bench(self, tmp_path: Path):
        from factory.task_registry import TaskRegistry

        task_dir = tmp_path / ".factory" / "tasks"
        task_dir.mkdir(parents=True)

        import shutil
        src = Path(__file__).parent.parent / "examples" / "swe_bench_task.py"
        shutil.copy(src, task_dir / "swe_bench.py")

        TaskRegistry.reset()
        entries = TaskRegistry.discover(tmp_path)
        assert "swe-bench" in entries

    def test_task_meta_and_factory_function(self):
        from swe_bench_task import meta, task

        assert meta["name"] == "swe-bench"
        assert "description" in meta
        t = task()
        assert isinstance(t, SWEBenchTask)
        assert t.name == "swe-bench"

    def test_has_no_run_method(self):
        """Task no longer has run() — execution is driven by InnerLoop."""
        assert not hasattr(Task, "run")


# ── Instances ────────────────────────────────────────────────────


class TestInstances:
    def test_yields_task_instances(self):
        t = SWEBenchTask()
        instances = list(t.instances())
        assert len(instances) >= 3
        for inst in instances:
            assert isinstance(inst, TaskInstance)
            assert inst.id

    def test_instance_ids_unique(self):
        t = SWEBenchTask()
        ids = [inst.id for inst in t.instances()]
        assert len(ids) == len(set(ids))

    def test_instance_ids_follow_swe_bench_format(self):
        t = SWEBenchTask()
        for inst in t.instances():
            assert "__" in inst.id, f"Expected owner__repo-NNNNN format, got {inst.id}"
            parts = inst.id.split("__")
            assert len(parts) == 2

    def test_instances_have_required_metadata(self):
        t = SWEBenchTask()
        required_keys = {"repo", "base_commit", "problem_statement", "test_patch", "FAIL_TO_PASS"}
        for inst in t.instances():
            assert required_keys.issubset(inst.metadata.keys()), (
                f"Instance {inst.id} missing keys: {required_keys - inst.metadata.keys()}"
            )

    def test_fail_to_pass_is_nonempty_list(self):
        t = SWEBenchTask()
        for inst in t.instances():
            fail_tests = inst.metadata["FAIL_TO_PASS"]
            assert isinstance(fail_tests, list)
            assert len(fail_tests) >= 1

    def test_repo_format(self):
        t = SWEBenchTask()
        for inst in t.instances():
            repo = inst.metadata["repo"]
            assert "/" in repo, f"Expected owner/repo format, got {repo}"

    def test_base_commit_is_hex_string(self):
        t = SWEBenchTask()
        for inst in t.instances():
            commit = inst.metadata["base_commit"]
            assert len(commit) == 40
            assert all(c in "0123456789abcdef" for c in commit)


# ── Setup ────────────────────────────────────────────────────────


class TestSetup:
    def test_creates_workspace_files(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        assert (tmp_path / "instance.json").exists()
        assert (tmp_path / "test_patch.diff").exists()
        assert (tmp_path / "requirements.txt").exists()

    def test_instance_json_contents(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        data = json.loads((tmp_path / "instance.json").read_text())
        assert data["instance_id"] == inst.id
        assert "repo" in data
        assert "problem_statement" in data

    def test_creates_stub_repo_structure(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        repo = inst.metadata["repo"]
        project_name = repo.split("/")[-1]
        assert (tmp_path / project_name / "__init__.py").exists()
        assert (tmp_path / "tests" / "__init__.py").exists()

    def test_setup_idempotent(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        # Write custom content to a file that setup should NOT overwrite
        init_file = tmp_path / inst.metadata["repo"].split("/")[-1] / "__init__.py"
        init_file.write_text("# custom\n")

        t.setup(inst, tmp_path)
        assert init_file.read_text() == "# custom\n"

    def test_writes_test_patch(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        patch_content = (tmp_path / "test_patch.diff").read_text()
        assert len(patch_content) > 0
        assert patch_content == inst.metadata["test_patch"]


# ── Prompt ───────────────────────────────────────────────────────


class TestPrompt:
    def test_returns_nonempty_string(self):
        t = SWEBenchTask()
        inst = next(t.instances())
        p = t.prompt(inst)
        assert isinstance(p, str)
        assert len(p) > 0

    def test_references_repo_name(self):
        t = SWEBenchTask()
        inst = next(t.instances())
        p = t.prompt(inst)
        repo = inst.metadata["repo"]
        assert repo in p

    def test_references_failing_tests(self):
        t = SWEBenchTask()
        inst = next(t.instances())
        p = t.prompt(inst)
        for test_id in inst.metadata["FAIL_TO_PASS"]:
            assert test_id in p

    def test_includes_problem_statement(self):
        t = SWEBenchTask()
        inst = next(t.instances())
        p = t.prompt(inst)
        problem = inst.metadata["problem_statement"]
        assert problem in p

    def test_includes_instructions(self):
        t = SWEBenchTask()
        inst = next(t.instances())
        p = t.prompt(inst)
        assert "minimal fix" in p.lower() or "fix" in p.lower()


# ── Verify ───────────────────────────────────────────────────────


class TestVerify:
    def test_returns_verify_result(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)
        result = t.verify(inst, tmp_path)

        assert isinstance(result, VerifyResult)
        assert isinstance(result.passed, bool)
        assert result.score in (0.0, 1.0)

    def test_missing_workspace_returns_zero(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        # Don't call setup — workspace has no instance.json
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = t.verify(inst, empty_dir)

        assert result.passed is False
        assert result.score == 0.0
        assert "instance.json missing" in result.details.get("error", "")

    def test_no_changes_returns_zero(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        # __init__.py is empty so no "code changes" detected
        result = t.verify(inst, tmp_path)
        assert result.score == 0.0

    def test_with_patch_file_returns_one(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        # Simulate a fix by writing a patch file
        (tmp_path / "fix.patch").write_text("--- a/file.py\n+++ b/file.py\n")
        result = t.verify(inst, tmp_path)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_patch"] is True

    def test_with_code_changes_returns_one(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)

        repo = inst.metadata["repo"]
        project_name = repo.split("/")[-1]
        (tmp_path / project_name / "fix.py").write_text("def fix(): pass\n")
        result = t.verify(inst, tmp_path)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_code_changes"] is True

    def test_verify_details_include_test_name(self, tmp_path: Path):
        t = SWEBenchTask()
        inst = next(t.instances())
        t.setup(inst, tmp_path)
        (tmp_path / "fix.patch").write_text("patch content")
        result = t.verify(inst, tmp_path)

        assert "test_name" in result.details
        assert "test_status" in result.details


# ── Compose integration ─────────────────────────────────────────


class TestComposeIntegration:
    def test_compose_produces_inner_loop(self, tmp_path: Path):
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="improve",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                ),
            },
            edges=[],
            start_node="builder",
        )

        t = SWEBenchTask()
        loop = compose(wf, t, tmp_path)
        assert isinstance(loop, InnerLoop)
        assert loop.task is t

    def test_inner_loop_step_iterates_instances(self, tmp_path: Path):
        from unittest.mock import patch

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="improve",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                ),
            },
            edges=[],
            start_node="builder",
        )

        t = _SingleInstanceSWEBenchTask()
        loop = compose(wf, t, tmp_path)

        from factory.workflow.executor import ExecutionResult

        async def fake_execute(self_exec):
            r = ExecutionResult()
            r.success = True
            r.item_results = [
                {"item_id": "django__django-16379", "score": 0.5, "status": "ok"},
            ]
            return r

        with patch.object(
            __import__("factory.workflow.executor", fromlist=["WorkflowExecutor"]).WorkflowExecutor,
            "execute",
            fake_execute,
        ):
            record = loop.step()

        assert record.instance_results is not None
        assert len(record.instance_results) >= 1
        assert record.score_end is not None
        assert 0.0 <= record.score_end <= 1.0

        for ir in record.instance_results:
            assert "item_id" in ir
            assert "score" in ir


# ── Helper: single-instance variant for fast integration test ────


class _SingleInstanceSWEBenchTask(SWEBenchTask):
    """Yields only one instance for fast integration tests."""

    def instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(
            id="django__django-16379",
            metadata={
                "repo": "django/django",
                "base_commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
                "problem_statement": "FileBasedCache has_key race condition.",
                "test_patch": "--- a/tests/cache/tests.py\n+++ b/tests/cache/tests.py\n",
                "FAIL_TO_PASS": ["tests.cache.tests.FileBasedCacheTests.test_has_key_race_condition"],
                "PASS_TO_PASS": [],
                "hints_text": "",
            },
        )
