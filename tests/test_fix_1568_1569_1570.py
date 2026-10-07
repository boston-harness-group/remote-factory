"""Tests for issues #1568, #1569, #1570 — holdout parsing, empty-set scoring,
split filtering, setup failure propagation, verify details, instance substitution,
and inner loop train default.

These tests cover both execution paths:
- InnerLoop._step_with_task (factory/inner_loop.py)
- WorkflowExecutor._execute_data (factory/workflow/executor.py)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from factory.task import (
    InstancesConfig,
    PromptConfig,
    ScoringContract,
    Task,
    TaskDefinition,
    TaskInstance,
    VerifyConfig,
    VerifyResult,
)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    FnNode,
    Workflow,
)


# ── Fix 1: holdout_ids round-trips through TOML parsing (#1568) ────────────


class TestFromTomlHoldoutIds:
    def test_holdout_ids_parsed(self, tmp_path: Path) -> None:
        """TOML with holdout_ids round-trips through from_toml() parsing."""
        toml_content = """\
[task]
name = "holdout-task"
description = "Task with holdout instances"

[instances]
format = "directory"
source = "data/"
holdout_ids = ["inst_a", "inst_b"]

[setup]
command = "echo setup"

[prompt]
text = "Fix the bug."

[verify]
command = "pytest -xvs"

[scoring]
method = "exit_code"
"""
        toml_file = tmp_path / "holdout-task.toml"
        toml_file.write_text(toml_content)

        defn = TaskDefinition.from_toml(toml_file)
        assert defn.instances_config.holdout_ids == ["inst_a", "inst_b"]
        assert defn.instances_config.format == "directory"
        assert defn.instances_config.source == "data/"

    def test_holdout_ids_empty_by_default(self, tmp_path: Path) -> None:
        """TOML without holdout_ids gives empty list (default)."""
        toml_content = """\
[task]
name = "no-holdout"

[instances]
format = "directory"
source = "data/"
"""
        toml_file = tmp_path / "no-holdout.toml"
        toml_file.write_text(toml_content)

        defn = TaskDefinition.from_toml(toml_file)
        assert defn.instances_config.holdout_ids == []

    def test_model_validate_used_for_all_sections(self, tmp_path: Path) -> None:
        """All four config sections (instances, setup, prompt, verify) use
        model_validate() and reject unknown fields."""
        toml_content = """\
[task]
name = "strict-task"

[instances]
format = "directory"
bogus_field = "should fail"
"""
        toml_file = tmp_path / "strict.toml"
        toml_file.write_text(toml_content)

        with pytest.raises(Exception):  # Pydantic ValidationError
            TaskDefinition.from_toml(toml_file)


# ── Fix 2: Empty filtered items raises ValueError (#1570) ─────────────────


class TestEmptyFilteredItemsRaises:
    def test_executor_raises_on_empty_items(self, tmp_path: Path) -> None:
        """DataNode with filter excluding all items raises ValueError
        (propagated as halt with 'resolved 0 items')."""
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="empty_set_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="a", metadata={"split": "train"}),
                        DataItem(id="b", metadata={"split": "train"}),
                    ],
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                    split="val",  # filter excludes all (all are train)
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.halted
        assert "resolved 0 items" in result.halt_reason

    def test_inner_loop_catches_empty_items_error(self, tmp_path: Path) -> None:
        """InnerLoop._step_with_data_node catches the ValueError and returns score=0.0."""
        from factory.inner_loop import InnerLoop

        wf = Workflow(
            name="empty_set_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="a", metadata={"split": "train"})],
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                    split="val",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        loop = InnerLoop(project_dir=tmp_path, workflow=wf)
        record = loop._step_with_data_node()
        assert record.score_end == 0.0


# ── Fix 3: Task-backed split filter uses TaskInstance.split (#1570) ───────


class _SplitTask:
    """Minimal Task-like object that yields instances with split labels."""

    def __init__(self, instances_data: list[dict[str, Any]]) -> None:
        self._instances_data = instances_data

    def instances(self):
        for d in self._instances_data:
            yield TaskInstance(
                id=d["id"],
                path=d.get("path"),
                metadata=d.get("metadata", {}),
                split=d.get("split"),
            )

    def setup(self, instance: Any, workspace: Path) -> None:
        pass

    def prompt(self, instance: Any) -> str:
        return f"prompt for {instance.id}"

    def verify(self, instance: Any, workspace: Path) -> VerifyResult:
        return VerifyResult(passed=True, score=1.0)


class TestTaskBackedSplitFilter:
    def test_task_instances_with_split_kept(self, tmp_path: Path) -> None:
        """Task-backed DataNode with split='train' keeps items whose
        TaskInstance.split == 'train' (not checking item.metadata)."""
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _SplitTask(
            instances_data=[
                {"id": "train1", "split": "train"},
                {"id": "val1", "split": "val"},
                {"id": "train2", "split": "train"},
            ]
        )

        wf = Workflow(
            name="split_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake.module:SplitTask",
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                    split="train",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

        with patch("factory.task.TaskRef.resolve", return_value=fake_task):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        item_ids = [r["item_id"] for r in parsed]
        assert "train1" in item_ids
        assert "train2" in item_ids
        assert "val1" not in item_ids
        assert len(parsed) == 2

    def test_inline_items_still_use_metadata_split(self, tmp_path: Path) -> None:
        """Inline DataItems (no TaskInstance) still use metadata.split."""
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="inline_split_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="a", metadata={"split": "train"}),
                        DataItem(id="b", metadata={"split": "val"}),
                    ],
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                    split="train",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())
        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 1
        assert parsed[0]["item_id"] == "a"


# ── Fix 4: Setup failure skips instance with score=0.0 (#1569) ────────────


class _FailingSetupTask:
    """Task whose setup() fails for specific instances."""

    def __init__(self, fail_ids: set[str]) -> None:
        self._fail_ids = fail_ids

    def instances(self):
        for iid in ["ok1", "fail_setup", "ok2"]:
            yield TaskInstance(id=iid)

    def setup(self, instance: Any, workspace: Path) -> None:
        if instance.id in self._fail_ids:
            raise RuntimeError("setup exploded")

    def prompt(self, instance: Any) -> str:
        return f"prompt for {instance.id}"

    def verify(self, instance: Any, workspace: Path) -> VerifyResult:
        return VerifyResult(passed=True, score=0.9)


class TestSetupFailureSkipsInstance:
    def test_executor_path_setup_failure(self, tmp_path: Path) -> None:
        """_execute_data: setup failure → item result has score=0.0,
        error='setup_failed', subgraph not executed."""
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _FailingSetupTask(fail_ids={"fail_setup"})

        wf = Workflow(
            name="setup_fail_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake.module:FailTask",
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

        with patch("factory.task.TaskRef.resolve", return_value=fake_task):
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())

        assert result.success  # partial failure → DataNode continues
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 3

        failed = next(r for r in parsed if r["item_id"] == "fail_setup")
        assert failed["score"] == 0.0
        assert failed["passed"] is False
        assert failed["error"] == "setup_failed"

        ok_items = [r for r in parsed if r["item_id"] != "fail_setup"]
        assert all(r["score"] == 0.9 for r in ok_items)

    def test_inner_loop_path_setup_failure(self, tmp_path: Path) -> None:
        """_step_with_task: setup failure → item result has score=0.0,
        error='setup_failed', subgraph not executed."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        task = MagicMock()
        task.instances.return_value = [
            TaskInstance(id="ok1"),
            TaskInstance(id="fail_setup"),
            TaskInstance(id="ok2"),
        ]
        task._definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"),
        )

        call_count = {"setup": 0, "prompt": 0, "verify": 0}

        def track_setup(inst, ws):
            call_count["setup"] += 1
            if inst.id == "fail_setup":
                raise RuntimeError("setup exploded")

        def track_prompt(inst):
            call_count["prompt"] += 1
            return "test prompt"

        def track_verify(inst, ws):
            call_count["verify"] += 1
            return VerifyResult(passed=True, score=0.9)

        task.setup.side_effect = track_setup
        task.prompt.side_effect = track_prompt
        task.verify.side_effect = track_verify

        wf = Workflow(
            name="test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}",
                ),
            },
            edges=[],
            start_node="builder",
        )

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        assert record.instance_results is not None
        failed = next(r for r in record.instance_results if r["instance_id"] == "fail_setup")
        assert failed["score"] == 0.0
        assert failed["error"] == "setup_failed"

        # prompt and verify should NOT be called for the failed instance
        assert call_count["prompt"] == 2  # only ok1 and ok2
        assert call_count["verify"] == 2


# ── Fix 5: Verify details include stdout on success (#1569) ───────────────


class TestVerifyDetailsOnSuccess:
    def test_exit_code_success_includes_stdout(self, tmp_path: Path) -> None:
        """verify() with passing command includes stdout in details."""
        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="exit_code"),
            verify_config=VerifyConfig(command="echo hello-world"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="t1")
        result = task.verify(inst, tmp_path)

        assert result.passed is True
        assert result.score == 1.0
        assert "stdout" in result.details
        assert "hello-world" in result.details["stdout"]

    def test_exit_code_failure_still_includes_stdout(self, tmp_path: Path) -> None:
        """verify() with failing command still includes stdout in details."""
        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="exit_code"),
            verify_config=VerifyConfig(command="echo failure-output; exit 1"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="t1")
        result = task.verify(inst, tmp_path)

        assert result.passed is False
        assert "stdout" in result.details
        assert "failure-output" in result.details["stdout"]


# ── Fix 6a: verify() substitutes {instance_id} into command (#1569) ──────


class TestVerifyInstanceSubstitution:
    def test_verify_substitutes_instance_id(self, tmp_path: Path) -> None:
        """verify() substitutes {instance_id} into the command."""
        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="exit_code"),
            verify_config=VerifyConfig(command="echo {instance_id}"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="test42")
        result = task.verify(inst, tmp_path)

        assert result.passed is True
        assert "test42" in result.details["stdout"]

    def test_verify_substitutes_instance_dir(self, tmp_path: Path) -> None:
        """verify() substitutes {instance_dir} into the command."""
        inst_dir = tmp_path / "instances" / "my_inst"
        inst_dir.mkdir(parents=True)

        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="exit_code"),
            verify_config=VerifyConfig(command="echo {instance_dir}"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="my_inst", path=inst_dir)
        result = task.verify(inst, tmp_path)

        assert result.passed is True
        assert str(inst_dir) in result.details["stdout"]

    def test_verify_uses_shlex_quote(self, tmp_path: Path) -> None:
        """verify() quotes values with shlex.quote() for shell safety."""
        defn = TaskDefinition(
            name="test",
            scoring=ScoringContract(method="exit_code"),
            verify_config=VerifyConfig(command="echo {instance_id}"),
        )
        task = Task(definition=defn)
        # Instance ID with spaces — should be shell-quoted
        inst = TaskInstance(id="test with spaces")
        result = task.verify(inst, tmp_path)
        # The command should have been quoted, so it runs without error
        assert result.passed is True
        assert "test with spaces" in result.details["stdout"]


# ── Fix 6b: prompt() substitutes {instance_id} into text (#1569) ─────────


class TestPromptInstanceSubstitution:
    def test_prompt_substitutes_instance_id(self) -> None:
        """prompt() substitutes {instance_id} into prompt text."""
        defn = TaskDefinition(
            name="test",
            prompt_config=PromptConfig(text="Fix {instance_id}"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="bug99")
        assert task.prompt(inst) == "Fix bug99"

    def test_prompt_substitutes_instance_dir(self) -> None:
        """prompt() substitutes {instance_dir} into prompt text."""
        defn = TaskDefinition(
            name="test",
            prompt_config=PromptConfig(text="Work in {instance_dir}"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="x", path=Path("/data/instances/x"))
        assert task.prompt(inst) == "Work in /data/instances/x"

    def test_prompt_no_quoting(self) -> None:
        """prompt() does NOT use shlex.quote() — prompt text is not shell-executed."""
        defn = TaskDefinition(
            name="test",
            prompt_config=PromptConfig(text="Fix {instance_id} now"),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="test with spaces")
        result = task.prompt(inst)
        # No quoting — raw substitution
        assert result == "Fix test with spaces now"

    def test_prompt_without_placeholder_unchanged(self) -> None:
        """prompt() without placeholders returns text unchanged."""
        defn = TaskDefinition(
            name="test",
            prompt_config=PromptConfig(text="Just do it."),
        )
        task = Task(definition=defn)
        inst = TaskInstance(id="any")
        assert task.prompt(inst) == "Just do it."


# ── Fix 7: Inner loop defaults to train split (#1568) ────────────────────


class TestInnerLoopTrainDefault:
    def test_uses_train_split_when_holdout_ids_configured(self, tmp_path: Path) -> None:
        """InnerLoop with holdout_ids defaults to train instances
        when no subset_selector is set."""
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        # Create a task with holdout_ids configured
        defn = TaskDefinition(
            name="test-task",
            scoring=ScoringContract(method="exit_code"),
            instances_config=InstancesConfig(
                format="directory",
                holdout_ids=["val1"],
            ),
        )

        task = MagicMock()
        task._definition = defn

        # instances() should be called with split="train"
        task.instances.return_value = [
            TaskInstance(id="train1", split="train"),
        ]
        task.setup.return_value = None
        task.prompt.return_value = "p"
        task.verify.return_value = VerifyResult(passed=True, score=1.0)

        wf = Workflow(
            name="test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}",
                ),
            },
            edges=[],
            start_node="builder",
        )

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            from factory.inner_loop import InnerLoop

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            loop.step()  # we only care about the .instances() call args

        # Verify instances() was called with split="train"
        task.instances.assert_called_once_with(split="train")

    def test_uses_all_instances_when_no_holdout_ids(self, tmp_path: Path) -> None:
        """InnerLoop without holdout_ids calls instances() without split arg."""
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        defn = TaskDefinition(
            name="test-task",
            scoring=ScoringContract(method="exit_code"),
            instances_config=InstancesConfig(format="directory"),
            # No holdout_ids
        )

        task = MagicMock()
        task._definition = defn
        task.instances.return_value = [
            TaskInstance(id="inst1", split="train"),
        ]
        task.setup.return_value = None
        task.prompt.return_value = "p"
        task.verify.return_value = VerifyResult(passed=True, score=1.0)

        wf = Workflow(
            name="test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}",
                ),
            },
            edges=[],
            start_node="builder",
        )

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            from factory.inner_loop import InnerLoop

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            loop.step()  # we only care about the .instances() call args

        # Verify instances() was called without split arg (defaults to "all")
        task.instances.assert_called_once_with()

    def test_subset_selector_overrides_train_default(self, tmp_path: Path) -> None:
        """When _subset_selector is set, instances() uses default (all)
        even with holdout_ids configured."""
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        defn = TaskDefinition(
            name="test-task",
            scoring=ScoringContract(method="exit_code"),
            instances_config=InstancesConfig(
                format="directory",
                holdout_ids=["val1"],
            ),
        )

        task = MagicMock()
        task._definition = defn
        task.instances.return_value = [
            TaskInstance(id="train1"),
            TaskInstance(id="val1"),
        ]
        task.setup.return_value = None
        task.prompt.return_value = "p"
        task.verify.return_value = VerifyResult(passed=True, score=1.0)

        wf = Workflow(
            name="test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}",
                ),
            },
            edges=[],
            start_node="builder",
        )

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            from factory.inner_loop import InnerLoop

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            # Attach a mock subset_selector
            loop._subset_selector = MagicMock()
            loop._subset_selector.select.return_value = ["train1", "val1"]
            loop.step()  # we only care about the .instances() call args

        # With subset_selector present, instances() should be called without split
        task.instances.assert_called_once_with()


# ── Fix 8: DataNode path uses real verify scores instead of binary scoring ──


class _ScoringTask:
    """Task that returns configurable per-instance scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def instances(self, **kwargs: Any):
        for iid in self._scores:
            yield TaskInstance(id=iid)

    def setup(self, instance: Any, workspace: Path) -> None:
        pass

    def prompt(self, instance: Any) -> str:
        return f"prompt for {instance.id}"

    def verify(self, instance: Any, workspace: Path) -> VerifyResult:
        s = self._scores.get(instance.id, 0.0)
        return VerifyResult(passed=s >= 0.5, score=s)


def _make_data_node_workflow() -> Workflow:
    """Build a minimal DataNode workflow for testing."""
    return Workflow(
        name="dn_score_test",
        nodes={
            "data": DataNode(
                id="data",
                task_ref="fake:Task",
                subgraph_entry="sub",
                subgraph_exit="sub",
            ),
            "sub": FnNode(id="sub", command="echo x"),
        },
        edges=[],
        start_node="data",
    )


def _mock_exec_result(
    success: bool,
    item_results: list[dict[str, Any]],
) -> MagicMock:
    """Build a mock ExecutionResult with item_results populated."""
    from factory.workflow.executor import ExecutionResult

    r = ExecutionResult()
    r.success = success
    r.halted = not success
    r.halt_reason = "" if success else "partial failure"
    r.nodes_executed = 3
    r.duration_ms = 200.0
    r.item_results = item_results
    r.node_outputs = {}
    return r


class TestDataNodeVerifyScores:
    """Fix: _step_with_data_node must use real task.verify() scores,
    not binary 1.0/0.0 from exec_result_wf.success."""

    def test_datanode_path_uses_verify_scores(self, tmp_path: Path) -> None:
        """Real verify scores (0.85, 0.72, 0.93) → mean ≈ 0.8333, not 1.0."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        wf = _make_data_node_workflow()

        item_results = [
            {"item_id": "a", "score": 0.85, "passed": True, "success": True},
            {"item_id": "b", "score": 0.72, "passed": True, "success": True},
            {"item_id": "c", "score": 0.93, "passed": True, "success": True},
        ]
        mock_result = _mock_exec_result(success=True, item_results=item_results)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            async def _fake_exec():
                return mock_result

            mock_exec = MagicMock()
            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, workflow=wf)
            record = loop._step_with_data_node()

        expected = (0.85 + 0.72 + 0.93) / 3
        assert record.score_end is not None
        assert abs(record.score_end - expected) < 1e-6
        # Must NOT be binary 1.0
        assert record.score_end != 1.0

    def test_datanode_path_scores_when_executor_fails(self, tmp_path: Path) -> None:
        """Even when exec_result_wf.success=False (partial failure),
        real scores are used instead of falling back to 0.0."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        wf = _make_data_node_workflow()

        item_results = [
            {"item_id": "a", "score": 0.85, "passed": True, "success": True},
            {"item_id": "b", "score": 0.0, "passed": False, "success": False},
            {"item_id": "c", "score": 0.72, "passed": True, "success": True},
        ]
        mock_result = _mock_exec_result(success=False, item_results=item_results)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            async def _fake_exec():
                return mock_result

            mock_exec = MagicMock()
            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, workflow=wf)
            record = loop._step_with_data_node()

        expected = (0.85 + 0.0 + 0.72) / 3
        assert record.score_end is not None
        assert abs(record.score_end - expected) < 1e-6
        # Must NOT be binary 0.0
        assert record.score_end != 0.0

    def test_datanode_path_instance_results_populated(self, tmp_path: Path) -> None:
        """instance_results on CycleRecord contains per-item data."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        wf = _make_data_node_workflow()

        item_results = [
            {"item_id": "x", "score": 0.9, "passed": True, "success": True},
            {"item_id": "y", "score": 0.3, "passed": False, "success": True},
        ]
        mock_result = _mock_exec_result(success=True, item_results=item_results)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            async def _fake_exec():
                return mock_result

            mock_exec = MagicMock()
            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, workflow=wf)
            record = loop._step_with_data_node()

        assert record.instance_results is not None
        assert len(record.instance_results) == 2
        by_id = {r["instance_id"]: r for r in record.instance_results}
        assert by_id["x"]["score"] == 0.9
        assert by_id["x"]["passed"] is True
        assert by_id["y"]["score"] == 0.3
        assert by_id["y"]["passed"] is False

    def test_datanode_path_fallback_when_no_item_results(self, tmp_path: Path) -> None:
        """When item_results is empty (executor crashed early), score → 0.0."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        wf = _make_data_node_workflow()

        mock_result = _mock_exec_result(success=False, item_results=[])

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            async def _fake_exec():
                return mock_result

            mock_exec = MagicMock()
            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, workflow=wf)
            record = loop._step_with_data_node()

        assert record.score_end == 0.0
        assert record.instance_results is None


# ── Fix 9: Executor calls task.verify() for non-task_ref DataNodes ──────


class _InlineVerifyTask:
    """Task whose verify() returns a configurable score."""

    def __init__(self, score: float = 0.85) -> None:
        self._score = score
        self.verify_calls: list[str] = []
        self.setup_calls: list[str] = []

    def instances(self, **kwargs: Any):
        yield TaskInstance(id="inline1")

    def setup(self, instance: Any, workspace: Path) -> None:
        self.setup_calls.append(instance.id)

    def prompt(self, instance: Any) -> str:
        return f"prompt for {instance.id}"

    def verify(self, instance: Any, workspace: Path) -> VerifyResult:
        self.verify_calls.append(instance.id)
        return VerifyResult(passed=self._score >= 0.5, score=self._score)


class TestExecutorUsesInnerLoopTaskForVerify:
    """Fix: When executor has task= and DataNode uses inline_items (no task_ref),
    task.verify() should be called and its score used (not binary 1.0/0.0)."""

    def test_executor_uses_inner_loop_task_for_verify(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        fake_task = _InlineVerifyTask(score=0.85)

        wf = Workflow(
            name="inline_verify_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="item_a", prompt="do a"),
                        DataItem(id="item_b", prompt="do b"),
                    ],
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

        executor = WorkflowExecutor(wf, tmp_path, dry_run=True, task=fake_task)
        result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2
        # Scores should be 0.85 from verify(), not binary 1.0
        assert all(r["score"] == 0.85 for r in parsed)
        assert all(r["passed"] is True for r in parsed)
        # verify() was called for each item
        assert sorted(fake_task.verify_calls) == ["item_a", "item_b"]
        # setup() was called for each item
        assert sorted(fake_task.setup_calls) == ["item_a", "item_b"]


class TestExecutorTaskRefTakesPriority:
    """When DataNode has task_ref AND executor has task=,
    the DataNode's task_ref should be used (not the fallback)."""

    def test_executor_task_ref_takes_priority(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        # task_ref task returns 0.7
        task_ref_task = _ScoringTask(scores={"inst_x": 0.7, "inst_y": 0.7})
        # fallback task returns 0.85
        fallback_task = _InlineVerifyTask(score=0.85)

        wf = Workflow(
            name="priority_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake.module:PriorityTask",
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

        with patch("factory.task.TaskRef.resolve", return_value=task_ref_task):
            executor = WorkflowExecutor(
                wf, tmp_path, dry_run=True, task=fallback_task,
            )
            result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        # Should use task_ref's verify (0.7), not fallback (0.85)
        assert all(r["score"] == 0.7 for r in parsed)
        # Fallback task's verify should NOT have been called
        assert fallback_task.verify_calls == []


class TestExecutorNoTaskStaysBinary:
    """When no task is passed and DataNode has no task_ref,
    scores remain binary (1.0 for success, 0.0 for failure)."""

    def test_executor_no_task_stays_binary(self, tmp_path: Path) -> None:
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="binary_score_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="x", prompt="go"),
                        DataItem(id="y", prompt="go"),
                    ],
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

        # No task= passed
        executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
        result = asyncio.run(executor.execute())

        assert result.success
        parsed = json.loads(result.node_outputs["data"])
        # Binary scoring: all succeed → 1.0
        assert all(r["score"] == 1.0 for r in parsed)
        assert all(r["verify_details"] == {} for r in parsed)


# ── Fix 10: TaskRef.resolve() sys.path for eval worktrees (#1571) ──────


class TestTaskRefResolveWithSysPath:
    """TaskRef.resolve() uses importlib.import_module() which requires the
    module on sys.path.  In eval worktrees .factory/tasks/ exists but is NOT
    on sys.path, so resolve() throws ImportError.  The executor must add the
    tasks dir to sys.path before calling resolve().
    """

    def test_taskref_resolve_fails_without_syspath(self, tmp_path: Path) -> None:
        """Without sys.path manipulation, TaskRef.resolve() fails on a
        module that lives only in .factory/tasks/."""
        import sys

        from factory.task import TaskRef

        tasks_dir = tmp_path / ".factory" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "eval_worktree_task.py").write_text(
            "from factory.task import Task, TaskDefinition\n"
            "class EvalWorktreeTask(Task):\n"
            "    def __init__(self):\n"
            "        super().__init__(TaskDefinition(name='eval-worktree'))\n"
        )

        # Ensure the tasks dir is NOT on sys.path
        original_path = sys.path[:]
        sys.path = [p for p in sys.path if str(tasks_dir) not in p]
        try:
            ref = TaskRef(ref="eval_worktree_task:EvalWorktreeTask")
            with pytest.raises(ImportError):
                ref.resolve()
        finally:
            sys.path = original_path

    def test_taskref_resolve_succeeds_with_syspath(self, tmp_path: Path) -> None:
        """With .factory/tasks/ on sys.path, TaskRef.resolve() succeeds."""
        import sys

        from factory.task import TaskRef

        tasks_dir = tmp_path / ".factory" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "eval_worktree_task2.py").write_text(
            "from factory.task import Task, TaskDefinition\n"
            "class EvalWorktreeTask2(Task):\n"
            "    def __init__(self):\n"
            "        super().__init__(TaskDefinition(name='eval-worktree-2'))\n"
        )

        original_path = sys.path[:]
        try:
            sys.path.insert(0, str(tasks_dir))
            ref = TaskRef(ref="eval_worktree_task2:EvalWorktreeTask2")
            task = ref.resolve()
            assert task.name == "eval-worktree-2"
        finally:
            sys.path = original_path

    def test_executor_adds_syspath_before_resolve(self, tmp_path: Path) -> None:
        """WorkflowExecutor._execute_data adds .factory/tasks/ to sys.path
        before calling TaskRef.resolve()."""
        import sys

        from factory.workflow.executor import WorkflowExecutor

        tasks_dir = tmp_path / ".factory" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "exec_task.py").write_text(
            "from factory.task import Task, TaskDefinition, TaskInstance\n"
            "class ExecTask(Task):\n"
            "    def __init__(self):\n"
            "        super().__init__(TaskDefinition(name='exec-task'))\n"
            "    def instances(self, **kw):\n"
            "        yield TaskInstance(id='inst1')\n"
        )

        wf = Workflow(
            name="syspath_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="exec_task:ExecTask",
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo ok"),
            },
            edges=[],
            start_node="data",
        )

        # Remove tasks_dir from sys.path if present
        original_path = sys.path[:]
        sys.path = [p for p in sys.path if str(tasks_dir) not in p]
        try:
            executor = WorkflowExecutor(wf, tmp_path, dry_run=True)
            result = asyncio.run(executor.execute())
            assert result.success
            parsed = json.loads(result.node_outputs["data"])
            assert len(parsed) == 1
            assert parsed[0]["item_id"] == "inst1"
        finally:
            sys.path = original_path
            # Clean up imported module to avoid polluting other tests
            sys.modules.pop("exec_task", None)


# ── Fix 11: _create_worktree copies .factory/tasks/ (#1571) ──────


class TestCreateWorktreeCopiesTasks:
    """_create_worktree must copy .factory/tasks/ to the eval worktree
    so TaskRef.resolve() can find task module files."""

    def test_create_worktree_copies_tasks(self, tmp_path: Path) -> None:
        """A project with .factory/tasks/my_task.py should have that file
        copied into the eval worktree at .factory/tasks/my_task.py."""
        import subprocess

        from factory.outer_loop.evaluator import SwarmEvaluator

        # Set up a minimal git repo as the source project
        project = tmp_path / "project"
        project.mkdir()
        subprocess.run(["git", "init", str(project)], capture_output=True, check=True)
        subprocess.run(
            ["git", "-C", str(project), "config", "user.email", "test@test.com"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(project), "config", "user.name", "Test"],
            capture_output=True, check=True,
        )

        # Create .factory/tasks/my_task.py (gitignored, copied by _create_worktree)
        tasks_dir = project / ".factory" / "tasks"
        tasks_dir.mkdir(parents=True)
        task_file = tasks_dir / "my_task.py"
        task_file.write_text("class MyTask: pass\n")

        # Also create outer_loop/modes and workflows to verify existing behavior
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        (modes_dir / "test_mode.json").write_text("{}")

        workflows_dir = project / ".factory" / "workflows"
        workflows_dir.mkdir(parents=True)
        (workflows_dir / "test_wf.py").write_text("# wf\n")

        # Create a dummy commit so HEAD exists
        dummy = project / "README.md"
        dummy.write_text("test\n")
        subprocess.run(
            ["git", "-C", str(project), "add", "README.md"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(project), "commit", "-m", "init"],
            capture_output=True, check=True,
        )

        wt_path: Path | None = None
        try:
            wt_path = SwarmEvaluator._create_worktree(str(project), "test")

            # tasks/ copied
            assert (wt_path / ".factory" / "tasks" / "my_task.py").exists()
            assert (wt_path / ".factory" / "tasks" / "my_task.py").read_text() == "class MyTask: pass\n"

            # existing behavior preserved: modes and workflows copied
            assert (wt_path / ".factory" / "outer_loop" / "modes" / "test_mode.json").exists()
            assert (wt_path / ".factory" / "workflows" / "test_wf.py").exists()
        finally:
            if wt_path is not None:
                SwarmEvaluator._cleanup_worktree(str(project), wt_path)

    def test_create_worktree_no_tasks_dir_still_works(self, tmp_path: Path) -> None:
        """When .factory/tasks/ does not exist, _create_worktree should
        still succeed (no crash on missing dir)."""
        import subprocess

        from factory.outer_loop.evaluator import SwarmEvaluator

        project = tmp_path / "project"
        project.mkdir()
        subprocess.run(["git", "init", str(project)], capture_output=True, check=True)
        subprocess.run(
            ["git", "-C", str(project), "config", "user.email", "test@test.com"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(project), "config", "user.name", "Test"],
            capture_output=True, check=True,
        )

        dummy = project / "README.md"
        dummy.write_text("test\n")
        subprocess.run(
            ["git", "-C", str(project), "add", "README.md"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(project), "commit", "-m", "init"],
            capture_output=True, check=True,
        )

        wt_path: Path | None = None
        try:
            wt_path = SwarmEvaluator._create_worktree(str(project), "test")
            # No .factory/tasks in source → no .factory/tasks in worktree
            assert not (wt_path / ".factory" / "tasks").exists()
        finally:
            if wt_path is not None:
                SwarmEvaluator._cleanup_worktree(str(project), wt_path)
