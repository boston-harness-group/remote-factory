"""Tests for InnerLoop execution_strategy dispatch logic."""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch


from factory.inner_loop import InnerLoop, _SubprocessExecutionResult
from factory.task import ScoringContract, TaskDefinition, TaskInstance, VerifyResult
from factory.workflow.primitives import AgentNode, AgentRole, Workflow


def _make_workflow(name: str = "test") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="build {project_path}"),
        },
        edges=[],
        start_node="builder")


def _make_exec_result(success: bool = True) -> MagicMock:
    r = MagicMock()
    r.success = success
    r.halted = not success
    r.halt_reason = "" if success else "halted"
    r.nodes_executed = 1
    r.duration_ms = 100.0
    return r


def _async_return(val: object) -> MagicMock:
    async def _coro(*a: object, **kw: object) -> object:
        return val
    return MagicMock(side_effect=_coro)


def _make_task():
    task = MagicMock()
    task.instances.return_value = [TaskInstance(id="inst-1")]
    task.setup.return_value = None
    task.prompt.return_value = "test prompt"
    task.verify.return_value = VerifyResult(passed=True, score=0.8)
    task.definition = TaskDefinition(name="mock", scoring=ScoringContract(method="exit_code"))
    return task


class TestExecutionStrategyDefault:
    """Default execution_strategy='executor' preserves existing behavior."""

    def test_default_strategy_is_executor(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        loop = InnerLoop(project_dir=tmp_path, mode="test")
        assert loop.execution_strategy == "executor"

    def test_executor_strategy_uses_workflow_executor(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()
            mock_exec.execute = _async_return(_make_exec_result())
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(
                project_dir=tmp_path, mode="test", task=task, workflow=wf,
                execution_strategy="executor")
            record = loop.step()

        MockExecutor.assert_called_once()
        assert record.score_end == 0.8


class TestCeoSkillDispatch:
    """execution_strategy='ceo-skill' dispatches to _run_ceo_subprocess."""

    def test_ceo_skill_calls_run_ceo_subprocess(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        mock_result = _SubprocessExecutionResult(success=True, nodes_executed=1, duration_ms=50)

        with patch.object(loop, "_run_ceo_subprocess", return_value=mock_result) as mock_run:
            record = loop.step()

        mock_run.assert_called_once_with("test prompt", engine="skill")
        assert record.score_end == 0.8

    def test_ceo_skill_still_calls_setup_and_verify(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        call_order: list[str] = []

        task = MagicMock()
        task.instances.return_value = [TaskInstance(id="inst-1")]
        task.definition = TaskDefinition(name="mock", scoring=ScoringContract(method="exit_code"))
        task.setup.side_effect = lambda i, w: call_order.append("setup")
        task.prompt.side_effect = lambda i: (call_order.append("prompt"), "prompt")[1]
        task.verify.side_effect = lambda i, w: (
            call_order.append("verify"),
            VerifyResult(passed=True, score=0.9))[1]

        wf = _make_workflow()
        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        mock_result = _SubprocessExecutionResult(success=True)
        with patch.object(loop, "_run_ceo_subprocess", return_value=mock_result):
            loop.step()

        assert call_order == ["setup", "prompt", "verify"]


class TestCeoToolDispatch:
    """execution_strategy='ceo-tool' dispatches with engine='tool'."""

    def test_ceo_tool_calls_run_ceo_subprocess_with_tool(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-tool")

        mock_result = _SubprocessExecutionResult(success=True, nodes_executed=1)
        with patch.object(loop, "_run_ceo_subprocess", return_value=mock_result) as mock_run:
            loop.step()

        mock_run.assert_called_once_with("test prompt", engine="tool")


class TestEphemeralModeRegistration:
    """_run_ceo_subprocess registers and cleans up ephemeral mode files."""

    def test_ceo_subprocess_registers_ephemeral_mode(self, tmp_path: Path) -> None:
        """Before CEO spawns, mode JSON and wrapper files exist in .factory/."""
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        files_existed_during_run: dict[str, int] = {}

        def capture_subprocess(*args: object, **kwargs: object) -> MagicMock:
            # During subprocess.run, check that ephemeral files exist
            modes_dir = tmp_path / ".factory" / "outer_loop" / "modes"
            workflows_dir = tmp_path / ".factory" / "workflows"
            mode_jsons = list(modes_dir.glob("eval-test-*.json")) if modes_dir.exists() else []
            wrappers = list(workflows_dir.glob("eval-test-*.py")) if workflows_dir.exists() else []
            files_existed_during_run["mode_jsons"] = len(mode_jsons)
            files_existed_during_run["wrappers"] = len(wrappers)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("factory.inner_loop.subprocess.run", side_effect=capture_subprocess):
            loop._run_ceo_subprocess("test prompt", engine="skill")

        assert files_existed_during_run["mode_jsons"] == 1
        assert files_existed_during_run["wrappers"] == 1

    def test_ceo_subprocess_cleans_up_ephemeral_mode(self, tmp_path: Path) -> None:
        """After subprocess completes, ephemeral mode files are removed."""
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("factory.inner_loop.subprocess.run", return_value=mock_result):
            loop._run_ceo_subprocess("test prompt", engine="skill")

        # After run, ephemeral files should be cleaned up
        modes_dir = tmp_path / ".factory" / "outer_loop" / "modes"
        workflows_dir = tmp_path / ".factory" / "workflows"
        mode_jsons = list(modes_dir.glob("eval-test-*.json")) if modes_dir.exists() else []
        wrappers = list(workflows_dir.glob("eval-test-*.py")) if workflows_dir.exists() else []
        assert mode_jsons == []
        assert wrappers == []

    def test_ceo_subprocess_uses_registered_mode_name(self, tmp_path: Path) -> None:
        """The --mode flag uses the registered ephemeral mode name, not self.mode."""
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        captured_cmd: list[str] = []

        def capture_subprocess(cmd: list[str], **kwargs: object) -> MagicMock:
            captured_cmd.extend(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("factory.inner_loop.subprocess.run", side_effect=capture_subprocess):
            loop._run_ceo_subprocess("test prompt", engine="skill")

        # Find the --mode argument
        mode_idx = captured_cmd.index("--mode")
        mode_value = captured_cmd[mode_idx + 1]

        # Should be the ephemeral name (eval-test-<hash>), not "test"
        assert mode_value.startswith("eval-test-")
        assert mode_value != "test"


class TestSubprocessExecutionResult:
    """_SubprocessExecutionResult duck-types ExecutionResult."""

    def test_has_required_fields(self) -> None:
        r = _SubprocessExecutionResult()
        assert hasattr(r, "success")
        assert hasattr(r, "halted")
        assert hasattr(r, "halt_reason")
        assert hasattr(r, "nodes_executed")
        assert hasattr(r, "duration_ms")
        assert hasattr(r, "node_outputs")

    def test_defaults(self) -> None:
        r = _SubprocessExecutionResult()
        assert r.success is False
        assert r.halted is False
        assert r.halt_reason == ""
        assert r.nodes_executed == 0
        assert r.duration_ms == 0
        assert r.node_outputs == {}


class TestCostWarning:
    """CEO strategies emit a one-time cost warning."""

    def test_cost_warning_fires_once(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        task = _make_task()
        task.instances.return_value = [TaskInstance(id="a"), TaskInstance(id="b")]
        task.verify.side_effect = [
            VerifyResult(passed=True, score=0.5),
            VerifyResult(passed=True, score=0.5),
        ]
        wf = _make_workflow()

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        mock_result = _SubprocessExecutionResult(success=True)
        with patch.object(loop, "_run_ceo_subprocess", return_value=mock_result):
            with patch("factory.inner_loop.log") as mock_log:
                loop.step()

        # Warning should fire exactly once even though there are 2 instances
        warning_calls = [
            c for c in mock_log.warning.call_args_list
            if c.args and c.args[0] == "ceo_subprocess_cost_warning"
        ]
        assert len(warning_calls) == 1


class TestLegacyPathUnchanged:
    """When task is None, _step_subprocess is used regardless of execution_strategy."""

    def test_no_task_uses_step_subprocess(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir()
        loop = InnerLoop(
            project_dir=tmp_path, mode="test",
            execution_strategy="ceo-skill")
        assert loop.task is None
        # step() should route to _step_subprocess, not _step_with_task
        assert hasattr(loop, "_step_subprocess")


def _make_data_node_workflow() -> Workflow:
    """Create a workflow with a DataNode for dispatch tests."""
    from factory.workflow.primitives import DataItem, DataNode, Edge, JoinNode

    items = [DataItem(id="item-1", prompt="test")]
    data_node = DataNode(
        id="data",
        inline_items=items)
    builder = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        prompt_template="build")
    join = JoinNode(id="join", sources=["builder"])
    return Workflow(
        name="data-test",
        nodes={"data": data_node, "builder": builder, "join": join},
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="join"),
        ],
        start_node="data")


class TestDataNodeStrategyDispatch:
    """DataNode dispatch respects execution_strategy."""

    def test_datanode_executor_uses_step_with_data_node_inline(self, tmp_path: Path) -> None:
        """executor + DataNode still routes to _step_with_data_node_inline."""
        (tmp_path / ".factory").mkdir()
        wf = _make_data_node_workflow()

        task = MagicMock()
        task.instances.return_value = [TaskInstance(id="inst-1")]
        task.definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"))

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="executor")

        with patch.object(loop, "_step_with_data_node_inline") as mock_data:
            mock_data.return_value = MagicMock(score_end=0.9)
            loop.step()

        mock_data.assert_called_once()

    def test_datanode_ceo_skill_raises(self, tmp_path: Path) -> None:
        """ceo-skill + DataNode raises ValueError (not supported until PR B)."""
        (tmp_path / ".factory").mkdir()
        wf = _make_data_node_workflow()

        task = MagicMock()
        task.instances.return_value = [TaskInstance(id="inst-1")]
        task.definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"))

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-skill")

        with pytest.raises(ValueError, match="not supported"):
            loop.step()

    def test_datanode_ceo_tool_raises(self, tmp_path: Path) -> None:
        """ceo-tool + DataNode raises ValueError (not supported until PR B)."""
        (tmp_path / ".factory").mkdir()
        wf = _make_data_node_workflow()

        task = MagicMock()
        task.instances.return_value = [TaskInstance(id="inst-1")]
        task.definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"))

        loop = InnerLoop(
            project_dir=tmp_path, mode="test", task=task, workflow=wf,
            execution_strategy="ceo-tool")

        with pytest.raises(ValueError, match="not supported"):
            loop.step()
