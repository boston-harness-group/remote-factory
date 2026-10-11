"""Tests for factory/workflow/cli.py — full coverage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from factory.workflow.cli import (
    _cmd_export_skills,
    _cmd_lint_contributed,
    _cmd_list,
    _cmd_run,
    _cmd_show,
    _cmd_validate,
    cmd_workflow,
)
from factory.workflow.executor import ExecutionResult
from factory.workflow.primitives import (
    DEFAULT_AGENT_POOL,
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    Study,
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry state before each test."""
    WorkflowRegistry.reset()
    yield
    WorkflowRegistry.reset()


def _make_args(name: str, project_path: str, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(name=name, project_path=project_path, dry_run=dry_run)


def _success_result() -> ExecutionResult:
    r = ExecutionResult()
    r.success = True
    r.halted = False
    r.nodes_executed = 3
    r.duration_ms = 42.0
    r.completed_files = {"a.txt", "b.txt"}
    return r


def _failure_result() -> ExecutionResult:
    r = ExecutionResult()
    r.success = False
    r.halted = True
    r.halt_reason = "gate rejected"
    r.nodes_executed = 2
    r.duration_ms = 10.0
    return r


class TestCmdRun:
    def test_unknown_workflow_returns_1(self, tmp_path: Path) -> None:
        args = _make_args("nonexistent", str(tmp_path))
        with patch.object(WorkflowRegistry, "get_workflow", return_value=None):
            assert _cmd_run(args) == 1

    def test_success_returns_0(self, tmp_path: Path) -> None:
        mock_wf = MagicMock()
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=_success_result())

        with (
            patch.object(WorkflowRegistry, "get_workflow", return_value=mock_wf),
            patch("factory.workflow.cli.WorkflowExecutor", return_value=mock_executor),
            patch("factory.agents.runner.begin_cycle_session", return_value="span-123") as mock_begin,
            patch("factory.agents.runner.complete_cycle_session") as mock_complete,
        ):
            result = _cmd_run(_make_args("design", str(tmp_path)))

        assert result == 0
        mock_begin.assert_called_once_with(tmp_path.resolve(), cycle_id="design")
        mock_complete.assert_called_once_with(tmp_path.resolve(), "span-123")

    def test_failure_returns_1(self, tmp_path: Path) -> None:
        mock_wf = MagicMock()
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=_failure_result())

        with (
            patch.object(WorkflowRegistry, "get_workflow", return_value=mock_wf),
            patch("factory.workflow.cli.WorkflowExecutor", return_value=mock_executor),
            patch("factory.agents.runner.begin_cycle_session", return_value=None),
            patch("factory.agents.runner.complete_cycle_session"),
        ):
            result = _cmd_run(_make_args("design", str(tmp_path)))

        assert result == 1

    def test_complete_called_on_exception(self, tmp_path: Path) -> None:
        mock_wf = MagicMock()
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch.object(WorkflowRegistry, "get_workflow", return_value=mock_wf),
            patch("factory.workflow.cli.WorkflowExecutor", return_value=mock_executor),
            patch("factory.agents.runner.begin_cycle_session", return_value="span-456") as mock_begin,
            patch("factory.agents.runner.complete_cycle_session") as mock_complete,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                _cmd_run(_make_args("design", str(tmp_path)))

        mock_begin.assert_called_once()
        mock_complete.assert_called_once_with(tmp_path.resolve(), "span-456")

    def test_executor_receives_correct_params(self, tmp_path: Path) -> None:
        mock_wf = MagicMock()
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=_success_result())

        with (
            patch.object(WorkflowRegistry, "get_workflow", return_value=mock_wf),
            patch("factory.workflow.cli.WorkflowExecutor", return_value=mock_executor) as mock_cls,
            patch("factory.agents.runner.begin_cycle_session", return_value=None),
            patch("factory.agents.runner.complete_cycle_session"),
        ):
            _cmd_run(_make_args("design", str(tmp_path), dry_run=True))

        mock_cls.assert_called_once_with(
            mock_wf,
            tmp_path.resolve(),
            agent_pool=DEFAULT_AGENT_POOL,
            dry_run=True,
        )


# ── helpers for new tests ──────────────────────────────────────


def _build_simple_workflow() -> Workflow:
    """Build a small workflow with various node types for testing."""
    nodes: dict[str, AgentNode | FnNode | GateNode | ForkNode | JoinNode | Study] = {
        "study": Study(id="study", command="factory study {project_path}", reads=set(), writes={"observations"}, focus="code"),
        "research": AgentNode(
            id="research", role=AgentRole.RESEARCHER, reads={"observations"}, writes={"findings"}
        ),
        "gate": GateNode(
            id="gate", evaluator_type="agent", reads={"findings"}, writes=set()
        ),
        "fork": ForkNode(id="fork", targets=["build_a", "build_b"], reads=set(), writes=set()),
        "join": JoinNode(id="join", sources=["build_a", "build_b"], reads=set(), writes=set()),
        "build_fn": FnNode(id="build_fn", command="echo build", reads=set(), writes={"artifact"}),
    }
    edges = [
        Edge(source="study", target="research"),
        Edge(source="research", target="gate"),
        Edge(source="gate", target="fork", condition=VerdictType.PROCEED),
        Edge(source="gate", target="study", condition=VerdictType.HALT),
        Edge(source="fork", target="join"),
    ]
    return Workflow(name="test_wf", nodes=nodes, edges=edges, start_node="study")


# ── cmd_workflow dispatch ──────────────────────────────────────


class TestCmdWorkflow:
    def test_no_subcommand_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace()  # no workflow_command attr
        assert cmd_workflow(args) == 1
        assert "Usage:" in capsys.readouterr().out

    def test_none_subcommand_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(workflow_command=None)
        assert cmd_workflow(args) == 1
        assert "Usage:" in capsys.readouterr().out

    def test_unknown_subcommand_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(workflow_command="bogus")
        assert cmd_workflow(args) == 1
        assert "Unknown workflow subcommand: bogus" in capsys.readouterr().out

    def test_dispatches_to_list(self) -> None:
        args = argparse.Namespace(workflow_command="list", project_path=None)
        with patch("factory.workflow.cli._cmd_list", return_value=0) as m:
            assert cmd_workflow(args) == 0
            m.assert_called_once_with(args)

    def test_dispatches_to_show(self) -> None:
        args = argparse.Namespace(workflow_command="show", name="design", project_path=None)
        with patch("factory.workflow.cli._cmd_show", return_value=0) as m:
            assert cmd_workflow(args) == 0
            m.assert_called_once_with(args)

    def test_dispatches_to_validate(self) -> None:
        args = argparse.Namespace(workflow_command="validate", name="design", project_path=None)
        with patch("factory.workflow.cli._cmd_validate", return_value=0) as m:
            assert cmd_workflow(args) == 0
            m.assert_called_once_with(args)

    def test_dispatches_to_export_skills(self) -> None:
        args = argparse.Namespace(workflow_command="export-skills")
        with patch("factory.workflow.cli._cmd_export_skills", return_value=0) as m:
            assert cmd_workflow(args) == 0
            m.assert_called_once_with(args)

    def test_dispatches_to_lint_contributed(self) -> None:
        args = argparse.Namespace(workflow_command="lint-contributed")
        with patch("factory.workflow.cli._cmd_lint_contributed", return_value=0) as m:
            assert cmd_workflow(args) == 0
            m.assert_called_once_with(args)


# ── _cmd_list ──────────────────────────────────────────────────


class TestCmdList:
    def test_lists_workflows(self, capsys: pytest.CaptureFixture[str]) -> None:
        wf = _build_simple_workflow()

        @dataclass
        class FakeEntry:
            name: str
            description: str = ""
            path: str = ""
            source: str = "builtin"

        entries = [FakeEntry(name="test_wf")]

        with (
            patch.object(WorkflowRegistry, "list_workflows", return_value=entries),
            patch.object(WorkflowRegistry, "get_workflow", return_value=wf),
        ):
            args = argparse.Namespace(project_path=None)
            result = _cmd_list(args)

        assert result == 0
        out = capsys.readouterr().out
        assert "test_wf" in out
        assert "study" in out  # start_node

    def test_list_with_project_path(self, tmp_path: Path) -> None:
        with (
            patch.object(WorkflowRegistry, "list_workflows", return_value=[]),
        ):
            args = argparse.Namespace(project_path=str(tmp_path))
            result = _cmd_list(args)
        assert result == 0

    def test_list_skips_none_workflows(self, capsys: pytest.CaptureFixture[str]) -> None:
        @dataclass
        class FakeEntry:
            name: str
            description: str = ""
            path: str = ""
            source: str = "builtin"

        entries = [FakeEntry(name="missing")]

        with (
            patch.object(WorkflowRegistry, "list_workflows", return_value=entries),
            patch.object(WorkflowRegistry, "get_workflow", return_value=None),
        ):
            args = argparse.Namespace(project_path=None)
            result = _cmd_list(args)

        assert result == 0
        out = capsys.readouterr().out
        assert "missing" not in out.split("\n")[-1]  # not printed as a row


# ── _cmd_show ──────────────────────────────────────────────────


class TestCmdShow:
    def test_unknown_workflow_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(WorkflowRegistry, "get_workflow", return_value=None):
            args = argparse.Namespace(name="nope", project_path=None)
            assert _cmd_show(args) == 1
        assert "Unknown workflow: nope" in capsys.readouterr().out

    def test_show_prints_graph(self, capsys: pytest.CaptureFixture[str]) -> None:
        wf = _build_simple_workflow()
        with patch.object(WorkflowRegistry, "get_workflow", return_value=wf):
            args = argparse.Namespace(name="test_wf", project_path=None)
            assert _cmd_show(args) == 0

        out = capsys.readouterr().out
        assert "Workflow: test_wf" in out
        assert "Start:    study" in out
        assert "Nodes:" in out
        assert "Edges:" in out
        # Check node types are rendered
        assert "Agent(researcher)" in out
        assert "Gate(agent)" in out
        assert "Fork(2)" in out
        assert "Join(2)" in out
        assert "Study" in out
        assert "Fn" in out
        # Check edge conditions
        assert "proceed" in out
        assert "halt" in out

    def test_show_truncates_long_reads_writes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify reads/writes longer than 28 chars are truncated."""
        long_reads = {f"very_long_read_name_{i}" for i in range(5)}
        nodes: dict[str, FnNode] = {
            "fn": FnNode(id="fn", command="echo x", reads=long_reads, writes=long_reads),
        }
        wf = Workflow(
            name="long_wf",
            nodes=nodes,
            edges=[],
            start_node="fn",
        )
        with patch.object(WorkflowRegistry, "get_workflow", return_value=wf):
            args = argparse.Namespace(name="long_wf", project_path=None)
            assert _cmd_show(args) == 0

        out = capsys.readouterr().out
        assert "..." in out


# ── _cmd_validate ──────────────────────────────────────────────


class TestCmdValidate:
    def test_unknown_workflow_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(WorkflowRegistry, "get_workflow", return_value=None):
            args = argparse.Namespace(name="nope", project_path=None, file=None)
            assert _cmd_validate(args) == 1
        assert "Unknown workflow: nope" in capsys.readouterr().out

    def test_valid_workflow_returns_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        wf = MagicMock()
        wf.validate_graph.return_value = []
        wf.nodes = {"a": MagicMock(), "b": MagicMock()}
        wf.edges = [MagicMock()]

        with patch.object(WorkflowRegistry, "get_workflow", return_value=wf):
            args = argparse.Namespace(name="ok_wf", project_path=None, file=None)
            assert _cmd_validate(args) == 0

        out = capsys.readouterr().out
        assert "VALID" in out
        assert "2 nodes" in out

    def test_invalid_workflow_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        wf = MagicMock()
        wf.validate_graph.return_value = ["orphan node X", "missing edge Y"]

        with patch.object(WorkflowRegistry, "get_workflow", return_value=wf):
            args = argparse.Namespace(name="bad_wf", project_path=None, file=None)
            assert _cmd_validate(args) == 1

        out = capsys.readouterr().out
        assert "2 issue(s)" in out
        assert "orphan node X" in out
        assert "missing edge Y" in out

    def test_file_flag_loads_and_validates(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--file flag loads a standalone workflow .py file and validates it."""
        wf_file = tmp_path / "my_workflow.py"
        wf_file.write_text(
            "from factory.workflow.primitives import FnNode, Workflow\n"
            "\n"
            'meta = {"name": "my_wf", "description": "Test workflow"}\n'
            "\n"
            "def workflow():\n"
            "    return Workflow(\n"
            '        name="my_wf",\n'
            '        nodes={"start": FnNode(id="start", command="echo hi")},\n'
            "        edges=[],\n"
            '        start_node="start",\n'
            "    )\n"
        )
        args = argparse.Namespace(name=None, project_path=None, file=str(wf_file))
        assert _cmd_validate(args) == 0

        out = capsys.readouterr().out
        assert "VALID" in out
        assert "my_wf" in out

    def test_file_flag_missing_file_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            name=None, project_path=None, file=str(tmp_path / "missing.py")
        )
        assert _cmd_validate(args) == 1
        assert "File not found" in capsys.readouterr().out

    def test_file_flag_invalid_file_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--file with a file missing meta dict returns 1."""
        wf_file = tmp_path / "bad.py"
        wf_file.write_text("def workflow(): pass\n")
        args = argparse.Namespace(name=None, project_path=None, file=str(wf_file))
        assert _cmd_validate(args) == 1
        assert "Failed to load" in capsys.readouterr().out

    def test_no_name_no_file_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Neither name nor --file provided returns an error."""
        args = argparse.Namespace(name=None, project_path=None, file=None)
        assert _cmd_validate(args) == 1
        assert "provide a workflow name or --file" in capsys.readouterr().out


# ── _cmd_export_skills ─────────────────────────────────────────


class TestCmdExportSkills:
    def test_export_no_verify(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fake_paths = [tmp_path / "skill-a" / "SKILL.md", tmp_path / "skill-b" / "SKILL.md"]
        for p in fake_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# skill content")

        with (
            patch.object(WorkflowRegistry, "discover", return_value={"wf1": MagicMock()}),
            patch.object(WorkflowRegistry, "get_workflow", return_value=MagicMock()),
            patch(
                "factory.workflow.skill_export.export_all_skills",
                return_value=fake_paths,
            ),
        ):
            args = argparse.Namespace(
                output_dir=str(tmp_path), verify=False, project_path=None
            )
            assert _cmd_export_skills(args) == 0

        out = capsys.readouterr().out
        assert "Exported 2 skills" in out

    def test_export_verify_pass(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fake_path = tmp_path / "skill-a" / "SKILL.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.write_text("# valid skill")

        with (
            patch.object(WorkflowRegistry, "discover", return_value={"wf1": MagicMock()}),
            patch.object(WorkflowRegistry, "get_workflow", return_value=MagicMock()),
            patch("factory.workflow.skill_export.export_all_skills", return_value=[fake_path]),
            patch("factory.workflow.skill_export.validate_skill", return_value=[]),
        ):
            args = argparse.Namespace(
                output_dir=str(tmp_path), verify=True, project_path=None
            )
            assert _cmd_export_skills(args) == 0

        out = capsys.readouterr().out
        assert "All skills valid" in out

    def test_export_verify_fail(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fake_path = tmp_path / "skill-bad" / "SKILL.md"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.write_text("# broken")

        with (
            patch.object(WorkflowRegistry, "discover", return_value={"wf1": MagicMock()}),
            patch.object(WorkflowRegistry, "get_workflow", return_value=MagicMock()),
            patch("factory.workflow.skill_export.export_all_skills", return_value=[fake_path]),
            patch("factory.workflow.skill_export.validate_skill", return_value=["missing section X"]),
        ):
            args = argparse.Namespace(
                output_dir=str(tmp_path), verify=True, project_path=None
            )
            assert _cmd_export_skills(args) == 1

        out = capsys.readouterr().out
        assert "INVALID" in out
        assert "missing section X" in out
        assert "1 validation issue(s)" in out

    def test_export_skips_none_workflows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(
                WorkflowRegistry, "discover", return_value={"wf1": MagicMock(), "wf2": MagicMock()}
            ),
            patch.object(WorkflowRegistry, "get_workflow", side_effect=[MagicMock(), None]),
            patch("factory.workflow.skill_export.export_all_skills", return_value=[]) as mock_export,
        ):
            args = argparse.Namespace(
                output_dir=str(tmp_path), verify=False, project_path=None
            )
            _cmd_export_skills(args)

        # Only 1 workflow should be passed (the non-None one)
        workflows_arg = mock_export.call_args[0][1]
        assert len(workflows_arg) == 1


# ── _cmd_lint_contributed ──────────────────────────────────────


class TestCmdLintContributed:
    def test_clean_returns_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("factory.workflow.lint.lint_contributed", return_value=[]):
            args = argparse.Namespace(path=str(tmp_path))
            assert _cmd_lint_contributed(args) == 0
        assert "clean" in capsys.readouterr().out

    def test_issues_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        @dataclass
        class FakeLintIssue:
            directory: str
            check: str
            message: str

        issues = [
            FakeLintIssue(directory="foo", check="missing_file", message="no README.md"),
            FakeLintIssue(directory="bar", check="bad_meta", message="invalid meta dict"),
        ]
        with patch("factory.workflow.lint.lint_contributed", return_value=issues):
            args = argparse.Namespace(path=str(tmp_path))
            assert _cmd_lint_contributed(args) == 1

        out = capsys.readouterr().out
        assert "2 issue(s)" in out
        assert "foo" in out
        assert "no README.md" in out

    def test_default_path_used(self) -> None:
        """When path is None, uses the default contributed directory."""
        with patch("factory.workflow.lint.lint_contributed", return_value=[]) as m:
            args = argparse.Namespace(path=None)
            _cmd_lint_contributed(args)

        called_path = m.call_args[0][0]
        assert "contributed" in str(called_path)
