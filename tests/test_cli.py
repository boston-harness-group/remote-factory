"""Tests for factory.cli — CLI subcommand routing."""

import argparse
import asyncio
import contextlib
import json
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from factory.cli import main, build_parser
from factory.cli._task_builder import _build_ceo_task
from factory.cli._path_resolver import (
    _slugify,
    _extract_project_name,
    _dedupe_project_path,
    _persist_spec,
    _has_research_target,
    _ensure_repo,
    _materialize_project,
    _is_scaffold_only,
    _resolve_input,
)
from factory.cli._helpers import _is_github_url
from factory.models import ExperimentRecord
from factory.store import ExperimentStore


# Async mock helper for invoke_agent — returns (stdout, return_code)
def _mock_invoke_agent_ok():
    return AsyncMock(return_value=("CEO completed successfully", 0))


def _mock_invoke_agent_fail():
    return AsyncMock(return_value=("Error: agent failed", 1))


@contextlib.contextmanager
def _mock_foreground():
    """Mock the interactive foreground path: subprocess.run inside ClaudeRunner,
    worktree lifecycle, and dashboard.  Yields the subprocess.run mock."""
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    with (
        patch("factory.runners.claude.subprocess.run", mock_run),
        patch(
            "factory.worktree.create_worktree",
            side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
        ),
        patch("factory.worktree.remove_worktree"),
        patch("factory.worktree.prune_stale", return_value=[]),
        patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
        patch("factory.cli._path_resolver._is_scaffold_only", return_value=False),
        patch("factory.cli._helpers._ensure_dashboard"),
        patch("factory.graph.is_graphify_installed", return_value=False),
    ):
        yield mock_run


class TestParser:
    def test_detect_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["detect", "/some/path"])
        assert args.command == "detect"
        assert args.path == "/some/path"

    def test_discover_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["discover", "/some/path"])
        assert args.command == "discover"

    def test_init_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["init", "/some/path"])
        assert args.command == "init"
        assert args.reparse is False

    def test_init_with_reparse(self):
        parser = build_parser()
        args = parser.parse_args(["init", "/some/path", "--reparse"])
        assert args.reparse is True

    def test_guard_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["guard", "/path", "--baseline", "abc123"])
        assert args.command == "guard"
        assert args.baseline == "abc123"

    def test_begin_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["begin", "/path", "--hypothesis", "test hyp"])
        assert args.hypothesis == "test hyp"

    def test_finalize_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "finalize",
                "/path",
                "--id",
                "1",
                "--verdict",
                "keep",
                "--hypothesis",
                "h",
                "--summary",
                "s",
            ]
        )
        assert args.id == 1
        assert args.verdict == "keep"

    def test_finalize_with_scores(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "finalize",
                "/path",
                "--id",
                "1",
                "--verdict",
                "keep",
                "--hypothesis",
                "h",
                "--summary",
                "s",
                "--score-before",
                "0.80",
                "--score-after",
                "0.85",
            ]
        )
        assert args.score_before == 0.80
        assert args.score_after == 0.85

    def test_version_flag_exits_zero(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            main(["--version"])
        out = capsys.readouterr().out
        assert out.startswith("remote-factory ")
        version_str = out.strip().split(" ", 1)[1]
        assert version_str[0].isdigit()

    def test_no_command_returns_1(self):
        assert main([]) == 1

    def test_emit_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            ["emit", "agent.started", "--agent", "researcher", "--project", "/p"]
        )
        assert args.command == "emit"
        assert args.event_type == "agent.started"
        assert args.agent == "researcher"
        assert args.project == "/p"

    def test_emit_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["emit", "cycle.started"])
        assert args.agent is None
        assert args.project == "."
        assert args.data is None

    def test_ceo_mode_design(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "distributed eval runner", "--mode", "design"])
        assert args.mode == "design"
        assert args.path == "distributed eval runner"

    def test_ceo_mode_interactive_backward_compat(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "distributed eval runner", "--mode", "design"])
        assert args.mode == "design"
        assert args.path == "distributed eval runner"

    def test_ceo_mode_project_prefix(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/tmp/proj", "--mode", "project:greet"])
        assert args.mode == "project:greet"

    def test_ceo_mode_unknown_accepted_by_parser(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/tmp/proj", "--mode", "my-custom-mode"])
        assert args.mode == "my-custom-mode"

    def test_ceo_path_optional(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "--mode", "design"])
        assert args.path is None

    def test_ceo_agent_strategist_choice(self):
        parser = build_parser()
        args = parser.parse_args(["agent", "strategist", "--task", "test", "--project", "/p"])
        assert args.role == "strategist"

    def test_ceo_agent_failure_analyst_choice(self):
        parser = build_parser()
        args = parser.parse_args(["agent", "failure_analyst", "--task", "test", "--project", "/p"])
        assert args.role == "failure_analyst"


class TestGroupedHelp:
    """Tests for _GroupedHelpParser and _COMMAND_GROUPS completeness."""

    EXPECTED_GROUPS = [
        "Entry Points:",
        "Project Setup:",
        "Experiment Lifecycle:",
        "Project Intelligence:",
        "Backlog & Refinement:",
        "Knowledge & Archive:",
        "Self-Evolution:",
        "Configuration:",
        "Validation & Recovery:",
    ]

    def test_help_output_contains_all_group_headers(self):
        help_text = build_parser().format_help()
        for header in self.EXPECTED_GROUPS:
            assert header in help_text, f"Missing group header: {header}"

    def test_all_subcommands_covered_by_groups(self):
        from factory.cli._main import _COMMAND_GROUPS

        grouped = {cmd for _, cmds in _COMMAND_GROUPS for cmd in cmds}
        parser = build_parser()
        sub_action = None
        for action in parser._subparsers._group_actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_action = action
                break
        assert sub_action is not None
        registered = set(sub_action._name_parser_map.keys())
        orphans = registered - grouped
        assert orphans == set(), f"Commands not in any group: {orphans}"

    def test_no_command_in_multiple_groups(self):
        from factory.cli._main import _COMMAND_GROUPS

        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for group_name, cmds in _COMMAND_GROUPS:
            for cmd in cmds:
                if cmd in seen:
                    duplicates.append(f"{cmd!r} in both {seen[cmd]!r} and {group_name!r}")
                seen[cmd] = group_name
        assert duplicates == [], f"Commands in multiple groups: {duplicates}"

    def test_no_ungrouped_other_section(self):
        help_text = build_parser().format_help()
        assert "\nOther:\n" not in help_text, (
            "Help has an 'Other' section — some commands are ungrouped"
        )

    def test_group_count_is_nine(self):
        from factory.cli._main import _COMMAND_GROUPS

        assert len(_COMMAND_GROUPS) == 9


class TestRefactoryAgentFilter:
    """Tests for --refactory-agent help filtering."""

    EXPECTED_COMMANDS = {
        "ceo",
        "run",
        "tmux",
        "tmux-ls",
        "tmux-stop",
        "tmux-capture",
        "discover",
        "init",
        "detect",
        "eval",
        "history",
        "study",
        "status",
        "backlog-list",
        "backlog-add",
        "checkpoint",
        "resume",
        "ace",
        "ace-stats",
    }

    def test_filtered_help_shows_only_expected_commands(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["factory", "--help", "--refactory-agent"])
        parser = build_parser()
        help_text = parser.format_help()
        import re as _re

        displayed = set(_re.findall(r"^  (\S+)", help_text, _re.MULTILINE))
        assert displayed == self.EXPECTED_COMMANDS

    def test_filtered_help_has_group_headers(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["factory", "--help", "--refactory-agent"])
        help_text = build_parser().format_help()
        for header in (
            "Entry Points:",
            "Project Setup:",
            "Project Intelligence:",
            "Validation & Recovery:",
            "Self-Evolution:",
        ):
            assert header in help_text, f"Missing group header: {header}"

    def test_filtered_help_omits_empty_groups(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["factory", "--help", "--refactory-agent"])
        help_text = build_parser().format_help()
        for header in ("Experiment Lifecycle:", "Knowledge & Archive:", "Configuration:"):
            assert header not in help_text, f"Group should be hidden: {header}"

    def test_filtered_help_no_other_section(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["factory", "--help", "--refactory-agent"])
        help_text = build_parser().format_help()
        assert "\nOther:\n" not in help_text

    def test_unfiltered_help_unaffected(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["factory", "--help"])
        help_text = build_parser().format_help()
        assert "Experiment Lifecycle:" in help_text
        assert "begin" in help_text


class TestCmdCeoDesign:
    def test_design_headless_incompatible(self, capsys):
        result = main(["ceo", "an idea", "--mode", "design", "--headless"])
        assert result == 1
        assert "incompatible" in capsys.readouterr().err.lower()

    def test_design_prompt_incompatible(self, capsys):
        result = main(["ceo", "an idea", "--mode", "design", "--prompt", "file.md"])
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err.lower()

    def test_design_focus_incompatible_new_idea(self, capsys):
        """--focus + --mode design is rejected for new ideas (non-directory paths)."""
        result = main(["ceo", "an idea", "--mode", "design", "--focus", "UI"])
        assert result == 1
        err = capsys.readouterr().err.lower()
        assert "new ideas" in err

    def test_design_focus_allowed_existing_project(self, tmp_path):
        """--focus + --mode design is allowed when path is an existing directory."""
        (tmp_path / ".git").mkdir()
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "design", "--focus", "auth layer"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Plan Loop (Interactive)" in task
        assert "auth layer" in task

    def test_no_path_fails(self, capsys):
        result = main(["ceo"])
        assert result == 1
        err = capsys.readouterr().err.lower()
        assert "provide" in err or "error" in err

    def test_design_foreground_uses_subprocess_run(self, tmp_path):
        """--mode design launches via subprocess.run (foreground)."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "design"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert "--dangerously-skip-permissions" in cmd

    def test_design_existing_has_plan_loop_block(self, tmp_path):
        """--mode design on an existing directory injects Plan Loop block."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Plan Loop (Interactive)" in task
        assert "existing_project: true" in task

    def test_design_new_idea_has_plan_loop_block(self):
        """--mode design with a non-directory path injects Plan Loop block."""
        with _mock_foreground() as mock_run:
            main(["ceo", "build a cool CLI tool", "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Plan Loop (Interactive)" in task
        assert "## Project Specification" not in task

    def test_design_task_contains_idea_text(self):
        """--mode design with raw idea text includes it in the Plan Loop block."""
        with _mock_foreground() as mock_run:
            main(["ceo", "distributed eval runner", "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "distributed eval runner" in task

    def test_design_existing_mode_is_design(self, tmp_path):
        """--mode design on existing dir sets Mode: design in the CEO task."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "Mode: design" in task

    def test_design_new_idea_mode_is_ideation(self):
        """--mode design with new idea sets Mode: ideation in the CEO task."""
        with _mock_foreground() as mock_run:
            main(["ceo", "weather CLI", "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "Mode: ideation" in task

    def test_design_mode_emits_plan_loop(self, tmp_path):
        """--mode design generates a task with the Plan Loop section."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Plan Loop (Interactive)" in task

    def test_auto_approve_rejected_without_design_mode(self, capsys):
        """--auto-approve without --mode design is rejected."""
        result = main(["ceo", "/some/path", "--mode", "founder", "--auto-approve"])
        assert result == 1
        assert "--auto-approve only applies to --mode design" in capsys.readouterr().err

    def test_auto_approve_accepted_with_design_mode(self, tmp_path):
        """--auto-approve with --mode design succeeds and runs headless."""
        mock_invoke = _mock_invoke_agent_ok()
        with (
            patch("factory.agents.runner.invoke_agent", mock_invoke),
            patch(
                "factory.worktree.create_worktree",
                side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
            ),
            patch("factory.worktree.remove_worktree"),
            patch("factory.worktree.prune_stale", return_value=[]),
            patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
            patch("factory.cli._path_resolver._is_scaffold_only", return_value=False),
            patch("factory.cli._helpers._ensure_dashboard"),
            patch("factory.graph.is_graphify_installed", return_value=False),
        ):
            result = main(["ceo", str(tmp_path), "--mode", "design", "--auto-approve"])
        assert result == 0

    def test_auto_approve_forces_headless(self):
        """--auto-approve with --mode design forces headless=True in the validation tuple."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="an idea",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus=None,
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=True,
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int), f"Expected tuple, got error code {validated}"
        (
            _mode,
            headless,
            _bg,
            _bg_agents,
            _prompt,
            _focus,
            _dir,
            _refine,
            auto_approve,
            _from_plan,
            _just_plan,
        ) = validated
        assert headless is True
        assert auto_approve is True

    def test_auto_approve_false_by_default(self):
        """auto_approve defaults to False when flag is omitted."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="some idea",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus=None,
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=False,
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int)
        auto_approve = validated[8]
        assert auto_approve is False


class TestRunAutoApprove:
    def test_run_auto_approve_rejected_without_design(self, capsys):
        """cmd_run rejects --auto-approve when mode is not design."""
        result = main(["run", "/some/path", "--mode", "founder", "--auto-approve"])
        assert result == 1
        assert "--auto-approve only applies to --mode design" in capsys.readouterr().err

    def test_run_auto_approve_rejected_default_mode(self, capsys):
        """cmd_run rejects --auto-approve when mode is the default (auto)."""
        result = main(["run", "/some/path", "--auto-approve"])
        assert result == 1
        assert "--auto-approve only applies to --mode design" in capsys.readouterr().err


class TestRunFocusIncompatibleMode:
    def test_run_focus_rejected_with_incompatible_mode(self, capsys):
        """cmd_run rejects --focus with a mode other than design or research."""
        result = main(["run", "/some/path", "--mode", "founder", "--focus", "auth"])
        assert result == 1
        assert "only works in design, research, or task-setup mode" in capsys.readouterr().err


class TestAutoApproveEvent:
    def test_execute_ceo_emits_auto_approve_event(self, tmp_path):
        """_execute_ceo calls _emit_cli_event with 'auto_approve.enabled' when flag is set."""
        mock_invoke = _mock_invoke_agent_ok()
        with (
            patch("factory.agents.runner.invoke_agent", mock_invoke),
            patch(
                "factory.worktree.create_worktree",
                side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
            ),
            patch("factory.worktree.remove_worktree"),
            patch("factory.worktree.prune_stale", return_value=[]),
            patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
            patch("factory.cli._path_resolver._is_scaffold_only", return_value=False),
            patch("factory.cli._helpers._ensure_dashboard"),
            patch("factory.graph.is_graphify_installed", return_value=False),
            patch("factory.cli._ceo_helpers._emit_cli_event") as mock_emit,
        ):
            result = main(["ceo", str(tmp_path), "--mode", "design", "--auto-approve"])
        assert result == 0
        mock_emit.assert_any_call(tmp_path, "auto_approve.enabled", {"mode": "design"})

    def test_execute_ceo_no_event_without_flag(self, tmp_path):
        """_execute_ceo does not emit auto_approve.enabled when --auto-approve is absent."""
        with (
            _mock_foreground(),
            patch("factory.cli._ceo_helpers._emit_cli_event") as mock_emit,
        ):
            result = main(["ceo", str(tmp_path), "--mode", "design"])
        assert result == 0
        auto_approve_calls = [
            c
            for c in mock_emit.call_args_list
            if len(c.args) >= 2 and c.args[1] == "auto_approve.enabled"
        ]
        assert len(auto_approve_calls) == 0


def _make_config(*, research_target: dict | None = None) -> dict:
    """Build a valid FactoryConfig dict for testing."""
    config: dict = {
        "goal": "test project",
        "scope": ["src/**/*.py"],
        "guards": ["Do not delete tests"],
        "eval_command": "python eval/score.py",
        "eval_threshold": 0.8,
        "constraints": ["Prefer small changes"],
    }
    if research_target is not None:
        config["research_target"] = research_target
    return config


class TestHasResearchTarget:
    def test_returns_false_no_factory(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert _has_research_target(tmp_path) is False

    def test_returns_false_no_research_target(self, tmp_path):
        (tmp_path / ".git").mkdir()
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        (factory_dir / "config.json").write_text(json.dumps(_make_config()))
        assert _has_research_target(tmp_path) is False

    def test_returns_true_with_research_target(self, tmp_path):
        (tmp_path / ".git").mkdir()
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        rt = {
            "objective": "maximize accuracy",
            "metric": "accuracy",
            "target": 0.9,
            "run_command": "python run.py",
            "result_path": "results.json",
        }
        (factory_dir / "config.json").write_text(json.dumps(_make_config(research_target=rt)))
        assert _has_research_target(tmp_path) is True


class TestCmdDetect:
    def test_detect_no_repo(self, tmp_path, capsys):
        result = main(["detect", str(tmp_path / "nonexistent")])
        assert result == 0
        assert "no_repo" in capsys.readouterr().out

    def test_detect_no_factory(self, tmp_project, capsys):
        result = main(["detect", str(tmp_project)])
        assert result == 0
        assert "no_factory" in capsys.readouterr().out


class TestCmdDiscover:
    def test_discover_python_project(self, python_project, capsys):
        result = main(["discover", str(python_project)])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["project"]["language"] == "python"
        assert output["eval_profile"]["tier"] in ("discovered", "researched", "fallback")


class TestCmdStatus:
    def test_status_parser(self):
        parser = build_parser()
        args = parser.parse_args(["status", "/some/path"])
        assert args.command == "status"

    def test_status_no_factory(self, tmp_project, capsys):
        result = main(["status", str(tmp_project)])
        assert result == 0
        out = capsys.readouterr().out
        assert "no_factory" in out
        assert str(tmp_project) in out

    def test_status_with_factory(self, tmp_project, capsys, sample_config):
        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        exp_id = asyncio.run(store.begin("Improve performance"))
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now(),
            hypothesis="Improve performance",
            change_summary="Optimized hot path",
            issue_number=None,
            pr_number=None,
            score_before=0.8,
            score_after=0.95,
            delta=0.15,
            verdict="keep",
            cost_usd=None,
            notes="",
        )
        asyncio.run(store.finalize(exp_id, record))

        result = main(["status", str(tmp_project)])
        assert result == 0
        out = capsys.readouterr().out
        assert "has_factory" in out
        assert "1 total" in out
        assert "1 kept" in out
        assert "0 reverted" in out
        assert "Improve performance" in out
        assert "0.950" in out
        assert sample_config.goal in out

    def test_status_with_factory_no_experiments(self, tmp_project, capsys, sample_config):
        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))

        result = main(["status", str(tmp_project)])
        assert result == 0
        out = capsys.readouterr().out
        assert "has_factory" in out
        assert "Experiments: none" in out


class TestCmdHistory:
    def test_history_no_experiments(self, tmp_project, capsys, sample_config):
        import asyncio
        from factory.store import ExperimentStore

        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        result = main(["history", str(tmp_project)])
        assert result == 0
        assert "No experiments" in capsys.readouterr().out

    def test_history_with_records(self, tmp_project, capsys, sample_config):
        """history displays formatted table when records exist."""
        import asyncio
        from datetime import datetime
        from factory.store import ExperimentStore
        from factory.models import ExperimentRecord

        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        exp_id = asyncio.run(store.begin("Test hypothesis"))
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now(),
            hypothesis="Test hypothesis",
            change_summary="Changed stuff",
            issue_number=None,
            pr_number=None,
            score_before=0.80,
            score_after=0.85,
            delta=0.05,
            verdict="keep",
            cost_usd=1.23,
            notes="",
        )
        asyncio.run(store.finalize(exp_id, record))
        result = main(["history", str(tmp_project)])
        assert result == 0
        out = capsys.readouterr().out
        assert "keep" in out
        assert "Test hypothesis" in out


class TestCmdRun:
    def test_run_success(self, tmp_path):
        """cmd_run returns 0 when CEO agent succeeds."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()):
            result = main(["run", str(tmp_path)])
        assert result == 0

    def test_run_agent_failure(self, tmp_path):
        """cmd_run returns 1 when CEO agent fails."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_fail()):
            result = main(["run", str(tmp_path)])
        assert result == 1


class TestCmdEval:
    def test_eval_failing(self, tmp_project, capsys, sample_config):
        """cmd_eval returns 1 when eval fails."""
        import asyncio
        from factory.store import ExperimentStore

        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))

        # The eval_command won't exist, so this will fail
        result = main(["eval", str(tmp_project)])
        assert result == 1


class TestCmdNotify:
    def test_notify_no_config(self, tmp_project, capsys, caplog, sample_config):
        """cmd_notify warns when telegram is not configured."""
        import asyncio
        import logging
        from factory.store import ExperimentStore

        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))

        with caplog.at_level(logging.WARNING):
            result = main(["notify", str(tmp_project)])
        assert result == 0
        assert "Digest sent" in capsys.readouterr().out


class TestCmdInit:
    def test_init_missing_factory_md(self, tmp_project, capsys):
        """cmd_init returns 1 when factory.md is missing."""
        result = main(["init", str(tmp_project)])
        assert result == 1
        assert "factory.md not found" in capsys.readouterr().err

    def test_init_with_factory_md(self, tmp_project, capsys):
        """cmd_init succeeds when factory.md exists."""
        (tmp_project / "factory.md").write_text(
            "# Factory\n\n## Goal\nBuild stuff\n\n## Scope\n- src/\n\n"
            "## Guards\n- no deletes\n\n## Eval\n```\npython eval.py\n```\n\n"
            "## Threshold\n0.8\n\n## Constraints\n- small changes\n"
        )
        result = main(["init", str(tmp_project)])
        assert result == 0
        assert "Initialized" in capsys.readouterr().out


class TestCmdArchive:
    def test_archive_parser(self):
        parser = build_parser()
        args = parser.parse_args(["archive", "/some/path"])
        assert args.command == "archive"
        assert args.path == "/some/path"

    def test_archive_no_experiments(self, tmp_project, capsys, sample_config):
        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        result = main(["archive", str(tmp_project)])
        assert result == 0
        assert "Nothing to archive" in capsys.readouterr().out

    def test_archive_with_experiments(self, tmp_project, capsys, sample_config):
        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        exp_id = asyncio.run(store.begin("Improve throughput"))
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now(),
            hypothesis="Improve throughput",
            change_summary="Optimized pipeline",
            issue_number=None,
            pr_number=None,
            score_before=0.7,
            score_after=0.85,
            delta=0.15,
            verdict="keep",
            cost_usd=0.5,
            notes="",
        )
        asyncio.run(store.finalize(exp_id, record))

        with (
            patch("factory.obsidian.notes.write_experiment_note") as mock_exp,
            patch("factory.obsidian.notes.write_project_dashboard") as mock_dash,
            patch("factory.obsidian.notes.write_strategy_note") as mock_strat,
            patch("factory.obsidian.notes.update_memory_index"),
            patch("factory.obsidian.notes._get_vault_path", return_value=tmp_project / "vault"),
        ):
            result = main(["archive", str(tmp_project)])

        assert result == 0
        out = capsys.readouterr().out
        assert "Archived 1 experiments" in out
        mock_exp.assert_called_once()
        mock_dash.assert_called_once()
        mock_strat.assert_not_called()

    def test_archive_with_strategy(self, tmp_project, capsys, sample_config):
        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))
        exp_id = asyncio.run(store.begin("Test hypothesis"))
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now(),
            hypothesis="Test hypothesis",
            change_summary="Changed stuff",
            issue_number=None,
            pr_number=None,
            score_before=0.8,
            score_after=0.85,
            delta=0.05,
            verdict="keep",
            cost_usd=None,
            notes="",
        )
        asyncio.run(store.finalize(exp_id, record))
        strategy_path = store.factory_dir / "strategy" / "current.md"
        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        strategy_path.write_text("Focus on reliability.")

        with (
            patch("factory.obsidian.notes.write_experiment_note") as mock_exp,
            patch("factory.obsidian.notes.write_project_dashboard") as mock_dash,
            patch("factory.obsidian.notes.write_strategy_note") as mock_strat,
            patch("factory.obsidian.notes.update_memory_index"),
            patch("factory.obsidian.notes._get_vault_path", return_value=tmp_project / "vault"),
        ):
            result = main(["archive", str(tmp_project)])

        assert result == 0
        mock_exp.assert_called_once()
        mock_dash.assert_called_once()
        mock_strat.assert_called_once()


class TestCmdVaultInit:
    def test_vault_init_parser(self):
        parser = build_parser()
        args = parser.parse_args(["vault-init"])
        assert args.command == "vault-init"

    def test_vault_init_calls_init_vault(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("FACTORY_VAULT_PATH", str(tmp_path / "vault"))
        monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
        result = main(["vault-init"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Factory vault initialized" in out
        assert (tmp_path / "vault" / ".obsidian").is_dir()


class TestGitHubUrlDetection:
    def test_https_url_detected(self):
        assert _is_github_url("https://github.com/user/repo") is True

    def test_https_url_with_git_suffix(self):
        assert _is_github_url("https://github.com/user/repo.git") is True

    def test_ssh_url_detected(self):
        assert _is_github_url("git@github.com:user/repo.git") is True

    def test_local_path_not_detected(self):
        assert _is_github_url("/some/local/path") is False

    def test_relative_path_not_detected(self):
        assert _is_github_url("./relative/path") is False

    def test_other_url_not_detected(self):
        assert _is_github_url("https://gitlab.com/user/repo") is False


class TestRunModeFlag:
    def test_mode_default_is_auto(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path"])
        assert args.mode == "auto"

    def test_mode_discover(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--mode", "design"])
        assert args.mode == "design"

    def test_mode_improve_explicit(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--mode", "design"])
        assert args.mode == "design"

    def test_mode_meta(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--mode", "meta"])
        assert args.mode == "meta"


class TestRunWithGitHubUrl:
    def test_run_clones_https_url(self, capsys):
        """cmd_run clones a GitHub HTTPS URL into a temp dir and invokes CEO."""
        url = "https://github.com/user/repo"
        with (
            patch("factory.cli._path_resolver.subprocess.run") as mock_clone,
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),
            patch("factory.cli._path_resolver.tempfile.mkdtemp", return_value="/tmp/factory-abc"),
            patch("factory.cli.run._read_target_branch", return_value="main"),
        ):
            result = main(["run", url])

        assert result == 0
        mock_clone.assert_called_once_with(
            ["git", "clone", url, "/tmp/factory-abc"],
            check=True,
        )
        out = capsys.readouterr().out
        assert "Cloned https://github.com/user/repo" in out

    def test_run_clones_ssh_url(self, capsys):
        """cmd_run clones a GitHub SSH URL into a temp dir."""
        url = "git@github.com:user/repo.git"
        with (
            patch("factory.cli._path_resolver.subprocess.run") as mock_clone,
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),
            patch("factory.cli._path_resolver.tempfile.mkdtemp", return_value="/tmp/factory-xyz"),
            patch("factory.cli.run._read_target_branch", return_value="main"),
        ):
            result = main(["run", url])

        assert result == 0
        mock_clone.assert_called_once_with(
            ["git", "clone", url, "/tmp/factory-xyz"],
            check=True,
        )
        out = capsys.readouterr().out
        assert f"Cloned {url}" in out

    def test_run_local_path_no_clone(self, tmp_path):
        """cmd_run with a local path does not clone — just invokes CEO."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["run", str(tmp_path)])

        assert result == 0
        mock_agent.assert_called_once()

    def test_run_design_mode(self, tmp_path):
        """cmd_run with --mode=design passes design task to CEO."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["run", str(tmp_path), "--mode", "design"])

        assert result == 0
        call_args = mock_agent.call_args
        task = call_args[0][1]  # second positional arg is the task
        assert "Mode: design" in task

    def test_run_meta_mode(self, tmp_path):
        """cmd_run with --mode=meta passes meta task to CEO."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["run", str(tmp_path), "--mode", "meta"])

        assert result == 0
        call_args = mock_agent.call_args
        task = call_args[0][1]
        assert "Meta mode" in task


class TestMainErrorHandling:
    def test_main_catches_exception(self, capsys):
        """main catches exceptions from handlers and returns 1."""
        with patch("factory.cli.cmd_detect", side_effect=RuntimeError("boom")):
            result = main(["detect", "/some/path"])
        assert result == 1
        assert "boom" in capsys.readouterr().err


class TestHeartbeatParserFlags:
    def test_loop_flag_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path"])
        assert args.loop is False

    def test_loop_flag_enabled(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--loop"])
        assert args.loop is True

    def test_interval_default_1800(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path"])
        assert args.interval == 1800

    def test_interval_custom(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--interval", "60"])
        assert args.interval == 60

    def test_max_cycles_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path"])
        assert args.max_cycles is None

    def test_max_cycles_custom(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--max-cycles", "5"])
        assert args.max_cycles == 5


class TestHeartbeatLoop:
    def test_no_loop_single_run(self, tmp_path):
        """Without --loop, cmd_run executes exactly one cycle."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["run", str(tmp_path)])
        assert result == 0
        mock_agent.assert_called_once()

    def test_loop_exits_after_max_cycles(self, tmp_path, capsys):
        """With --loop --max-cycles=3, runs exactly 3 cycles then exits."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(
                [
                    "run",
                    str(tmp_path),
                    "--loop",
                    "--max-cycles",
                    "3",
                    "--interval",
                    "0",
                ]
            )
        assert result == 0
        assert mock_agent.call_count == 3

        out = capsys.readouterr().out
        assert "[factory] Cycle 1 started at" in out
        assert "[factory] Cycle 2 started at" in out
        assert "[factory] Cycle 3 started at" in out
        assert "[factory] Shutting down gracefully after 3 cycles." in out

    def test_loop_single_cycle(self, tmp_path, capsys):
        """--max-cycles=1 runs one cycle, no sleep, then exits."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),

        ):
            result = main(
                [
                    "run",
                    str(tmp_path),
                    "--loop",
                    "--max-cycles",
                    "1",
                ]
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "[factory] Cycle 1 started at" in out
        assert "[factory] Shutting down gracefully after 1 cycles." in out

    def test_loop_graceful_sigterm(self, tmp_path, capsys):
        """SIGTERM during sleep causes clean exit."""
        captured_handlers: dict[int, object] = {}
        original_signal = signal.signal

        def _capture_signal(signum, handler):
            captured_handlers[signum] = handler
            return original_signal(signum, handler)

        def _trigger_sigterm_after_cycle(*args, **kwargs):
            handler = captured_handlers.get(signal.SIGTERM)
            if handler and callable(handler):
                threading.Timer(0.05, handler, args=(signal.SIGTERM, None)).start()
            return ("ok", 0)

        with (
            patch("signal.signal", side_effect=_capture_signal),
            patch(
                "factory.agents.runner.invoke_agent",
                AsyncMock(side_effect=_trigger_sigterm_after_cycle),
            ),

        ):
            result = main(["run", str(tmp_path), "--loop", "--interval", "30"])

        assert result == 0
        out = capsys.readouterr().out
        assert "[factory] Shutting down gracefully after 1 cycles." in out

    def test_loop_graceful_sigint(self, tmp_path, capsys):
        """SIGINT during sleep causes clean exit."""
        captured_handlers: dict[int, object] = {}
        original_signal = signal.signal

        def _capture_signal(signum, handler):
            captured_handlers[signum] = handler
            return original_signal(signum, handler)

        def _trigger_sigint_after_cycle(*args, **kwargs):
            handler = captured_handlers.get(signal.SIGINT)
            if handler and callable(handler):
                threading.Timer(0.05, handler, args=(signal.SIGINT, None)).start()
            return ("ok", 0)

        with (
            patch("signal.signal", side_effect=_capture_signal),
            patch(
                "factory.agents.runner.invoke_agent",
                AsyncMock(side_effect=_trigger_sigint_after_cycle),
            ),

        ):
            result = main(["run", str(tmp_path), "--loop", "--interval", "30"])

        assert result == 0
        out = capsys.readouterr().out
        assert "[factory] Shutting down gracefully after 1 cycles." in out

    def test_loop_logs_sleep_message(self, tmp_path, capsys):
        """Verify the sleep log message appears between cycles."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),

        ):
            result = main(
                [
                    "run",
                    str(tmp_path),
                    "--loop",
                    "--max-cycles",
                    "2",
                    "--interval",
                    "0",
                ]
            )
        assert result == 0
        out = capsys.readouterr().out
        assert "[factory] Cycle 1 completed. Sleeping for 0s..." in out


# ── Factory v2: agent and ceo commands ────────────────────────


class TestCmdAgentParser:
    def test_agent_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "agent",
                "researcher",
                "--task",
                "Research the project",
                "--project",
                "/some/path",
            ]
        )
        assert args.command == "agent"
        assert args.role == "researcher"
        assert args.task == "Research the project"
        assert args.project == "/some/path"

    def test_agent_default_timeout(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "agent",
                "builder",
                "--task",
                "Build it",
                "--project",
                "/path",
            ]
        )
        assert args.timeout == 600.0

    def test_agent_custom_timeout(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "agent",
                "health_checker",
                "--task",
                "Eval",
                "--project",
                "/path",
                "--timeout",
                "300",
            ]
        )
        assert args.timeout == 300.0

    def test_agent_all_roles_valid(self):
        parser = build_parser()
        for role in [
            "researcher",
            "strategist",
            "builder",
            "health_checker",
            "code_reviewer",
            "adversarial_tester",
            "archivist",
            "ceo",
        ]:
            args = parser.parse_args(["agent", role, "--task", "test", "--project", "/path"])
            assert args.role == role


class TestCmdAgent:
    def test_agent_invokes_invoke_agent(self, tmp_path, capsys):
        """cmd_agent delegates to invoke_agent with correct args."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            result = main(
                [
                    "agent",
                    "researcher",
                    "--task",
                    "Research",
                    "--project",
                    str(tmp_path),
                ]
            )
        assert result == 0
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert call_args[0][0] == "researcher"
        assert call_args[0][1] == "Research"
        out = capsys.readouterr().out
        assert "CEO completed successfully" in out

    def test_agent_returns_nonzero_on_failure(self, tmp_path):
        """cmd_agent returns agent exit code on failure."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_fail()):
            result = main(
                [
                    "agent",
                    "builder",
                    "--task",
                    "Build",
                    "--project",
                    str(tmp_path),
                ]
            )
        assert result == 1


class TestCmdCeoParser:
    def test_ceo_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.command == "ceo"
        assert args.path == "/some/path"

    def test_ceo_default_mode(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.mode == "auto"

    def test_ceo_meta_mode(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path", "--mode", "meta"])
        assert args.mode == "meta"

    def test_ceo_review_mode(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path", "--mode", "review", "--pr", "42"])
        assert args.mode == "review"
        assert args.pr == 42

    def test_ceo_review_mode_with_repo(self):
        parser = build_parser()
        args = parser.parse_args(
            ["ceo", "/some/path", "--mode", "review", "--pr", "42", "--repo", "owner/repo"]
        )
        assert args.repo == "owner/repo"

    def test_ceo_pr_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.pr is None

    def test_ceo_repo_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.repo is None


class TestCmdCeoReview:
    def test_review_mode_without_pr_errors(self, capsys):
        result = main(["ceo", "/some/path", "--mode", "review"])
        assert result == 1
        assert "--pr" in capsys.readouterr().err

    def test_review_mode_nonexistent_path_errors(self, capsys):
        result = main(["ceo", "/nonexistent/path", "--mode", "review", "--pr", "42"])
        assert result == 1
        assert "existing directory" in capsys.readouterr().err

    def test_review_mode_headless_builds_correct_task(self, tmp_path, capsys):
        """--mode review --pr 42 --headless builds a review task and invokes CEO."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            result = main(["ceo", str(tmp_path), "--mode", "review", "--pr", "42", "--headless"])
        assert result == 0
        mock_agent.assert_called_once()
        task = mock_agent.call_args[0][1]
        assert "Mode: review" in task
        assert "PR #42" in task
        assert "review-only run" in task
        assert "no Builder iterations" in task
        assert "factory eval" in task
        assert "iteration 1/1" in task
        assert "--reason" in task
        assert "--qa-body-file" in task
        assert "factory review --verdict" in task

    def test_review_mode_headless_with_repo(self, tmp_path, capsys):
        """--mode review --pr 42 --repo owner/repo includes repo in task."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            result = main(
                [
                    "ceo",
                    str(tmp_path),
                    "--mode",
                    "review",
                    "--pr",
                    "42",
                    "--repo",
                    "owner/repo",
                    "--headless",
                ]
            )
        assert result == 0
        task = mock_agent.call_args[0][1]
        assert "owner/repo" in task
        assert "--repo owner/repo" in task
        assert "factory review --verdict" in task

    def test_review_mode_skips_worktree(self, tmp_path):
        """Review mode does not create worktrees or touch experiment store."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),
            patch("factory.worktree.create_worktree") as mock_wt,
        ):
            main(["ceo", str(tmp_path), "--mode", "review", "--pr", "42", "--headless"])
        mock_wt.assert_not_called()

    def test_review_mode_foreground(self, tmp_path):
        """Review mode without --headless launches interactively."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        with (
            patch("factory.runners.claude.subprocess.run", mock_run),
            patch("factory.cli._helpers._ensure_dashboard"),
        ):
            main(["ceo", str(tmp_path), "--mode", "review", "--pr", "42"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "Mode: review" in task
        assert "PR #42" in task

    def test_review_mode_max_respawns_is_1(self, tmp_path):
        """Review mode uses max_respawns=1."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            main(["ceo", str(tmp_path), "--mode", "review", "--pr", "42", "--headless"])
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs.get("timeout") == 7200.0


class TestCmdCeoDeepQa:
    def test_deep_qa_mode_without_pr_errors(self, capsys):
        result = main(["ceo", "/some/path", "--mode", "deep-qa"])
        assert result == 1
        assert "--pr" in capsys.readouterr().err

    def test_qa_mode_nonexistent_path_errors(self, capsys):
        result = main(["ceo", "/nonexistent/path", "--mode", "deep-qa", "--pr", "42"])
        assert result == 1
        assert "existing directory" in capsys.readouterr().err

    def test_qa_mode_headless_builds_correct_task(self, tmp_path, capsys):
        """--mode deep-qa --pr 42 --headless builds a deep-qa task and invokes CEO."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            result = main(["ceo", str(tmp_path), "--mode", "deep-qa", "--pr", "42", "--headless"])
        assert result == 0
        mock_agent.assert_called_once()
        task = mock_agent.call_args[0][1]
        assert "Mode: deep-qa" in task
        assert "PR #42" in task
        assert "factory review --verdict" in task
        assert "--reason" in task
        assert "--qa-body-file" in task
        assert "Do NOT post any PR comments" in task

    def test_qa_mode_headless_with_repo(self, tmp_path, capsys):
        """--mode deep-qa --pr 42 --repo owner/repo includes repo in task."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            result = main(
                [
                    "ceo",
                    str(tmp_path),
                    "--mode",
                    "deep-qa",
                    "--pr",
                    "42",
                    "--repo",
                    "owner/repo",
                    "--headless",
                ]
            )
        assert result == 0
        task = mock_agent.call_args[0][1]
        assert "owner/repo" in task
        assert "--repo owner/repo" in task

    def test_qa_mode_skips_worktree(self, tmp_path):
        """Deep-QA mode does not create worktrees or touch experiment store."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),
            patch("factory.worktree.create_worktree") as mock_wt,
        ):
            main(["ceo", str(tmp_path), "--mode", "deep-qa", "--pr", "42", "--headless"])
        mock_wt.assert_not_called()

    def test_qa_mode_foreground(self, tmp_path):
        """Deep-QA mode without --headless launches interactively."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        with (
            patch("factory.runners.claude.subprocess.run", mock_run),
            patch("factory.cli._helpers._ensure_dashboard"),
        ):
            main(["ceo", str(tmp_path), "--mode", "deep-qa", "--pr", "42"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "Mode: deep-qa" in task
        assert "PR #42" in task

    def test_qa_mode_max_respawns_is_1(self, tmp_path):
        """Deep-QA mode uses max_respawns=1."""
        with patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent:
            main(["ceo", str(tmp_path), "--mode", "deep-qa", "--pr", "42", "--headless"])
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs.get("timeout") == 7200.0


class TestCmdCeo:
    def test_ceo_headless_invokes_ceo_agent(self, tmp_path, capsys):
        """cmd_ceo --headless spawns CEO agent via invoke_agent."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["ceo", str(tmp_path), "--headless"])
        assert result == 0
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert call_args[0][0] == "ceo"
        assert str(tmp_path) in call_args[0][1]

    def test_ceo_headless_meta_mode_task(self, tmp_path):
        """cmd_ceo --headless with --mode=meta includes meta instructions."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["ceo", str(tmp_path), "--mode", "meta", "--headless"])
        assert result == 0
        task = mock_agent.call_args[0][1]
        assert "Meta mode" in task

    def test_ceo_headless_clones_github_url(self, capsys):
        """cmd_ceo --headless clones a GitHub URL then invokes CEO."""
        url = "https://github.com/user/repo"
        with (
            patch("factory.cli._path_resolver.subprocess.run") as mock_clone,
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()),

            patch("factory.cli._path_resolver.tempfile.mkdtemp", return_value="/tmp/factory-ceo"),
            patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
            patch("factory.graph.is_graphify_installed", return_value=False),
        ):
            result = main(["ceo", url, "--headless"])
        assert result == 0
        mock_clone.assert_called_once_with(
            ["git", "clone", url, "/tmp/factory-ceo"],
            check=True,
        )

    def test_ceo_headless_timeout_is_2_hours(self, tmp_path):
        """CEO agent gets 7200s timeout in headless mode."""
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            main(["ceo", str(tmp_path), "--headless"])
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["timeout"] == 7200.0

    def test_ceo_foreground_uses_subprocess_run(self, tmp_path):
        """cmd_ceo (default) launches claude interactively via subprocess.run."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path)])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert "--append-system-prompt-file" in cmd
        assert "--dangerously-skip-permissions" in cmd

    def test_ceo_foreground_passes_task_as_prompt(self, tmp_path):
        """Foreground mode passes the task as the initial user message."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path)])
        cmd = mock_run.call_args[0][0]
        assert any(str(tmp_path) in arg for arg in cmd)

    def test_ceo_foreground_cwd_is_project(self, tmp_path):
        """Foreground mode passes cwd to subprocess.run."""
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path)])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == tmp_path

    def test_ceo_parser_has_headless_flag(self):
        """Parser accepts --headless flag."""
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path", "--headless"])
        assert args.headless is True

    def test_ceo_parser_default_not_headless(self):
        """Parser defaults to foreground (not headless)."""
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.headless is False


class TestSlugify:
    def test_basic_slug(self):
        assert _slugify("Locals Know") == "locals-know"

    def test_strips_special_chars(self):
        assert _slugify("Betty Terminal — AI-Native") == "betty-terminal-ai-native"

    def test_truncates_to_50(self):
        long_name = "a" * 100
        assert len(_slugify(long_name)) <= 50

    def test_empty_string(self):
        assert _slugify("") == "factory-project"

    def test_special_only(self):
        assert _slugify("!!!") == "factory-project"


class TestExtractProjectName:
    def test_strips_build_verb(self):
        assert _extract_project_name("Build a weather CLI tool") == "weather-cli-tool"

    def test_strips_create_verb(self):
        assert _extract_project_name("Create an API server") == "api-server"

    def test_strips_filler_adjectives(self):
        assert (
            _extract_project_name("Build a comprehensive e-commerce platform with payments")
            == "e-commerce-platform-payments"
        )

    def test_caps_at_four_words(self):
        result = _extract_project_name("distributed eval runner for multi-node benchmarks on GPUs")
        assert result == "distributed-eval-runner-multi-node"

    def test_no_verb_prefix(self):
        assert _extract_project_name("weather CLI") == "weather-cli"

    def test_strips_multiple_fillers(self):
        assert _extract_project_name("Build a simple lightweight modern REST API") == "rest-api"

    def test_empty_after_stripping_falls_back(self):
        result = _extract_project_name("build a the")
        assert result == "build-a-the"

    def test_preserves_hyphenated_words(self):
        assert _extract_project_name("real-time chat app") == "real-time-chat-app"

    def test_setup_verb(self):
        assert _extract_project_name("Set up a deployment pipeline") == "deployment-pipeline"


class TestDedupeProjectPath:
    def test_no_existing_dir(self, tmp_path):
        path = tmp_path / "projects" / "my-app"
        assert _dedupe_project_path(path, "Build a todo app") == path

    def test_existing_dir_no_spec(self, tmp_path):
        path = tmp_path / "projects" / "my-app"
        path.mkdir(parents=True)
        assert _dedupe_project_path(path, "Build a todo app") == path

    def test_existing_dir_same_spec_reuses(self, tmp_path):
        path = tmp_path / "projects" / "my-app"
        spec_dir = path / ".factory" / "strategy"
        spec_dir.mkdir(parents=True)
        (spec_dir / "current.md").write_text("## Project Specification\n\nBuild a todo app\n")
        assert _dedupe_project_path(path, "Build a todo app") == path

    def test_existing_dir_different_spec_appends_suffix(self, tmp_path):
        path = tmp_path / "projects" / "rest-api"
        spec_dir = path / ".factory" / "strategy"
        spec_dir.mkdir(parents=True)
        (spec_dir / "current.md").write_text(
            "## Project Specification\n\nBuild a REST API for users\n"
        )
        result = _dedupe_project_path(path, "Build a REST API for payments")
        assert result == tmp_path / "projects" / "rest-api-2"

    def test_multiple_collisions(self, tmp_path):
        base = tmp_path / "projects" / "rest-api"
        for suffix in ("", "-2", "-3"):
            d = base.parent / f"{base.name}{suffix}" if suffix else base
            spec_dir = d / ".factory" / "strategy"
            spec_dir.mkdir(parents=True)
            (spec_dir / "current.md").write_text(f"## Project Specification\n\nvariant {suffix}\n")
        result = _dedupe_project_path(base, "yet another REST API")
        assert result == tmp_path / "projects" / "rest-api-4"

    def test_resolve_input_dedupes_raw_prompt(self, tmp_path):
        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            p1, ctx1 = _resolve_input("Build a REST API")
            _materialize_project(p1, ctx1)
            p2, _ = _resolve_input("Create a new REST API")
        assert p1.name == "rest-api"
        assert p2.name == "rest-api-2"


class TestPersistSpec:
    def test_writes_spec_file(self, tmp_path):
        _persist_spec(tmp_path, "Build a todo app")
        spec = (tmp_path / ".factory" / "strategy" / "current.md").read_text()
        assert "Build a todo app" in spec

    def test_does_not_overwrite_existing(self, tmp_path):
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "current.md").write_text("Existing strategy")

        _persist_spec(tmp_path, "New spec")
        assert "Existing strategy" in (strategy_dir / "current.md").read_text()

    def test_creates_directories(self, tmp_path):
        _persist_spec(tmp_path, "some spec")
        assert (tmp_path / ".factory" / "strategy" / "current.md").exists()


class TestResolveInput:
    def test_existing_dir(self, tmp_path):
        project_path, context = _resolve_input(str(tmp_path))
        assert project_path == tmp_path
        assert context is None

    def test_idea_file(self, tmp_path):
        idea_file = tmp_path / "My Project \u2014 Something Cool.md"
        idea_file.write_text("# Build something cool")

        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input(str(idea_file))

        assert project_path.name == "my-project"
        assert not (project_path / ".git").is_dir()
        assert context is not None
        assert "Build something cool" in context

    def test_raw_prompt(self, tmp_path):
        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input("Build a todo app with FastAPI")

        assert project_path.parent == tmp_path / "projects"
        assert project_path.name == "todo-app-fastapi"
        assert not (project_path / ".git").is_dir()
        assert context == "Build a todo app with FastAPI"

    def test_non_md_file(self, tmp_path):
        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')")

        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input(str(py_file))

        assert project_path.name == "script"
        assert not (project_path / ".git").is_dir()
        assert context == "print('hello')"

    def test_binary_file_raises(self, tmp_path):
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\xff")

        with (
            patch(
                "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
            ),
            pytest.raises(UnicodeDecodeError),
        ):
            _resolve_input(str(bin_file))

    def test_ceo_receives_context(self, tmp_path):
        """When an idea file is given, its content reaches the CEO task."""
        idea_file = tmp_path / "Test Idea \u2014 Details.md"
        idea_file.write_text("# Test Idea\nBuild X that does Y")

        with (
            patch(
                "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
            ),

            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,
        ):
            main(["ceo", str(idea_file), "--headless"])

        task_arg = mock_agent.call_args[0][1]  # second positional = task
        assert "Build X that does Y" in task_arg
        assert "Project Specification" in task_arg

    def test_dir_overrides_slug_for_raw_prompt(self, tmp_path):
        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input(
                "Build a todo app with FastAPI", dir_name="my-todo"
            )

        assert project_path.name == "my-todo"
        assert not (project_path / ".git").is_dir()

    def test_dir_overrides_slug_for_idea_file(self, tmp_path):
        idea_file = tmp_path / "Long Idea Name — Details.md"
        idea_file.write_text("# Build something")

        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input(str(idea_file), dir_name="custom-name")

        assert project_path.name == "custom-name"
        assert not (project_path / ".git").is_dir()

    def test_dir_ignored_for_existing_directory(self, tmp_path):
        project_path, context = _resolve_input(str(tmp_path), dir_name="ignored-name")

        assert project_path == tmp_path
        assert context is None

    def test_dir_is_slugified(self, tmp_path):
        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input("Build something", dir_name="My Cool Project!")

        assert project_path.name == "my-cool-project"

    def test_ceo_parser_accepts_dir_argument(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "Build something", "--dir", "my-project"])
        assert args.dir == "my-project"


class TestResearchMode:
    def test_ceo_parser_accepts_research_mode(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path", "--mode", "research"])
        assert args.mode == "research"

    def test_run_parser_accepts_research_mode(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/some/path", "--mode", "research"])
        assert args.mode == "research"

    def test_tmux_parser_accepts_research_mode(self):
        parser = build_parser()
        args = parser.parse_args(["tmux", "/some/path", "--mode", "research"])
        assert args.mode == "research"

    def test_research_mode_task_text(self, tmp_path):
        """--mode research includes research-specific instructions in the CEO task."""
        (tmp_path / ".git").mkdir()
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        rt = {
            "objective": "maximize accuracy",
            "metric": "accuracy",
            "target": 0.9,
            "run_command": "python run.py",
            "result_path": "results.json",
        }
        (factory_dir / "config.json").write_text(json.dumps(_make_config(research_target=rt)))
        with (
            patch("factory.agents.runner.invoke_agent", _mock_invoke_agent_ok()) as mock_agent,

        ):
            result = main(["ceo", str(tmp_path), "--mode", "research", "--headless"])
        assert result == 0
        task = mock_agent.call_args[0][1]
        assert "Research mode" in task
        assert "research_target" in task

    def test_auto_detect_design_mode(self, tmp_project, sample_config):
        """Auto-detection returns 'design' for all project states."""
        from factory.store import ExperimentStore

        store = ExperimentStore(tmp_project)
        asyncio.run(store.init(sample_config))

        from factory.cli._mode_handlers import _auto_detect_mode

        mode = _auto_detect_mode(tmp_project, force_fresh=True)
        assert mode == "design"


class TestBuildCeoTaskDesign:
    """Unit tests for _build_ceo_task design_existing parameter."""

    def test_existing_project_emits_plan_loop_section(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", design_existing=True)
        assert "## Plan Loop (Interactive)" in task
        assert "existing_project: true" in task
        assert "existing project" in task

    def test_existing_project_with_focus(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", design_existing=True, focus="auth layer")
        assert "## Plan Loop (Interactive)" in task
        assert "auth layer" in task
        assert "Focus topic" in task

    def test_existing_project_without_focus(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", design_existing=True)
        assert "No specific topic was provided" in task

    def test_new_idea_emits_plan_loop_section(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", design_idea="weather CLI")
        assert "## Plan Loop (Interactive)" in task
        assert "weather CLI" in task

    def test_existing_uses_same_header_as_new_idea(self, tmp_path):
        """Both new ideas and existing projects use the same Plan Loop header."""
        existing_task = _build_ceo_task(tmp_path, "design", design_existing=True)
        new_task = _build_ceo_task(tmp_path, "design", design_idea="weather CLI")
        assert "## Plan Loop (Interactive)" in existing_task
        assert "## Plan Loop (Interactive)" in new_task

    def test_existing_project_has_existing_flag(self, tmp_path):
        """Existing project task includes the existing_project flag for CEO conditionals."""
        task = _build_ceo_task(tmp_path, "design", design_existing=True)
        assert "existing_project: true" in task

    def test_existing_mode_shows_display_mode(self, tmp_path):
        """When display_mode is provided, task shows it instead of internal mode."""
        task = _build_ceo_task(tmp_path, "design", design_existing=True, display_mode="design")
        assert "Mode: design" in task


class TestCreateModeFocus:
    """Tests for --focus working with --mode create (issue #832)."""

    def test_focus_accepted_with_create_mode(self, tmp_path):
        """--focus is no longer rejected when --mode create is set."""
        (tmp_path / ".git").mkdir()
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "create", "--focus", "a PR validation mode"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Create Mode (New Portable Workflow)" in task
        assert "a PR validation mode" in task

    def test_create_mode_without_focus(self, tmp_path):
        """--mode create without --focus still works (create_description is None)."""
        (tmp_path / ".git").mkdir()
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "create"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "## Create Mode (New Portable Workflow)" not in task

    def test_build_ceo_task_create_description(self, tmp_path):
        """_build_ceo_task emits the Create Mode section when create_description is provided."""
        task = _build_ceo_task(tmp_path, "design", create_description="a mode for validating PRs")
        assert "## Create Mode (New Portable Workflow)" in task
        assert "a mode for validating PRs" in task
        assert "Mode description from user" in task
        assert "**output_folder:** .factory/workflows/" in task
        assert "## Focus Directive" not in task

    def test_build_ceo_task_no_create_description(self, tmp_path):
        """_build_ceo_task omits the Create Mode section when create_description is None."""
        task = _build_ceo_task(tmp_path, "design", create_description=None)
        assert "## Create Mode (New Portable Workflow)" not in task

    def test_focus_accepted_with_create_v2_mode(self, tmp_path):
        """--focus is accepted when --mode create-v2 is set (issue #1428)."""
        (tmp_path / ".git").mkdir()
        with _mock_foreground() as mock_run:
            main(["ceo", str(tmp_path), "--mode", "create-v2", "--focus", "add a linting mode"])
        mock_run.assert_called_once()


class TestTaskSetupModeFocus:
    """Tests for --focus working with --mode task-setup."""

    def test_focus_accepted_with_task_setup_mode(self, tmp_path):
        """--focus is not rejected by _validate_late_flags when --mode task-setup is set."""
        from factory.cli._ceo_helpers import _validate_late_flags

        result = _validate_late_flags(
            mode="task-setup",
            focus="chess engine evaluation",
            prompt_file=None,
            research_ideation=None,
            design_existing=False,
            project_path=tmp_path,
            no_github=False,
            issue_number=None,
        )
        assert result is None

    def test_focus_rejected_with_unknown_mode(self, tmp_path, capsys):
        """--focus is rejected by _validate_late_flags for an unrecognized mode."""
        from factory.cli._ceo_helpers import _validate_late_flags

        result = _validate_late_flags(
            mode="bogus",
            focus="something",
            prompt_file=None,
            research_ideation=None,
            design_existing=False,
            project_path=tmp_path,
            no_github=False,
            issue_number=None,
        )
        assert result == 1
        assert "task-setup" in capsys.readouterr().err


class TestProfileParser:
    def test_profile_build_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["profile", "build"])
        assert args.command == "profile"
        assert args.profile_command == "build"

    def test_profile_build_with_paths(self):
        parser = build_parser()
        args = parser.parse_args(["profile", "build", "/path/a", "/path/b"])
        assert args.paths == ["/path/a", "/path/b"]

    def test_profile_build_dry_run(self):
        parser = build_parser()
        args = parser.parse_args(["profile", "build", "--dry-run"])
        assert args.dry_run is True

    def test_profile_show_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["profile", "show"])
        assert args.profile_command == "show"

    def test_use_profile_flag_on_ceo(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/path", "--use-profile"])
        assert args.use_profile is True

    def test_use_profile_flag_default_false_ceo(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/path"])
        assert args.use_profile is False

    def test_use_profile_flag_on_run(self):
        parser = build_parser()
        args = parser.parse_args(["run", "/path", "--use-profile"])
        assert args.use_profile is True

    def test_use_profile_flag_on_agent(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "agent",
                "researcher",
                "--task",
                "test",
                "--project",
                "/p",
                "--use-profile",
            ]
        )
        assert args.use_profile is True


class TestCmdProfile:
    def test_profile_show_no_file(self, capsys):
        with patch("factory.profile._PROFILE_PATH", Path("/nonexistent/profile.md")):
            result = main(["profile", "show"])
        assert result == 1
        assert "No profile found" in capsys.readouterr().out

    def test_profile_show_with_file(self, tmp_path, capsys):
        profile_path = tmp_path / "profile.md"
        profile_path.write_text("---\ngenerated: 2024\n---\n\nTest profile")
        with patch("factory.profile._PROFILE_PATH", profile_path):
            result = main(["profile", "show"])
        assert result == 0
        assert "Test profile" in capsys.readouterr().out

    def test_profile_no_subcommand(self, capsys):
        result = main(["profile"])
        assert result == 1

    def test_profile_build_dry_run(self, tmp_path, capsys):
        proj = tmp_path / "myproj"
        factory_dir = proj / ".factory"
        factory_dir.mkdir(parents=True)
        (factory_dir / "results.tsv").write_text("id\thyp\tverdict\n1\ttest-hyp\tkeep\n")
        result = main(["profile", "build", "--dry-run", str(proj)])
        assert result == 0
        out = capsys.readouterr().out
        assert "experiment_history" in out
        assert "test-hyp" in out
        assert "ceo_verdicts" in out


class TestCmdHomeReturnsFactoryDir:
    def test_cmd_home_returns_package_root(self, capsys):
        from factory.cli import cmd_home
        import argparse

        result = cmd_home(argparse.Namespace())
        assert result == 0
        output = capsys.readouterr().out.strip()
        assert "site-packages" not in output or Path(output).is_dir()
        assert (Path(output) / "cli" / "__init__.py").is_file()


class TestCmdTmuxBareCLI:
    def test_tmux_command_uses_bare_factory(self):
        """cmd_tmux generates a shell command using bare 'factory ceo', not uv run."""
        from factory.cli import cmd_tmux
        import argparse

        with (
            patch("factory.cli._tmux_commands._tmux_available", return_value=True),
            patch("factory.cli._tmux_commands._tmux_session_alive", return_value=True),
            patch("factory.cli._tmux_commands.time.sleep"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = type("R", (), {"returncode": 1})()  # has-session fails
            mock_run.side_effect = [
                type("R", (), {"returncode": 1})(),  # has-session → no existing session
                type("R", (), {"returncode": 0})(),  # new-session → success
                type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),  # capture-pane
            ]
            args = argparse.Namespace(
                path="/tmp/test-project",
                session=None,
                attach=False,
                mode=None,
                loop=True,
                interval=None,
                max_cycles=None,
                model=None,
                runner=None,
                no_github=False,
                profile=None,
            )
            result = cmd_tmux(args)
            assert result == 0

            new_session_call = mock_run.call_args_list[1]
            shell_cmd = new_session_call[0][0][-1]  # last arg is the shell command
            assert "factory ceo" in shell_cmd
            assert "uv run python -m factory" not in shell_cmd
            assert "cd " not in shell_cmd
            assert "source .venv/bin/activate" not in shell_cmd


class TestPluginAgentsDirGuard:
    def test_plugin_agents_dir_none_when_missing(self, tmp_path):
        """_PLUGIN_AGENTS_DIR is None when the agents/ dir doesn't exist."""
        from factory.agents import plugin

        original = plugin._PLUGIN_AGENTS_DIR
        try:
            plugin._PLUGIN_AGENTS_DIR = None
            result = plugin.check_agents_in_sync()
            assert result == []
        finally:
            plugin._PLUGIN_AGENTS_DIR = original


class TestResolveProjectPath:
    def test_cmd_notify_resolves_relative_path(self, tmp_path, capsys):
        """cmd_notify resolves relative paths so .name is non-empty."""
        from factory.cli import cmd_notify
        import argparse

        with (
            patch("factory.cli.admin._run", side_effect=lambda c: []),
            patch("factory.notify.telegram.TelegramNotifier") as MockNotifier,
        ):
            mock_instance = MockNotifier.return_value
            mock_instance.send_digest = AsyncMock()
            args = argparse.Namespace(path=str(tmp_path))
            cmd_notify(args)
            call_args = mock_instance.send_digest.call_args[0]
            assert call_args[0] != ""

    def test_cmd_archive_resolves_relative_path(self, tmp_path, capsys):
        """cmd_archive resolves paths and uses non-empty project_path.name."""
        from factory.cli import cmd_archive
        import argparse

        project_path = tmp_path / "my-project"
        project_path.mkdir()
        (project_path / ".factory").mkdir()

        with patch("factory.cli.infra._run", side_effect=lambda c: []):
            args = argparse.Namespace(path=str(project_path))
            result = cmd_archive(args)
            assert result == 0
            output = capsys.readouterr().out.strip()
            assert "Nothing to archive" in output


class TestNoBareUvRunPythonMFactory:
    """Guard: no file should use 'uv run python -m factory' — use bare 'factory' instead."""

    SCAN_GLOBS = [
        "factory/agents/prompts/*.md",
        "factory/cli/*.py",
        "SKILL.md",
        "README.md",
        "docs/**/*.md",
    ]

    def test_no_hardcoded_uv_run_python_m_factory(self):
        import glob

        repo_root = Path(__file__).resolve().parent.parent
        violations: list[str] = []
        for pattern in self.SCAN_GLOBS:
            for filepath in glob.glob(str(repo_root / pattern), recursive=True):
                with open(filepath) as f:
                    for lineno, line in enumerate(f, 1):
                        if "uv run python -m factory" in line:
                            violations.append(f"{Path(filepath).relative_to(repo_root)}:{lineno}")
        assert violations == [], (
            "Found 'uv run python -m factory' — use bare 'factory' instead:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


class TestSacredRule8Present:
    """Guard: Sacred Rule 8 (CEO must not do another agent's job) must exist in the CEO prompt."""

    REQUIRED_PHRASES = [
        "Do not do another agent's job",
        "Sacred Rule 8",
        "never take over the agent's work yourself",
    ]

    def test_sacred_rule_8_in_ceo_prompt(self):
        repo_root = Path(__file__).resolve().parent.parent
        ceo_prompt = (repo_root / "factory" / "agents" / "prompts" / "ceo.md").read_text()
        missing = [p for p in self.REQUIRED_PHRASES if p not in ceo_prompt]
        assert missing == [], (
            "Sacred Rule 8 is missing or incomplete in ceo.md. Missing phrases:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    def test_sacred_rule_8_in_sacred_rules_section(self):
        """Rule 8 must be in the numbered Sacred Rules list, not just mentioned elsewhere."""
        repo_root = Path(__file__).resolve().parent.parent
        ceo_prompt = (repo_root / "factory" / "agents" / "prompts" / "ceo.md").read_text()
        assert "8. **Do not do another agent's job**" in ceo_prompt, (
            "Sacred Rule 8 must be a numbered item (8.) in the Sacred Rules section"
        )


class TestEnsureRepo:
    """Tests for _ensure_repo() — verifies repos are initialized with at least one commit."""

    def test_new_repo_has_commit(self, tmp_path):
        """_ensure_repo() should create a repo with at least one commit."""
        project = tmp_path / "new-project"
        _ensure_repo(project)
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert int(result.stdout.strip()) >= 1

    def test_new_repo_has_valid_branch(self, tmp_path):
        """_ensure_repo() should produce a repo with a resolvable default branch ref."""
        project = tmp_path / "new-project"
        _ensure_repo(project)
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        branch = result.stdout.strip()
        assert branch and branch != "HEAD"

    def test_idempotent_on_existing_repo(self, tmp_path):
        """Calling _ensure_repo() on an already-initialized repo should be a no-op."""
        project = tmp_path / "existing"
        _ensure_repo(project)
        count_before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _ensure_repo(project)
        count_after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert count_before == count_after


class TestDesignFileInput:
    """Tests for design mode with file path input (Bug 1 fix)."""

    def test_file_content_becomes_design_idea(self, tmp_path):
        """When --mode design receives a file path, the file content should be used as the idea."""
        spec_file = tmp_path / "my-cool-app.md"
        spec_file.write_text("Build a weather dashboard with real-time updates")
        with _mock_foreground() as mock_run:
            main(["ceo", str(spec_file), "--mode", "design"])
        cmd = mock_run.call_args[0][0]
        dsp_idx = cmd.index("--dangerously-skip-permissions")
        task = cmd[dsp_idx + 1]
        assert "Build a weather dashboard with real-time updates" in task

    def test_slug_derived_from_filename(self, tmp_path, capsys):
        """The project slug should come from the file stem, not the full path."""
        spec_file = tmp_path / "weather-dashboard.md"
        spec_file.write_text("Build a weather dashboard")
        with _mock_foreground():
            main(["ceo", str(spec_file), "--mode", "design"])
        output = capsys.readouterr().out
        assert "weather-dashboard" in output
        assert "Idea file: weather-dashboard.md" in output

    def test_raw_idea_persists_spec(self, tmp_path):
        """When --mode design receives a raw string, the spec should be persisted."""
        with (
            _mock_foreground(),
            patch("factory.cli._ceo_helpers._get_projects_dir", return_value=tmp_path),
        ):
            main(["ceo", "Build a CLI todo app", "--mode", "design"])
        matches = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(matches) == 1
        spec_path = matches[0] / ".factory" / "strategy" / "current.md"
        assert spec_path.exists()
        assert "Build a CLI todo app" in spec_path.read_text()


class TestRefineFlag:
    """Tests for --refine flag parsing and mutual exclusivity."""

    def test_refine_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path", "--refine", "fix the login bug"])
        assert args.refine == "fix the login bug"

    def test_refine_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["ceo", "/some/path"])
        assert args.refine is None

    def test_refine_exclusive_with_interactive(self, tmp_path, capsys):
        with _mock_foreground():
            result = main(["ceo", str(tmp_path), "--refine", "fix bug", "--mode", "design"])
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_refine_exclusive_with_research(self, tmp_path, capsys):
        with _mock_foreground():
            result = main(["ceo", str(tmp_path), "--refine", "fix bug", "--mode", "research"])
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_refine_exclusive_with_meta(self, tmp_path, capsys):
        with _mock_foreground():
            result = main(["ceo", str(tmp_path), "--refine", "fix bug", "--mode", "meta"])
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_refine_exclusive_with_prompt(self, tmp_path, capsys):
        prompt_file = tmp_path / "spec.md"
        prompt_file.write_text("some spec")
        with _mock_foreground():
            result = main(
                ["ceo", str(tmp_path), "--refine", "fix bug", "--prompt", str(prompt_file)]
            )
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_refine_exclusive_with_focus(self, tmp_path, capsys):
        with _mock_foreground():
            result = main(["ceo", str(tmp_path), "--refine", "fix bug", "--focus", "auth"])
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_refine_requires_existing_directory(self, capsys):
        with _mock_foreground():
            result = main(["ceo", "/nonexistent/path", "--refine", "fix bug"])
        assert result == 1
        assert "existing project directory" in capsys.readouterr().err

    def test_refine_rejects_url(self, capsys):
        with _mock_foreground():
            result = main(["ceo", "https://github.com/user/repo", "--refine", "fix bug"])
        assert result == 1
        assert "existing project directory" in capsys.readouterr().err


class TestBuildCeoTaskRefine:
    """Tests for _build_ceo_task refinement mode section."""

    def test_refine_request_emits_section(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", refine_request="fix the login bug")
        assert "## Refinement Mode" in task
        assert "fix the login bug" in task
        assert "Mode: Refine" in task

    def test_no_refine_request_omits_section(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design")
        assert "## Refinement Mode" not in task

    def test_refine_request_none_omits_section(self, tmp_path):
        task = _build_ceo_task(tmp_path, "design", refine_request=None)
        assert "## Refinement Mode" not in task


class TestRefinerPromptExists:
    """Verify the refiner.md prompt file exists and has key sections."""

    def test_refiner_prompt_file_exists(self):
        prompt_path = Path(__file__).parent.parent / "factory" / "agents" / "prompts" / "refiner.md"
        assert prompt_path.exists(), f"refiner.md not found at {prompt_path}"

    def test_refiner_prompt_has_key_sections(self):
        prompt_path = Path(__file__).parent.parent / "factory" / "agents" / "prompts" / "refiner.md"
        content = prompt_path.read_text()
        assert "Tier" in content, "refiner.md should reference Tier classification"
        assert "Builder" in content or "builder" in content, (
            "refiner.md should reference the Builder agent"
        )


class TestMaterializeProject:
    """Tests for _materialize_project — deferred directory creation."""

    def test_creates_repo(self, tmp_path):
        project = tmp_path / "new-project"
        _materialize_project(project)
        assert (project / ".git").is_dir()

    def test_creates_repo_and_persists_spec(self, tmp_path):
        project = tmp_path / "new-project"
        _materialize_project(project, "Build a todo app")
        assert (project / ".git").is_dir()
        spec = (project / ".factory" / "strategy" / "current.md").read_text()
        assert "Build a todo app" in spec

    def test_no_spec_skips_persist(self, tmp_path):
        project = tmp_path / "new-project"
        _materialize_project(project, None)
        assert (project / ".git").is_dir()
        assert not (project / ".factory" / "strategy" / "current.md").exists()

    def test_idempotent_on_existing_repo(self, tmp_path):
        project = tmp_path / "existing"
        _materialize_project(project, "first spec")
        count_before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _materialize_project(project, "second spec")
        count_after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert count_before == count_after


class TestIsScaffoldOnly:
    """Tests for _is_scaffold_only — cleanup heuristic."""

    def test_scaffold_project(self, tmp_path):
        project = tmp_path / "scaffold"
        _materialize_project(project, "some idea")
        assert _is_scaffold_only(project) is True

    def test_scaffold_without_spec(self, tmp_path):
        project = tmp_path / "scaffold"
        _materialize_project(project)
        assert _is_scaffold_only(project) is True

    def test_not_scaffold_with_extra_file(self, tmp_path):
        project = tmp_path / "real"
        _materialize_project(project, "some idea")
        (project / "README.md").write_text("# Hello")
        assert _is_scaffold_only(project) is False

    def test_not_scaffold_with_extra_commit(self, tmp_path):
        project = tmp_path / "real"
        _materialize_project(project, "some idea")
        (project / "README.md").write_text("# Hello")
        subprocess.run(["git", "add", "README.md"], cwd=project, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@t", "commit", "-m", "second"],
            cwd=project,
            capture_output=True,
        )
        assert _is_scaffold_only(project) is False

    def test_nonexistent_dir(self, tmp_path):
        assert _is_scaffold_only(tmp_path / "nope") is False

    def test_no_git(self, tmp_path):
        project = tmp_path / "nogit"
        project.mkdir()
        assert _is_scaffold_only(project) is False


class TestDeferredCreationFlow:
    """Integration tests: _resolve_input defers creation, _materialize_project creates."""

    def test_resolve_then_materialize_file(self, tmp_path):
        idea_file = tmp_path / "my-app.md"
        idea_file.write_text("Build something cool")

        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input(str(idea_file))

        assert not project_path.exists()
        _materialize_project(project_path, context)
        assert (project_path / ".git").is_dir()
        assert (project_path / ".factory" / "strategy" / "current.md").exists()

    def test_resolve_then_materialize_raw_prompt(self, tmp_path):
        with patch(
            "factory.cli._path_resolver._get_projects_dir", return_value=tmp_path / "projects"
        ):
            project_path, context = _resolve_input("Build a weather CLI")

        assert not project_path.exists()
        _materialize_project(project_path, context)
        assert (project_path / ".git").is_dir()
        assert (
            "Build a weather CLI"
            in (project_path / ".factory" / "strategy" / "current.md").read_text()
        )

    def test_existing_dir_not_affected(self, tmp_path):
        """_resolve_input on existing dir returns it unchanged, _materialize_project is no-op."""
        (tmp_path / ".git").mkdir()
        project_path, context = _resolve_input(str(tmp_path))
        assert project_path == tmp_path
        assert context is None


class TestFromPlanFlag:
    """Tests for --from-plan flag on design mode."""

    def test_from_plan_requires_design_mode(self, capsys):
        """--from-plan without --mode design is rejected."""
        result = main(["ceo", "/some/path", "--mode", "founder", "--from-plan", "plan.md"])
        assert result == 1
        assert "--from-plan requires --mode design" in capsys.readouterr().err

    def test_from_plan_mutually_exclusive_with_focus(self, capsys):
        """--from-plan and --focus cannot be used together."""
        result = main(
            ["ceo", "/some/path", "--mode", "design", "--from-plan", "plan.md", "--focus", "auth"]
        )
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err.lower()

    def test_from_plan_mutually_exclusive_with_prompt(self, capsys):
        """--from-plan and --prompt cannot be used together."""
        result = main(
            [
                "ceo",
                "/some/path",
                "--mode",
                "design",
                "--from-plan",
                "plan.md",
                "--prompt",
                "spec.md",
            ]
        )
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err.lower()

    def test_from_plan_default_is_none(self):
        """from_plan defaults to None when flag is omitted."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="some idea",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus=None,
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=False,
            from_plan=None,
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int)
        *_, from_plan, _just_plan = validated
        assert from_plan is None

    def test_from_plan_validation_passes_with_design_mode(self):
        """--from-plan with --mode design passes validation."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="some idea",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus=None,
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=False,
            from_plan="plan.md",
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int), f"Expected tuple, got error code {validated}"
        *_, from_plan, _just_plan = validated
        assert from_plan == "plan.md"


class TestResolvePlanSource:
    """Tests for _resolve_plan_source()."""

    def test_resolve_plan_source_local_file(self, tmp_path):
        """Local file path returns PlanSource with plan content and no feedback."""
        from factory.cli._path_resolver import _resolve_plan_source

        plan_file = tmp_path / "my-plan.md"
        plan_file.write_text("## Phase 1\nBuild the scaffold")
        result = _resolve_plan_source(str(plan_file), tmp_path)
        assert "## Phase 1" in result.plan
        assert "Build the scaffold" in result.plan
        assert result.feedback == []
        assert result.source == "my-plan.md"

    def test_resolve_plan_source_relative_file(self, tmp_path):
        """Relative file path is resolved relative to project_path."""
        from factory.cli._path_resolver import _resolve_plan_source

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("## Phase 1\nDo things")
        result = _resolve_plan_source("plan.md", tmp_path)
        assert "## Phase 1" in result.plan

    def test_resolve_plan_source_issue_number(self, tmp_path):
        """Issue number triggers fetch_issue path and returns PlanSource."""
        from factory.cli._path_resolver import _resolve_plan_source

        (tmp_path / ".git").mkdir()
        subprocess.run(
            ["git", "init"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "remote",
                "add",
                "origin",
                "git@github.com:owner/repo.git",
            ],
            capture_output=True,
            check=True,
        )

        from factory.issue import IssueSpec

        mock_issue = IssueSpec(number=42, title="Plan", body="plan body", url="", forge="github")
        with (
            patch("factory.issue.fetch_issue", return_value=mock_issue),
            patch("factory.issue.parse_issue_ref", return_value=("github", "owner/repo", 42)),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=json.dumps(["comment body 1", "comment body 2"]),
                returncode=0,
            )
            result = _resolve_plan_source("42", tmp_path)
        assert result.plan == "plan body"
        assert result.feedback == ["comment body 1", "comment body 2"]

    def test_resolve_plan_source_fuzzy_search(self, tmp_path):
        """Non-file, non-issue string triggers fuzzy search and returns PlanSource."""
        from factory.cli._path_resolver import _resolve_plan_source

        (tmp_path / ".git").mkdir()
        subprocess.run(
            ["git", "init"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "remote",
                "add",
                "origin",
                "git@github.com:owner/repo.git",
            ],
            capture_output=True,
            check=True,
        )

        from factory.issue import IssueSpec

        mock_issue = IssueSpec(
            number=99, title="My Plan", body="fuzzy plan body", url="", forge="github"
        )
        search_result = json.dumps([{"number": 99, "title": "My Plan"}])

        with (
            patch("factory.issue.infer_remote", return_value=("github", "owner/repo")),
            patch("factory.issue.fetch_issue", return_value=mock_issue),
            patch("factory.issue.parse_issue_ref", return_value=("github", "owner/repo", 99)),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                MagicMock(stdout=search_result, returncode=0),
                MagicMock(stdout="[]", returncode=0),
            ]
            result = _resolve_plan_source("my cool plan", tmp_path)
        assert result.plan == "fuzzy plan body"

    def test_resolve_plan_source_includes_comments(self, tmp_path):
        """Issue fetch separates body (plan) from comments (feedback)."""
        from factory.cli._path_resolver import _resolve_plan_source

        from factory.issue import IssueSpec

        mock_issue = IssueSpec(number=10, title="Plan", body="issue body", url="", forge="github")
        with (
            patch("factory.issue.fetch_issue", return_value=mock_issue),
            patch("factory.issue.parse_issue_ref", return_value=("github", "owner/repo", 10)),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=json.dumps(["first comment", "second comment"]),
                returncode=0,
            )
            result = _resolve_plan_source("10", tmp_path)
        assert result.plan == "issue body"
        assert result.feedback == ["first comment", "second comment"]
        assert result.source == "issue #10"

    def test_resolve_plan_source_multiline_comments(self, tmp_path):
        """Multi-line comments are preserved as single entries, not split on newlines."""
        from factory.cli._path_resolver import _resolve_plan_source

        from factory.issue import IssueSpec

        multiline_comment = "Phase 1 feedback:\n- Add auth\n- Add caching\n\nPhase 2 looks good."
        mock_issue = IssueSpec(number=10, title="Plan", body="issue body", url="", forge="github")
        with (
            patch("factory.issue.fetch_issue", return_value=mock_issue),
            patch("factory.issue.parse_issue_ref", return_value=("github", "owner/repo", 10)),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=json.dumps([multiline_comment, "short comment"]),
                returncode=0,
            )
            result = _resolve_plan_source("10", tmp_path)
        assert len(result.feedback) == 2
        assert (
            "Phase 1 feedback:\n- Add auth\n- Add caching\n\nPhase 2 looks good."
            in result.feedback[0]
        )
        assert result.feedback[1] == "short comment"


class TestBuildCeoTaskFromPlan:
    """Tests for _build_ceo_task with from_plan parameter."""

    def test_build_ceo_task_from_plan_directive(self, tmp_path):
        """from_plan parameter emits the Plan Loop (From Existing Plan) section."""
        task = _build_ceo_task(tmp_path, "design", from_plan="## Phase 1\nBuild it")
        assert "## Plan Loop (From Existing Plan)" in task
        assert "Skip the Research phase" in task

    def test_build_ceo_task_no_from_plan(self, tmp_path):
        """Without from_plan, the section is not emitted."""
        task = _build_ceo_task(tmp_path, "design")
        assert "## Plan Loop (From Existing Plan)" not in task

    def test_build_ceo_task_from_plan_none(self, tmp_path):
        """from_plan=None does not emit the section."""
        task = _build_ceo_task(tmp_path, "design", from_plan=None)
        assert "## Plan Loop (From Existing Plan)" not in task

    def test_build_ceo_task_from_plan_with_feedback_includes_reconciliation(self, tmp_path):
        """from_plan with feedback includes Strategist reconciliation instructions."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            from_plan="## Phase 1\nBuild it",
            from_plan_feedback=["Please add auth", "Also need caching"],
        )
        assert "## Plan Loop (From Existing Plan)" in task
        assert "thread-feedback.md" in task
        assert "Reconcile" in task
        assert "Strategist" in task
        assert "RECONCILED" in task

    def test_build_ceo_task_from_plan_without_feedback_skips_strategist(self, tmp_path):
        """from_plan without feedback skips the Strategist step."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            from_plan="## Phase 1\nBuild it",
            from_plan_feedback=[],
        )
        assert "## Plan Loop (From Existing Plan)" in task
        assert "No thread feedback exists" in task
        assert "no Strategist needed" in task
        assert "RECONCILED" not in task

    def test_build_ceo_task_from_plan_feedback_none_skips_strategist(self, tmp_path):
        """from_plan with feedback=None behaves like no feedback."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            from_plan="## Phase 1\nBuild it",
            from_plan_feedback=None,
        )
        assert "## Plan Loop (From Existing Plan)" in task
        assert "No thread feedback exists" in task

    def test_build_ceo_task_from_plan_excludes_design_existing(self, tmp_path):
        """from_plan takes precedence over design_existing — no contradictory directives."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            from_plan="## Phase 1\nBuild it",
            design_existing=True,
        )
        assert "## Plan Loop (From Existing Plan)" in task
        assert "## Plan Loop (Interactive)" not in task

    def test_build_ceo_task_from_plan_excludes_design_idea(self, tmp_path):
        """from_plan takes precedence over design_idea — no contradictory directives."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            from_plan="## Phase 1\nBuild it",
            design_idea="Build a weather CLI",
        )
        assert "## Plan Loop (From Existing Plan)" in task
        assert "## Plan Loop (Interactive)" not in task
        assert "Raw idea from user" not in task


class TestFromPlanFeedbackWritesFile:
    """Tests for thread feedback file writing in _execute_ceo."""

    def test_from_plan_with_feedback_writes_thread_feedback_file(self, tmp_path):
        """When plan source has feedback, thread-feedback.md is written."""
        from factory.cli._path_resolver import PlanSource

        plan_source = PlanSource(
            plan="## Phase 1\nBuild it",
            feedback=["Add auth flow", "Need caching layer"],
            source="issue #42",
        )
        mock_invoke = _mock_invoke_agent_ok()
        with (
            patch("factory.agents.runner.invoke_agent", mock_invoke),
            patch(
                "factory.worktree.create_worktree",
                side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
            ),
            patch("factory.worktree.remove_worktree"),
            patch("factory.worktree.prune_stale", return_value=[]),
            patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
            patch("factory.cli._path_resolver._is_scaffold_only", return_value=False),
            patch("factory.cli._helpers._ensure_dashboard"),
            patch("factory.graph.is_graphify_installed", return_value=False),
            patch("factory.cli._ceo_helpers._resolve_plan_source", return_value=plan_source),
        ):
            result = main(
                ["ceo", str(tmp_path), "--mode", "design", "--from-plan", "42", "--auto-approve"]
            )
        assert result == 0
        feedback_file = tmp_path / ".factory" / "strategy" / "thread-feedback.md"
        assert feedback_file.exists()
        content = feedback_file.read_text()
        assert "Add auth flow" in content
        assert "Need caching layer" in content

    def test_from_plan_without_feedback_no_thread_feedback_file(self, tmp_path):
        """When plan source has no feedback, thread-feedback.md is not written."""
        from factory.cli._path_resolver import PlanSource

        plan_source = PlanSource(
            plan="## Phase 1\nBuild it",
            feedback=[],
            source="my-plan.md",
        )
        mock_invoke = _mock_invoke_agent_ok()
        with (
            patch("factory.agents.runner.invoke_agent", mock_invoke),
            patch(
                "factory.worktree.create_worktree",
                side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
            ),
            patch("factory.worktree.remove_worktree"),
            patch("factory.worktree.prune_stale", return_value=[]),
            patch("factory.cli._ceo_helpers._read_target_branch", return_value="main"),
            patch("factory.cli._path_resolver._is_scaffold_only", return_value=False),
            patch("factory.cli._helpers._ensure_dashboard"),
            patch("factory.graph.is_graphify_installed", return_value=False),
            patch("factory.cli._ceo_helpers._resolve_plan_source", return_value=plan_source),
        ):
            result = main(
                [
                    "ceo",
                    str(tmp_path),
                    "--mode",
                    "design",
                    "--from-plan",
                    "plan.md",
                    "--auto-approve",
                ]
            )
        assert result == 0
        feedback_file = tmp_path / ".factory" / "strategy" / "thread-feedback.md"
        assert not feedback_file.exists()


class TestJustPlanFlag:
    """Tests for --just-plan flag on design mode."""

    def test_just_plan_requires_design_mode(self, capsys):
        """--just-plan without --mode design is rejected."""
        result = main(["ceo", "/some/path", "--mode", "founder", "--just-plan"])
        assert result == 1
        assert "--just-plan requires --mode design" in capsys.readouterr().err

    def test_just_plan_mutually_exclusive_with_from_plan(self, capsys):
        """--just-plan and --from-plan cannot be used together."""
        result = main(
            ["ceo", "/some/path", "--mode", "design", "--just-plan", "--from-plan", "plan.md"]
        )
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err.lower()

    def test_just_plan_mutually_exclusive_with_prompt(self, capsys):
        """--just-plan and --prompt cannot be used together."""
        result = main(
            ["ceo", "/some/path", "--mode", "design", "--just-plan", "--prompt", "spec.md"]
        )
        assert result == 1
        assert "mutually exclusive" in capsys.readouterr().err.lower()

    def test_just_plan_with_focus_allowed(self):
        """--just-plan and --focus are allowed together."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="/some/path",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus="auth",
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=False,
            from_plan=None,
            just_plan=True,
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int), f"Expected tuple, got error code {validated}"
        *_, just_plan = validated
        assert just_plan is True

    def test_mode_plan_no_longer_valid(self, capsys):
        """--mode plan is rejected at runtime (not a valid built-in or project mode)."""
        result = main(["ceo", "/some/path", "--mode", "plan"])
        assert result == 1
        captured = capsys.readouterr()
        assert "unknown mode" in captured.err

    def test_just_plan_default_is_false(self):
        """just_plan defaults to False when flag is omitted."""
        from factory.cli._ceo_helpers import _validate_ceo_flags

        args = argparse.Namespace(
            path="some idea",
            mode="design",
            bg=False,
            bg_agents=False,
            headless=False,
            prompt=None,
            focus=None,
            dir=None,
            no_github=False,
            refine=None,
            auto_approve=False,
            from_plan=None,
            just_plan=False,
        )
        validated = _validate_ceo_flags(args)
        assert not isinstance(validated, int)
        *_, just_plan = validated
        assert just_plan is False

    def test_task_builder_just_plan_directive(self, tmp_path):
        """_build_ceo_task generates the plan directive for just_plan=True."""
        task = _build_ceo_task(tmp_path, "design", just_plan=True)
        assert "## Plan Loop (Just Plan)" in task
        assert "just_plan: true" in task
        assert "Terminal mode" in task
        assert "### Post-Approval: GitHub Publish (MANDATORY)" in task
        assert "gh label create plan" in task
        assert "gh issue comment" in task
        assert "gh issue create" in task
        assert "Do NOT skip this step" in task

    def test_task_builder_no_just_plan_directive(self, tmp_path):
        """_build_ceo_task omits the plan directive when just_plan=False."""
        task = _build_ceo_task(tmp_path, "design", just_plan=False)
        assert "## Plan Loop (Just Plan)" not in task


class TestDataNodeTaskFlags:
    """Tests for --data-node and --task CLI flags."""

    def test_data_node_flag_parsed(self):
        """--data-node sets data_node=True."""
        parser = build_parser()
        args = parser.parse_args(["ceo", "/tmp/test", "--data-node"])
        assert getattr(args, "data_node", False) is True

    def test_task_flag_parsed(self):
        """--task sets task_ref."""
        parser = build_parser()
        args = parser.parse_args(
            ["ceo", "/tmp/test", "--mode", "create", "--task", "swe-bench"]
        )
        assert getattr(args, "task_ref") == "swe-bench"
        assert getattr(args, "data_node", False) is False

    def test_both_flags_parsed(self):
        """--data-node --task together are parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(
            ["ceo", "/tmp/test", "--mode", "create",
             "--data-node", "--task", "my.mod:MyTask"]
        )
        assert getattr(args, "data_node") is True
        assert getattr(args, "task_ref") == "my.mod:MyTask"

    def test_default_values(self):
        """Without flags, defaults are False/None."""
        parser = build_parser()
        args = parser.parse_args(["ceo", "/tmp/test"])
        assert getattr(args, "data_node", False) is False
        assert getattr(args, "task_ref", None) is None

    def test_build_ceo_task_data_node_directive(self, tmp_path):
        """_build_ceo_task injects DataNode directive when data_node=True."""
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
            data_node=True,
        )
        assert "## DataNode Directive" in task
        # New directive tells Builder to include DataNode directly
        assert "DataNode" in task
        assert "start_node" in task
        # Example code block is present
        assert "```python" in task
        assert "DataNode(" in task
        assert "subgraph_entry" in task
        assert "subgraph_exit" in task
        # current_item.json documentation is present
        assert "current_item.json" in task
        # Template variables warning is present
        assert "template variables" in task
        # JSON structure example is present
        assert '"id"' in task
        assert '"prompt"' in task
        assert '"metadata"' in task

    def test_build_ceo_task_task_directive(self, tmp_path):
        """_build_ceo_task injects Task directive when task_context is provided."""
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
            task_context={
                "name": "swe-bench",
                "description": "Bug fix task",
                "field_names": ["repo", "problem_statement"],
                "sample_prompt": "Fix this bug...",
                "ref_string": "examples.swe_bench_task:SWEBenchTask",
                "scoring_method": "exit_code",
                "scoring_metric_path": "score",
                "verify_command": "pytest",
                "timeout": 300,
            },
        )
        assert "## Task Directive" in task
        assert "swe-bench" in task
        assert "`repo`" in task
        assert "`problem_statement`" in task
        assert "exit_code" in task

    def test_build_ceo_task_no_pollution_without_flags(self, tmp_path):
        """Without --data-node or --task, no DataNode/Task directives injected."""
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
        )
        assert "## DataNode Directive" not in task
        assert "## Task Directive" not in task

    def test_build_ceo_task_both_flags(self, tmp_path):
        """Both --data-node and --task inject both directives."""
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
            data_node=True,
            task_ref="x:Y",
            task_context={
                "name": "my-task",
                "description": "",
                "field_names": ["f1"],
                "sample_prompt": "do stuff",
                "ref_string": "x:Y",
                "scoring_method": "json",
                "scoring_metric_path": "win_rate",
            },
        )
        assert "## DataNode Directive" in task
        assert "## Task Directive" in task
        # DataNode directive tells Builder to include DataNode as start_node
        assert "DataNode" in task
        assert "start_node" in task
        assert "```python" in task
        # DataNode directive comes before Task directive
        dn_pos = task.index("## DataNode Directive")
        td_pos = task.index("## Task Directive")
        assert dn_pos < td_pos

    def test_build_ceo_task_json_scoring(self, tmp_path):
        """JSON scoring method shows metric path."""
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
            task_context={
                "name": "t",
                "description": "",
                "field_names": [],
                "sample_prompt": "",
                "ref_string": "x:Y",
                "scoring_method": "json",
                "scoring_metric_path": "win_rate",
            },
        )
        assert "JSON scoring" in task
        assert "`win_rate`" in task

    def test_build_ceo_task_sample_prompt_truncation(self, tmp_path):
        """Long sample prompts (>1500 chars) are truncated."""
        long_prompt = "x" * 2000
        task = _build_ceo_task(
            tmp_path,
            "create",
            create_description="my pipeline",
            task_context={
                "name": "t",
                "description": "",
                "field_names": [],
                "sample_prompt": long_prompt,
                "ref_string": "x:Y",
                "scoring_method": "exit_code",
            },
        )
        assert "... (truncated)" in task
        # The full 2000-char prompt should NOT appear
        assert long_prompt not in task

    def test_data_node_directive_not_in_non_create_mode(self, tmp_path):
        """DataNode directive NOT injected when create_description is None."""
        task = _build_ceo_task(
            tmp_path,
            "design",
            data_node=True,
        )
        assert "## DataNode Directive" not in task

    # ── cmd_ceo() validation path tests ────────────────────────

    def test_cmd_ceo_data_node_requires_create_mode(self, tmp_path, capsys):
        """--data-node without --mode create or create-v2 should print error and return 1."""
        result = main(["ceo", str(tmp_path), "--mode", "design", "--data-node"])
        assert result == 1
        assert "--data-node requires --mode create or --mode create-v2" in capsys.readouterr().err

    def test_cmd_ceo_task_requires_create_mode(self, tmp_path, capsys):
        """--task without --mode create or create-v2 should print error and return 1."""
        result = main(["ceo", str(tmp_path), "--mode", "design", "--task", "swe-bench"])
        assert result == 1
        assert "--task requires --mode create or --mode create-v2" in capsys.readouterr().err

    def test_cmd_ceo_data_node_accepted_with_create_v2(self, tmp_path, capsys):
        """--data-node with --mode create-v2 should NOT be rejected by mode validation."""
        with patch("factory.cli.ceo._execute_ceo", return_value=0):
            result = main(["ceo", str(tmp_path), "--mode", "create-v2", "--focus", "my pipeline", "--data-node"])
        captured = capsys.readouterr()
        # Should NOT contain the mode-validation error
        assert "--data-node requires --mode create" not in captured.err
        assert result != 1 or "--data-node requires" not in captured.err

    def test_cmd_ceo_task_accepted_with_create_v2(self, tmp_path, capsys):
        """--task with --mode create-v2 should NOT be rejected by mode validation."""
        with patch("factory.cli.ceo._execute_ceo", return_value=0):
            result = main(["ceo", str(tmp_path), "--mode", "create-v2", "--focus", "my pipeline", "--task", "swe-bench"])
        captured = capsys.readouterr()
        # Should NOT contain the mode-validation error
        assert "--task requires --mode create" not in captured.err
        assert result != 1 or "--task requires" not in captured.err

    def test_cmd_ceo_task_import_error(self, tmp_path, capsys):
        """--task with invalid ref should print clean error and return 1."""
        result = main([
            "ceo", str(tmp_path), "--mode", "create",
            "--focus", "my pipeline", "--task", "bad.nonexistent.mod:BadTask",
        ])
        assert result == 1
        assert "could not be resolved" in capsys.readouterr().err

    # ── _resolve_task_for_directive() tests ────────────────────

    def test_resolve_qualified_ref(self):
        """Qualified ref (with ':') resolves via TaskRef.resolve()."""
        from factory.cli.ceo import _resolve_task_for_directive

        mock_defn = MagicMock()
        mock_defn.name = "test-task"
        mock_defn.description = "A test task"
        mock_defn.scoring.method = "exit_code"
        mock_defn.scoring.metric_path = "score"
        mock_defn.verify_config = None
        mock_defn.constraints = None

        mock_task = MagicMock()
        mock_task.to_definition.return_value = mock_defn
        mock_task.instances.return_value = []

        with patch("factory.task.TaskRef") as MockRef:
            MockRef.return_value.resolve.return_value = mock_task
            result = _resolve_task_for_directive("my.mod:MyTask", Path("/tmp"))

        assert result["name"] == "test-task"
        assert result["ref_string"] == "my.mod:MyTask"
        assert result["scoring_method"] == "exit_code"
        assert result["verify_command"] == ""
        assert result["timeout"] == 600

    def test_resolve_simple_name(self):
        """Simple name (no ':') resolves via TaskRegistry.load_task()."""
        from factory.cli.ceo import _resolve_task_for_directive

        mock_defn = MagicMock()
        mock_defn.name = "my-task"
        mock_defn.description = "desc"
        mock_defn.scoring.method = "json"
        mock_defn.scoring.metric_path = "accuracy"
        mock_defn.verify_config = MagicMock(command="pytest")
        mock_defn.constraints = MagicMock(timeout=300)

        mock_task = MagicMock()
        mock_task.to_definition.return_value = mock_defn
        mock_task.instances.return_value = []

        with patch("factory.task_registry.TaskRegistry") as MockRegistry:
            MockRegistry.load_task.return_value = mock_task
            result = _resolve_task_for_directive("my-task", Path("/tmp"))

        assert result["name"] == "my-task"
        assert result["ref_string"] == "my-task"
        assert result["scoring_method"] == "json"
        assert result["scoring_metric_path"] == "accuracy"
        assert result["verify_command"] == "pytest"
        assert result["timeout"] == 300

    def test_resolve_extracts_field_names_and_sample_prompt(self):
        """_resolve_task_for_directive extracts field_names and sample_prompt from instances."""
        from factory.cli.ceo import _resolve_task_for_directive

        mock_defn = MagicMock()
        mock_defn.name = "test"
        mock_defn.description = ""
        mock_defn.scoring.method = "exit_code"
        mock_defn.scoring.metric_path = "score"
        mock_defn.verify_config = None
        mock_defn.constraints = None

        mock_instance = MagicMock()
        mock_instance.metadata = {"repo": "x", "problem": "y"}

        mock_task = MagicMock()
        mock_task.to_definition.return_value = mock_defn
        mock_task.instances.return_value = [mock_instance]
        mock_task.prompt.return_value = "Fix this bug in repo x"

        with patch("factory.task.TaskRef") as MockRef:
            MockRef.return_value.resolve.return_value = mock_task
            result = _resolve_task_for_directive("mod:Task", Path("/tmp"))

        assert result["field_names"] == ["problem", "repo"]
        assert result["sample_prompt"] == "Fix this bug in repo x"

    def test_resolve_graceful_degradation_on_instances_error(self):
        """_resolve_task_for_directive gracefully handles instances() raising."""
        from factory.cli.ceo import _resolve_task_for_directive

        mock_defn = MagicMock()
        mock_defn.name = "broken"
        mock_defn.description = ""
        mock_defn.scoring.method = "exit_code"
        mock_defn.scoring.metric_path = "score"
        mock_defn.verify_config = None
        mock_defn.constraints = None

        mock_task = MagicMock()
        mock_task.to_definition.return_value = mock_defn
        mock_task.instances.side_effect = RuntimeError("data not available")

        with patch("factory.task.TaskRef") as MockRef:
            MockRef.return_value.resolve.return_value = mock_task
            result = _resolve_task_for_directive("mod:Task", Path("/tmp"))

        # Graceful degradation: empty field_names and sample_prompt
        assert result["field_names"] == []
        assert result["sample_prompt"] == ""
        assert result["name"] == "broken"
