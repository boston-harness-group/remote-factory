"""Tests for CodexRunner — command building, AGENTS.md handling, and output parsing."""

from __future__ import annotations

from pathlib import Path

from factory.models import AgentRunRequest
from factory.runners.codex import CodexRunner, _parse_codex_usage


class TestMetadata:
    def test_name(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.name == "codex"

    def test_binary(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.binary == "codex"

    def test_no_required_env_vars(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.required_env_vars == []

    def test_no_session_support(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.supports_session_name is False
        assert meta.supports_session_resume is False

    def test_install_hint(self) -> None:
        meta = CodexRunner.metadata()
        assert "@openai/codex" in meta.install_hint


class TestBuildCommand:
    def _make_request(self, tmp_path: Path, **overrides: object) -> AgentRunRequest:
        defaults: dict[str, object] = {
            "prompt": "You are a helpful assistant.",
            "task": "Fix the bug",
            "cwd": tmp_path,
            "role": "builder",
        }
        defaults.update(overrides)
        return AgentRunRequest(**defaults)  # type: ignore[arg-type]

    def test_basic_command_structure(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, env, temp_files = runner.build_command(req)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "Fix the bug" in cmd[-1]
        assert "You are a helpful assistant." in cmd[-1]

    def test_model_flag(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path, model="o3")
        cmd, _env, _temp = runner.build_command(req)
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o3"

    def test_cd_flag(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, _env, _temp = runner.build_command(req)
        idx = cmd.index("-C")
        assert cmd[idx + 1] == str(tmp_path)

    def test_no_model_flag_when_none(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, _env, _temp = runner.build_command(req)
        assert "--model" not in cmd


class TestModelResolution:
    def test_strips_sonnet(self) -> None:
        assert CodexRunner._resolve_model("sonnet") is None

    def test_strips_opus(self) -> None:
        assert CodexRunner._resolve_model("opus") is None

    def test_strips_claude_prefixed(self) -> None:
        assert CodexRunner._resolve_model("claude-sonnet-4-5") is None

    def test_passes_openai_model(self) -> None:
        assert CodexRunner._resolve_model("o3") == "o3"

    def test_passes_gpt_model(self) -> None:
        assert CodexRunner._resolve_model("gpt-4o") == "gpt-4o"

    def test_none_stays_none(self) -> None:
        assert CodexRunner._resolve_model(None) is None

    def test_empty_string_becomes_none(self) -> None:
        assert CodexRunner._resolve_model("") is None


class TestCombinedPrompt:
    def test_prompt_and_task_combined(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = AgentRunRequest(
            prompt="You are a researcher.",
            task="Find all bugs",
            cwd=tmp_path,
            role="researcher",
        )
        cmd, _env, _temp = runner.build_command(req)
        combined = cmd[-1]
        assert "You are a researcher." in combined
        assert "Find all bugs" in combined

    def test_no_agents_md_created(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = AgentRunRequest(
            prompt="System instructions.",
            task="Do something",
            cwd=tmp_path,
            role="builder",
        )
        runner.build_command(req)
        assert not (tmp_path / "AGENTS.md").exists()

    def test_no_temp_files(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = AgentRunRequest(
            prompt="System instructions.",
            task="Do something",
            cwd=tmp_path,
            role="builder",
        )
        _cmd, _env, temp_files = runner.build_command(req)
        assert temp_files == []


class TestParseUsage:
    def test_openai_token_fields(self) -> None:
        data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model": "o3",
        }
        usage = _parse_codex_usage(data)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "o3"

    def test_standard_token_fields(self) -> None:
        data = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 75,
            },
        }
        usage = _parse_codex_usage(data)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 75

    def test_empty_usage(self) -> None:
        usage = _parse_codex_usage({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestRegistration:
    def test_codex_in_runners(self) -> None:
        from factory.runners import get_available_runners

        runners = get_available_runners()
        assert "codex" in runners

    def test_get_runner_codex(self) -> None:
        from factory.runners import get_runner

        runner = get_runner("codex")
        assert isinstance(runner, CodexRunner)

    def test_runner_choices_include_codex(self) -> None:
        from factory.runners import get_runner_choices

        choices = get_runner_choices()
        assert "codex" in choices
