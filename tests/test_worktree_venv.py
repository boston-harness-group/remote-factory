"""Tests for per-worktree venv isolation (issue #1365)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from factory.eval.languages.python import _resolve_python
from factory.eval.runner import _run_project_eval, _run_single_project_dimension
from factory.models import AgentRunRequest, ProjectEvalDimension
from factory.runners.claude import ClaudeRunner
from factory.worktree import WORKTREE_VENV_MARKER, _setup_worktree_venv


class TestSetupWorktreeVenv:
    def test_creates_venv_via_uv_sync(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / ".venv").mkdir()

        success = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("factory.worktree.subprocess.run", return_value=success) as mock_run:
            result = _setup_worktree_venv(tmp_path)

        assert result == tmp_path / ".venv"
        assert (tmp_path / ".venv" / WORKTREE_VENV_MARKER).exists()
        mock_run.assert_called_once_with(
            ["uv", "sync", "--directory", str(tmp_path)],
            capture_output=True,
            text=True,
        )

    def test_skips_when_no_pyproject_toml(self, tmp_path: Path) -> None:
        result = _setup_worktree_venv(tmp_path)
        assert result is None

    def test_fallback_on_uv_sync_failure(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / ".venv").mkdir()

        uv_sync_fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
        venv_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        pip_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch("factory.worktree.subprocess.run", side_effect=[uv_sync_fail, venv_ok, pip_ok]):
            result = _setup_worktree_venv(tmp_path)

        assert result == tmp_path / ".venv"
        assert (tmp_path / ".venv" / WORKTREE_VENV_MARKER).exists()

    def test_graceful_degradation_on_total_failure(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")

        with patch("factory.worktree.subprocess.run", return_value=fail):
            result = _setup_worktree_venv(tmp_path)

        assert result is None

    def test_fallback_venv_created_then_pip_fails(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        uv_fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
        venv_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        pip_fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")

        with patch("factory.worktree.subprocess.run", side_effect=[uv_fail, venv_ok, pip_fail]):
            result = _setup_worktree_venv(tmp_path)

        assert result is None


class TestResolvePython:
    def test_prefers_venv_python(self, tmp_path: Path) -> None:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        python = venv_bin / "python"
        python.touch()
        python.chmod(0o755)
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        result = _resolve_python(tmp_path)
        assert result == str(python)

    def test_falls_back_to_sys_executable(self, tmp_path: Path) -> None:
        result = _resolve_python(tmp_path)
        assert result == sys.executable

    def test_ignores_non_factory_venv(self, tmp_path: Path) -> None:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        result = _resolve_python(tmp_path)
        assert result == sys.executable


class TestRunCmdVenvEnv:
    def test_sets_venv_env_when_factory_venv_exists(self, tmp_path: Path) -> None:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        from factory.eval.languages.base import _run_cmd

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )
            _run_cmd(["echo", "test"], tmp_path)

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
        assert env["PATH"].startswith(str(venv_bin))

    def test_no_venv_env_when_no_venv(self, tmp_path: Path) -> None:
        from factory.eval.languages.base import _run_cmd

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )
            _run_cmd(["echo", "test"], tmp_path)

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert "VIRTUAL_ENV" not in env

    def test_no_venv_env_when_non_factory_venv(self, tmp_path: Path) -> None:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        from factory.eval.languages.base import _run_cmd

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )
            _run_cmd(["echo", "test"], tmp_path)

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert "VIRTUAL_ENV" not in env


class TestBuildCommandVenvEnv:
    def test_injects_venv_when_marker_exists(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        request = AgentRunRequest(
            prompt="test", task="test", cwd=tmp_path, role="builder", skip_permissions=False
        )
        cmd, env, temp_files = ClaudeRunner().build_command(request)
        try:
            assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
            assert env["PATH"].startswith(str(venv_bin))
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)

    def test_no_venv_without_marker(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]

        request = AgentRunRequest(
            prompt="test", task="test", cwd=tmp_path, role="builder", skip_permissions=False
        )
        cmd, env, temp_files = ClaudeRunner().build_command(request)
        try:
            assert "VIRTUAL_ENV" not in env
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)


class TestBuildInteractiveCommandVenvEnv:
    def test_injects_venv_when_marker_exists(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        request = AgentRunRequest(
            prompt="test", task="test", cwd=tmp_path, role="ceo", skip_permissions=False
        )
        cmd, env, temp_files = ClaudeRunner().build_interactive_command(request)
        try:
            assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
            assert env["PATH"].startswith(str(venv_bin))
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)

    def test_no_venv_without_marker(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]

        request = AgentRunRequest(
            prompt="test", task="test", cwd=tmp_path, role="ceo", skip_permissions=False
        )
        cmd, env, temp_files = ClaudeRunner().build_interactive_command(request)
        try:
            assert "VIRTUAL_ENV" not in env
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)


class TestRunProjectEvalVenvEnv:
    async def test_injects_venv_when_marker_exists(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"{}", b"")
        mock_proc.returncode = 0

        mock_exec = AsyncMock(return_value=mock_proc)
        with patch("factory.eval.runner.asyncio.create_subprocess_exec", mock_exec):
            await _run_project_eval("echo ok", tmp_path)

        env = mock_exec.call_args.kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
        assert env["PATH"].startswith(str(venv_bin))

    async def test_no_venv_without_marker(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"{}", b"")
        mock_proc.returncode = 0

        mock_exec = AsyncMock(return_value=mock_proc)
        with patch("factory.eval.runner.asyncio.create_subprocess_exec", mock_exec):
            await _run_project_eval("echo ok", tmp_path)

        env = mock_exec.call_args.kwargs["env"]
        assert "VIRTUAL_ENV" not in env


class TestRunSingleDimensionVenvEnv:
    async def test_injects_venv_when_marker_exists(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (tmp_path / ".venv" / WORKTREE_VENV_MARKER).touch()

        dim = ProjectEvalDimension(name="test_dim", command="echo ok", timeout=30.0)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'{"score": 1.0}', b"")
        mock_proc.returncode = 0

        mock_exec = AsyncMock(return_value=mock_proc)
        with patch("factory.eval.runner.asyncio.create_subprocess_exec", mock_exec):
            await _run_single_project_dimension(dim, tmp_path)

        env = mock_exec.call_args.kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
        assert env["PATH"].startswith(str(venv_bin))

    async def test_no_venv_without_marker(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # type: ignore[union-attr]

        dim = ProjectEvalDimension(name="test_dim", command="echo ok", timeout=30.0)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'{"score": 1.0}', b"")
        mock_proc.returncode = 0

        mock_exec = AsyncMock(return_value=mock_proc)
        with patch("factory.eval.runner.asyncio.create_subprocess_exec", mock_exec):
            await _run_single_project_dimension(dim, tmp_path)

        env = mock_exec.call_args.kwargs["env"]
        assert "VIRTUAL_ENV" not in env
