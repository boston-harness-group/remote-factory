"""ClaudeRunner — Claude Code CLI backend implementation."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.runners._subprocess import run_subprocess

if TYPE_CHECKING:
    from factory.models import AgentRunRequest, AgentRunResult, AgentUsage
    from factory.runners.protocol import RunnerMeta

log = structlog.get_logger()

# Boolean-flag env vars that switch Claude Code onto a 3P backend (their own credentials,
# never the OAuth keychain), as opposed to ANTHROPIC_API_KEY which carries the key itself.
_BARE_COMPATIBLE_BACKEND_FLAGS = ("CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_BEDROCK")


def _supports_bare_mode(env: dict[str, str]) -> bool:
    """True when the environment implies API-key-style auth, not OAuth/keychain login.

    `--bare` skips keychain reads entirely (see `claude --help`), so it only works when
    auth is ANTHROPIC_API_KEY or a 3P backend (Vertex/Bedrock). Passing it unconditionally
    breaks subscription (Pro/Max) logins, which live in the keychain — see #1437.
    """
    if env.get("ANTHROPIC_API_KEY", "").strip():
        return True
    return any(
        env.get(var, "").strip().lower() in ("1", "true", "yes")
        for var in _BARE_COMPATIBLE_BACKEND_FLAGS
    )


def _make_ceo_message_emitter(project_path: Path) -> Callable[[bytes], None]:
    """Return a callback that emits ceo.message events for assistant JSONL lines."""
    from factory.events import emit_event

    def _on_line(line: bytes) -> None:
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(parsed, dict) or parsed.get("type") != "assistant":
            return
        message = parsed.get("message", "")
        if isinstance(message, str):
            text = message
        elif isinstance(message, dict):
            content = message.get("content", [])
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            return
        if not text:
            return
        emit_event(
            project_path,
            "ceo.message",
            agent="ceo",
            data={"message": text, "message_type": "assistant"},
        )

    return _on_line


def _parse_usage(data: dict) -> AgentUsage:
    """Extract AgentUsage from Claude Code JSON output."""
    from factory.models import AgentUsage

    usage_block = data.get("usage", {})
    return AgentUsage(
        input_tokens=usage_block.get("input_tokens", 0),
        output_tokens=usage_block.get("output_tokens", 0),
        cache_read_tokens=usage_block.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage_block.get("cache_creation_input_tokens", 0),
        total_cost_usd=data.get("total_cost_usd", 0.0) or 0.0,
        duration_ms=data.get("duration_ms", 0.0) or 0.0,
        num_turns=data.get("num_turns", 0) or 0,
        model=data.get("model", ""),
    )


def _claude_bin() -> str:
    """Return the Claude CLI binary name, respecting FACTORY_CLAUDE_BIN override."""
    return os.environ.get("FACTORY_CLAUDE_BIN") or "claude"


def _claude_model() -> str:
    """Return the model to pass to internal `claude -p` LLM calls.

    Internal callers (outer-loop prompt rewriters, knob expanders, the
    reflector) historically hardcoded the "opus" alias. On gateways that
    only serve specific models (e.g. LiteLLM proxies), an explicit
    `--model opus` both overrides the user's configured default and 403s.
    Respect the standard model overrides first:

    1. FACTORY_MODEL — factory's own model knob (env var / config.toml
       `[defaults] model`, resolved like the rest of the CLI)
    2. ANTHROPIC_MODEL — the claude CLI's own default-model env var
    3. "opus" — the historical default
    """
    from factory.user_config import resolve

    return (
        resolve("model", env_var="FACTORY_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "opus"
    )


# `claude -p` prints auth/API errors to stdout (sometimes with exit 0),
# e.g. "Failed to authenticate. API Error: 403 ...". Heuristic prefixes —
# a sturdier fix is `--output-format json` and its structured is_error
# field (see follow-up note in the outer-loop PR).
_CLI_ERROR_PREFIXES = ("API Error", "Failed to authenticate")


def _cli_error_text(stdout: str) -> bool:
    """True if `claude -p` stdout looks like a CLI error, not model output."""
    return stdout.lstrip()[:200].startswith(_CLI_ERROR_PREFIXES)


def _cli_output_ok(returncode: int, stdout: str) -> bool:
    """Guard against accepting CLI error text as LLM output.

    Without this check an error message (or any failed run's output) can
    be captured as a rewritten prompt or knob value.
    """
    if returncode != 0:
        return False
    return not _cli_error_text(stdout)


class ClaudeRunner:
    """Runner implementation for Claude Code CLI."""

    name: str = "claude"

    @classmethod
    def metadata(cls) -> RunnerMeta:
        from factory.runners.protocol import RunnerMeta

        return RunnerMeta(
            name="claude",
            display_name="Claude Code",
            binary=_claude_bin(),
            install_hint="npm install -g @anthropic-ai/claude-code",
            supports_usage_telemetry=True,
            supports_session_name=True,
            supports_session_resume=True,
            supports_background=True,
        )

    def build_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        """Build the Claude CLI command, env dict, and temp files."""
        prompt_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            prefix="factory-prompt-",
            delete=False,
        )
        prompt_file.write(request.prompt)
        prompt_file.close()
        prompt_path = Path(prompt_file.name)

        cmd = [
            _claude_bin(),
            "--append-system-prompt-file",
            prompt_file.name,
            "-p",
            request.task,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if _supports_bare_mode(dict(os.environ)):
            cmd.append("--bare")
        cmd.extend(["--disallowedTools", "Agent"])
        settings_file = request.extras.get("settings_file")
        if settings_file:
            cmd.extend(["--settings", str(settings_file)])
        if request.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        if request.model:
            cmd.extend(["--model", request.model])
        if request.session_name:
            cmd.extend(["--name", request.session_name])
        if request.resume_session_id:
            cmd.extend(["--resume", request.resume_session_id])
        elif request.session_id:
            cmd.extend(["--session-id", request.session_id])

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            from factory.worktree import is_factory_venv
            if is_factory_venv(Path(request.cwd)):
                venv_path = Path(request.cwd) / ".venv"
                env["VIRTUAL_ENV"] = str(venv_path)
                env["PATH"] = str(venv_path / "bin") + os.pathsep + env.get("PATH", "")
        if request.model:
            env["FACTORY_MODEL"] = request.model
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())

        return cmd, env, [prompt_path]

    async def headless(self, request: AgentRunRequest) -> AgentRunResult:
        """Run a headless Claude Code invocation."""
        from factory.models import AgentRunResult

        background = request.extras.get("background", False)
        if background:
            from factory.runners._background import run_in_background

            stdout, rc, usage = await run_in_background(
                request.prompt,
                request.task,
                request.cwd,
                request.role,
                timeout=request.timeout,
                model=request.model,
                dangerously_skip_permissions=request.skip_permissions,
            )
            return AgentRunResult(stdout=stdout, return_code=rc, usage=usage)

        tmux_persist = request.extras.get("tmux_persist", False)
        if tmux_persist:
            from factory.runners._tmux_persist import find_project_path, run_in_tmux, tmux_available

            if tmux_available():
                stdout, rc, usage = await run_in_tmux(
                    request.prompt,
                    request.task,
                    request.cwd,
                    request.role,
                    find_project_path(request.cwd),
                    model=request.model,
                    dangerously_skip_permissions=request.skip_permissions,
                )
                return AgentRunResult(stdout=stdout, return_code=rc, usage=usage)
            log.warning("tmux_not_available")

        cmd, env, temp_files = self.build_command(request)
        env["TELEMETRY_PLATFORM"] = ""
        try:
            log.info("claude_headless", cwd=str(request.cwd), model=request.model)

            on_line = None
            if request.role == "ceo" and request.project_path is not None:
                on_line = _make_ceo_message_emitter(request.project_path)

            result = await run_subprocess(
                cmd,
                cwd=str(request.cwd),
                env=env,
                timeout=request.timeout,
                runner_name="claude",
                role=request.role,
                on_line=on_line,
                sanitize=True,
            )

            usage = None
            result_text = result.stdout
            metadata: dict[str, object] = {**result.metadata}

            data: dict[str, object] | None = None
            for line in reversed(result.stdout.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict) and "result" in parsed:
                    data = parsed
                    break

            if data is not None:
                result_value = data.get("result", result.stdout)
                result_text = result_value if isinstance(result_value, str) else result.stdout
                usage = _parse_usage(data)
                for key in (
                    "session_id",
                    "uuid",
                    "stop_reason",
                    "terminal_reason",
                    "duration_api_ms",
                    "ttft_ms",
                    "is_error",
                    "subtype",
                ):
                    metadata[key] = data.get(key)
                metadata["model_usage"] = data.get("modelUsage")
                metadata["permission_denials"] = data.get("permission_denials")

            return AgentRunResult(
                stdout=result_text,
                return_code=result.return_code,
                usage=usage,
                metadata=metadata,
            )
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)

    def build_interactive_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        """Build the CLI command, env dict, and temp files for an interactive invocation."""
        prompt_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            prefix="factory-prompt-",
            delete=False,
        )
        prompt_file.write(request.prompt)
        prompt_file.close()
        prompt_path = Path(prompt_file.name)

        temp_files: list[Path] = [prompt_path]

        # Write a slim CEO identity to .claude/CLAUDE.md so it survives session
        # transitions (background via ←, resume, daemon restart). The full prompt
        # is delivered via --append-system-prompt-file; CLAUDE.md only needs enough
        # to re-orient the CEO on resume.
        cwd = Path(request.cwd)
        claude_dir = cwd / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)

        claude_md_path = claude_dir / "CLAUDE.md"
        backup_path = claude_dir / "CLAUDE.md.factory-backup"
        if claude_md_path.exists():
            import shutil

            shutil.copy2(claude_md_path, backup_path)

        claude_md_content = request.prompt_core if request.prompt_core else request.prompt
        claude_md_path.write_text(claude_md_content)
        temp_files.append(claude_md_path)

        # Write disallowedTools to settings.local.json so it survives session
        # transitions (CLI flags are not carried over on background/resume).
        settings_path = claude_dir / "settings.local.json"
        settings: dict[str, object] = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
            except (json.JSONDecodeError, ValueError):
                settings = {}
        settings["disallowedTools"] = ["Agent"]
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        temp_files.append(settings_path)

        cmd = [
            _claude_bin(),
            "--append-system-prompt-file",
            prompt_file.name,
        ]
        if _supports_bare_mode(dict(os.environ)):
            cmd.append("--bare")
        settings_file = request.extras.get("settings_file")
        if settings_file:
            cmd.extend(["--settings", str(settings_file)])
        if request.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        cmd.append(request.task)
        if request.model:
            cmd.extend(["--model", request.model])
        if request.session_name:
            cmd.extend(["--name", request.session_name])
        if request.resume_session_id:
            cmd.extend(["--resume", request.resume_session_id])
        elif request.session_id:
            cmd.extend(["--session-id", request.session_id])

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            from factory.worktree import is_factory_venv
            if is_factory_venv(Path(request.cwd)):
                venv_path = Path(request.cwd) / ".venv"
                env["VIRTUAL_ENV"] = str(venv_path)
                env["PATH"] = str(venv_path / "bin") + os.pathsep + env.get("PATH", "")
        if request.model:
            env["FACTORY_MODEL"] = request.model
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())

        return cmd, env, temp_files

    def interactive_run(self, request: AgentRunRequest) -> int:
        """Run an interactive Claude Code session as a subprocess."""
        cmd, env, temp_files = self.build_interactive_command(request)
        if not env.get("FACTORY_TRACE_ID"):
            env["TELEMETRY_PLATFORM"] = ""
        cwd = Path(request.cwd)
        backup_path = cwd / ".claude" / "CLAUDE.md.factory-backup"
        claude_md_path = cwd / ".claude" / "CLAUDE.md"
        try:
            log.info("claude_interactive", cwd=str(request.cwd))
            result = subprocess.run(cmd, cwd=request.cwd, env=env)
            return result.returncode
        finally:
            for f in temp_files:
                if f == claude_md_path:
                    continue
                f.unlink(missing_ok=True)
            if backup_path.exists():
                import shutil

                shutil.move(str(backup_path), str(claude_md_path))
            else:
                claude_md_path.unlink(missing_ok=True)
