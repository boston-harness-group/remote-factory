"""CodexRunner — OpenAI Codex CLI backend implementation.

Codex CLI (https://github.com/openai/codex) is OpenAI's open-source
agentic coding tool. Key interface differences from Claude Code:

- Headless mode uses ``codex exec "<prompt>"`` (positional arg to exec subcommand)
- System prompt is injected via an ``AGENTS.md`` file in the project directory
  (no --append-system-prompt-file flag)
- JSON output via ``--json`` (JSONL to stdout)
- Headless approval bypass via ``--dangerously-bypass-approvals-and-sandbox``
- Interactive approval bypass via ``--ask-for-approval never``
- Working directory via ``-C <path>``
- No session management (--name, --resume, --session-id)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.runners._subprocess import run_subprocess

if TYPE_CHECKING:
    from factory.models import AgentRunRequest, AgentRunResult, AgentUsage
    from factory.runners.protocol import RunnerMeta

log = structlog.get_logger()


def _parse_codex_usage(data: dict) -> AgentUsage:
    """Extract AgentUsage from Codex JSON output."""
    from factory.models import AgentUsage

    usage_block = data.get("usage", {})
    return AgentUsage(
        input_tokens=usage_block.get("input_tokens", 0)
        or usage_block.get("prompt_tokens", 0),
        output_tokens=usage_block.get("output_tokens", 0)
        or usage_block.get("completion_tokens", 0),
        cache_read_tokens=usage_block.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage_block.get("cache_creation_input_tokens", 0),
        total_cost_usd=data.get("total_cost_usd", 0.0) or 0.0,
        duration_ms=data.get("duration_ms", 0.0) or 0.0,
        num_turns=data.get("num_turns", 0) or 0,
        model=data.get("model", ""),
    )


_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku", "claude", "fable"}


class CodexRunner:
    """Runner implementation for OpenAI Codex CLI."""

    name: str = "codex"

    @classmethod
    def metadata(cls) -> RunnerMeta:
        from factory.runners.protocol import RunnerMeta

        return RunnerMeta(
            name="codex",
            display_name="OpenAI Codex CLI",
            binary="codex",
            install_hint="npm install -g @openai/codex",

            supports_usage_telemetry=False,
            supports_session_name=False,
            supports_session_resume=False,
            supports_background=False,
            supports_interactive=True,
            supports_streaming=True,
            supports_model_override=False,
        )

    @staticmethod
    def _resolve_model(model: str | None) -> str | None:
        """Strip Claude-specific model aliases; let Codex use its own default."""
        if not model:
            return None
        if model.lower() in _CLAUDE_MODEL_ALIASES or model.lower().startswith("claude"):
            return None
        return model

    @staticmethod
    def _build_combined_prompt(prompt: str, task: str) -> str:
        """Combine system prompt and task into a single Codex prompt.

        Codex has no --append-system-prompt-file equivalent, so we prepend
        the agent role prompt (researcher.md, builder.md, etc.) to the task.
        """
        return f"{prompt}\n\n---\n\n## Task\n\n{task}"

    def build_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        """Build the Codex CLI command, env dict, and temp files.

        Returns an empty temp_files list — Codex prompt injection is inline,
        not file-based, so there's nothing to clean up.
        """
        model = self._resolve_model(request.model)
        combined_prompt = self._build_combined_prompt(request.prompt, request.task)
        temp_files: list[Path] = []

        cmd = [
            "codex",
            "exec",
            "--json",
        ]
        if request.skip_permissions:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if request.cwd:
            cmd.extend(["-C", str(request.cwd)])
        if model:
            cmd.extend(["--model", model])
        cmd.append(combined_prompt)

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())
        if model:
            env["FACTORY_MODEL"] = model

        return cmd, env, temp_files

    async def headless(self, request: AgentRunRequest) -> AgentRunResult:
        """Run a headless Codex CLI invocation."""
        from factory.models import AgentRunResult

        cmd, env, temp_files = self.build_command(request)
        try:
            log.info("codex_headless", cwd=str(request.cwd), model=request.model)

            result = await run_subprocess(
                cmd,
                cwd=str(request.cwd),
                env=env,
                timeout=request.timeout,
                runner_name="codex",
                role=request.role,
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
                if isinstance(parsed, dict) and ("result" in parsed or "message" in parsed):
                    data = parsed
                    break

            if data is not None:
                result_value = data.get("result", data.get("message", result.stdout))
                result_text = result_value if isinstance(result_value, str) else result.stdout
                usage = _parse_codex_usage(data)

            return AgentRunResult(
                stdout=result_text,
                return_code=result.return_code,
                usage=usage,
                metadata=metadata,
            )
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)

    def interactive_run(self, request: AgentRunRequest) -> int:
        """Run an interactive Codex session as a subprocess."""
        model = self._resolve_model(request.model)
        combined_prompt = self._build_combined_prompt(request.prompt, request.task)

        cmd = ["codex"]
        if request.skip_permissions:
            cmd.extend(["--ask-for-approval", "never"])
        if request.cwd:
            cmd.extend(["-C", str(request.cwd)])
        if model:
            cmd.extend(["--model", model])
        cmd.append(combined_prompt)

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())
        if model:
            env["FACTORY_MODEL"] = model

        try:
            log.info("codex_interactive", cwd=str(request.cwd))
            result = subprocess.run(cmd, cwd=request.cwd, env=env)
            return result.returncode
        finally:
            pass
