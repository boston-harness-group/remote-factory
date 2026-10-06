"""Tests for model resolution and output validation in outer-loop claude CLI calls.

Covers two related fixes:

1. Internal `claude -p` callers (prompt rewriters, knob expanders, the
   reflector) resolve their model via FACTORY_MODEL / ANTHROPIC_MODEL
   instead of hardcoding the "opus" alias (which 403s on gateways that
   only serve specific models, e.g. LiteLLM proxies).
2. CLI error text printed to stdout ("API Error: ...", "Failed to
   authenticate ...") must never be accepted as LLM output — previously a
   403 error message could become a mutated prompt or knob value.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from factory.runners.claude import _claude_model


class TestClaudeModelResolution:
    def test_defaults_to_opus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FACTORY_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        assert _claude_model() == "opus"

    def test_factory_model_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FACTORY_MODEL", "my-gateway-model")
        monkeypatch.setenv("ANTHROPIC_MODEL", "other-model")
        assert _claude_model() == "my-gateway-model"

    def test_anthropic_model_used_when_factory_model_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FACTORY_MODEL", raising=False)
        monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-default")
        assert _claude_model() == "anthropic-default"


class TestPromptRewriterCliErrors:
    def _error_proc(self) -> MagicMock:
        proc = MagicMock()
        proc.stdout = (
            "Failed to authenticate. API Error: 403 team not allowed to "
            "access model. This team can only access models=['x']. "
            "Tried to access claude-opus-4-6"
        )
        proc.returncode = 1
        return proc

    def test_error_text_not_returned_as_prompt(self) -> None:
        from factory.outer_loop.mutations import default_prompt_rewriter

        with patch("subprocess.run", return_value=self._error_proc()):
            result = default_prompt_rewriter("build", "current prompt", None)
        assert result is None

    def test_nonzero_returncode_rejected_even_with_text(self) -> None:
        from factory.outer_loop.mutations import default_prompt_rewriter

        proc = MagicMock()
        proc.stdout = "A perfectly plausible rewritten prompt."
        proc.returncode = 1
        with patch("subprocess.run", return_value=proc):
            result = default_prompt_rewriter("build", "current prompt", None)
        assert result is None

    def test_api_error_prefix_rejected_even_on_zero_returncode(self) -> None:
        from factory.outer_loop.mutations import default_prompt_rewriter

        proc = MagicMock()
        # claude -p has been observed printing auth errors to stdout with
        # exit code 0 on some gateway setups
        proc.stdout = "API Error: 500 internal error"
        proc.returncode = 0
        with patch("subprocess.run", return_value=proc):
            result = default_prompt_rewriter("build", "current prompt", None)
        assert result is None

    def test_model_env_var_passed_to_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from factory.outer_loop.mutations import default_prompt_rewriter

        monkeypatch.setenv("FACTORY_MODEL", "gateway/glm-5-3")
        proc = MagicMock()
        proc.stdout = "rewritten prompt"
        proc.returncode = 0

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = default_prompt_rewriter("build", "current prompt", None)

        assert result == "rewritten prompt"
        cmd = mock_run.call_args.args[0]
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gateway/glm-5-3"


class TestKnobExpanderCliErrors:
    def test_error_text_not_returned_as_value(self) -> None:
        from factory.outer_loop.mutations import default_knob_expander

        proc = MagicMock()
        proc.stdout = "Failed to authenticate. API Error: 403 ..."
        proc.returncode = 1
        with patch("subprocess.run", return_value=proc):
            result = default_knob_expander("threshold", "hint", 0.5, [0.1, 0.9])
        assert result is None

    def test_valid_numeric_output_still_works(self) -> None:
        from factory.outer_loop.mutations import default_knob_expander

        proc = MagicMock()
        proc.stdout = "0.75"
        proc.returncode = 0
        with patch("subprocess.run", return_value=proc):
            result = default_knob_expander("threshold", "hint", 0.5, [0.1, 0.9])
        assert result == 0.75


class TestReflectorCliErrors:
    def test_llm_reflect_fast_fails_on_cli_error(self) -> None:
        """A CLI-level auth/model error must not be retried 3x or parsed."""
        from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport
        from tests.test_outer_loop.test_reflector import _make_record, _make_step

        reflector = OuterLoopReflector(k=1, llm_reflect=True)
        recs_top = [("w1", 0.9, _make_record(0.9, [_make_step("builder")]))]
        recs_bottom = [("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)]))]
        report = ReflectionReport()

        proc = MagicMock()
        proc.stdout = "Failed to authenticate. API Error: 403 ..."
        proc.returncode = 1

        with patch(
            "factory.outer_loop.reflector.subprocess.run", return_value=proc
        ) as mock_run:
            reflector._llm_reflect(recs_top, recs_bottom, [], report)

        # fast-fail: exactly one attempt, no 3x retry loop
        assert mock_run.call_count == 1
        # nothing was extracted from the error text
        assert report.prompt_improvements == []

    def test_llm_reflect_uses_resolved_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport
        from tests.test_outer_loop.test_reflector import _make_record, _make_step

        monkeypatch.setenv("FACTORY_MODEL", "gateway/glm-5-3")
        reflector = OuterLoopReflector(k=1, llm_reflect=True)
        recs_top = [("w1", 0.9, _make_record(0.9, [_make_step("builder")]))]
        recs_bottom = [("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)]))]
        report = ReflectionReport()

        proc = MagicMock()
        proc.stdout = json.dumps(
            {"prompt_improvements": ["Be concise"], "failure_patterns": []}
        )
        proc.returncode = 0

        with patch("factory.outer_loop.reflector.subprocess.run", return_value=proc) as mock_run:
            reflector._llm_reflect(recs_top, recs_bottom, [], report)

        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("--model") + 1] == "gateway/glm-5-3"
        assert "Be concise" in report.prompt_improvements
