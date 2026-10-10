"""Tests for SkillOpt meta-skill module."""
from __future__ import annotations

import json

from skillopt.optimizer.meta_skill import (
    MAX_META_SKILL_TOKENS,
    format_meta_skill_context,
    load_meta_skill_content,
    run_meta_skill,
    should_generate_meta_skill,
    validate_deployment_gate,
    _enforce_token_cap,
    _extract_json,
    _format_comparison_text,
)
from skillopt.engine.trainer import (
    generate_epoch_meta_skill,
    load_active_meta_skill,
)
from skillopt.gradient.reflect import (
    reflect_on_errors,
    reflect_on_successes,
    reflect_and_merge,
)


class TestFormatMetaSkillContext:
    def test_empty_input_returns_empty(self):
        assert format_meta_skill_context("") == ""

    def test_none_input_returns_empty(self):
        assert format_meta_skill_context(None) == ""

    def test_whitespace_only_returns_empty(self):
        assert format_meta_skill_context("   \n  ") == ""

    def test_non_empty_returns_formatted_block(self):
        result = format_meta_skill_context("Prefer concrete edits over vague ones.")
        assert "## Optimizer Meta Skill" in result
        assert "Prefer concrete edits over vague ones." in result
        assert "optimizer-side memory" in result

    def test_content_is_token_capped(self):
        long_content = "x" * (MAX_META_SKILL_TOKENS * 4 + 1000)
        result = format_meta_skill_context(long_content)
        content_part = result.split("## Optimizer Meta Skill\n")[1]
        header_end = content_part.find("\n\n") + 2
        raw_content = content_part[header_end:]
        assert len(raw_content) // 4 <= MAX_META_SKILL_TOKENS


class TestRunMetaSkill:
    def test_returns_none_without_chat_fn(self):
        result = run_meta_skill(
            prev_skill="prev",
            curr_skill="curr",
            comparison_pairs=[],
            chat_fn=None,
        )
        assert result is None

    def test_returns_parsed_result_with_mock_chat(self):
        response_json = json.dumps({
            "reasoning": "Concrete edits worked better",
            "meta_skill_content": "Prefer specific over vague edits.",
        })

        def mock_chat(system, user, **kwargs):
            return (response_json, {})

        result = run_meta_skill(
            prev_skill="skill v1",
            curr_skill="skill v2",
            comparison_pairs=[{"id": "1", "category": "improved", "task": "test"}],
            chat_fn=mock_chat,
        )
        assert result is not None
        assert result["reasoning"] == "Concrete edits worked better"
        assert result["meta_skill_content"] == "Prefer specific over vague edits."

    def test_returns_none_on_empty_meta_skill_content(self):
        response_json = json.dumps({
            "reasoning": "nothing useful",
            "meta_skill_content": "",
        })

        def mock_chat(system, user, **kwargs):
            return (response_json, {})

        result = run_meta_skill(
            prev_skill="prev",
            curr_skill="curr",
            comparison_pairs=[],
            chat_fn=mock_chat,
        )
        assert result is None

    def test_returns_none_on_chat_exception(self):
        def failing_chat(system, user, **kwargs):
            raise RuntimeError("API error")

        result = run_meta_skill(
            prev_skill="prev",
            curr_skill="curr",
            comparison_pairs=[],
            chat_fn=failing_chat,
        )
        assert result is None

    def test_custom_system_prompt_is_used(self):
        captured = {}

        def mock_chat(system, user, **kwargs):
            captured["system"] = system
            return (json.dumps({"reasoning": "x", "meta_skill_content": "y"}), {})

        run_meta_skill(
            prev_skill="prev",
            curr_skill="curr",
            comparison_pairs=[],
            system_prompt="Custom prompt here",
            chat_fn=mock_chat,
        )
        assert captured["system"] == "Custom prompt here"

    def test_token_cap_applied_to_result(self):
        long_content = "x" * (MAX_META_SKILL_TOKENS * 4 + 5000)
        response_json = json.dumps({
            "reasoning": "test",
            "meta_skill_content": long_content,
        })

        def mock_chat(system, user, **kwargs):
            return (response_json, {})

        result = run_meta_skill(
            prev_skill="prev",
            curr_skill="curr",
            comparison_pairs=[],
            chat_fn=mock_chat,
        )
        assert result is not None
        assert len(result["meta_skill_content"]) // 4 <= MAX_META_SKILL_TOKENS


class TestLoadMetaSkillContent:
    def test_returns_empty_for_epoch_zero(self, tmp_path):
        assert load_meta_skill_content(str(tmp_path), 0) == ""

    def test_returns_empty_when_no_file(self, tmp_path):
        assert load_meta_skill_content(str(tmp_path), 3) == ""

    def test_loads_existing_content(self, tmp_path):
        epoch_dir = tmp_path / "meta_skill" / "epoch_02"
        epoch_dir.mkdir(parents=True)
        result_file = epoch_dir / "meta_skill_result.json"
        result_file.write_text(json.dumps({
            "meta_skill_content": "Use concrete edits.",
        }))
        assert load_meta_skill_content(str(tmp_path), 2) == "Use concrete edits."

    def test_recency_window_skips_old_epochs(self, tmp_path):
        # Write content at epoch 1 only
        epoch_dir = tmp_path / "meta_skill" / "epoch_01"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "meta_skill_result.json").write_text(json.dumps({
            "meta_skill_content": "Old content.",
        }))
        # Requesting epoch 5 should not find epoch 1 (outside 3-epoch window)
        assert load_meta_skill_content(str(tmp_path), 5) == ""

    def test_recency_window_finds_recent_epoch(self, tmp_path):
        epoch_dir = tmp_path / "meta_skill" / "epoch_03"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "meta_skill_result.json").write_text(json.dumps({
            "meta_skill_content": "Recent content.",
        }))
        # Requesting epoch 5 should find epoch 3 (within 3-epoch window)
        assert load_meta_skill_content(str(tmp_path), 5) == "Recent content."

    def test_handles_malformed_json(self, tmp_path):
        epoch_dir = tmp_path / "meta_skill" / "epoch_02"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "meta_skill_result.json").write_text("not valid json{{{")
        assert load_meta_skill_content(str(tmp_path), 2) == ""


class TestValidateDeploymentGate:
    def test_clean_skill_returns_no_warnings(self):
        skill = "# Task Rules\n\n- Do X when Y happens.\n- Always check Z."
        assert validate_deployment_gate(skill) == []

    def test_detects_meta_skill_markers(self):
        skill = "# Task Rules\n\n## Optimizer Meta Skill\nSome guidance here."
        warnings = validate_deployment_gate(skill)
        assert len(warnings) >= 1
        assert any("Optimizer Meta Skill" in w for w in warnings)

    def test_detects_optimizer_memory_marker(self):
        skill = "This uses optimizer memory from prior epochs."
        warnings = validate_deployment_gate(skill)
        assert len(warnings) >= 1

    def test_detects_meta_skill_underscore_marker(self):
        skill = "The meta_skill content was generated."
        warnings = validate_deployment_gate(skill)
        assert len(warnings) >= 1


class TestShouldGenerateMetaSkill:
    def test_epoch_1_returns_false(self):
        assert should_generate_meta_skill(1, score_delta=0.05) is False

    def test_epoch_0_returns_false(self):
        assert should_generate_meta_skill(0, score_delta=0.1) is False

    def test_epoch_2_positive_delta_returns_true(self):
        assert should_generate_meta_skill(2, score_delta=0.01) is True

    def test_epoch_2_negative_delta_returns_false(self):
        assert should_generate_meta_skill(2, score_delta=-0.05) is False

    def test_epoch_2_zero_delta_returns_true(self):
        assert should_generate_meta_skill(2, score_delta=0.0) is True

    def test_none_delta_returns_true(self):
        assert should_generate_meta_skill(3, score_delta=None) is True


class TestTokenCapEnforcement:
    def test_short_content_unchanged(self):
        content = "Short content."
        assert _enforce_token_cap(content) == content

    def test_long_content_truncated(self):
        content = "x" * (MAX_META_SKILL_TOKENS * 4 + 2000)
        result = _enforce_token_cap(content)
        assert len(result) <= MAX_META_SKILL_TOKENS * 4
        assert len(result) // 4 <= MAX_META_SKILL_TOKENS

    def test_exact_boundary_unchanged(self):
        content = "x" * (MAX_META_SKILL_TOKENS * 4)
        assert _enforce_token_cap(content) == content


class TestExtractJson:
    def test_extracts_json_from_text(self):
        text = 'Here is the result: {"key": "value"} done.'
        assert _extract_json(text) == {"key": "value"}

    def test_returns_none_for_no_json(self):
        assert _extract_json("no json here") is None

    def test_handles_nested_json(self):
        text = '{"outer": {"inner": "val"}}'
        result = _extract_json(text)
        assert result == {"outer": {"inner": "val"}}

    def test_returns_none_for_malformed_json(self):
        assert _extract_json("{broken: json}") is None


class TestFormatComparisonText:
    def test_empty_pairs(self):
        result = _format_comparison_text([])
        assert "No comparison data" in result

    def test_formats_categories(self):
        pairs = [
            {"id": "1", "category": "improved", "task": "task A"},
            {"id": "2", "category": "regressed", "task": "task B"},
        ]
        result = _format_comparison_text(pairs)
        assert "Improved" in result or "improved" in result.lower()
        assert "Regressed" in result or "regressed" in result.lower()
        assert "task A" in result
        assert "task B" in result


class TestGenerateEpochMetaSkill:
    def test_skips_first_epoch(self, tmp_path):
        result = generate_epoch_meta_skill(
            str(tmp_path), epoch=1,
            prev_skill="prev", curr_skill="curr",
            comparison_pairs=[],
        )
        assert result["action"] == "skip_first_epoch"
        done_path = tmp_path / "meta_skill" / "epoch_01" / "meta_skill_result.json"
        assert done_path.exists()

    def test_resumes_from_existing(self, tmp_path):
        meta_dir = tmp_path / "meta_skill" / "epoch_02"
        meta_dir.mkdir(parents=True)
        existing = {"action": "write_meta_skill", "meta_skill_content": "cached"}
        (meta_dir / "meta_skill_result.json").write_text(json.dumps(existing))
        result = generate_epoch_meta_skill(
            str(tmp_path), epoch=2,
            prev_skill="prev", curr_skill="curr",
            comparison_pairs=[],
        )
        assert result["action"] == "write_meta_skill"

    def test_skips_negative_delta(self, tmp_path):
        result = generate_epoch_meta_skill(
            str(tmp_path), epoch=3,
            prev_skill="prev", curr_skill="curr",
            comparison_pairs=[],
            score_delta=-0.1,
        )
        assert result["action"] == "skip_negative_delta"

    def test_generates_with_chat_fn(self, tmp_path):
        response = json.dumps({
            "reasoning": "test",
            "meta_skill_content": "Use targeted edits.",
        })

        def mock_chat(system, user, **kwargs):
            return (response, {})

        result = generate_epoch_meta_skill(
            str(tmp_path), epoch=2,
            prev_skill="prev", curr_skill="curr",
            comparison_pairs=[],
            score_delta=0.05,
            chat_fn=mock_chat,
        )
        assert result is not None
        assert result["action"] == "write_meta_skill"
        assert result["meta_skill_content"] == "Use targeted edits."
        done_path = tmp_path / "meta_skill" / "epoch_02" / "meta_skill_result.json"
        assert done_path.exists()


class TestLoadActiveMetaSkill:
    def test_returns_empty_when_disabled(self, tmp_path):
        result = load_active_meta_skill(str(tmp_path), epoch=3, use_meta_skill=False)
        assert result == ""

    def test_returns_empty_when_no_file(self, tmp_path):
        result = load_active_meta_skill(str(tmp_path), epoch=3, use_meta_skill=True)
        assert result == ""

    def test_loads_previous_epoch(self, tmp_path):
        epoch_dir = tmp_path / "meta_skill" / "epoch_02"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "meta_skill_result.json").write_text(json.dumps({
            "meta_skill_content": "Loaded content.",
        }))
        result = load_active_meta_skill(str(tmp_path), epoch=3, use_meta_skill=True)
        assert result == "Loaded content."


class TestReflectFunctions:
    def test_reflect_on_errors_returns_none_without_items(self):
        result = reflect_on_errors(
            skill_content="skill",
            failed_items=[],
            prediction_dir="/tmp",
        )
        assert result is None

    def test_reflect_on_errors_returns_none_without_chat_fn(self):
        result = reflect_on_errors(
            skill_content="skill",
            failed_items=[{"id": "1", "task_description": "test"}],
            prediction_dir="/tmp",
        )
        assert result is None

    def test_reflect_on_errors_with_meta_skill_context(self):
        captured = {}

        def mock_chat(system, user, **kwargs):
            captured["user"] = user
            return (json.dumps({"patch": {"edits": []}}), {})

        reflect_on_errors(
            skill_content="skill content",
            failed_items=[{"id": "1", "task_description": "test"}],
            prediction_dir="/tmp",
            meta_skill_context="Use concrete edits.",
            chat_fn=mock_chat,
        )
        assert "Optimizer Meta Skill" in captured["user"]
        assert "Use concrete edits." in captured["user"]

    def test_reflect_on_successes_returns_none_without_items(self):
        result = reflect_on_successes(
            skill_content="skill",
            success_items=[],
            prediction_dir="/tmp",
        )
        assert result is None

    def test_reflect_and_merge_returns_none_without_chat_fn(self):
        result = reflect_and_merge(
            skill_content="skill",
            failure_patches=[],
            success_patches=[],
        )
        assert result is None

    def test_reflect_and_merge_includes_meta_skill(self):
        captured = {}

        def mock_chat(system, user, **kwargs):
            captured["user"] = user
            return (json.dumps({"merged_patch": {"edits": []}}), {})

        reflect_and_merge(
            skill_content="skill",
            failure_patches=[{"edits": []}],
            success_patches=[],
            meta_skill_context="Merge carefully.",
            chat_fn=mock_chat,
        )
        assert "Optimizer Meta Skill" in captured["user"]
        assert "Merge carefully." in captured["user"]
