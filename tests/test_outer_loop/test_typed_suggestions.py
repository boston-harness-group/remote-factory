"""Tests for typed mutation suggestions, LLM reflection, and knob propagation.

Covers Parts A, B, and C of the H1 hypothesis:
- Part A: LLM contrastive reflection populates prompt_improvements
- Part B: MutationSuggestion typed contract, select_guided_operator, mutate_knob
- Part C: OptKnob propagation through seed path
- Integration: reflector typed_suggestions consumed by select_guided_operator
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import patch

from factory.cycle_analyzer import AgentStep, CycleRecord
from factory.outer_loop.models import MutationType
from factory.outer_loop.mutations import (
    WeightedRandomStrategy,
    _parse_knob_suggestion,
    mutate_knob,
)
from factory.outer_loop.reflector import (
    MutationSuggestion,
    OuterLoopReflector,
    ReflectionReport,
)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Workflow,
)


def _make_record(
    score: float,
    steps: list[AgentStep] | None = None,
    kept: int = 0,
    reverted: int = 0,
    errored: int = 0,
    eval_details: dict | None = None,
    instance_results: list[dict] | None = None,
) -> CycleRecord:
    return CycleRecord(
        cycle_number=1,
        mode="test",
        started_at=None,
        ended_at=None,
        duration_s=10.0,
        score_start=0.0,
        score_end=score,
        score_delta=score,
        steps=steps or [],
        kept=kept,
        reverted=reverted,
        errored=errored,
        eval_details=eval_details,
        instance_results=instance_results,
    )


def _make_step(role: str, succeeded: bool = True, error: str | None = None, duration: float = 10.0) -> AgentStep:
    return AgentStep(
        order=0,
        role=role,
        started_at="2024-01-01T00:00:00",
        duration_s=duration,
        cost_usd=0.1,
        output_tokens=100,
        succeeded=succeeded,
        error=error,
    )


def _make_workflow_with_knobs() -> Workflow:
    return Workflow(
        name="knobbed-seed",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                model="opus",
                timeout=600,
            ),
        },
        edges=[],
        start_node="builder",
        terminal=True,
        knob_values={"max_retries": 3, "style": "focused"},
        knob_bounds={"max_retries": [1, 2, 3, 5], "style": ["focused", "broad", "creative"]},
        knob_expandable={"style": "Generation style for the builder agent"},
    )


# ── Part B: MutationSuggestion dataclass ──────────────────────


class TestMutationSuggestion:
    def test_basic_creation(self) -> None:
        ms = MutationSuggestion(
            operator="node_insert",
            target="researcher",
            rationale="Present in winners but not losers",
        )
        assert ms.operator == "node_insert"
        assert ms.target == "researcher"
        assert ms.value is None

    def test_with_value(self) -> None:
        ms = MutationSuggestion(
            operator="knob_mutate",
            target="max_retries",
            rationale="Higher retries correlated with success",
            value="5",
        )
        assert ms.value == "5"


class TestTypedSuggestionsProduced:
    def test_role_diff_produces_typed_suggestions(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("researcher"), _make_step("builder")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [_make_step("builder")], reverted=1)),
        ]
        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0)

        assert len(report.typed_suggestions) > 0
        insert_suggestions = [s for s in report.typed_suggestions if s.operator == "node_insert"]
        assert any(s.target == "researcher" for s in insert_suggestions)

    def test_timeout_produces_typed_param_mutate(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=1)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False, duration=600.0)])),
        ]
        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0)

        param_suggestions = [s for s in report.typed_suggestions if s.operator == "param_mutate"]
        assert len(param_suggestions) > 0

    def test_knob_patterns_produce_typed_knob_mutate(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]
        kvbi = {
            "w1": {"style": "focused"},
            "l1": {"style": "broad"},
        }
        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0, knob_values_by_id=kvbi)

        knob_suggestions = [s for s in report.typed_suggestions if s.operator == "knob_mutate"]
        assert len(knob_suggestions) > 0
        assert knob_suggestions[0].target == "style"
        assert knob_suggestions[0].value is not None

    def test_backward_compat_string_fields_populated(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("researcher"), _make_step("builder")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [_make_step("builder")], reverted=1)),
        ]
        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0)

        assert len(report.mutation_suggestions) > 0
        assert len(report.typed_suggestions) > 0


# ── Part B: select_guided_operator with typed suggestions ─────


class TestSelectGuidedOperatorTyped:
    def test_typed_suggestions_direct_enum_match(self) -> None:
        strategy = WeightedRandomStrategy()
        wf = _make_workflow_with_knobs()

        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(operator="knob_mutate", target="style", rationale="test"),
                MutationSuggestion(operator="knob_mutate", target="max_retries", rationale="test"),
                MutationSuggestion(operator="node_insert", target="researcher", rationale="test"),
            ],
        )

        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1

        # knob_mutate is filtered out by the guided-selection guard
        assert MutationType.KNOB_MUTATE not in counts, "knob_mutate should be filtered out"
        assert MutationType.NODE_INSERT in counts

    def test_falls_back_to_string_when_no_typed(self) -> None:
        strategy = WeightedRandomStrategy()
        wf = _make_workflow_with_knobs()

        report = ReflectionReport(
            mutation_suggestions=["NODE_INSERT: Add researcher agent"],
            typed_suggestions=[],
        )

        counts: dict[MutationType, int] = {}
        for _ in range(50):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1

        assert MutationType.NODE_INSERT in counts

    def test_invalid_operator_in_typed_ignored(self) -> None:
        strategy = WeightedRandomStrategy()
        wf = _make_workflow_with_knobs()

        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(operator="invalid_op", target="x", rationale="test"),
                MutationSuggestion(operator="knob_mutate", target="style", rationale="test"),
            ],
        )

        # knob_mutate is filtered by the guided-selection guard and
        # invalid_op is dropped as unrecognised, so the result falls
        # back to random selection from all valid operators.
        op = strategy.select_guided_operator(wf, 0, report)
        assert op in MutationType


# ── Part B: _parse_knob_suggestion with typed input ──────────


class TestParseKnobSuggestion:
    def test_typed_mutation_suggestion(self) -> None:
        ms = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value="creative",
        )
        result = _parse_knob_suggestion(ms)
        assert result == ("style", "creative")

    def test_typed_without_value_returns_none(self) -> None:
        ms = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value=None,
        )
        result = _parse_knob_suggestion(ms)
        assert result is None

    def test_string_format_still_works(self) -> None:
        result = _parse_knob_suggestion("KNOB_MUTATE: style=focused (avg score +3) outperforms ...")
        assert result == ("style", "focused")

    def test_string_non_knob_returns_none(self) -> None:
        result = _parse_knob_suggestion("NODE_INSERT: Add researcher")
        assert result is None


# ── Part B: mutate_knob prefers typed suggestions ──────────


class TestMutateKnobTyped:
    def test_typed_knob_suggestion_used(self) -> None:
        wf = _make_workflow_with_knobs()
        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(
                    operator="knob_mutate",
                    target="style",
                    rationale="focused works best",
                    value="creative",
                ),
            ],
        )

        random.seed(1)
        result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None
        mutated_wf, rec = result
        assert rec.operator == MutationType.KNOB_MUTATE

    def test_falls_back_to_string_suggestions(self) -> None:
        wf = _make_workflow_with_knobs()
        report = ReflectionReport(
            mutation_suggestions=["KNOB_MUTATE: style=creative (avg score +5) outperforms ..."],
            typed_suggestions=[],
        )

        random.seed(1)
        result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None


# ── Part A: LLM contrastive reflection ──────────────────────


class TestLLMReflect:
    def test_llm_reflect_populates_prompt_improvements(self) -> None:
        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=1))]
        bottom_k = [("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)]))]
        records = list(top_k) + list(bottom_k)
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [
                "Focus on reading error messages before attempting fixes",
                "Use step-by-step reasoning for complex problems",
            ],
            "failure_patterns": [
                "Bottom candidates skip test verification",
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, records, report)

        assert len(report.prompt_improvements) == 2
        assert "step-by-step" in report.prompt_improvements[1]
        assert len(report.failure_patterns) == 1

    def test_llm_reflect_graceful_on_timeout(self) -> None:
        import subprocess as sp

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=sp.TimeoutExpired("cmd", 120)):
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []

    def test_llm_reflect_graceful_on_bad_json(self) -> None:
        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "not valid json at all"
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []

    def test_llm_reflect_graceful_on_empty_response(self) -> None:
        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []

    def test_llm_reflect_graceful_on_missing_claude(self) -> None:
        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=FileNotFoundError):
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []

    def test_collect_individual_details_generic_keys(self) -> None:
        rec = _make_record(
            0.3,
            eval_details={
                "verify": {
                    "verify_count": 3,
                    "failed_count": 2,
                    "instance_results": [
                        {"index": 0, "passed": False, "score": 0.0, "details": {"returncode": 1, "stderr": "error"}},
                        {"index": 1, "passed": True, "score": 1.0, "details": {"returncode": 0}},
                    ],
                },
                "custom_field": "some_domain_specific_value",
            },
        )
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.3, rec)
        assert "abc12345" in result
        assert "custom_field" in result
        assert "verify_count" in result


# ── Part A: Full reflect() with mocked LLM ──────────────────


class TestReflectWithLLM:
    def test_reflect_calls_llm_reflect(self) -> None:
        reflector = OuterLoopReflector(k=1, llm_reflect=True)
        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=1)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]

        fake_response = json.dumps({
            "prompt_improvements": ["Use targeted approach"],
            "failure_patterns": [],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            report = reflector.reflect(records, generation=0)

        assert "Use targeted approach" in report.prompt_improvements


# ── Part C: Knob propagation ────────────────────────────────


class TestKnobPropagation:
    def test_workflow_to_dict_preserves_knobs(self) -> None:
        wf = _make_workflow_with_knobs()
        data = wf.to_dict()
        assert "knob_values" in data
        assert data["knob_values"]["max_retries"] == 3
        assert data["knob_values"]["style"] == "focused"
        assert "knob_bounds" in data
        assert "knob_expandable" in data

    def test_workflow_round_trip_preserves_knobs(self) -> None:
        wf = _make_workflow_with_knobs()
        data = wf.to_dict()
        restored = Workflow.from_dict(data)
        assert restored.knob_values == {"max_retries": 3, "style": "focused"}
        assert restored.knob_bounds == {"max_retries": [1, 2, 3, 5], "style": ["focused", "broad", "creative"]}
        assert restored.knob_expandable == {"style": "Generation style for the builder agent"}

    def test_make_individual_preserves_knobs(self) -> None:
        from factory.outer_loop.population import Population

        wf = _make_workflow_with_knobs()
        ind = Population.make_individual(wf, generation=0)
        assert "knob_values" in ind.workflow_data
        assert ind.workflow_data["knob_values"]["max_retries"] == 3

    def test_seed_preserves_knobs_through_population(self) -> None:
        from factory.outer_loop.population import Population

        wf = _make_workflow_with_knobs()
        ind = Population.make_individual(wf, generation=0)

        restored_wf = Workflow.from_dict(ind.workflow_data)
        assert restored_wf.knob_values == wf.knob_values
        assert restored_wf.knob_bounds == wf.knob_bounds
        assert restored_wf.knob_expandable == wf.knob_expandable

    def test_workflow_without_knobs_seeds_empty(self) -> None:
        from factory.outer_loop.population import Population

        wf = Workflow(
            name="no-knobs",
            nodes={"builder": AgentNode(id="builder", role=AgentRole.BUILDER)},
            edges=[],
            start_node="builder",
            terminal=True,
        )
        ind = Population.make_individual(wf, generation=0)
        assert "knob_values" not in ind.workflow_data


# ── Part C: Serialization round-trip via save/load ───────────


class TestKnobSerializationRoundTrip:
    def test_population_save_load_preserves_knobs(self, tmp_path: Path) -> None:
        from factory.outer_loop.population import Population

        wf = _make_workflow_with_knobs()
        pop = Population()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        pop.save(tmp_path / "pop")
        loaded = Population.load(tmp_path / "pop")
        loaded_ind = loaded.individuals[0]

        assert loaded_ind.workflow_data["knob_values"]["max_retries"] == 3
        assert loaded_ind.workflow_data["knob_bounds"]["style"] == ["focused", "broad", "creative"]


# ── Integration: reflector → select_guided_operator E2E ──────


class TestReflectorToMutationIntegration:
    """End-to-end test that typed suggestions from the reflector flow correctly
    through select_guided_operator. This is the critical integration test
    that prevents the PR #1449 failure class: 'two parallel implementations
    drifted without a shared integration test'.
    """

    def test_reflector_typed_suggestions_consumed_by_select_guided_operator(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.8, _make_record(
                0.8,
                [_make_step("researcher"), _make_step("builder")],
                kept=1,
            )),
            ("l1", 0.2, _make_record(
                0.2,
                [_make_step("builder")],
                reverted=1,
            )),
        ]

        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0)

        assert len(report.typed_suggestions) > 0, "Reflector must produce typed suggestions"

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"builder": AgentNode(id="builder", role=AgentRole.BUILDER)},
            edges=[],
            start_node="builder",
        )

        op_counts: dict[MutationType, int] = {}
        for _ in range(200):
            op = strategy.select_guided_operator(wf, 0, report)
            op_counts[op] = op_counts.get(op, 0) + 1

        suggestion_ops = {MutationType(s.operator) for s in report.typed_suggestions}
        for expected_op in suggestion_ops:
            assert expected_op in op_counts, (
                f"Expected {expected_op} from typed suggestions to appear in "
                f"select_guided_operator results, got {op_counts}"
            )

    def test_reflector_knob_suggestions_consumed_by_mutate_knob(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]
        kvbi = {
            "w1": {"style": "focused"},
            "l1": {"style": "broad"},
        }
        with patch.object(reflector, "_llm_reflect"):
            report = reflector.reflect(records, generation=0, knob_values_by_id=kvbi)

        knob_typed = [s for s in report.typed_suggestions if s.operator == "knob_mutate"]
        assert len(knob_typed) > 0, "Reflector must produce knob_mutate typed suggestions"

        wf = _make_workflow_with_knobs()
        random.seed(42)
        result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None, "mutate_knob should use typed suggestion"

    def test_serialized_reflection_json_round_trip(self, tmp_path: Path) -> None:
        """Test that typed_suggestions survive JSON serialization and
        deserialization, as happens between _cmd_reflect and _cmd_evolve."""
        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)
        records = [
            ("w1", 0.8, _make_record(
                0.8, [_make_step("researcher"), _make_step("builder")], kept=1,
            )),
            ("l1", 0.2, _make_record(
                0.2, [_make_step("builder")], reverted=1,
            )),
        ]

        with patch.object(reflector, "_llm_reflect"):
            original_report = reflector.reflect(records, generation=5)

        json_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen5.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert "typed_suggestions" in data

        raw_typed = data.pop("typed_suggestions", [])
        filtered = {k: v for k, v in data.items() if k != "generation"}
        loaded_report = ReflectionReport(**filtered)
        for item in raw_typed:
            if isinstance(item, dict):
                loaded_report.typed_suggestions.append(
                    MutationSuggestion(
                        operator=item.get("operator", ""),
                        target=item.get("target", ""),
                        rationale=item.get("rationale", ""),
                        value=item.get("value"),
                    )
                )

        assert len(loaded_report.typed_suggestions) == len(original_report.typed_suggestions)
        for orig, loaded in zip(original_report.typed_suggestions, loaded_report.typed_suggestions):
            assert orig.operator == loaded.operator
            assert orig.target == loaded.target

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"builder": AgentNode(id="builder", role=AgentRole.BUILDER)},
            edges=[],
            start_node="builder",
        )
        op = strategy.select_guided_operator(wf, 0, loaded_report)
        assert isinstance(op, MutationType)
