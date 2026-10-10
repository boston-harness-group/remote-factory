"""Tests for OuterLoopReflector contrastive reflection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from factory.cycle_analyzer import AgentStep, CycleRecord
from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport


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


class TestOuterLoopReflector:
    def test_basic_reflection(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("winner1", 0.9, _make_record(0.9, [_make_step("builder"), _make_step("researcher")], kept=2)),
            ("loser1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False, error="timeout")], errored=1)),
        ]

        report = reflector.reflect(records, generation=0)

        assert len(report.failure_patterns) > 0
        assert len(report.success_patterns) > 0
        assert report.top_k_ids == ["winner1"]
        assert report.bottom_k_ids == ["loser1"]

    def test_mutation_suggestions_from_role_diff(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("researcher"), _make_step("builder")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [_make_step("builder")], reverted=1)),
        ]

        report = reflector.reflect(records, generation=0)

        role_suggestions = [s for s in report.mutation_suggestions if "researcher" in s.lower()]
        assert len(role_suggestions) > 0

    def test_insufficient_data(self) -> None:
        reflector = OuterLoopReflector(k=1)
        records = [("only1", 0.5, _make_record(0.5))]
        report = reflector.reflect(records, generation=0)

        assert len(report.failure_patterns) == 0
        assert len(report.success_patterns) == 0

    def test_none_records_filtered(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("builder")], kept=1)),
            ("n1", 0.5, None),
            ("l1", 0.2, _make_record(0.2, [_make_step("builder", succeeded=False)], errored=1)),
        ]

        report = reflector.reflect(records, generation=0)
        assert len(report.top_k_ids) == 1
        assert len(report.bottom_k_ids) == 1

    def test_save_report(self, tmp_path: Path) -> None:
        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)

        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("builder")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [], errored=1)),
        ]

        reflector.reflect(records, generation=3)

        json_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen3.json"
        md_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen3.md"
        assert json_path.exists()
        assert md_path.exists()

    def test_structural_recommendations_timeout(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=1)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False, duration=600.0)])),
        ]

        report = reflector.reflect(records, generation=0)
        timeout_recs = [r for r in report.structural_recommendations if "timeout" in r.lower()]
        assert len(timeout_recs) > 0

    def test_multiple_winners_losers(self) -> None:
        reflector = OuterLoopReflector(k=2)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("w2", 0.85, _make_record(0.85, [_make_step("builder"), _make_step("researcher")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [], errored=1)),
            ("l2", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], reverted=2)),
        ]

        report = reflector.reflect(records, generation=0)
        assert len(report.top_k_ids) == 2
        assert len(report.bottom_k_ids) == 2

    def test_reflect_with_knob_values_produces_suggestions(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]

        kvbi = {
            "w1": {"style": "focused", "_prompt_builder": "Be precise"},
            "l1": {"style": "broad", "_prompt_builder": "Be creative"},
        }

        report = reflector.reflect(records, generation=0, knob_values_by_id=kvbi)

        knob_suggestions = [
            s for s in report.mutation_suggestions if "KNOB_MUTATE" in s or "PROMPT_MUTATE" in s
        ]
        assert len(knob_suggestions) > 0

        assert len(report.prompt_improvements) > 0

    def test_llm_reflect_non_dict_json_gracefully_returns(self) -> None:
        """_llm_reflect should not crash when LLM returns non-dict JSON."""
        from unittest.mock import patch, MagicMock

        reflector = OuterLoopReflector(k=1, llm_reflect=True)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]

        non_dict_payloads = [
            '[1, 2, 3]',       # JSON array
            '42',              # JSON number
            '"hello"',         # JSON string
            'null',            # JSON null
            'true',            # JSON boolean
        ]

        for payload in non_dict_payloads:
            mock_proc = MagicMock()
            mock_proc.stdout = payload
            mock_proc.returncode = 0

            with patch("subprocess.run", return_value=mock_proc), \
                 patch("time.sleep"):
                report = reflector.reflect(records, generation=0)
                assert isinstance(report, ReflectionReport)

    def test_reflect_without_knob_values_no_knob_suggestions(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]

        report = reflector.reflect(records, generation=0)

        knob_suggestions = [
            s for s in report.mutation_suggestions if "KNOB_MUTATE" in s
        ]
        assert len(knob_suggestions) == 0


class TestCollectIndividualDetails:
    """Cover branches in _collect_individual_details."""

    def test_none_record(self) -> None:
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.5, None)
        assert "abc12345" in result
        assert "0.500" in result

    def test_eval_details_with_verify_and_instances(self) -> None:
        """Instance data now comes from rec.instance_results (BUG 2 fix)."""
        rec = _make_record(
            0.4,
            instance_results=[
                {"item_id": "i0", "split": "train", "status": "ok",
                 "passed": False, "score": 0.0},
                {"item_id": "i1", "split": "train", "status": "ok",
                 "passed": True, "score": 1.0},
            ],
        )
        result = OuterLoopReflector._collect_individual_details("def12345", 0.4, rec)
        assert "instance" in result
        assert "i0" in result
        assert "i1" in result

    def test_instance_results_on_record(self) -> None:
        rec = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=0.0,
            score_end=0.6,
            score_delta=0.6,
            steps=[],
            instance_results=[
                {"item_id": "t1", "split": "train", "status": "ok",
                 "passed": True, "score": 1.0},
                {"item_id": "t2", "split": "train", "status": "failed",
                 "passed": False, "score": 0.0},
            ],
        )
        result = OuterLoopReflector._collect_individual_details("ghi12345", 0.6, rec)
        assert "item_id" in result

    def test_steps_in_details(self) -> None:
        rec = _make_record(
            0.7,
            steps=[_make_step("builder"), _make_step("researcher", succeeded=False)],
        )
        result = OuterLoopReflector._collect_individual_details("jkl12345", 0.7, rec)
        assert "builder(ok)" in result
        assert "researcher(FAIL)" in result

    def test_non_dict_instance_skipped(self) -> None:
        """Non-dict items in instance_results are skipped."""
        rec = _make_record(
            0.3,
            instance_results=[
                "not a dict",
                42,
                {"item_id": "i0", "split": "train", "status": "ok",
                 "passed": True, "score": 1.0},
            ],
        )
        result = OuterLoopReflector._collect_individual_details("skip123", 0.3, rec)
        assert "i0" in result

    def test_verify_not_a_dict(self) -> None:
        """When instance_results is empty, only header is returned."""
        rec = _make_record(0.3, instance_results=[])
        result = OuterLoopReflector._collect_individual_details("verstr12", 0.3, rec)
        assert "verstr12" in result

    def test_eval_details_without_verify(self) -> None:
        """Instance results with verify_details appear in output."""
        rec = _make_record(
            0.5,
            instance_results=[
                {"item_id": "c1", "split": "train", "status": "ok",
                 "passed": True, "score": 1.0,
                 "verify_details": {"custom": "data"}},
            ],
        )
        result = OuterLoopReflector._collect_individual_details("noverify", 0.5, rec)
        assert "custom" in result
        assert "score" in result


class TestMutationSuggestionsAvgSteps:
    """Cover typed_suggestions from avg step count comparison."""

    def test_top_more_steps_produces_node_insert(self) -> None:
        """When top-K has significantly more steps than bottom-K, suggest NODE_INSERT."""
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(
                0.9,
                [_make_step("researcher"), _make_step("strategist"), _make_step("builder")],
                kept=2,
            )),
            ("l1", 0.1, _make_record(
                0.1,
                [_make_step("builder", succeeded=False)],
                errored=1,
            )),
        ]
        report = reflector.reflect(records, generation=0)

        insert_typed = [s for s in report.typed_suggestions if s.operator == "node_insert" and s.target == "any"]
        assert len(insert_typed) > 0

    def test_bottom_more_steps_produces_node_remove(self) -> None:
        """When bottom-K has significantly more steps than top-K, suggest NODE_REMOVE."""
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(
                0.9,
                [_make_step("builder")],
                kept=2,
            )),
            ("l1", 0.1, _make_record(
                0.1,
                [_make_step("researcher"), _make_step("strategist"), _make_step("builder", succeeded=False)],
                errored=1,
            )),
        ]
        report = reflector.reflect(records, generation=0)

        remove_typed = [s for s in report.typed_suggestions if s.operator == "node_remove" and s.target == "any"]
        assert len(remove_typed) > 0

    def test_roles_in_bottom_not_top_produces_node_remove(self) -> None:
        """When bottom has succeeded roles not in top, produce NODE_REMOVE typed suggestions."""
        reflector = OuterLoopReflector(k=1)
        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(
                0.1,
                [_make_step("researcher", succeeded=True), _make_step("builder", succeeded=True)],
                reverted=2,
            )),
        ]
        report = reflector.reflect(records, generation=0)

        remove_suggestions = [s for s in report.typed_suggestions if s.operator == "node_remove" and s.target == "researcher"]
        assert len(remove_suggestions) > 0


class TestLLMReflectPayloadTruncation:
    """Cover the payload truncation branch in _llm_reflect."""

    def test_long_payload_truncated(self) -> None:
        """Payload exceeding _LLM_PAYLOAD_BUDGET (8000) gets truncated."""
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=3)
        long_details = {f"key_{i}": "x" * 600 for i in range(10)}
        recs_top = [
            (f"w{i}", 0.9, _make_record(0.9, [_make_step("builder")], eval_details=long_details))
            for i in range(3)
        ]
        recs_bottom = [
            (f"l{i}", 0.1, _make_record(0.1, [_make_step("builder")], eval_details=long_details))
            for i in range(3)
        ]
        report = ReflectionReport()

        captured_args: list = []

        def fake_run(*args, **kwargs):
            from unittest.mock import MagicMock
            captured_args.append(args)
            mock = MagicMock()
            mock.stdout = '{"prompt_improvements": ["Be concise"], "failure_patterns": []}'
            mock.returncode = 0
            return mock

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=fake_run):
            reflector._llm_reflect(recs_top, recs_bottom, [], report)

        assert "Be concise" in report.prompt_improvements

    def test_llm_reflect_with_failure_and_improvement_items(self) -> None:
        """Cover the full extraction path: both improvements and failures populated."""
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = (
            '{"prompt_improvements": ["Check errors first", "Use step-by-step"], '
            '"failure_patterns": ["Skipped tests", "Wrong file"]}'
        )

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.prompt_improvements) == 2
        assert len(report.failure_patterns) == 2

    def test_llm_reflect_non_string_items_skipped(self) -> None:
        """Non-string items in improvements/failures lists should be skipped."""
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = (
            '{"prompt_improvements": ["Valid", 42, null, ""], '
            '"failure_patterns": [123, "Real failure"]}'
        )

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == ["Valid"]
        assert report.failure_patterns == ["Real failure"]

    def test_llm_reflect_non_list_improvements_ignored(self) -> None:
        """Non-list prompt_improvements/failure_patterns handled gracefully."""
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = '{"prompt_improvements": "not a list", "failure_patterns": 42}'

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []
        assert report.failure_patterns == []

    def test_llm_reflect_value_error_handled(self) -> None:
        """ValueError from subprocess should be caught."""
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=ValueError("bad")):
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []


class TestStructuralRecommendationsForkNode:
    """Cover the ForkNode parallel execution recommendation path."""

    def test_fork_node_in_top_k_produces_parallelize_suggestion(self) -> None:
        from factory.cycle_analyzer import NodeTrace

        reflector = OuterLoopReflector(k=1)
        rec_with_fork = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=10.0,
            score_start=0.0,
            score_end=0.9,
            score_delta=0.9,
            steps=[_make_step("builder")],
            kept=2,
            node_trace={
                "fork_1": NodeTrace(
                    node_id="fork_1",
                    node_type="ForkNode",
                    role=None,
                    declared_writes=set(),
                    declared_reads=set(),
                ),
                "builder": NodeTrace(
                    node_id="builder",
                    node_type="AgentNode",
                    role="builder",
                    declared_writes=set(),
                    declared_reads=set(),
                ),
            },
        )
        records = [
            ("w1", 0.9, rec_with_fork),
            ("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)], errored=1)),
        ]
        report = reflector.reflect(records, generation=0)

        parallelize_typed = [s for s in report.typed_suggestions if s.operator == "parallelize"]
        assert len(parallelize_typed) > 0
        parallelize_recs = [r for r in report.structural_recommendations if "PARALLELIZE" in r]
        assert len(parallelize_recs) > 0


class TestSaveReportAllSections:
    """Cover all sections of _save_report markdown rendering."""

    def test_save_report_typed_suggestion_without_value(self, tmp_path: Path) -> None:
        from factory.outer_loop.reflector import MutationSuggestion

        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)
        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(
                    operator="node_insert",
                    target="researcher",
                    rationale="Add researcher for coverage",
                    value=None,
                ),
            ],
            top_k_ids=["w1"],
            bottom_k_ids=["l1"],
        )
        reflector._save_report(report, generation=11)

        md_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen11.md"
        md_content = md_path.read_text()
        assert "## Typed Suggestions" in md_content
        assert "[node_insert] researcher:" in md_content
        assert "value=" not in md_content

    def test_save_report_empty_report(self, tmp_path: Path) -> None:
        import json

        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)
        report = ReflectionReport(top_k_ids=["w1"], bottom_k_ids=["l1"])
        reflector._save_report(report, generation=12)

        json_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen12.json"
        data = json.loads(json_path.read_text())
        assert data["typed_suggestions"] == []
        assert data["prompt_improvements"] == []


class TestLLMReflectJSONExtraction:
    """Cover JSON extraction branches in _llm_reflect."""

    def test_json_embedded_in_text(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9, [_make_step("builder")]))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        response_with_prefix = (
            'Here is my analysis:\n'
            '{"prompt_improvements": ["Try harder"], "failure_patterns": ["Bad strategy"]}\n'
            'Hope this helps!'
        )

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = response_with_prefix
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert "Try harder" in report.prompt_improvements
        assert "Bad strategy" in report.failure_patterns

    def test_non_dict_json_returns_early(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "[1, 2, 3]"
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []

    def test_os_error_handled(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=OSError("no such file")):
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert report.prompt_improvements == []


class TestSaveReportTypedSuggestions:
    """Cover typed_suggestions serialization and markdown rendering in _save_report."""

    def test_save_report_with_typed_suggestions(self, tmp_path: Path) -> None:
        import json

        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)
        records = [
            ("w1", 0.8, _make_record(0.8, [_make_step("researcher"), _make_step("builder")], kept=1)),
            ("l1", 0.2, _make_record(0.2, [_make_step("builder", succeeded=False, duration=600.0)])),
        ]
        report = reflector.reflect(records, generation=7)

        json_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen7.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "typed_suggestions" in data
        assert isinstance(data["typed_suggestions"], list)
        for ts in data["typed_suggestions"]:
            assert "operator" in ts
            assert "target" in ts
            assert "rationale" in ts

        md_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen7.md"
        assert md_path.exists()
        md_content = md_path.read_text()
        if report.typed_suggestions:
            assert "## Typed Suggestions" in md_content

    def test_save_report_with_prompt_improvements(self, tmp_path: Path) -> None:
        from factory.outer_loop.reflector import MutationSuggestion

        reflector = OuterLoopReflector(k=1, project_dir=tmp_path)

        report = ReflectionReport(
            failure_patterns=["f1"],
            success_patterns=["s1"],
            mutation_suggestions=["m1"],
            prompt_improvements=["Focus on error messages"],
            structural_recommendations=["r1"],
            top_k_ids=["w1"],
            bottom_k_ids=["l1"],
            typed_suggestions=[
                MutationSuggestion(
                    operator="knob_mutate",
                    target="style",
                    rationale="Focused works best",
                    value="focused",
                ),
            ],
        )
        reflector._save_report(report, generation=9)

        md_path = tmp_path / ".factory" / "outer_loop" / "reflections" / "gen9.md"
        md_content = md_path.read_text()
        assert "## Prompt Improvements" in md_content
        assert "Focus on error messages" in md_content
        assert "## Typed Suggestions" in md_content
        assert "knob_mutate" in md_content
        assert "value=focused" in md_content


class TestExtractEvalPatternsVerify:
    """Cover verify/instance_results branches in _extract_eval_patterns."""

    def test_verify_failure_patterns_with_instance_details(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(
                0.9,
                [_make_step("builder")],
                kept=2,
                eval_details={
                    "verify": {
                        "verify_count": 3,
                        "passed_count": 3,
                    },
                },
            )),
            ("l1", 0.1, _make_record(
                0.1,
                [_make_step("builder", succeeded=False)],
                eval_details={
                    "verify": {
                        "verify_count": 3,
                        "failed_count": 2,
                        "instance_results": [
                            {"index": 0, "passed": False, "details": {"returncode": 1}},
                            {"index": 1, "passed": False, "details": {"returncode": 2}},
                            {"index": 2, "passed": True, "details": {"returncode": 0}},
                        ],
                    },
                },
            )),
        ]
        report = reflector.reflect(records, generation=0)

        returncode_patterns = [p for p in report.failure_patterns if "returncode=" in p]
        assert len(returncode_patterns) > 0

    def test_verify_score_comparison_mutation_suggestion(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(
                0.9,
                [_make_step("builder")],
                kept=2,
                eval_details={
                    "verify": {
                        "verify_count": 2,
                        "passed_count": 2,
                        "instance_results": [
                            {"index": 0, "passed": True, "score": 0.9},
                            {"index": 1, "passed": True, "score": 0.95},
                        ],
                    },
                },
            )),
            ("l1", 0.1, _make_record(
                0.1,
                [_make_step("builder", succeeded=False)],
                eval_details={
                    "verify": {
                        "verify_count": 2,
                        "failed_count": 2,
                        "instance_results": [
                            {"index": 0, "passed": False, "score": 0.1},
                            {"index": 1, "passed": False, "score": 0.2},
                        ],
                    },
                },
            )),
        ]
        report = reflector.reflect(records, generation=0)

        verify_suggestions = [s for s in report.mutation_suggestions if "verify" in s.lower()]
        assert len(verify_suggestions) > 0
        typed_prompt = [s for s in report.typed_suggestions if s.operator == "prompt_mutate"]
        assert len(typed_prompt) > 0

    def test_test_details_failure_pattern(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(
                0.9, [_make_step("builder")], kept=2,
                eval_details={"test_details": {"returncode": 0, "total": 10}},
            )),
            ("l1", 0.1, _make_record(
                0.1, [_make_step("builder", succeeded=False)],
                eval_details={"test_details": {"returncode": 1, "failed": 3, "total": 10}},
            )),
        ]
        report = reflector.reflect(records, generation=0)

        test_patterns = [p for p in report.failure_patterns if "test" in p.lower()]
        assert len(test_patterns) > 0

    def test_rejected_and_error_patterns(self) -> None:
        reflector = OuterLoopReflector(k=1)

        records = [
            ("w1", 0.9, _make_record(0.9, [_make_step("builder")], kept=2)),
            ("l1", 0.1, _make_record(
                0.1, [_make_step("builder", succeeded=False)],
                eval_details={
                    "rejected": "timeout exceeded",
                    "error": "process crashed with SIGSEGV",
                },
            )),
        ]
        report = reflector.reflect(records, generation=0)

        rejected_patterns = [p for p in report.failure_patterns if "rejected" in p.lower()]
        assert len(rejected_patterns) > 0
        error_patterns = [p for p in report.failure_patterns if "error" in p.lower()]
        assert len(error_patterns) > 0


class TestLLMReflectMutationSuggestions:
    """Tests for LLM-generated typed mutation suggestions."""

    def test_llm_mutation_suggestions_parsed(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9, [_make_step("builder")]))]
        bottom_k = [("l1", 0.1, _make_record(0.1, [_make_step("builder", succeeded=False)]))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": ["Use step-by-step reasoning"],
            "failure_patterns": ["Missing error handling"],
            "mutation_suggestions": [
                {
                    "operator": "prompt_mutate",
                    "target": "builder",
                    "rationale": "Builder needs step-by-step instructions",
                },
                {
                    "operator": "knob_mutate",
                    "target": "temperature",
                    "rationale": "Lower temperature for more deterministic output",
                    "value": "0.3",
                },
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 2
        assert report.typed_suggestions[0].operator == "prompt_mutate"
        assert report.typed_suggestions[0].target == "builder"
        assert report.typed_suggestions[1].operator == "knob_mutate"
        assert report.typed_suggestions[1].value == "0.3"

    def test_invalid_operators_filtered_out(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [],
            "failure_patterns": [],
            "mutation_suggestions": [
                {"operator": "prompt_mutate", "target": "builder", "rationale": "Valid"},
                {"operator": "invalid_op", "target": "x", "rationale": "Should be filtered"},
                {"operator": "magic_improve", "target": "y", "rationale": "Also invalid"},
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 1
        assert report.typed_suggestions[0].operator == "prompt_mutate"

    def test_empty_mutation_suggestions_handled(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": ["Improve prompts"],
            "failure_patterns": [],
            "mutation_suggestions": [],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 0
        assert len(report.prompt_improvements) == 1

    def test_missing_mutation_suggestions_field_handled(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": ["Advice"],
            "failure_patterns": ["Problem"],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 0
        assert len(report.prompt_improvements) == 1

    def test_non_dict_items_in_mutation_suggestions_skipped(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [],
            "failure_patterns": [],
            "mutation_suggestions": [
                "not a dict",
                42,
                None,
                {"operator": "node_insert", "target": "researcher", "rationale": "Valid"},
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 1
        assert report.typed_suggestions[0].operator == "node_insert"

    def test_missing_target_or_operator_skipped(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [],
            "failure_patterns": [],
            "mutation_suggestions": [
                {"operator": "prompt_mutate", "rationale": "No target"},
                {"target": "builder", "rationale": "No operator"},
                {"operator": 123, "target": "builder", "rationale": "Non-string op"},
                {"operator": "prompt_mutate", "target": 456, "rationale": "Non-string target"},
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 0

    def test_node_ids_in_prompt_context(self) -> None:
        from unittest.mock import patch
        from factory.cycle_analyzer import NodeTrace

        reflector = OuterLoopReflector(k=1)
        rec_with_trace = CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=10.0, score_start=0.0, score_end=0.9, score_delta=0.9,
            steps=[_make_step("builder")], kept=2,
            node_trace={
                "builder": NodeTrace(
                    node_id="builder", node_type="AgentNode", role="builder",
                    declared_writes=set(), declared_reads=set(),
                ),
                "researcher": NodeTrace(
                    node_id="researcher", node_type="AgentNode", role="researcher",
                    declared_writes=set(), declared_reads=set(),
                ),
            },
        )
        top_k = [("w1", 0.9, rec_with_trace)]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        captured_prompts: list[str] = []

        def fake_run(*args, **kwargs):
            from unittest.mock import MagicMock
            # BUG 3 fix: prompt now arrives via input= kwarg, not as CLI arg
            prompt_text = kwargs.get("input", "")
            captured_prompts.append(prompt_text)
            mock = MagicMock()
            mock.stdout = '{"prompt_improvements": [], "failure_patterns": [], "mutation_suggestions": []}'
            mock.returncode = 0
            return mock

        with patch("factory.outer_loop.reflector.subprocess.run", side_effect=fake_run):
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(captured_prompts) == 1
        assert "builder" in captured_prompts[0]
        assert "researcher" in captured_prompts[0]
        assert "AVAILABLE WORKFLOW NODES" in captured_prompts[0]

    def test_mutation_suggestion_value_coerced_to_string(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [],
            "failure_patterns": [],
            "mutation_suggestions": [
                {"operator": "knob_mutate", "target": "temp", "rationale": "Lower", "value": 0.5},
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 1
        assert report.typed_suggestions[0].value == "0.5"

    def test_mutation_suggestion_null_value_stays_none(self) -> None:
        from unittest.mock import patch

        reflector = OuterLoopReflector(k=1)
        top_k = [("w1", 0.9, _make_record(0.9))]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        fake_response = json.dumps({
            "prompt_improvements": [],
            "failure_patterns": [],
            "mutation_suggestions": [
                {"operator": "prompt_mutate", "target": "builder", "rationale": "Improve", "value": None},
            ],
        })

        with patch("factory.outer_loop.reflector.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_response
            mock_run.return_value.returncode = 0
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(report.typed_suggestions) == 1
        assert report.typed_suggestions[0].value is None


class TestCollectNodeIds:
    """Tests for _collect_node_ids static method."""

    def test_collects_from_node_trace(self) -> None:
        from factory.cycle_analyzer import NodeTrace

        rec = CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=10.0, score_start=0.0, score_end=0.9, score_delta=0.9,
            steps=[], kept=1,
            node_trace={
                "builder": NodeTrace(
                    node_id="builder", node_type="AgentNode", role="builder",
                    declared_writes=set(), declared_reads=set(),
                ),
            },
        )
        result = OuterLoopReflector._collect_node_ids([("w1", 0.9, rec)])
        assert len(result) == 1
        assert result[0]["node_id"] == "builder"
        assert result[0]["role"] == "builder"

    def test_collects_from_steps_fallback(self) -> None:
        rec = _make_record(0.9, [_make_step("researcher"), _make_step("builder")])
        result = OuterLoopReflector._collect_node_ids([("w1", 0.9, rec)])
        roles = {n["node_id"] for n in result}
        assert "researcher" in roles
        assert "builder" in roles

    def test_none_records_skipped(self) -> None:
        result = OuterLoopReflector._collect_node_ids([("w1", 0.9, None)])
        assert len(result) == 0

    def test_deduplicates(self) -> None:
        from factory.cycle_analyzer import NodeTrace

        rec1 = CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=10.0, score_start=0.0, score_end=0.9, score_delta=0.9,
            steps=[_make_step("builder")], kept=1,
            node_trace={
                "builder": NodeTrace(
                    node_id="builder", node_type="AgentNode", role="builder",
                    declared_writes=set(), declared_reads=set(),
                ),
            },
        )
        rec2 = CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=10.0, score_start=0.0, score_end=0.5, score_delta=0.5,
            steps=[_make_step("builder")], kept=0,
            node_trace={
                "builder": NodeTrace(
                    node_id="builder", node_type="AgentNode", role="builder",
                    declared_writes=set(), declared_reads=set(),
                ),
            },
        )
        result = OuterLoopReflector._collect_node_ids([("w1", 0.9, rec1), ("l1", 0.5, rec2)])
        assert len(result) == 1


class TestExperimentContextInDetails:
    """Tests for rec.experiments appearing in _collect_individual_details."""

    def _make_record_with_experiments(
        self,
        experiments: list | None = None,
        steps: list[AgentStep] | None = None,
    ) -> CycleRecord:
        from factory.cycle_analyzer import ExperimentRecord

        if experiments is None:
            experiments = [
                ExperimentRecord(
                    exp_id=1, hypothesis="Add retry logic for flaky tests",
                    verdict="keep", score_before=0.5, score_after=0.55,
                    score_delta=0.05, cost_usd=0.1, duration_s=30.0,
                ),
                ExperimentRecord(
                    exp_id=2, hypothesis="Refactor error handling",
                    verdict="revert", score_before=0.55, score_after=0.50,
                    score_delta=-0.05, cost_usd=0.2, duration_s=45.0,
                ),
            ]
        return CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=60.0, score_start=0.5, score_end=0.55, score_delta=0.05,
            steps=steps or [], experiments=experiments,
            kept=1, reverted=1,
        )

    def test_experiments_appear_in_output(self) -> None:
        rec = self._make_record_with_experiments()
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.55, rec)
        assert "exp1(keep" in result
        assert "exp2(revert" in result
        assert "+0.050" in result
        assert "-0.050" in result

    def test_hypothesis_text_included(self) -> None:
        rec = self._make_record_with_experiments()
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.55, rec)
        assert "retry logic" in result

    def test_hypothesis_truncated_under_tight_budget(self) -> None:
        from factory.cycle_analyzer import ExperimentRecord

        long_hyp = "A" * 500
        experiments = [
            ExperimentRecord(
                exp_id=1, hypothesis=long_hyp, verdict="keep",
                score_before=0.5, score_after=0.6, score_delta=0.1,
                cost_usd=0.1, duration_s=30.0,
            ),
        ]
        rec = self._make_record_with_experiments(experiments=experiments)
        # Budget large enough to include the experiment tag but not the
        # full 500-char hypothesis.  Hypothesis is capped at 200 chars
        # by the implementation.
        result = OuterLoopReflector._collect_individual_details(
            "abc12345", 0.6, rec, char_budget=500,
        )
        assert "exp1(keep" in result
        assert long_hyp not in result

    def test_experiments_omitted_when_budget_exhausted(self) -> None:
        from factory.cycle_analyzer import ExperimentRecord

        experiments = [
            ExperimentRecord(
                exp_id=1, hypothesis="Should not appear", verdict="keep",
                score_before=0.5, score_after=0.6, score_delta=0.1,
                cost_usd=0.1, duration_s=30.0,
            ),
        ]
        # Budget is just enough for the header (30 chars), not the experiment
        rec = CycleRecord(
            cycle_number=1, mode="test", started_at=None, ended_at=None,
            duration_s=60.0, score_start=0.5, score_end=0.6, score_delta=0.1,
            steps=[], experiments=experiments,
        )
        result = OuterLoopReflector._collect_individual_details(
            "abc12345", 0.6, rec, char_budget=30,
        )
        assert "exp1" not in result

    def test_char_budget_none_still_includes_experiments(self) -> None:
        rec = self._make_record_with_experiments()
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.55, rec)
        assert "exp1" in result
        assert "exp2" in result

    def test_no_experiments_field_unchanged_output(self) -> None:
        rec = _make_record(0.7, steps=[_make_step("builder")])
        result_without_budget = OuterLoopReflector._collect_individual_details(
            "abc12345", 0.7, rec,
        )
        result_with_budget = OuterLoopReflector._collect_individual_details(
            "abc12345", 0.7, rec, char_budget=5000,
        )
        assert result_without_budget == result_with_budget

    def test_experiment_without_hypothesis(self) -> None:
        from factory.cycle_analyzer import ExperimentRecord

        experiments = [
            ExperimentRecord(
                exp_id=3, hypothesis=None, verdict="keep",
                score_before=0.5, score_after=0.6, score_delta=0.1,
                cost_usd=0.1, duration_s=30.0,
            ),
        ]
        rec = self._make_record_with_experiments(experiments=experiments)
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.6, rec)
        assert "exp3(keep Δ=+0.100)" in result

    def test_experiment_without_score_delta(self) -> None:
        from factory.cycle_analyzer import ExperimentRecord

        experiments = [
            ExperimentRecord(
                exp_id=4, hypothesis="Test hypothesis", verdict="error",
                score_before=None, score_after=None, score_delta=None,
                cost_usd=0.0, duration_s=10.0,
            ),
        ]
        rec = self._make_record_with_experiments(experiments=experiments)
        result = OuterLoopReflector._collect_individual_details("abc12345", 0.5, rec)
        assert "exp4(error)" in result
        assert "Δ=" not in result.split("exp4")[1].split(";")[0]

    def test_llm_reflect_passes_per_individual_budget(self) -> None:
        from unittest.mock import patch, MagicMock

        reflector = OuterLoopReflector(k=1)
        rec = self._make_record_with_experiments()
        top_k = [("w1", 0.9, rec)]
        bottom_k = [("l1", 0.1, _make_record(0.1))]
        report = ReflectionReport()

        captured_budgets: list[int | None] = []
        original_fn = OuterLoopReflector._collect_individual_details

        def spy(id_: str, score: float, rec: CycleRecord | None, *, char_budget: int | None = None) -> str:
            captured_budgets.append(char_budget)
            return original_fn(id_, score, rec, char_budget=char_budget)

        fake_response = '{"prompt_improvements": [], "failure_patterns": [], "mutation_suggestions": []}'

        with (
            patch.object(OuterLoopReflector, "_collect_individual_details", staticmethod(spy)),
            patch("factory.outer_loop.reflector.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout=fake_response, returncode=0)
            reflector._llm_reflect(top_k, bottom_k, [], report)

        assert len(captured_budgets) == 2
        assert all(b is not None for b in captured_budgets)
        expected = int(8000 * 0.85) // 2
        assert all(b == expected for b in captured_budgets)


# ── Tests moved from test_coverage_gaps.py ─────────────────────────


class TestReflectorHandlesEmptyHistory:
    """Verifies reflector degrades gracefully with no prior generations."""

    def test_empty_records_returns_empty_report(self) -> None:
        reflector = OuterLoopReflector(k=2)
        report = reflector.reflect([], generation=0)
        assert report.failure_patterns == []
        assert report.success_patterns == []
        assert report.mutation_suggestions == []
        assert report.top_k_ids == []
        assert report.bottom_k_ids == []

    def test_single_record_returns_empty_report(self) -> None:
        reflector = OuterLoopReflector(k=2)
        records = [("only1", 0.5, _make_record(0.5, [_make_step("builder")], kept=1))]
        report = reflector.reflect(records, generation=0)
        assert report.failure_patterns == []
        assert report.success_patterns == []

    def test_all_none_records_returns_empty_report(self) -> None:
        reflector = OuterLoopReflector(k=2)
        records: list[tuple[str, float, CycleRecord | None]] = [
            ("a", 0.5, None),
            ("b", 0.3, None),
            ("c", 0.7, None),
        ]
        report = reflector.reflect(records, generation=0)
        assert report.failure_patterns == []
        assert report.success_patterns == []
        assert report.top_k_ids == []
        assert report.bottom_k_ids == []


# ── Bug-fix tests (BUG 1, 2, 3) ───────────────────────────────────


def test_long_verify_details_in_prompt():
    """BUG 1: ItemResult with long lists → every value appears in prompt."""
    long_terms = [f"term_{i}" for i in range(50)]
    items = [
        {
            "item_id": "x",
            "split": "train",
            "status": "ok",
            "score": 0.5,
            "passed": False,
            "verify_details": {
                "missing_terms": long_terms,
                "missing_sections": ["Overview", "Installation", "Usage", "Errors"],
                "word_count": 500,
            },
        },
    ]
    rec = CycleRecord.from_run(items, aggregate="mean")
    reflector = OuterLoopReflector()
    top_k: list[tuple[str, float, CycleRecord | None]] = [("top1", 0.8, rec)]
    bottom_k: list[tuple[str, float, CycleRecord | None]] = [("bot1", 0.3, rec)]
    prompt = reflector.build_reflection_prompt(top_k, bottom_k, top_k + bottom_k)
    # Every term must appear — no truncation inside item blocks
    for t in long_terms:
        assert t in prompt, f"{t} missing from prompt"
    assert "Overview" in prompt
    assert "Installation" in prompt


def test_over_budget_drops_whole_items():
    """BUG 1: Over budget → whole items dropped (passed first), omitted message added."""
    items = []
    for i in range(100):
        items.append(
            {
                "item_id": f"item_{i}",
                "split": "train",
                "status": "ok",
                "score": 0.9,
                "passed": True,
                "verify_details": {"data": "x" * 500},
            }
        )
    rec = CycleRecord.from_run(items, aggregate="mean")
    reflector = OuterLoopReflector()
    top_k: list[tuple[str, float, CycleRecord | None]] = [("top1", 0.8, rec)]
    bottom_k: list[tuple[str, float, CycleRecord | None]] = [("bot1", 0.3, rec)]
    prompt = reflector.build_reflection_prompt(top_k, bottom_k, top_k + bottom_k)
    assert "omitted" in prompt.lower()
    # Should not exceed 60KB total
    assert len(prompt) < 60000


def test_items_not_duplicated():
    """BUG 2: 2 items → each item_id appears exactly once per candidate section."""
    items = [
        {
            "item_id": "alpha",
            "split": "train",
            "status": "ok",
            "score": 0.8,
            "passed": True,
            "verify_details": {"x": 1},
        },
        {
            "item_id": "beta",
            "split": "train",
            "status": "failed",
            "score": 0.2,
            "passed": False,
            "verify_details": {"y": 2},
        },
    ]
    rec = CycleRecord.from_run(items, aggregate="mean")
    reflector = OuterLoopReflector()
    top_k: list[tuple[str, float, CycleRecord | None]] = [("ind1", 0.5, rec)]
    bottom_k: list[tuple[str, float, CycleRecord | None]] = [("ind2", 0.3, rec)]
    prompt = reflector.build_reflection_prompt(top_k, bottom_k, top_k + bottom_k)
    # Each item_id should appear exactly 2 times — once per candidate section
    assert prompt.count("alpha") == 2, (
        f"alpha should appear exactly 2 times (once per candidate), got {prompt.count('alpha')}"
    )
    assert prompt.count("beta") == 2


def test_prompt_via_stdin_not_argv():
    """BUG 3: Prompt passed via stdin, not as CLI arg. 200KB prompt does not raise."""
    items = [{"item_id": "a", "status": "ok", "score": 0.5, "passed": True}]
    rec = CycleRecord.from_run(items, aggregate="mean")
    reflector = OuterLoopReflector(llm_reflect=True)
    report = ReflectionReport()
    top_k: list[tuple[str, float, CycleRecord | None]] = [("t", 0.8, rec)]
    bottom_k: list[tuple[str, float, CycleRecord | None]] = [("b", 0.3, rec)]

    mock_proc = MagicMock()
    mock_proc.stdout = json.dumps(
        {
            "prompt_improvements": ["test"],
            "failure_patterns": ["f1"],
            "mutation_suggestions": [],
        }
    )
    mock_proc.returncode = 0

    with (
        patch(
            "factory.outer_loop.reflector.subprocess.run", return_value=mock_proc,
        ) as mock_run,
        patch("factory.runners.claude._claude_bin", return_value="claude"),
        patch("factory.runners.claude._claude_model", return_value="haiku"),
        patch("factory.runners.claude._cli_error_text", return_value=False),
    ):
        reflector._llm_reflect(top_k, bottom_k, list(top_k + bottom_k), report)

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    # Prompt must NOT be in the command args
    cmd_list = call_args[0][0]
    assert "-p" not in cmd_list, f"-p found in command args: {cmd_list}"
    # Prompt must be in input= kwarg
    assert call_args.kwargs.get("input") is not None, "prompt not passed via input="
