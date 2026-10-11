"""Tests for mutation operators and MutationStrategy."""

from __future__ import annotations


import random
from unittest.mock import patch

from factory.outer_loop.models import MutationType
from factory.outer_loop.mutations import (
    MutationStrategy,
    WeightedRandomStrategy,
    _generate_unique_agent_id,
    _try_mutation,
    apply_random_mutation,
    insert_node,
    mutate_knob,
    mutate_params,
    mutate_prompt,
    parallelize,
    redirect_edge,
    remove_node,
    serialize,
    validate_and_repair)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow)


class TestInsertNode:
    def test_insert_between_nodes(self, simple_workflow: Workflow) -> None:
        new_node = AgentNode(id="reviewer", role=AgentRole.CODE_REVIEWER)
        result = insert_node(simple_workflow, new_node, "strategist")
        assert result is not None
        wf, rec = result
        assert "reviewer" in wf.nodes
        assert rec.operator == MutationType.NODE_INSERT

    def test_insert_respects_frozen(self, simple_workflow: Workflow) -> None:
        new_node = AgentNode(id="new", role=AgentRole.RESEARCHER)
        result = insert_node(
            simple_workflow, new_node, "researcher", frozen_nodes={"researcher"}
        )
        assert result is None

    def test_insert_after_nonexistent(self, simple_workflow: Workflow) -> None:
        new_node = AgentNode(id="new", role=AgentRole.RESEARCHER)
        result = insert_node(simple_workflow, new_node, "nonexistent")
        assert result is None


class TestRemoveNode:
    def test_remove_middle_node(self, simple_workflow: Workflow) -> None:
        result = remove_node(simple_workflow, "strategist")
        assert result is not None
        wf, rec = result
        assert "strategist" not in wf.nodes
        assert rec.operator == MutationType.NODE_REMOVE
        has_edge = any(
            e.source == "researcher" and e.target == "builder" for e in wf.edges
        )
        assert has_edge

    def test_remove_start_node_fails(self, simple_workflow: Workflow) -> None:
        result = remove_node(simple_workflow, "study")
        assert result is None

    def test_remove_frozen_fails(self, simple_workflow: Workflow) -> None:
        result = remove_node(simple_workflow, "builder", frozen_nodes={"builder"})
        assert result is None


class TestRedirectEdge:
    def test_redirect_edge(self, simple_workflow: Workflow) -> None:
        result = redirect_edge(simple_workflow, "researcher", "strategist", "builder")
        assert result is not None
        wf, rec = result
        assert rec.operator == MutationType.EDGE_REDIRECT
        has_new = any(
            e.source == "researcher" and e.target == "builder" for e in wf.edges
        )
        assert has_new

    def test_redirect_nonexistent_target(self, simple_workflow: Workflow) -> None:
        result = redirect_edge(simple_workflow, "researcher", "strategist", "nonexistent")
        assert result is None

    def test_redirect_frozen_source(self, simple_workflow: Workflow) -> None:
        result = redirect_edge(
            simple_workflow, "researcher", "strategist", "builder",
            frozen_nodes={"researcher"})
        assert result is None


class TestParallelize:
    def test_parallelize_two_nodes(self, simple_workflow: Workflow) -> None:
        result = parallelize(simple_workflow, ["researcher", "strategist"])
        assert result is not None
        wf, rec = result
        assert rec.operator == MutationType.PARALLELIZE
        fork_nodes = [nid for nid, n in wf.nodes.items() if type(n).__name__ == "ForkNode"]
        join_nodes = [nid for nid, n in wf.nodes.items() if type(n).__name__ == "JoinNode"]
        assert len(fork_nodes) >= 1
        assert len(join_nodes) >= 1

    def test_parallelize_single_node_fails(self, simple_workflow: Workflow) -> None:
        result = parallelize(simple_workflow, ["researcher"])
        assert result is None

    def test_parallelize_frozen_fails(self, simple_workflow: Workflow) -> None:
        result = parallelize(
            simple_workflow, ["researcher", "strategist"],
            frozen_nodes={"researcher"})
        assert result is None


class TestSerialize:
    def test_serialize_reverses_parallelize(self, simple_workflow: Workflow) -> None:
        par_result = parallelize(simple_workflow, ["researcher", "strategist"])
        assert par_result is not None
        wf_par, _ = par_result

        fork_ids = [nid for nid, n in wf_par.nodes.items() if type(n).__name__ == "ForkNode"]
        assert len(fork_ids) >= 1

        ser_result = serialize(wf_par, fork_ids[0])
        assert ser_result is not None
        wf_ser, rec = ser_result
        assert rec.operator == MutationType.SERIALIZE
        assert not any(type(n).__name__ == "ForkNode" for n in wf_ser.nodes.values())

    def test_serialize_nonexistent_fails(self, simple_workflow: Workflow) -> None:
        result = serialize(simple_workflow, "nonexistent")
        assert result is None

    def test_serialize_non_fork_fails(self, simple_workflow: Workflow) -> None:
        result = serialize(simple_workflow, "researcher")
        assert result is None


class TestMutateParams:
    def test_change_timeout(self, simple_workflow: Workflow) -> None:
        result = mutate_params(simple_workflow, "researcher", {"timeout": 1200})
        assert result is not None
        wf, rec = result
        assert rec.operator == MutationType.PARAM_MUTATE
        node = wf.nodes["researcher"]
        assert hasattr(node, "timeout")
        assert node.timeout == 1200  # type: ignore[union-attr]

    def test_model_is_not_mutable(self, simple_workflow: Workflow) -> None:
        result = mutate_params(simple_workflow, "researcher", {"model": "opus"})
        assert result is None

    def test_disallowed_param_ignored(self, simple_workflow: Workflow) -> None:
        result = mutate_params(simple_workflow, "researcher", {"role": "builder"})
        assert result is None

    def test_frozen_fails(self, simple_workflow: Workflow) -> None:
        result = mutate_params(
            simple_workflow, "researcher", {"timeout": 900},
            frozen_nodes={"researcher"})
        assert result is None


class TestValidateAndRepair:
    def test_valid_workflow_passes(self, simple_workflow: Workflow) -> None:
        result = validate_and_repair(simple_workflow)
        assert result is not None

    def test_prunes_unreachable(self) -> None:
        nodes = {
            "start": FnNode(id="start", command="echo start"),
            "reachable": FnNode(id="reachable", command="echo r"),
            "orphan": FnNode(id="orphan", command="echo orphan"),
        }
        edges = [Edge(source="start", target="reachable")]
        wf = Workflow(name="test", nodes=nodes, edges=edges, start_node="start")
        result = validate_and_repair(wf)
        assert result is not None
        assert "orphan" not in result.nodes

    def test_cycle_without_gate_returns_none(self) -> None:
        nodes = {
            "a": FnNode(id="a", command="echo a"),
            "b": FnNode(id="b", command="echo b"),
        }
        edges = [
            Edge(source="a", target="b"),
            Edge(source="b", target="a"),
        ]
        wf = Workflow(name="test", nodes=nodes, edges=edges, start_node="a")
        result = validate_and_repair(wf)
        assert result is None


class TestWeightedRandomStrategy:
    def test_implements_protocol(self) -> None:
        strategy = WeightedRandomStrategy()
        assert isinstance(strategy, MutationStrategy)

    def test_select_operator_returns_valid(self, simple_workflow: Workflow) -> None:
        strategy = WeightedRandomStrategy()
        op = strategy.select_operator(simple_workflow, 0, {})
        assert isinstance(op, MutationType)

    def test_mutation_rate(self) -> None:
        strategy = WeightedRandomStrategy(mutation_rate=0.5)
        assert strategy.get_mutation_rate(0) == 0.5
        assert strategy.get_mutation_rate(10) == 0.5

    def test_designer_ratio(self) -> None:
        strategy = WeightedRandomStrategy(designer_ratio=0.4)
        assert strategy.get_designer_ratio(0) == 0.4

    def test_operator_weights(self) -> None:
        weights = {t.value: (1.0 if t == MutationType.NODE_INSERT else 0.0) for t in MutationType}
        strategy = WeightedRandomStrategy(weights=weights)
        ops = [strategy.select_operator(Workflow(
            name="dummy",
            nodes={"a": FnNode(id="a", command="x")},
            edges=[],
            start_node="a"), 0, {}) for _ in range(20)]
        assert all(op == MutationType.NODE_INSERT for op in ops)


class TestApplyRandomMutation:
    def test_produces_valid_result(self, simple_workflow: Workflow) -> None:
        strategy = WeightedRandomStrategy()
        result = apply_random_mutation(
            simple_workflow, strategy, generation=0, max_attempts=20)
        if result is not None:
            wf, rec = result
            assert isinstance(rec.operator, MutationType)
            assert wf.start_node in wf.nodes

    def test_with_frozen_nodes(self, simple_workflow: Workflow) -> None:
        strategy = WeightedRandomStrategy()
        all_nodes = set(simple_workflow.nodes.keys())
        result = apply_random_mutation(
            simple_workflow, strategy, generation=0,
            frozen_nodes=all_nodes,
            max_attempts=5)
        assert result is None


class TestKnobMutate:
    def test_mutates_knob_within_bounds(self, simple_workflow: Workflow) -> None:
        wf = simple_workflow.model_copy(update={
            "knob_values": {"style": "broad"},
            "knob_bounds": {"style": ["broad", "focused", "deep"]},
        })
        result = mutate_knob(wf, expander=None)
        assert result is not None
        child_wf, rec = result
        assert rec.operator == MutationType.KNOB_MUTATE
        assert child_wf.knob_values["style"] != "broad"
        assert child_wf.knob_values["style"] in ["focused", "deep"]

    def test_returns_none_without_knob_values(self, simple_workflow: Workflow) -> None:
        result = mutate_knob(simple_workflow, expander=None)
        assert result is None

    def test_calls_expander_when_bounds_exhausted(self, simple_workflow: Workflow) -> None:
        wf = simple_workflow.model_copy(update={
            "knob_values": {"style": "only_option"},
            "knob_bounds": {"style": ["only_option"]},
            "knob_expandable": {"style": "invent a new style"},
        })
        expanded_value = None
        def fake_expander(name, hint, current, bounds):
            nonlocal expanded_value
            expanded_value = "invented_style"
            return "invented_style"
        result = mutate_knob(wf, expander=fake_expander)
        assert result is not None
        child_wf, rec = result
        assert child_wf.knob_values["style"] == "invented_style"
        assert "invented_style" in child_wf.knob_bounds["style"]

    def test_expander_not_called_when_not_expandable(self, simple_workflow: Workflow) -> None:
        wf = simple_workflow.model_copy(update={
            "knob_values": {"style": "only_option"},
            "knob_bounds": {"style": ["only_option"]},
        })
        called = False
        def spy_expander(name, hint, current, bounds):
            nonlocal called
            called = True
            return "should_not_appear"
        mutate_knob(wf, expander=spy_expander)
        assert not called


class TestParseKnobSuggestionEmptyValue:
    def test_empty_string_value_returns_none(self) -> None:
        from factory.outer_loop.mutations import _parse_knob_suggestion
        from factory.outer_loop.reflector import MutationSuggestion

        suggestion = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value="")
        assert _parse_knob_suggestion(suggestion) is None

    def test_whitespace_only_value_returns_none(self) -> None:
        from factory.outer_loop.mutations import _parse_knob_suggestion
        from factory.outer_loop.reflector import MutationSuggestion

        suggestion = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value="   ")
        assert _parse_knob_suggestion(suggestion) is None

    def test_valid_value_returns_tuple(self) -> None:
        from factory.outer_loop.mutations import _parse_knob_suggestion
        from factory.outer_loop.reflector import MutationSuggestion

        suggestion = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value="focused")
        result = _parse_knob_suggestion(suggestion)
        assert result == ("style", "focused")

    def test_none_value_returns_none(self) -> None:
        from factory.outer_loop.mutations import _parse_knob_suggestion
        from factory.outer_loop.reflector import MutationSuggestion

        suggestion = MutationSuggestion(
            operator="knob_mutate",
            target="style",
            rationale="test",
            value=None)
        assert _parse_knob_suggestion(suggestion) is None


class TestApplyRandomMutationWithReflection:
    def test_guided_operator_selection_with_reflection(self, simple_workflow: Workflow) -> None:
        from unittest.mock import patch

        from factory.outer_loop.reflector import ReflectionReport

        report = ReflectionReport(
            mutation_suggestions=["NODE_INSERT: Add researcher agent"],
            structural_recommendations=[])
        strategy = WeightedRandomStrategy()

        with patch("factory.outer_loop.mutations.random") as mock_random:
            mock_random.random.return_value = 0.3
            mock_random.choice = __import__("random").choice
            mock_random.choices = __import__("random").choices
            mock_random.sample = __import__("random").sample
            mock_random.randint = __import__("random").randint

            apply_random_mutation(
                simple_workflow, strategy, generation=1,
                reflection_report=report,
                max_attempts=20)
            mock_random.random.assert_called()

    def test_no_reflection_uses_random_operator(self, simple_workflow: Workflow) -> None:
        strategy = WeightedRandomStrategy()
        result = apply_random_mutation(
            simple_workflow, strategy, generation=1,
            reflection_report=None,
            max_attempts=20)
        if result is not None:
            _, rec = result
            assert isinstance(rec.operator, MutationType)


class TestPromptVariantsDomainNeutral:
    def test_no_coding_specific_keywords(self) -> None:
        from factory.outer_loop.mutations import _PROMPT_VARIANTS

        coding_phrases = ["codebase", "failing tests", "tests first"]
        for variant in _PROMPT_VARIANTS:
            lower = variant.lower()
            for phrase in coding_phrases:
                assert phrase not in lower, (
                    f"_PROMPT_VARIANTS contains coding-specific phrase '{phrase}' in: {variant}"
                )


class TestSelectGuidedOperatorStringFallback:
    """Cover the string fallback path in select_guided_operator (typed_suggestions empty)."""

    def test_string_fallback_node_insert(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=["NODE_INSERT: Add researcher agent"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.NODE_INSERT in counts

    def test_string_fallback_node_remove(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=["NODE_REMOVE: Consider removing strategist"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.NODE_REMOVE in counts

    def test_string_fallback_parallelize(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=[],
            structural_recommendations=["PARALLELIZE: Use parallel execution for agents"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.PARALLELIZE in counts

    def test_string_fallback_param_mutate(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=["PARAM_MUTATE: Increase timeout for builder"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.PARAM_MUTATE in counts

    def test_string_fallback_prompt_mutate(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=["PROMPT_MUTATE: Improve builder prompt"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.PROMPT_MUTATE in counts

    def test_string_fallback_knob(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=["KNOB_MUTATE: style=focused outperforms broad"],
            typed_suggestions=[])
        counts: dict[MutationType, int] = {}
        for _ in range(100):
            op = strategy.select_guided_operator(wf, 0, report)
            counts[op] = counts.get(op, 0) + 1
        assert MutationType.KNOB_MUTATE in counts

    def test_empty_suggestions_falls_back_to_random(self) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        strategy = WeightedRandomStrategy()
        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b")
        report = ReflectionReport(
            mutation_suggestions=[],
            structural_recommendations=[],
            typed_suggestions=[])
        op = strategy.select_guided_operator(wf, 0, report)
        assert isinstance(op, MutationType)


class TestMutateKnobTypedPreference:
    """Cover typed_knobs preference path and string fallback in mutate_knob."""

    def test_typed_knobs_preferred_over_strings(self) -> None:
        import random

        from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b",
            knob_values={"style": "broad", "depth": 3},
            knob_bounds={"style": ["broad", "focused", "creative"], "depth": [1, 2, 3, 5]})
        report = ReflectionReport(
            mutation_suggestions=["KNOB_MUTATE: depth=5 outperforms others"],
            typed_suggestions=[
                MutationSuggestion(
                    operator="knob_mutate",
                    target="style",
                    rationale="focused works best",
                    value="focused"),
            ])
        random.seed(0)
        result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None

    def test_string_fallback_when_no_typed_knobs(self) -> None:
        import random

        from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b",
            knob_values={"style": "broad"},
            knob_bounds={"style": ["broad", "focused", "creative"]})
        report = ReflectionReport(
            mutation_suggestions=["KNOB_MUTATE: style=creative (avg score +5) outperforms ..."],
            typed_suggestions=[
                MutationSuggestion(
                    operator="node_insert",
                    target="researcher",
                    rationale="not a knob suggestion"),
            ])
        random.seed(1)
        result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None


class TestMutateKnobGuidedNoOp:
    """Cover guided_val == current_val no-op branch and float coercion in mutate_knob."""

    def test_guided_value_same_as_current_falls_through(self) -> None:
        """When guided value equals current, guided_knob is reset and random path takes over."""
        from unittest.mock import patch

        from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b",
            knob_values={"style": "focused"},
            knob_bounds={"style": ["focused", "broad", "creative"]})
        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(
                    operator="knob_mutate",
                    target="style",
                    rationale="focused works best",
                    value="focused"),
            ])
        with patch("factory.outer_loop.mutations.random") as mock_rng:
            mock_rng.random.return_value = 0.3
            mock_rng.choice.side_effect = lambda seq: seq[0]
            result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None

    def test_float_coercion_in_guided_knob(self) -> None:
        """When the current value is float and guided_val is string, coerce to float."""
        from unittest.mock import patch

        from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

        wf = Workflow(
            name="test",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER)},
            edges=[], start_node="b",
            knob_values={"temperature": 0.7},
            knob_bounds={"temperature": [0.5, 0.7, 0.9]})
        report = ReflectionReport(
            typed_suggestions=[
                MutationSuggestion(
                    operator="knob_mutate",
                    target="temperature",
                    rationale="higher works better",
                    value="0.9"),
            ])
        with patch("factory.outer_loop.mutations.random") as mock_rng:
            mock_rng.random.return_value = 0.3
            mock_rng.choice.side_effect = lambda seq: seq[0]
            result = mutate_knob(wf, expander=None, reflection_report=report)
        assert result is not None
        mutated_wf, _ = result
        assert mutated_wf.knob_values["temperature"] == 0.9


class TestApplyRandomMutationTypedSuggestions:
    """Cover the typed_suggestions check in apply_random_mutation's use_guided condition."""

    def test_use_guided_with_only_typed_suggestions(self, simple_workflow: Workflow) -> None:
        from unittest.mock import patch

        from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

        report = ReflectionReport(
            mutation_suggestions=[],
            structural_recommendations=[],
            typed_suggestions=[
                MutationSuggestion(operator="node_insert", target="researcher", rationale="test"),
            ])
        strategy = WeightedRandomStrategy()

        with patch("factory.outer_loop.mutations.random") as mock_random:
            mock_random.random.return_value = 0.3
            mock_random.choice = __import__("random").choice
            mock_random.choices = __import__("random").choices
            mock_random.sample = __import__("random").sample
            mock_random.randint = __import__("random").randint

            apply_random_mutation(
                simple_workflow, strategy, generation=1,
                reflection_report=report,
                max_attempts=20)
            mock_random.random.assert_called()

    def test_no_typed_no_string_suggestions_uses_random(self, simple_workflow: Workflow) -> None:
        from factory.outer_loop.reflector import ReflectionReport

        report = ReflectionReport(
            mutation_suggestions=[],
            structural_recommendations=[],
            typed_suggestions=[])
        strategy = WeightedRandomStrategy()

        result = apply_random_mutation(
            simple_workflow, strategy, generation=1,
            reflection_report=report,
            max_attempts=20)
        if result is not None:
            _, rec = result
            assert isinstance(rec.operator, MutationType)


class TestKnobPreservation:
    def test_insert_node_preserves_knobs(self, simple_workflow: Workflow) -> None:
        wf = simple_workflow.model_copy(update={
            "knob_values": {"style": "broad", "depth": 3.0},
            "knob_bounds": {"style": ["broad", "deep"], "depth": [1.0, 5.0]},
            "knob_expandable": {"style": "prompt hint"},
        })
        new_node = AgentNode(id="new_agent", role=AgentRole.RESEARCHER)
        result = insert_node(wf, new_node, "strategist")
        assert result is not None
        child_wf, _ = result
        assert child_wf.knob_values == {"style": "broad", "depth": 3.0}
        assert child_wf.knob_bounds == {"style": ["broad", "deep"], "depth": [1.0, 5.0]}
        assert child_wf.knob_expandable == {"style": "prompt hint"}

    def test_remove_node_preserves_knobs(self, simple_workflow: Workflow) -> None:
        wf = simple_workflow.model_copy(update={
            "knob_values": {"mode": "parallel"},
            "knob_bounds": {"mode": ["parallel", "serial"]},
        })
        result = remove_node(wf, "strategist")
        assert result is not None
        child_wf, _ = result
        assert child_wf.knob_values == {"mode": "parallel"}


def _chess_evolve_workflow() -> Workflow:
    """Chess-evolve-like fixture: 1 AgentNode (start) + GateNode + FnNode, gated reloop."""
    nodes: dict[str, AgentNode | GateNode | FnNode] = {
        "solver": AgentNode(
            id="solver",
            role=AgentRole.BUILDER,
            reads={"problem.md"},
            writes={"solution.py"},
            prompt_template="Solve the problem."),
        "gate": GateNode(
            id="gate",
            evaluator_type="fn",
            reads={"solution.py"}),
        "record": FnNode(
            id="record",
            command="echo done",
            reads={"solution.py"}),
    }
    edges = [
        Edge(source="solver", target="gate"),
        Edge(source="gate", target="record"),
        Edge(source="gate", target="solver", condition=VerdictType.RELOOP),
    ]
    return Workflow(
        name="chess_evolve_like",
        nodes=nodes,
        edges=edges,
        start_node="solver")


class TestPromptMutateStartNode:
    def test_prompt_mutate_succeeds_on_start_only_agent(self) -> None:
        wf = _chess_evolve_workflow()
        result = _try_mutation(wf, MutationType.PROMPT_MUTATE, set(), prompt_hint="be concise")
        assert result is not None
        child_wf, rec = result
        assert rec.operator == MutationType.PROMPT_MUTATE
        assert rec.target_node == "solver"

    def test_prompt_mutate_with_rewriter_none(self) -> None:
        wf = _chess_evolve_workflow()
        result = mutate_prompt(wf, "solver", frozen_nodes=set(), rewriter=None, prompt_hint="be fast")
        assert result is not None
        child_wf, rec = result
        node = child_wf.nodes["solver"]
        assert isinstance(node, AgentNode)
        assert "be fast" in node.prompt_template


class TestParamMutateStartNode:
    def test_param_mutate_succeeds_on_start_only_agent(self) -> None:
        wf = _chess_evolve_workflow()
        result = _try_mutation(wf, MutationType.PARAM_MUTATE, set())
        assert result is not None
        _, rec = result
        assert rec.operator == MutationType.PARAM_MUTATE
        assert rec.target_node == "solver"


class TestNodeInsertContextual:
    def test_insert_creates_complementary_role(self) -> None:
        wf = _chess_evolve_workflow()
        random.seed(42)
        result = _try_mutation(wf, MutationType.NODE_INSERT, set())
        assert result is not None
        child_wf, rec = result
        assert rec.operator == MutationType.NODE_INSERT
        new_id = rec.target_node
        new_node = child_wf.nodes[new_id]
        assert isinstance(new_node, AgentNode)
        assert new_node.role != AgentRole.CEO

    def test_insert_inherits_file_wiring(self) -> None:
        wf = _chess_evolve_workflow()
        # Run many times and check that inserted nodes have non-empty file wiring
        # when inserted after an AgentNode with writes
        random.seed(0)
        found_reads = False
        for _ in range(30):
            result = _try_mutation(wf, MutationType.NODE_INSERT, set())
            if result is None:
                continue
            child_wf, rec = result
            new_node = child_wf.nodes.get(rec.target_node)
            if isinstance(new_node, AgentNode) and new_node.reads:
                found_reads = True
                # solver.writes = {"solution.py"}, so new reads should come from there
                assert "solution.py" in new_node.reads
                break
        assert found_reads, "Expected at least one insertion to inherit file wiring"

    def test_insert_uses_role_prompt_template(self) -> None:
        wf = _chess_evolve_workflow()
        result = _try_mutation(wf, MutationType.NODE_INSERT, set())
        assert result is not None
        child_wf, rec = result
        new_node = child_wf.nodes[rec.target_node]
        assert isinstance(new_node, AgentNode)
        assert new_node.prompt_template != ""

    def test_insert_fallback_when_no_agent_nodes(self) -> None:
        nodes: dict[str, FnNode] = {
            "start": FnNode(id="start", command="echo start"),
            "end": FnNode(id="end", command="echo end"),
        }
        edges = [Edge(source="start", target="end")]
        wf = Workflow(name="no_agents", nodes=nodes, edges=edges, start_node="start")
        result = _try_mutation(wf, MutationType.NODE_INSERT, set())
        assert result is not None
        child_wf, rec = result
        new_node = child_wf.nodes[rec.target_node]
        assert isinstance(new_node, AgentNode)


class TestNodeRemoveLastAgentGuard:
    def test_remove_last_agent_returns_none(self) -> None:
        wf = _chess_evolve_workflow()
        # "solver" is the only AgentNode but it's also the start_node
        # so it can't be in structurally_mutable. Add it manually to test the guard.
        result = _try_mutation(wf, MutationType.NODE_REMOVE, set())
        # With chess-evolve, structurally_mutable = ["gate", "record"] (not AgentNodes),
        # so NODE_REMOVE can target them but not solver. Let's test with a workflow
        # where the last agent IS in the structurally_mutable list.
        nodes: dict[str, AgentNode | FnNode] = {
            "start": FnNode(id="start", command="echo start"),
            "agent": AgentNode(id="agent", role=AgentRole.BUILDER),
        }
        edges = [Edge(source="start", target="agent")]
        wf2 = Workflow(name="one_agent", nodes=nodes, edges=edges, start_node="start")
        result = _try_mutation(wf2, MutationType.NODE_REMOVE, set())
        assert result is None

    def test_remove_non_last_agent_succeeds(self, simple_workflow: Workflow) -> None:
        # simple_workflow has 3 AgentNodes (researcher, strategist, builder)
        # removing one should succeed
        result = remove_node(simple_workflow, "strategist")
        assert result is not None


class TestEdgeRedirectCycleFilter:
    def test_filters_ungated_cycle_targets(self) -> None:
        wf = _chess_evolve_workflow()
        # The only unconditional edges: solver→gate, gate→record
        # If we pick edge solver→gate, ancestors of solver in unconditional graph = {}
        # possible_targets = [record] (gate excluded as current target, solver excluded as source)
        # So redirect should succeed with target=record
        result = _try_mutation(wf, MutationType.EDGE_REDIRECT, set())
        if result is not None:
            child_wf, rec = result
            assert rec.operator == MutationType.EDGE_REDIRECT
            validated = validate_and_repair(child_wf)
            assert validated is not None

    def test_returns_none_when_all_targets_create_cycles(self) -> None:
        # Two nodes, one unconditional edge A→B. Redirecting A→B to A→A is self-loop,
        # and there are no other targets.
        nodes: dict[str, FnNode] = {
            "a": FnNode(id="a", command="echo a"),
            "b": FnNode(id="b", command="echo b"),
        }
        edges = [Edge(source="a", target="b")]
        wf = Workflow(name="tiny", nodes=nodes, edges=edges, start_node="a")
        # edge a→b: possible_targets excludes b (current target) and a (source) → empty
        result = _try_mutation(wf, MutationType.EDGE_REDIRECT, set())
        assert result is None


class TestUniqueIdGeneration:
    def test_generates_unique_id(self) -> None:
        existing = {f"builder_{i}" for i in range(100, 200)}
        new_id = _generate_unique_agent_id(existing, AgentRole.BUILDER)
        assert new_id not in existing
        assert new_id.startswith("builder_")

    def test_avoids_collision(self) -> None:
        existing = {"builder_100"}
        # Patch randint to return 100 first (collision), then 200 (unique)
        with patch("factory.outer_loop.mutations.random.randint", side_effect=[100, 200]):
            new_id = _generate_unique_agent_id(existing, AgentRole.BUILDER)
        assert new_id == "builder_200"


class TestMutateParamsValidation:
    def test_invalid_field_value_returns_none(self, simple_workflow: Workflow) -> None:
        # AgentNode has strict=True, extra="forbid". Passing a bad type for timeout
        # should fail validation with constructor-based creation.
        result = mutate_params(
            simple_workflow, "researcher", {"timeout": "not_a_number"}
        )
        assert result is None


class TestNodeInsertDataSubgraphAwareness:
    """Tests for NODE_INSERT data subgraph awareness in mutations."""

    @staticmethod
    def _workflow_with_data_subgraph() -> Workflow:
        """Workflow with a DataNode whose subgraph is researcher → builder.

        Includes an explicit edge from DataNode to subgraph entry for
        validate_and_repair reachability (mirrors real Designer output).
        """
        from factory.workflow.primitives import DataItem, DataNode, JoinNode

        return Workflow(
            name="data_wf",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="item1", prompt="test")]),
                "researcher": AgentNode(
                    id="researcher",
                    role=AgentRole.RESEARCHER,
                    reads={".factory/current_item.json"},
                    writes={".factory/strategy/research.md"},
                    prompt_template="Research the data item."),
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    reads={".factory/strategy/research.md"},
                    writes={".factory/reviews/builder-latest.md"},
                    prompt_template="Build the solution."),
                "join": JoinNode(id="join", sources=["builder"]),
            },
            edges=[
                Edge(source="data", target="researcher"),
                Edge(source="researcher", target="builder"),
                Edge(source="builder", target="join"),
            ],
            start_node="data")

    def test_node_insert_in_data_subgraph_reads_current_item(self) -> None:
        """Insert a node into a DataNode subgraph → new node has
        .factory/current_item.json in reads and data-aware prompt."""
        wf = self._workflow_with_data_subgraph()

        # Directly insert after 'researcher' (inside subgraph, not exit node)
        # to verify data-awareness is applied
        from factory.outer_loop.mutations import _is_in_data_subgraph

        assert _is_in_data_subgraph(wf, "researcher")

        random.seed(42)
        found_data_aware = False
        for _ in range(50):
            result = _try_mutation(wf, MutationType.NODE_INSERT, {"data"})
            if result is None:
                continue
            child_wf, rec = result
            new_node = child_wf.nodes.get(rec.target_node)
            if isinstance(new_node, AgentNode):
                inserted_after = rec.after.get("inserted_after", "")
                # When inserted after a node inside the subgraph,
                # the new node should be data-aware
                if inserted_after == "researcher":
                    assert ".factory/current_item.json" in new_node.reads
                    prompt = new_node.prompt_template or ""
                    assert "current_item.json" in prompt
                    assert "data item" not in prompt
                    # Role framing: prompt starts with 'As a <role>,'
                    role_name = new_node.role.value.replace("_", " ")
                    assert f"As a {role_name}," in prompt
                    found_data_aware = True
                    break
        assert found_data_aware, "Expected at least one data-aware insertion in subgraph"

    def test_node_insert_outside_subgraph_no_current_item(self) -> None:
        """Insert a node NOT in a DataNode subgraph → no current_item.json in reads."""
        from factory.workflow.primitives import DataItem, DataNode, JoinNode

        wf = Workflow(
            name="data_wf_ext",
            nodes={
                "external": FnNode(
                    id="external",
                    command="echo start",
                    writes={".factory/strategy/observations.md"}),
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="item1", prompt="test")]),
                "sub_builder": AgentNode(
                    id="sub_builder",
                    role=AgentRole.BUILDER,
                    reads={".factory/current_item.json"},
                    writes={".factory/reviews/builder-latest.md"},
                    prompt_template="Build."),
                "join": JoinNode(id="join", sources=["sub_builder"]),
                "post": AgentNode(
                    id="post",
                    role=AgentRole.RESEARCHER,
                    reads={".factory/strategy/observations.md"},
                    writes={".factory/strategy/research.md"},
                    prompt_template="Research after data processing."),
            },
            edges=[
                Edge(source="external", target="data"),
                Edge(source="data", target="sub_builder"),
                Edge(source="sub_builder", target="join"),
                Edge(source="join", target="post"),
            ],
            start_node="external")
        random.seed(0)
        found_non_data = False
        for _ in range(50):
            result = _try_mutation(wf, MutationType.NODE_INSERT, {"data"})
            if result is None:
                continue
            child_wf, rec = result
            new_node = child_wf.nodes.get(rec.target_node)
            if isinstance(new_node, AgentNode):
                inserted_after = rec.after.get("inserted_after", "")
                if inserted_after not in ("sub_builder"):
                    # Inserted outside the subgraph
                    assert ".factory/current_item.json" not in (new_node.reads or set())
                    found_non_data = True
                    break
        assert found_non_data, "Expected at least one insertion outside the subgraph"


class TestAutoFrozenNodes:
    """Tests for _auto_frozen_nodes and DataNode auto-freeze in engine."""

    def test_auto_frozen_nodes_returns_data_node_ids(self) -> None:
        from factory.outer_loop.engine import _auto_frozen_nodes
        from factory.workflow.primitives import DataNode, DataItem, JoinNode

        nodes: dict = {
            "data_loader": DataNode(
                id="data_loader",
                inline_items=[DataItem(id="item1", prompt="test")]),
            "builder": AgentNode(id="builder", role=AgentRole.BUILDER),
            "join": JoinNode(id="join", sources=["builder"]),
            "study": FnNode(id="study", command="echo hi"),
        }
        edges = [
            Edge(source="study", target="data_loader"),
            Edge(source="data_loader", target="builder"),
            Edge(source="builder", target="join"),
        ]
        wf = Workflow(name="with_data", nodes=nodes, edges=edges, start_node="study")
        frozen = _auto_frozen_nodes(wf)
        # DataNode + JoinNode are frozen; branch nodes are the evolution surface
        assert frozen == {"data_loader", "join"}

    def test_auto_frozen_nodes_empty_when_no_data_nodes(self) -> None:
        from factory.outer_loop.engine import _auto_frozen_nodes

        nodes: dict[str, AgentNode | FnNode] = {
            "builder": AgentNode(id="builder", role=AgentRole.BUILDER),
            "study": FnNode(id="study", command="echo hi"),
        }
        edges = [Edge(source="study", target="builder")]
        wf = Workflow(name="no_data", nodes=nodes, edges=edges, start_node="study")
        frozen = _auto_frozen_nodes(wf)
        assert frozen == set()

    def test_data_node_protected_from_direct_removal(self) -> None:
        """Frozen DataNode cannot be directly removed or param-mutated."""
        from factory.outer_loop.engine import _auto_frozen_nodes
        from factory.workflow.primitives import DataNode, DataItem, JoinNode

        nodes: dict = {
            "study": FnNode(
                id="study",
                command="factory study",
                writes={".factory/strategy/observations.md"}),
            "data_loader": DataNode(
                id="data_loader",
                inline_items=[DataItem(id="item1", prompt="test")],
                reads={".factory/strategy/observations.md"}),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                reads={".factory/strategy/observations.md"},
                writes={".factory/reviews/builder-latest.md"}),
            "join": JoinNode(id="join", sources=["builder"]),
        }
        edges = [
            Edge(source="study", target="data_loader"),
            Edge(source="data_loader", target="builder"),
            Edge(source="builder", target="join"),
        ]
        wf = Workflow(name="data_test", nodes=nodes, edges=edges, start_node="study")

        frozen = _auto_frozen_nodes(wf)
        assert "data_loader" in frozen

        assert remove_node(wf, "data_loader", frozen_nodes=frozen) is None
        assert mutate_params(
            wf, "data_loader", {"timeout": 999}, frozen_nodes=frozen) is None

    def test_auto_frozen_nodes_includes_only_data_and_join(self) -> None:
        """DataNode + JoinNode are frozen; branch nodes are the evolution surface."""
        from factory.outer_loop.engine import _auto_frozen_nodes
        from factory.workflow.primitives import DataNode, DataItem, JoinNode

        nodes: dict = {
            "data_loader": DataNode(
                id="data_loader",
                inline_items=[DataItem(id="item1", prompt="test")]),
            "sub_entry": AgentNode(id="sub_entry", role=AgentRole.BUILDER),
            "sub_mid": FnNode(id="sub_mid", command="echo mid"),
            "sub_exit": AgentNode(id="sub_exit", role=AgentRole.CODE_REVIEWER),
            "join": JoinNode(id="join", sources=["sub_exit"]),
            "external": FnNode(id="external", command="echo external"),
        }
        edges = [
            Edge(source="external", target="data_loader"),
            Edge(source="data_loader", target="sub_entry"),
            Edge(source="sub_entry", target="sub_mid"),
            Edge(source="sub_mid", target="sub_exit"),
            Edge(source="sub_exit", target="join"),
        ]
        wf = Workflow(
            name="multi_subgraph", nodes=nodes, edges=edges, start_node="external")
        frozen = _auto_frozen_nodes(wf)
        # DataNode + JoinNode are frozen
        assert frozen == {"data_loader", "join"}
        # Subgraph nodes are NOT frozen — they are the evolution surface
        assert "sub_entry" not in frozen
        assert "sub_mid" not in frozen
        assert "sub_exit" not in frozen
        assert "external" not in frozen

    def test_subgraph_node_is_mutable(self) -> None:
        """Subgraph nodes are the evolution surface — NOT auto-frozen."""
        from factory.outer_loop.engine import _auto_frozen_nodes
        from factory.workflow.primitives import DataNode, DataItem, JoinNode

        nodes: dict = {
            "start": FnNode(id="start", command="echo start"),
            "data_loader": DataNode(
                id="data_loader",
                inline_items=[DataItem(id="item1", prompt="test")]),
            "sub_builder": AgentNode(id="sub_builder", role=AgentRole.BUILDER),
            "join": JoinNode(id="join", sources=["sub_builder"]),
        }
        edges = [
            Edge(source="start", target="data_loader"),
            Edge(source="data_loader", target="sub_builder"),
            Edge(source="sub_builder", target="join"),
        ]
        wf = Workflow(
            name="protected_subgraph", nodes=nodes, edges=edges, start_node="start")

        # DataNode + JoinNode are frozen; sub_builder is mutable
        frozen = _auto_frozen_nodes(wf)
        assert "data_loader" in frozen
        assert "join" in frozen
        assert "sub_builder" not in frozen

        # DataNode itself is protected from removal
        assert remove_node(wf, "data_loader", frozen_nodes=frozen) is None
        # Subgraph node is not frozen but removal may be rejected
        # by validate_and_repair if it would break the fork structure
        assert "sub_builder" not in frozen  # confirms it's mutable
