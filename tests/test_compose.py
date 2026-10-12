"""Tests for factory/compose.py — composition validation, capability inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.compose import (
    Capability,
    IncompatibleCompositionError,
    ModeCapabilities,
    TaskCapabilities,
    TaskProtocol,
    check_mode_task_compat,
    compose,
    validate_composition,
)
from factory.task import (
    CAPABILITY_ALIASES,
    ScoringContract,
    Task,
    TaskConstraints,
    TaskDefinition,
)


_CHESS_EVOLVE_TOML = Path(__file__).resolve().parent.parent / "benchmarks" / "configs" / "chess-evolve.toml"


# ── Capability StrEnum tests ────────────────────────────────────


class TestCapability:
    def test_members(self):
        assert Capability.CAN_MODIFY_CODE == "can_modify_code"
        assert Capability.HAS_BUILDER == "has_builder"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Capability("nonexistent_capability")


# ── ModeCapabilities tests ──────────────────────────────────────


def _make_workflow(**node_kwargs):
    """Build a minimal Workflow with specified nodes."""
    from factory.workflow.primitives import (
        AgentNode,
        AgentRole,
        Edge,
        FnNode,
        ForkNode,
        GateNode,
        Workflow,
    )

    nodes: dict[str, AgentNode | FnNode | GateNode | ForkNode] = {}

    if node_kwargs.get("builder"):
        nodes["builder"] = AgentNode(
            id="builder", role=AgentRole.BUILDER, prompt_template="build"
        )
    if node_kwargs.get("researcher"):
        nodes["researcher"] = AgentNode(
            id="researcher", role=AgentRole.RESEARCHER, prompt_template="research"
        )
    if node_kwargs.get("strategist"):
        nodes["strategist"] = AgentNode(
            id="strategist", role=AgentRole.STRATEGIST, prompt_template="strategize"
        )
    if node_kwargs.get("code_reviewer"):
        nodes["code_reviewer"] = AgentNode(
            id="code_reviewer", role=AgentRole.CODE_REVIEWER, prompt_template="review"
        )
    if node_kwargs.get("gate"):
        nodes["gate"] = GateNode(id="gate", gate_prompt="check quality")
    if node_kwargs.get("fork"):
        targets = list(nodes.keys())[:1] or ["builder"]
        nodes["fork"] = ForkNode(id="fork", targets=targets)
    if node_kwargs.get("fn"):
        nodes["fn"] = FnNode(id="fn", command="echo test")

    # Build edges (simple chain)
    edges = []
    node_ids = list(nodes.keys())
    for i in range(len(node_ids) - 1):
        edges.append(Edge(source=node_ids[i], target=node_ids[i + 1]))

    return Workflow(
        name=node_kwargs.get("name", "test-workflow"),
        nodes=nodes,
        edges=edges,
        start_node=node_ids[0] if node_ids else "start",
    )


class TestModeCapabilities:
    def test_builder_provides_code_and_tests(self):
        wf = _make_workflow(builder=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.CAN_MODIFY_CODE in caps.provides
        assert Capability.HAS_BUILDER in caps.provides
        assert Capability.CAN_RUN_TESTS in caps.provides

    def test_researcher_provides_researcher(self):
        wf = _make_workflow(researcher=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.HAS_RESEARCHER in caps.provides

    def test_gate_provides_quality_gate(self):
        wf = _make_workflow(builder=True, gate=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.HAS_QUALITY_GATE in caps.provides

    def test_fork_provides_parallelism(self):
        wf = _make_workflow(builder=True, fork=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.HAS_PARALLELISM in caps.provides

    def test_fn_provides_subprocess(self):
        wf = _make_workflow(fn=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.CAN_RUN_SUBPROCESS in caps.provides

    def test_empty_workflow(self):
        from factory.workflow.primitives import Workflow

        wf = Workflow(name="empty", nodes={}, edges=[], start_node="start")
        caps = ModeCapabilities.from_workflow(wf)
        assert len(caps.provides) == 0


# ── TaskCapabilities tests ──────────────────────────────────────


class TestTaskCapabilities:
    def test_exit_code_requires_builder(self):
        """Task with required_capabilities=None (default) and exit_code scoring infers builder caps."""
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        assert task.constraints.required_capabilities is None
        caps = TaskCapabilities.from_task(task)
        assert Capability.CAN_MODIFY_CODE in caps.requires
        assert Capability.CAN_RUN_TESTS in caps.requires
        assert Capability.HAS_BUILDER in caps.requires

    def test_json_scoring_minimal_requirements(self):
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="json")))
        caps = TaskCapabilities.from_task(task)
        assert Capability.CAN_MODIFY_CODE not in caps.requires

    def test_explicit_empty_capabilities_skips_inference(self):
        """Task with required_capabilities=[] should require nothing, even with exit_code scoring."""
        task = Task(
            definition=TaskDefinition(
                name="domain-task",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(required_capabilities=[]),
            )
        )
        caps = TaskCapabilities.from_task(task)
        assert len(caps.requires) == 0

    def test_none_capabilities_uses_inference(self):
        """Task with required_capabilities=None and exit_code scoring should infer software-dev caps."""
        task = Task(
            definition=TaskDefinition(
                name="sw-task",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(required_capabilities=None),
            )
        )
        caps = TaskCapabilities.from_task(task)
        assert Capability.CAN_MODIFY_CODE in caps.requires
        assert Capability.CAN_RUN_TESTS in caps.requires
        assert Capability.HAS_BUILDER in caps.requires

    def test_explicit_empty_caps_passes_any_workflow(self):
        """A task with required_capabilities=[] should pass validation with ANY workflow."""
        wf = _make_workflow(researcher=True, name="research-only")
        task = Task(
            definition=TaskDefinition(
                name="domain-task",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(required_capabilities=[]),
            )
        )
        validate_composition(wf, task)


# ── validate_composition tests ───────────────────────────────────


class TestValidateComposition:
    def test_compatible(self):
        """Builder workflow + exit_code task = compatible."""
        wf = _make_workflow(builder=True)
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        validate_composition(wf, task)

    def test_incompatible_research_only(self):
        """Research-only workflow + exit_code task = incompatible."""
        wf = _make_workflow(researcher=True, name="research")
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        with pytest.raises(IncompatibleCompositionError) as exc_info:
            validate_composition(wf, task)
        assert "can_modify_code" in str(exc_info.value).lower() or "has_builder" in str(exc_info.value).lower()

    def test_json_task_with_any_workflow(self):
        """JSON scoring requires minimal capabilities."""
        wf = _make_workflow(researcher=True, fn=True)
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="json")))
        validate_composition(wf, task)

    def test_error_message_has_mode_and_task_name(self):
        wf = _make_workflow(researcher=True, name="design")
        task = Task(definition=TaskDefinition(name="chess-evolve", scoring=ScoringContract(method="exit_code")))
        with pytest.raises(IncompatibleCompositionError) as exc_info:
            validate_composition(wf, task)
        err = exc_info.value
        assert err.mode_name == "design"
        assert err.task_name == "chess-evolve"
        assert len(err.mode_missing) > 0


# ── compose() tests ─────────────────────────────────────────────


class TestCompose:
    def test_compose_returns_inner_loop(self, tmp_path: Path):
        wf = _make_workflow(builder=True)
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        loop = compose(wf, task, tmp_path)
        from factory.inner_loop import InnerLoop

        assert isinstance(loop, InnerLoop)

    def test_compose_wraps_non_datanode_workflow(self, tmp_path: Path):
        """compose() auto-wraps workflows with DataNode, so research-only workflows succeed."""
        from factory.workflow.primitives import DataNode
        wf = _make_workflow(researcher=True, name="research-only")
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        loop = compose(wf, task, tmp_path)
        # Should succeed — DataNode wrapping bypasses build capabilities
        assert any(isinstance(n, DataNode) for n in loop.workflow.nodes.values())

    def test_compose_non_task_raises(self, tmp_path: Path):
        wf = _make_workflow(builder=True)
        with pytest.raises(TypeError, match="four hooks"):
            compose(wf, "not a task", tmp_path)

    def test_compose_sets_task_on_loop(self, tmp_path: Path):
        wf = _make_workflow(builder=True)
        task = Task(definition=TaskDefinition(name="t", scoring=ScoringContract(method="exit_code")))
        loop = compose(wf, task, tmp_path)
        assert loop.task is task


# ── TaskProtocol tests ──────────────────────────────────────────


class TestTaskProtocol:
    def test_task_satisfies_protocol(self):
        task = Task()
        assert isinstance(task, TaskProtocol)

    def test_plain_object_not_protocol(self):
        assert not isinstance("hello", TaskProtocol)


# ── TOML task capability tests ──────────────────────────────────


class TestTomlTaskCapabilities:
    def test_chess_evolve_toml_no_builder_required(self):
        """chess-evolve.toml with required_capabilities=[] should need no capabilities."""
        from factory.task import TaskDefinition

        defn = TaskDefinition.from_toml(_CHESS_EVOLVE_TOML)
        assert defn.constraints.required_capabilities == []
        task = Task(definition=defn)
        caps = TaskCapabilities.from_task(task)
        assert len(caps.requires) == 0

    def test_chess_evolve_toml_passes_any_workflow(self):
        """chess-evolve.toml should pass composition with a research-only workflow."""
        from factory.task import TaskDefinition

        defn = TaskDefinition.from_toml(_CHESS_EVOLVE_TOML)
        task = Task(definition=defn)
        wf = _make_workflow(researcher=True, name="research-only")
        validate_composition(wf, task)

    def test_toml_without_required_capabilities_uses_inference(self, tmp_path):
        """A TOML task without required_capabilities key should get None → inference."""
        toml_content = """
[task]
name = "sw-task"

[scoring]
method = "exit_code"

[constraints]
timeout = 600
"""
        toml_file = tmp_path / "sw-task.toml"
        toml_file.write_text(toml_content)
        defn = TaskDefinition.from_toml(toml_file)
        assert defn.constraints.required_capabilities is None
        task = Task(definition=defn)
        caps = TaskCapabilities.from_task(task)
        assert Capability.CAN_MODIFY_CODE in caps.requires
        assert Capability.HAS_BUILDER in caps.requires


# ── TaskConstraints default tests ─────────────────────────────


class TestTaskConstraintsDefault:
    def test_default_required_capabilities_is_none(self):
        """Default TaskConstraints should have required_capabilities=None, not []."""
        tc = TaskConstraints()
        assert tc.required_capabilities is None

    def test_explicit_empty_list_preserved(self):
        """Setting required_capabilities=[] should be preserved, not coerced to None."""
        tc = TaskConstraints(required_capabilities=[])
        assert tc.required_capabilities == []
        assert tc.required_capabilities is not None

    def test_explicit_capabilities_preserved(self):
        tc = TaskConstraints(required_capabilities=[Capability.CAN_MODIFY_CODE])
        assert tc.required_capabilities == [Capability.CAN_MODIFY_CODE]


# ── Composition Matrix ──────────────────────────────────────────


class TestCompositionMatrix:
    """Test known mode × task combinations."""

    @pytest.mark.parametrize(
        "workflow_kwargs,scoring,should_pass",
        [
            # Full workflow (builder + gate) × exit_code → pass
            (dict(builder=True, gate=True), ScoringContract(method="exit_code"), True),
            # Builder-only × exit_code → pass
            (dict(builder=True), ScoringContract(method="exit_code"), True),
            # Research-only × exit_code → fail
            (dict(researcher=True), ScoringContract(method="exit_code"), False),
            # Builder × json → pass
            (dict(builder=True), ScoringContract(method="json"), True),
            # Research × json → pass (no code mod needed)
            (dict(researcher=True, fn=True), ScoringContract(method="json"), True),
            # Empty workflow × exit_code → fail
            (dict(fn=True), ScoringContract(method="exit_code"), False),
        ],
    )
    def test_composition(self, workflow_kwargs, scoring, should_pass):
        wf = _make_workflow(**workflow_kwargs)
        task = Task(definition=TaskDefinition(name="t", scoring=scoring))
        if should_pass:
            validate_composition(wf, task)  # should not raise
        else:
            with pytest.raises(IncompatibleCompositionError):
                validate_composition(wf, task)


# ── CAPABILITY_ALIASES mapping tests ───────────────────────────


class TestCapabilityAliases:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("codebase-analysis", Capability.HAS_RESEARCHER),
            ("code-generation", Capability.CAN_MODIFY_CODE),
            ("health-check", Capability.HAS_HEALTH_CHECK),
            ("code-review", Capability.HAS_CODE_REVIEW),
            ("adversarial-qa", Capability.HAS_ADVERSARIAL_QA),
            ("observation", Capability.HAS_RESEARCHER),
        ],
    )
    def test_mapping_coverage(self, alias, expected):
        assert CAPABILITY_ALIASES[alias] is expected

    def test_all_values_are_capability_members(self):
        for key, val in CAPABILITY_ALIASES.items():
            assert isinstance(val, Capability), f"{key} maps to non-Capability {val!r}"


# ── Package + Task integration tests ──────────────────────────


class TestPackageTaskIntegration:
    def _make_package_with_caps(self, capabilities: list[str]):
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow
        from factory.workflow.package import Package, StateContract

        builder = AgentNode(id="b", role=AgentRole.BUILDER, prompt_template="build")
        wf = Workflow(
            name="pkg-wf",
            nodes={"b": builder},
            edges=[],
            start_node="b",
        )
        return Package(
            name="test-pkg",
            graph=wf,
            entry_node="b",
            exit_node="b",
            contract=StateContract(capabilities=capabilities),
        )

    def test_compile_preserves_declared_capabilities(self):
        pkg = self._make_package_with_caps(["code-generation", "health-check"])
        compiled = pkg.compile()
        assert compiled.declared_capabilities == frozenset(["code-generation", "health-check"])

    def test_check_mode_task_compat_with_package(self):
        pkg = self._make_package_with_caps(["code-generation", "health-check"])
        compiled = pkg.compile()
        task = Task(
            definition=TaskDefinition(
                name="t",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(
                    required_capabilities=[Capability.CAN_MODIFY_CODE, Capability.HAS_HEALTH_CHECK],
                ),
            )
        )
        compat, missing = check_mode_task_compat(compiled, task)
        assert compat is True
        assert len(missing) == 0

    def test_check_mode_task_compat_missing_caps(self):
        pkg = self._make_package_with_caps(["observation"])
        compiled = pkg.compile()
        task = Task(
            definition=TaskDefinition(
                name="t",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(
                    required_capabilities=[Capability.HAS_CODE_REVIEW],
                ),
            )
        )
        compat, missing = check_mode_task_compat(compiled, task)
        assert compat is False
        assert Capability.HAS_CODE_REVIEW in missing


# ── Backward compat tests ─────────────────────────────────────


class TestDeclaredCapabilitiesBackwardCompat:
    def test_raw_workflow_no_declared_capabilities(self):
        wf = _make_workflow(builder=True, researcher=True)
        caps = ModeCapabilities.from_workflow(wf)
        assert Capability.CAN_MODIFY_CODE in caps.provides
        assert Capability.HAS_RESEARCHER in caps.provides
        assert Capability.HAS_BUILDER in caps.provides

    def test_empty_declared_capabilities_identical_to_none(self):
        from factory.workflow.primitives import Workflow

        wf = Workflow(name="empty-decl", nodes={}, edges=[], start_node="s")
        assert wf.declared_capabilities == frozenset()
        caps = ModeCapabilities.from_workflow(wf)
        assert len(caps.provides) == 0


# ── Unknown capability warning test ────────────────────────────


class TestUnmappedCapabilityWarning:
    def test_unmapped_capability_warns_and_excluded(self):
        import structlog

        captured = []
        def capture_processor(logger, method_name, event_dict):
            captured.append(event_dict)
            raise structlog.DropEvent

        old_config = structlog.get_config()
        try:
            structlog.configure(processors=[capture_processor])
            from factory.workflow.primitives import AgentNode, AgentRole, Workflow
            from factory.workflow.package import Package, StateContract

            builder = AgentNode(id="b", role=AgentRole.BUILDER, prompt_template="build")
            wf = Workflow(name="w", nodes={"b": builder}, edges=[], start_node="b")
            pkg = Package(
                name="p",
                graph=wf,
                entry_node="b",
                exit_node="b",
                contract=StateContract(capabilities=["unknown-xyz", "code-generation"]),
            )
            compiled = pkg.compile()
            caps = ModeCapabilities.from_workflow(compiled)
            assert Capability.CAN_MODIFY_CODE in caps.provides
            warnings = [e for e in captured if e.get("event") == "unmapped_capability"]
            assert len(warnings) == 1
            assert warnings[0]["capability"] == "unknown-xyz"
        finally:
            structlog.configure(**old_config)


# ── Round-trip serialization test ──────────────────────────────


class TestDeclaredCapabilitiesRoundTrip:
    def test_model_dump_validate_round_trip(self):
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow
        from factory.workflow.package import Package, StateContract

        builder = AgentNode(id="b", role=AgentRole.BUILDER, prompt_template="build")
        wf = Workflow(name="w", nodes={"b": builder}, edges=[], start_node="b")
        pkg = Package(
            name="p",
            graph=wf,
            entry_node="b",
            exit_node="b",
            contract=StateContract(capabilities=["code-generation", "health-check"]),
        )
        compiled = pkg.compile()
        assert compiled.declared_capabilities == frozenset(["code-generation", "health-check"])

        dumped = compiled.model_dump(mode="json")
        restored = Workflow.model_validate(dumped, strict=False)
        assert restored.declared_capabilities == frozenset(["code-generation", "health-check"])

    def test_to_dict_from_dict_round_trip(self):
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow
        from factory.workflow.package import Package, StateContract

        builder = AgentNode(id="b", role=AgentRole.BUILDER, prompt_template="build")
        wf = Workflow(name="w", nodes={"b": builder}, edges=[], start_node="b")
        pkg = Package(
            name="p",
            graph=wf,
            entry_node="b",
            exit_node="b",
            contract=StateContract(capabilities=["code-generation", "health-check"]),
        )
        compiled = pkg.compile()
        d = compiled.to_dict()
        restored = Workflow.from_dict(d)
        assert restored.declared_capabilities == frozenset(["code-generation", "health-check"])


class TestExplicitEmptyCapsValidation:
    """Gap 0: Tasks with required_capabilities=[] pass any workflow."""

    def test_explicit_empty_caps_passes_research_workflow(self):
        """A task with TaskConstraints(required_capabilities=[]) and exit_code scoring
        passes validate_composition() with a minimal workflow that has no builder node."""
        wf = _make_workflow(researcher=True, name="no-builder")
        task = Task(
            definition=TaskDefinition(
                name="eval-only-task",
                scoring=ScoringContract(method="exit_code"),
                constraints=TaskConstraints(required_capabilities=[]),
            )
        )
        # Should NOT raise IncompatibleCompositionError
        validate_composition(wf, task)
