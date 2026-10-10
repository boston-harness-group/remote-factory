"""Tests for plugin agent role registration (sanctioned AgentRole extension)."""

from __future__ import annotations

import pytest

from factory.plugins import PluginRegistry
from factory.workflow.primitives import (
    AgentConfig,
    AgentNode,
    AgentRole,
    Factory,
    GateNode,
    Workflow,
    _registered_plugin_roles,
    _rebuild_role_schemas,
    register_agent_role,
)


@pytest.fixture(autouse=True)
def _restore_agent_role():
    """Snapshot AgentRole state and restore it after each test.

    register_agent_role mutates a process-global enum; without this fixture
    roles registered in one test would leak into every later test (and into
    other suites, where e.g. the outer loop picks roles with
    random.choice(list(AgentRole))).
    """
    saved_members = dict(AgentRole._member_map_)
    saved_values = dict(AgentRole._value2member_map_)
    saved_names = list(AgentRole._member_names_)
    saved_attrs = [n for n in vars(AgentRole) if n.isupper() and isinstance(getattr(AgentRole, n), AgentRole)]
    saved_registered = dict(_registered_plugin_roles)
    yield

    AgentRole._member_map_.clear()
    AgentRole._member_map_.update(saved_members)
    AgentRole._value2member_map_.clear()
    AgentRole._value2member_map_.update(saved_values)
    AgentRole._member_names_[:] = saved_names
    for attr in saved_attrs:
        type.__setattr__(AgentRole, attr, saved_members[attr])
    for name in [n for n in vars(AgentRole) if n.isupper()]:
        if name not in saved_members:
            type.__delattr__(AgentRole, name)
    _registered_plugin_roles.clear()
    _registered_plugin_roles.update(saved_registered)
    _rebuild_role_schemas()


class TestRegisterAgentRole:
    def test_returns_member_with_derived_name(self):
        member = register_agent_role("paper-reader")
        assert member.name == "PAPER_READER"
        assert member.value == "paper-reader"
        assert AgentRole.PAPER_READER is member
        assert AgentRole("paper-reader") is member
        assert member in list(AgentRole)

    def test_custom_member_name(self):
        member = register_agent_role("cve-judge", name="CVE_JUDGE_ROLE")
        assert member.name == "CVE_JUDGE_ROLE"
        assert member.value == "cve-judge"

    def test_idempotent(self):
        first = register_agent_role("paper-reader")
        assert register_agent_role("paper-reader") is first

    def test_idempotent_with_same_custom_name(self):
        first = register_agent_role("x-role", name="X_ROLE")
        assert register_agent_role("x-role", name="X_ROLE") is first

    def test_builtin_value_collision_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            register_agent_role("builder")

    def test_builtin_name_collision_via_custom_name_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            register_agent_role("totally-new", name="BUILDER")

    def test_value_collision_with_plugin_role_raises(self):
        register_agent_role("dup-role", name="DUP_A")
        # same value under a different member name → rejected (name-mismatch error)
        with pytest.raises(ValueError, match="member name"):
            register_agent_role("dup-role", name="DUP_B")
        # same value under the same name → idempotent
        assert register_agent_role("dup-role", name="DUP_A") is AgentRole.DUP_A

    def test_name_mismatch_on_reregistration_raises(self):
        register_agent_role("y-role")
        with pytest.raises(ValueError, match="member name"):
            register_agent_role("y-role", name="OTHER_NAME")

    def test_invalid_identifier_raises(self):
        with pytest.raises(ValueError, match="valid enum member name"):
            register_agent_role("bad role!")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_agent_role("   ")


class TestGraphUsage:
    def test_agent_node_accepts_plugin_role(self):
        member = register_agent_role("paper-reader")
        node = AgentNode(id="read", role=member)
        assert node.role is member

    def test_lax_validation_accepts_role_string(self):
        """from_dict() validates node data with strict=False; plugin roles must
        behave like builtins there."""
        register_agent_role("paper-reader")
        node = AgentNode.model_validate(
            {"id": "read", "role": "paper-reader"}, strict=False
        )
        assert node.role is AgentRole.PAPER_READER

    def test_gate_node_accepts_plugin_role(self):
        member = register_agent_role("paper-reader")
        gate = GateNode(id="g", evaluator_role=member)
        assert gate.evaluator_role is member

    def test_agent_config_accepts_plugin_role(self):
        member = register_agent_role("paper-reader")
        config = AgentConfig(role=member, model="opus")
        assert config.role is member

    def test_workflow_json_roundtrip(self):
        """Plain enum mutation leaves Pydantic's frozen schema stale, so JSON
        validation of plugin roles fails. register_agent_role must fix that."""
        member = register_agent_role("paper-reader")
        wf = Workflow(
            name="t",
            nodes={"a": AgentNode(id="a", role=member)},
            edges=[],
            start_node="a",
        )
        restored = Workflow.model_validate_json(wf.model_dump_json())
        assert restored.nodes["a"].role is member

    def test_workflow_to_from_dict_roundtrip(self):
        member = register_agent_role("paper-reader")
        wf = Workflow(
            name="t",
            nodes={"a": AgentNode(id="a", role=member)},
            edges=[],
            start_node="a",
        )
        restored = Workflow.from_dict(wf.to_dict())
        assert restored.nodes["a"].role is member

    def test_gate_node_json_roundtrip(self):
        member = register_agent_role("paper-reader")
        gate = GateNode(id="g", evaluator_role=member)
        restored = GateNode.model_validate_json(gate.model_dump_json())
        assert restored.evaluator_role is member

    def test_factory_container_roundtrip(self):
        member = register_agent_role("paper-reader")
        factory = Factory(
            agent_pool={"paper-reader": AgentConfig(role=member, model="sonnet")},
            workflows={},
        )
        restored = Factory.model_validate_json(factory.model_dump_json())
        assert restored.agent_pool["paper-reader"].role is member

    def test_builtin_roles_still_validate_after_registration(self):
        register_agent_role("paper-reader")
        wf = Workflow(
            name="t",
            nodes={"a": AgentNode(id="a", role=AgentRole.BUILDER)},
            edges=[],
            start_node="a",
        )
        restored = Workflow.model_validate_json(wf.model_dump_json())
        assert restored.nodes["a"].role is AgentRole.BUILDER


class TestSkillExportRendering:
    def test_plugin_role_renders_agent_command(self):
        from factory.workflow.skill_export import workflow_to_skill_md

        member = register_agent_role("paper-reader")
        node = AgentNode(id="read", role=member)
        wf = Workflow(name="t", nodes={"read": node}, edges=[], start_node="read")
        content = workflow_to_skill_md(wf)
        assert "factory agent paper-reader" in content


class TestPluginRegistryIntegration:
    def test_add_agent_roles_registers_graph_usable_role(self):
        registry = PluginRegistry()
        registry.add_agent_roles(["paper-reader"])
        assert "paper-reader" in registry.agent_roles
        assert AgentRole.PAPER_READER is not None

        node = AgentNode(id="read", role=AgentRole.PAPER_READER)
        wf = Workflow(name="t", nodes={"read": node}, edges=[], start_node="read")
        restored = Workflow.model_validate_json(wf.model_dump_json())
        assert restored.nodes["read"].role is AgentRole.PAPER_READER

    def test_add_agent_roles_builtin_collision_skipped(self, caplog):
        import structlog

        registry = PluginRegistry()
        with structlog.testing.capture_logs() as logs:
            registry.add_agent_roles(["builder", "fresh-role"])
        assert "builder" not in registry.agent_roles
        assert "fresh-role" in registry.agent_roles
        events = [e for e in logs if e["event"] == "plugin_agent_role_collision_builtin"]
        assert events and events[0]["role"] == "builder"

    def test_add_agent_roles_duplicate_skipped(self):
        registry = PluginRegistry()
        registry.add_agent_roles(["dup-role"])
        registry.add_agent_roles(["dup-role"])
        assert registry.agent_roles.count("dup-role") == 1

    def test_add_agent_roles_invalid_role_skipped(self):
        import structlog

        registry = PluginRegistry()
        with structlog.testing.capture_logs() as logs:
            registry.add_agent_roles(["bad role!"])
        assert "bad role!" not in registry.agent_roles
        events = [
            e for e in logs if e["event"] == "plugin_agent_role_registration_failed"
        ]
        assert events and events[0]["role"] == "bad role!"


class TestPromptResolutionContract:
    """Review feedback on #1492: registering a role makes it valid, not
    behaved. A plugin role with no resolvable prompt should be caught at
    load time (not deep into a CEO cycle), and the runtime error should
    point at plugin packaging as the likely cause."""

    def test_load_plugins_warns_for_role_without_prompt(self, tmp_path):
        import structlog
        from factory.plugins import _warn_missing_role_prompts

        registry = PluginRegistry()
        registry.add_agent_roles(["ghost-role"])

        with structlog.testing.capture_logs() as logs:
            _warn_missing_role_prompts(registry)
        events = [e for e in logs if e["event"] == "plugin_agent_role_prompt_missing"]
        assert events and events[0]["role"] == "ghost-role"
        assert "escape" not in events[0]  # hint present, not an error

    def test_load_plugins_no_warning_when_user_prompt_exists(self, tmp_path, monkeypatch):
        import structlog
        from factory.agents import runner as runner_mod
        from factory.plugins import _warn_missing_role_prompts

        registry = PluginRegistry()
        registry.add_agent_roles(["settled-role"])

        fake_user_dir = tmp_path / ".factory" / "agents" / "prompts"
        fake_user_dir.mkdir(parents=True)
        (fake_user_dir / "settled-role.md").write_text("# settled role prompt\n")
        monkeypatch.setattr(runner_mod, "_USER_PROMPTS_DIR", fake_user_dir)
        # the check re-derives the path; patch Path.home for the check
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))

        with structlog.testing.capture_logs() as logs:
            _warn_missing_role_prompts(registry)
        assert not [
            e for e in logs if e["event"] == "plugin_agent_role_prompt_missing"
        ]

    def test_no_roles_means_no_check_output(self):
        import structlog
        from factory.plugins import _warn_missing_role_prompts

        with structlog.testing.capture_logs() as logs:
            _warn_missing_role_prompts(PluginRegistry())
        assert not [e for e in logs if "prompt_missing" in e["event"]]

    def test_resolve_prompt_error_mentions_plugin_registration(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from factory.agents.runner import resolve_prompt

        registry = PluginRegistry()
        registry.add_agent_roles(["ghost-role"])

        # No prompt anywhere: not a factory builtin, no user-global, no project
        with patch("factory.plugins.get_registry", return_value=registry):
            try:
                resolve_prompt("ghost-role", tmp_path)
            except FileNotFoundError as exc:
                assert "plugin-registered" in str(exc)
                assert "wheel" in str(exc)
            else:
                pytest.fail("expected FileNotFoundError for role with no prompt")

    def test_resolve_prompt_error_plain_for_builtin_role(self, tmp_path):
        from factory.agents.runner import resolve_prompt

        # A builtin role with no prompt file anywhere should NOT get the
        # plugin hint (patch the prompts dir to a nonexistent one).
        import factory.agents.runner as runner_mod

        original = runner_mod._PROMPTS_DIR
        runner_mod._PROMPTS_DIR = tmp_path / "nonexistent"
        try:
            with pytest.raises(FileNotFoundError) as excinfo:
                resolve_prompt("definitely-not-a-real-role", tmp_path)
            assert "plugin-registered" not in str(excinfo.value)
        finally:
            runner_mod._PROMPTS_DIR = original
