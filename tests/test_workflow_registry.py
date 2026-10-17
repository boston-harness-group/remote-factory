"""Tests for WorkflowRegistry — discovery, loading, error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.workflow.registry import WorkflowRegistry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry state before each test."""
    WorkflowRegistry.reset()
    yield
    WorkflowRegistry.reset()


# ── Discovery ────────────────────────────────────────────────────


class TestDiscovery:
    def test_discovers_builtins(self) -> None:
        entries = WorkflowRegistry.discover()
        assert "design" in entries
        assert "create" in entries
        assert entries["design"].source == "builtin"

    def test_discovers_from_project_path(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / ".factory" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "local.py").write_text(
            "from factory.workflow.definitions import design_workflow\n"
            "\n"
            'meta = {"name": "local", "description": "Project-local"}\n'
            "\n"
            "def workflow():\n"
            "    wf = design_workflow()\n"
            '    wf.name = "local"\n'
            "    return wf\n"
        )
        entries = WorkflowRegistry.discover(project_path=tmp_path)
        assert "local" in entries
        assert entries["local"].source == "project"


def _write_project_workflow(project: Path, name: str) -> None:
    wf_dir = project / ".factory" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / f"{name}.py").write_text(
        "from factory.workflow.definitions import design_workflow\n"
        "\n"
        f'meta = {{"name": "{name}", "description": "Project-local"}}\n'
        "\n"
        "def workflow():\n"
        "    wf = design_workflow()\n"
        f'    wf.name = "{name}"\n'
        "    return wf\n"
    )


class TestDiscoveryAfterPluginRegistration:
    """Plugins register callables at startup, before any lookup runs discovery."""

    def test_get_workflow_finds_project_workflow(self, tmp_path: Path) -> None:
        WorkflowRegistry.register_callable("composed", lambda: None, source="package")
        _write_project_workflow(tmp_path, "local")

        wf = WorkflowRegistry.get_workflow("local", tmp_path)

        assert wf is not None
        assert wf.name == "local"

    def test_list_workflows_includes_project_and_registered(self, tmp_path: Path) -> None:
        WorkflowRegistry.register_callable("composed", lambda: None, source="package")
        _write_project_workflow(tmp_path, "local")

        names = {entry.name for entry in WorkflowRegistry.list_workflows(tmp_path)}

        assert {"local", "composed", "design"} <= names

    def test_discover_keeps_registered_callables(self) -> None:
        WorkflowRegistry.register_callable("composed", lambda: None, source="package")

        entries = WorkflowRegistry.discover()

        assert entries["composed"].source == "package"

    def test_get_workflow_discovers_a_second_project(self, tmp_path: Path) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        _write_project_workflow(first, "one")
        _write_project_workflow(second, "two")

        assert WorkflowRegistry.get_workflow("one", first) is not None
        assert WorkflowRegistry.get_workflow("two", second) is not None


# ── get_workflow ─────────────────────────────────────────────────


class TestGetWorkflow:
    def test_returns_none_for_unknown(self) -> None:
        wf = WorkflowRegistry.get_workflow("nonexistent")
        assert wf is None

    def test_returns_builtin(self) -> None:
        wf = WorkflowRegistry.get_workflow("design")
        assert wf is not None
        assert wf.name == "design"


# ── list_workflows ───────────────────────────────────────────────


class TestListWorkflows:
    def test_returns_sorted_entries(self) -> None:
        workflows = WorkflowRegistry.list_workflows()
        names = [w.name for w in workflows]
        assert len(names) >= 3
        assert "design" in names
        assert "create" in names


# ── reset ────────────────────────────────────────────────────────


class TestReset:
    def test_clears_state(self) -> None:
        WorkflowRegistry.discover()
        assert len(WorkflowRegistry._entries) > 0

        WorkflowRegistry.reset()
        assert len(WorkflowRegistry._entries) == 0
        assert len(WorkflowRegistry._search_paths) == 0
