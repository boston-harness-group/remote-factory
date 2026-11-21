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


# ── Discovery after code registration (project workflows were invisible) ──


def _write_local(project: Path, name: str) -> None:
    wf_dir = project / ".factory" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
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


def _composed():
    from factory.workflow.definitions import design_workflow

    wf = design_workflow()
    wf.name = "composed-x"
    return wf


class TestDiscoveryAfterRegistration:
    """A callable registered at import (as packages.py does) must not hide project workflows.

    Before the fix, ``get_workflow``/``list_workflows`` discovered only when ``_entries`` was
    empty, and ``register_callable`` filled it at import, so ``factory workflow run <name>
    <project>`` answered "Unknown workflow" for every project-local workflow.
    """

    def test_project_workflow_found_after_a_callable_is_registered(self, tmp_path: Path) -> None:
        WorkflowRegistry.register_callable("composed-x", _composed, source="package")
        _write_local(tmp_path, "local")
        wf = WorkflowRegistry.get_workflow("local", tmp_path)
        assert wf is not None
        assert wf.name == "local"

    def test_list_includes_project_workflow_after_a_callable_is_registered(
        self, tmp_path: Path
    ) -> None:
        WorkflowRegistry.register_callable("composed-x", _composed, source="package")
        _write_local(tmp_path, "local")
        names = {e.name for e in WorkflowRegistry.list_workflows(tmp_path)}
        assert {"local", "composed-x", "design"} <= names

    def test_discover_keeps_registered_callables(self, tmp_path: Path) -> None:
        WorkflowRegistry.register_callable("composed-x", _composed, source="package")
        entries = WorkflowRegistry.discover(project_path=tmp_path)
        assert "composed-x" in entries
        assert WorkflowRegistry.get_workflow("composed-x", tmp_path) is not None

    def test_a_different_project_is_rediscovered(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        _write_local(a, "only_a")
        _write_local(b, "only_b")
        assert WorkflowRegistry.get_workflow("only_a", a) is not None
        assert WorkflowRegistry.get_workflow("only_b", b) is not None
        assert WorkflowRegistry.get_workflow("only_a", b) is None

    def test_project_workflow_still_wins_over_a_callable_of_the_same_name(
        self, tmp_path: Path
    ) -> None:
        WorkflowRegistry.register_callable("local", _composed, source="package")
        _write_local(tmp_path, "local")
        assert WorkflowRegistry.discover(project_path=tmp_path)["local"].source == "project"


class TestPackagePriority:
    def test_a_user_workflow_beats_a_package_mode_of_the_same_name(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        home = tmp_path / "home"
        _write_local(home, "composed-x")  # ~/.factory/workflows/composed-x.py
        monkeypatch.setenv("HOME", str(home))
        WorkflowRegistry.register_callable("composed-x", _composed, source="package")
        assert WorkflowRegistry.discover()["composed-x"].source == "user"

    def test_registering_a_callable_does_not_shadow_a_discovered_project_workflow(
        self, tmp_path: Path
    ) -> None:
        _write_local(tmp_path, "local")
        WorkflowRegistry.discover(project_path=tmp_path)
        WorkflowRegistry.register_callable("local", _composed, source="package")
        assert WorkflowRegistry._entries["local"].source == "project"
