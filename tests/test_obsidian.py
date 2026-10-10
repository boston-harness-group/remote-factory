"""Tests for factory.obsidian — note creation with Obsidian frontmatter."""

import subprocess
from datetime import datetime
from unittest.mock import Mock

import pytest

from factory.models import CompositeScore, EvalResult, ExperimentRecord
from factory.obsidian.notes import (
    _obsidian_create as _real_obsidian_create,
    init_vault,
    update_memory_index,
    write_experiment_note,
    write_project_dashboard,
    write_strategy_note,
)


@pytest.fixture(autouse=True)
def set_vault_path(obsidian_vault, monkeypatch):
    """Set OBSIDIAN_VAULT_PATH to temp dir and disable obsidian-cli for all tests."""
    monkeypatch.delenv("FACTORY_VAULT_PATH", raising=False)
    monkeypatch.setenv("FACTORY_VAULT_PATH", str(obsidian_vault))
    # Disable obsidian-cli so write functions always fall back to direct file I/O.
    # Individual tests in TestObsidianCli override this as needed.
    monkeypatch.setattr(
        "factory.obsidian.notes._obsidian_create",
        lambda name, content, vault="factory": False,
    )


@pytest.fixture
def sample_record() -> ExperimentRecord:
    return ExperimentRecord(
        id=1,
        timestamp=datetime(2026, 4, 11, 12, 0),
        hypothesis="Add session timeout handling",
        change_summary="Added timeout check in gateway.py",
        issue_number=11,
        pr_number=12,
        score_before=0.82,
        score_after=0.87,
        delta=0.05,
        verdict="keep",
        cost_usd=1.5,
        notes="",
    )


class TestExperimentNote:
    def test_creates_note(self, sample_record, obsidian_vault):
        path = write_experiment_note("cloud-gateway", sample_record)
        assert path.exists()
        assert "cloud-gateway-001.md" in path.name
        # Should be under 10-Projects/cloud-gateway/Experiments/
        assert "10-Projects" in str(path)
        assert "cloud-gateway" in str(path.parent.parent.name)
        assert path.parent.name == "Experiments"

    def test_note_has_frontmatter(self, sample_record, obsidian_vault):
        path = write_experiment_note("cloud-gateway", sample_record)
        content = path.read_text()
        assert content.startswith("---\n")
        assert "tags:" in content
        assert "  - factory" in content
        assert "  - experiment" in content
        assert "  - cloud-gateway" in content
        assert "verdict: keep" in content
        assert "experiment_id: 1" in content
        assert "source: factory-qa" in content

    def test_note_has_hypothesis(self, sample_record, obsidian_vault):
        path = write_experiment_note("cloud-gateway", sample_record)
        content = path.read_text()
        assert "Add session timeout handling" in content

    def test_note_with_eval_details(self, sample_record, obsidian_vault):
        before = CompositeScore(
            total=0.82,
            passed=True,
            guard_violations=[],
            results=[EvalResult(name="tests", score=1.0, weight=0.5, passed=True, details="ok")],
        )
        after = CompositeScore(
            total=0.87,
            passed=True,
            guard_violations=[],
            results=[EvalResult(name="tests", score=1.0, weight=0.5, passed=True, details="ok")],
        )
        path = write_experiment_note("cloud-gateway", sample_record, before, after)
        content = path.read_text()
        assert "Eval Details" in content
        assert "| tests" in content

    def test_creates_parent_dirs(self, sample_record, obsidian_vault):
        path = write_experiment_note("new-project", sample_record)
        assert path.parent.exists()


class TestProjectDashboard:
    def test_creates_dashboard(self, obsidian_vault):
        path = write_project_dashboard("cloud-gateway", "has_factory", 0.87, [])
        assert path.exists()
        assert "cloud-gateway.md" in path.name
        # Should be under 10-Projects/cloud-gateway/
        assert "10-Projects" in str(path)
        assert path.parent.name == "cloud-gateway"

    def test_dashboard_has_frontmatter(self, obsidian_vault):
        path = write_project_dashboard("cloud-gateway", "has_factory", 0.87, [])
        content = path.read_text()
        assert "  - factory" in content
        assert "  - project" in content

    def test_dashboard_with_records(self, sample_record, obsidian_vault):
        path = write_project_dashboard("cloud-gateway", "has_factory", 0.87, [sample_record])
        content = path.read_text()
        assert "Experiments Run**: 1" in content
        assert "Kept**: 1" in content
        assert "[[cloud-gateway-001]]" in content

    def test_dashboard_with_eval_dimensions(self, obsidian_vault):
        dims = [{"name": "tests", "weight": 0.5, "description": "Run tests"}]
        path = write_project_dashboard("cloud-gateway", "has_factory", 0.87, [], dims)
        content = path.read_text()
        assert "Eval Dimensions" in content
        assert "tests" in content


class TestStrategyNote:
    def test_creates_strategy_note(self, obsidian_vault):
        path = write_strategy_note("cloud-gateway", "## Strategy\nFocus on reliability.")
        assert path.exists()
        assert "cloud-gateway-" in path.name
        # Should be under 10-Projects/cloud-gateway/Strategies/
        assert "10-Projects" in str(path)
        assert path.parent.name == "Strategies"

    def test_strategy_has_frontmatter(self, obsidian_vault):
        path = write_strategy_note("cloud-gateway", "content")
        content = path.read_text()
        assert "  - strategy" in content
        assert "  - cloud-gateway" in content
        assert "source: factory-strategist" in content


class TestInitVault:
    def test_creates_vault_structure(self, obsidian_vault):
        vault = init_vault(obsidian_vault)
        assert vault == obsidian_vault
        assert (vault / "10-Projects").is_dir()
        assert (vault / "20-Knowledge" / "Concepts").is_dir()
        assert (vault / "20-Knowledge" / "Sources").is_dir()
        assert (vault / "00-Factory").is_dir()
        assert (vault / "00-Factory" / "Decisions").is_dir()

    def test_creates_obsidian_dir(self, obsidian_vault):
        init_vault(obsidian_vault)
        assert (obsidian_vault / ".obsidian").is_dir()

    def test_creates_templates(self, obsidian_vault):
        init_vault(obsidian_vault)
        templates = obsidian_vault / "_templates"
        assert templates.is_dir()
        template_files = list(templates.glob("*.md"))
        assert len(template_files) == 4
        names = {f.name for f in template_files}
        assert names == {"experiment.md", "decision.md", "strategy.md", "project.md"}

    def test_creates_memory_md(self, obsidian_vault):
        init_vault(obsidian_vault)
        memory = obsidian_vault / "MEMORY.md"
        assert memory.exists()
        content = memory.read_text()
        assert "Factory Memory Index" in content
        assert "(none yet)" in content

    def test_creates_dashboard(self, obsidian_vault):
        init_vault(obsidian_vault)
        dashboard = obsidian_vault / "00-Factory" / "Dashboard.md"
        assert dashboard.exists()
        content = dashboard.read_text()
        assert "Factory Dashboard" in content

    def test_idempotent(self, obsidian_vault):
        init_vault(obsidian_vault)
        # Write custom content to Dashboard.md
        dashboard = obsidian_vault / "00-Factory" / "Dashboard.md"
        dashboard.write_text("Custom content")

        # Second call should not overwrite existing files
        init_vault(obsidian_vault)
        assert dashboard.read_text() == "Custom content"


class TestUpdateMemoryIndex:
    def test_empty_vault(self, obsidian_vault):
        init_vault(obsidian_vault)
        path = update_memory_index()
        content = path.read_text()
        assert "(none yet)" in content

    def test_with_projects(self, obsidian_vault):
        init_vault(obsidian_vault)
        # Create a project with a dashboard
        proj_dir = obsidian_vault / "10-Projects" / "my-app"
        proj_dir.mkdir(parents=True)
        (proj_dir / "my-app.md").write_text(
            "---\ntags:\n  - factory\n---\n\n# Factory: my-app\n\n## Status\n"
            "- **State**: has_factory\n- **Current Score**: 0.9725\n"
            "- **Experiments Run**: 5\n"
        )

        path = update_memory_index()
        content = path.read_text()
        assert "[[my-app]]" in content
        assert "0.9725" in content
        assert "5 experiments" in content

    def test_with_explicit_projects(self, obsidian_vault):
        init_vault(obsidian_vault)
        projects = [{"name": "proj-a", "score": "0.95", "experiments": 3}]
        path = update_memory_index(projects=projects)
        content = path.read_text()
        assert "[[proj-a]]" in content
        assert "0.95" in content
        assert "3 experiments" in content


class TestAutoInit:
    def test_auto_creates_vault_on_write(self, tmp_path, monkeypatch):
        """Writing a note to nonexistent vault creates the vault structure."""
        vault = tmp_path / "new-vault"
        monkeypatch.setenv("FACTORY_VAULT_PATH", str(vault))
        assert not vault.exists()

        record = ExperimentRecord(
            id=1,
            timestamp=datetime(2026, 4, 11, 12, 0),
            hypothesis="Test auto-init",
            change_summary="Auto-init test",
            issue_number=None,
            pr_number=None,
            score_before=0.5,
            score_after=0.6,
            delta=0.1,
            verdict="keep",
            cost_usd=None,
            notes="",
        )
        path = write_experiment_note("test-project", record)
        assert path.exists()
        assert (vault / ".obsidian").is_dir()
        assert (vault / "10-Projects").is_dir()
        assert (vault / "MEMORY.md").exists()


class TestObsidianCli:
    def test_obsidian_create_success(self, monkeypatch):
        """obsidian_create returns True on success."""
        mock_run = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr("factory.obsidian.notes.subprocess.run", mock_run)
        # Restore real _obsidian_create (autouse fixture replaces it with a no-op)
        monkeypatch.setattr("factory.obsidian.notes._obsidian_create", _real_obsidian_create)

        assert _real_obsidian_create("test", "content") is True
        mock_run.assert_called_once()

    def test_obsidian_create_fallback(self, monkeypatch):
        """obsidian_create returns False when CLI not available."""
        monkeypatch.setattr(
            "factory.obsidian.notes.subprocess.run",
            Mock(side_effect=FileNotFoundError),
        )
        from factory.obsidian.notes import _obsidian_create

        assert _obsidian_create("test", "content") is False

    def test_write_experiment_tries_cli_first(
        self,
        monkeypatch,
        sample_record,
        obsidian_vault,
    ):
        """write_experiment_note tries obsidian-cli before file write."""
        calls: list[list[str]] = []
        original_run = subprocess.run

        def mock_run(*args, **kwargs):
            if args and args[0] and args[0][0] == "obsidian-cli":
                calls.append(args[0])
                return Mock(returncode=1)  # CLI fails
            return original_run(*args, **kwargs)

        monkeypatch.setattr("factory.obsidian.notes.subprocess.run", mock_run)
        # Restore real _obsidian_create so it actually calls subprocess.run
        monkeypatch.setattr("factory.obsidian.notes._obsidian_create", _real_obsidian_create)
        path = write_experiment_note("test-project", sample_record)
        # Should have tried CLI
        assert any("obsidian-cli" in str(c) for c in calls)
        # Should have fallen back to file write
        assert path.exists()

    def test_obsidian_search_vault(self, monkeypatch):
        """obsidian_search_vault returns results on success."""
        mock_run = Mock(return_value=Mock(returncode=0, stdout="result1\nresult2\n"))
        monkeypatch.setattr("factory.obsidian.notes.subprocess.run", mock_run)
        from factory.obsidian.notes import obsidian_search_vault

        result = obsidian_search_vault("test query")
        assert result is not None
        assert "result1" in result
