"""Tests for plugin package generation in CREATE mode."""

from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from factory.cli._task_builder import _build_ceo_task, _slug


class TestSlug:
    def test_basic(self):
        assert _slug("approval workflow") == "approval-workflow"

    def test_special_chars(self):
        assert _slug("My Cool Mode!!! v2") == "my-cool-mode-v2"

    def test_truncation(self):
        result = _slug("a" * 60)
        assert len(result) <= 40

    def test_strips_leading_trailing_hyphens(self):
        assert _slug("---hello---") == "hello"


class TestPluginTaskBuilder:
    def test_plugin_mode_produces_plugin_header(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="approval workflow",
            plugin_mode=True,
        )
        assert "## Create Mode (Plugin Package)" in task
        assert "plugin_mode:" in task
        assert "pyproject.toml" in task
        assert "register_plugin" in task

    def test_plugin_mode_with_folder(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="approval workflow",
            plugin_mode=True,
            plugin_folder="/tmp/my-plugin",
        )
        assert "/tmp/my-plugin" in task
        assert "## Create Mode (Plugin Package)" in task

    def test_plugin_mode_default_folder(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="approval workflow",
            plugin_mode=True,
        )
        assert "approval-workflow-plugin" in task

    def test_non_plugin_create_unchanged(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="approval workflow",
            plugin_mode=False,
        )
        assert "## Create Mode (New Portable Workflow)" in task
        assert "## Create Mode (Plugin Package)" not in task

    def test_update_mode_takes_precedence_over_plugin(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="add plugin support",
            update_existing_mode="create",
            plugin_mode=True,
        )
        assert "## Create Mode (Update Existing Mode)" in task
        assert "## Create Mode (Plugin Package)" not in task


# ── Editing a project's own workflow ("name: change") ───────────────


def _write_project_workflow(project: Path, name: str) -> Path:
    wf_dir = project / ".factory" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    path = wf_dir / f"{name.replace('-', '_')}.py"
    path.write_text(
        "from factory.workflow.definitions import design_workflow\n"
        "\n"
        f'meta = {{"name": "{name}", "description": "Project-local"}}\n'
        "\n"
        "def workflow():\n"
        "    wf = design_workflow()\n"
        f'    wf.name = "{name}"\n'
        "    return wf\n"
    )
    return path


class TestEditProjectWorkflow:
    """`--mode create --focus "<name>: <change>"` must edit the project's own mode.

    Before the fix only built-in modes (register_all) were update targets, so a project
    workflow's name fell through and the request silently created a second, duplicate mode.
    """

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        from factory.workflow.registry import WorkflowRegistry

        WorkflowRegistry.reset()
        yield
        WorkflowRegistry.reset()

    def test_project_workflow_name_routes_to_an_update(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _resolve_ceo_project

        _write_project_workflow(tmp_path, "growth-curve")
        resolved = _resolve_ceo_project(
            str(tmp_path), "create", False, False,
            "growth-curve: add a doubling-time column", None, None,
        )
        assert not isinstance(resolved, int)
        create_description, update_existing_mode = resolved[7], resolved[8]
        assert update_existing_mode == "growth-curve"
        assert create_description == "add a doubling-time column"

    def test_unknown_name_creates_a_new_mode_and_says_so(self, tmp_path: Path, capsys):
        from factory.cli._ceo_helpers import _resolve_ceo_project

        resolved = _resolve_ceo_project(
            str(tmp_path), "create", False, False,
            "growth-curve: add a doubling-time column", None, None,
        )
        assert not isinstance(resolved, int)
        assert resolved[8] is None
        assert resolved[7] == "growth-curve: add a doubling-time column"
        assert "no existing mode named 'growth-curve'" in capsys.readouterr().err

    def test_project_workflow_update_edits_its_file_not_factory_source(self, tmp_path: Path):
        path = _write_project_workflow(tmp_path, "growth-curve")
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="add a doubling-time column",
            update_existing_mode="growth-curve",
        )
        assert "## Create Mode (Update Existing Workflow File)" in task
        assert str(path.relative_to(tmp_path)) in task
        assert "ALL 20 registration points" not in task
        assert "Do not create a second workflow" in task

    def test_builtin_update_keeps_its_instructions(self, tmp_path: Path):
        task = _build_ceo_task(
            tmp_path, "create",
            create_description="tighten the gate",
            update_existing_mode="create",
        )
        assert "## Create Mode (Update Existing Mode)" in task
        assert "## Create Mode (Update Existing Workflow File)" not in task


class TestEditRoutingEdges:
    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        from factory.workflow.registry import WorkflowRegistry

        WorkflowRegistry.reset()
        yield
        WorkflowRegistry.reset()

    def test_a_mode_name_with_digits_routes_to_an_update(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _resolve_ceo_project

        _write_project_workflow(tmp_path, "growth2")
        resolved = _resolve_ceo_project(
            str(tmp_path), "create", False, False, "growth2: add a column", None, None
        )
        assert not isinstance(resolved, int)
        assert resolved[8] == "growth2"

    def test_a_user_workflow_is_refused_with_how_to_edit_it(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        from factory.cli._ceo_helpers import _resolve_ceo_project

        home = tmp_path / "home"
        _write_project_workflow(home, "shared-mode")  # ~/.factory/workflows/shared_mode.py
        monkeypatch.setenv("HOME", str(home))
        project = tmp_path / "proj"
        project.mkdir()
        resolved = _resolve_ceo_project(
            str(project), "create", False, False, "shared-mode: change it", None, None
        )
        assert resolved == 1
        err = capsys.readouterr().err
        assert "user workflow" in err and "copy it into" in err

    def test_prose_with_a_colon_gets_no_notice(self, tmp_path: Path, capsys):
        from factory.cli._ceo_helpers import _resolve_ceo_project

        resolved = _resolve_ceo_project(
            str(tmp_path), "create", False, False, "Note: build a triage mode", None, None
        )
        assert not isinstance(resolved, int)
        assert resolved[8] is None
        assert "no existing mode" not in capsys.readouterr().err


def test_review_prompts_allow_editing_the_workflow_being_updated():
    """Builder, code reviewer and refiner must agree, or review rejects the permitted edit."""
    root = Path(__file__).resolve().parent.parent / "factory" / "agents" / "prompts"
    for name in ("builder.md", "code_reviewer.md", "refiner.md"):
        text = (root / name).read_text()
        assert "except a workflow file under `.factory/workflows/`" in text, name


class TestCreateRunsInPlace:
    """Decided per file: a worktree's copy of a workflow file git does not track can't be
    committed on the run branch, and removing the worktree at exit deleted the edit."""

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        from factory.workflow.registry import WorkflowRegistry

        WorkflowRegistry.reset()
        yield
        WorkflowRegistry.reset()

    @staticmethod
    def _git(project: Path, *args: str) -> None:
        import subprocess

        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(project), "PATH": "/usr/bin:/bin"}
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True, env=env)

    def _repo(self, tmp_path: Path, *, ignore_factory: bool, tracked: tuple[str, ...] = (),
              untracked: tuple[str, ...] = ()) -> Path:
        project = tmp_path / "proj"
        project.mkdir()
        self._git(project, "init", "-q", "-b", "main")
        (project / "README.md").write_text("x")
        if ignore_factory:
            (project / ".gitignore").write_text(".factory/\n")
        for name in (*tracked, *untracked):
            _write_project_workflow(project, name)
        self._git(project, "add", "README.md", *([".gitignore"] if ignore_factory else []))
        for name in tracked:
            self._git(project, "add", "-f", f".factory/workflows/{name.replace('-', '_')}.py")
        self._git(project, "commit", "-q", "-m", "init")
        return project

    def test_an_untracked_workflow_is_updated_in_place(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _create_runs_in_place

        project = self._repo(tmp_path, ignore_factory=True, untracked=("growth-curve",))
        assert _create_runs_in_place(project, "create", False, "growth-curve")

    def test_mixed_layout_is_decided_per_file(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _create_runs_in_place

        project = self._repo(tmp_path, ignore_factory=True,
                             tracked=("kept-mode",), untracked=("loose-mode",))
        assert _create_runs_in_place(project, "create", False, "loose-mode")
        assert not _create_runs_in_place(project, "create", False, "kept-mode")
        assert _create_runs_in_place(project, "create", False, None)  # new file: ignored

    def test_a_new_workflow_in_an_unignored_dir_keeps_the_branch(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _create_runs_in_place

        project = self._repo(tmp_path, ignore_factory=False, tracked=("kept-mode",))
        assert not _create_runs_in_place(project, "create", False, None)
        assert not _create_runs_in_place(project, "create", False, "kept-mode")

    def test_builtin_updates_plugins_and_other_modes_keep_the_worktree(self, tmp_path: Path):
        from factory.cli._ceo_helpers import _create_runs_in_place

        project = self._repo(tmp_path, ignore_factory=True, untracked=("growth-curve",))
        assert not _create_runs_in_place(project, "create", False, "create")  # built-in
        assert not _create_runs_in_place(project, "create", True, None)       # plugin
        assert not _create_runs_in_place(project, "improve", False, None)     # not create

    def test_the_command_skips_the_worktree_and_tells_the_agent_not_to_commit(
        self, tmp_path: Path
    ):
        """End to end through `factory ceo`: the launcher uses the decision. (The decision
        itself is tested above with real git; `_mock_foreground` stubs subprocess.run for
        the whole process, so here the decision is pinned.)"""
        from tests.test_cli import _mock_foreground

        from factory.cli import main

        _write_project_workflow(tmp_path, "growth-curve")
        with _mock_foreground() as mock_run, \
                patch("factory.worktree.create_worktree") as cw, \
                patch("factory.cli._ceo_helpers._create_runs_in_place", return_value=True):
            rc = main(["ceo", str(tmp_path), "--mode", "create",
                       "--focus", "growth-curve: add a doubling-time column"])
        assert rc == 0
        cw.assert_not_called()
        sent = " ".join(str(a) for call in mock_run.call_args_list for a in call.args[0])
        assert "## Working in Place" in sent and "Do NOT commit" in sent
        assert "--output-dir .factory/skill-export" in sent
        assert "skills/workflow-<name>/SKILL.md" in sent
        # The CEO starts with FACTORY_IN_PLACE=1, so every agent it launches inherits it.
        claude_env_flagged = [
            (call.kwargs.get("env") or {}).get("FACTORY_IN_PLACE") == "1"
            for call in mock_run.call_args_list if call.args[0][0] == "claude"
        ]
        assert claude_env_flagged == [True]

    def test_the_command_keeps_the_worktree_when_not_in_place(self, tmp_path: Path):
        from tests.test_cli import _mock_foreground

        from factory.cli import main

        _write_project_workflow(tmp_path, "growth-curve")
        with _mock_foreground() as mock_run, patch(
            "factory.worktree.create_worktree",
            side_effect=lambda p, b="main", run_id=None: (p, "factory/run-test"),
        ) as cw, patch("factory.cli._ceo_helpers._create_runs_in_place", return_value=False):
            rc = main(["ceo", str(tmp_path), "--mode", "create",
                       "--focus", "growth-curve: add a doubling-time column"])
        assert rc == 0
        cw.assert_called_once()
        sent = " ".join(str(a) for call in mock_run.call_args_list for a in call.args[0])
        assert "## Working in Place" not in sent


def test_the_launcher_forces_in_place_and_says_so(tmp_path: Path, monkeypatch, capsys):
    import factory.cli._ceo_helpers as h

    monkeypatch.setattr(h, "_create_runs_in_place", lambda *a: True)
    assert h._resolve_no_worktree(False, tmp_path, "create", False, None) == (True, True)
    assert "works in place" in capsys.readouterr().err
    monkeypatch.setattr(h, "_create_runs_in_place", lambda *a: False)
    assert h._resolve_no_worktree(False, tmp_path, "create", False, None) == (False, False)
    assert h._resolve_no_worktree(True, tmp_path, "improve", False, None) == (True, False)


def test_a_user_workflow_never_gets_the_edit_in_place_task(tmp_path: Path, monkeypatch):
    """The task builder itself keeps file-edit instructions for project workflows only."""
    from factory.workflow.registry import WorkflowRegistry

    WorkflowRegistry.reset()
    home = tmp_path / "home"
    _write_project_workflow(home, "shared-mode")
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "proj"
    project.mkdir()
    task = _build_ceo_task(project, "create", create_description="change it",
                           update_existing_mode="shared-mode")
    assert "## Create Mode (Update Existing Workflow File)" not in task
    WorkflowRegistry.reset()


class TestInPlaceRunGuard:
    """An in-place run must not commit or switch branches in the person's checkout; if it
    does anyway, the end of the run says so loudly, with the command to undo it."""

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        TestCreateRunsInPlace._git(tmp_path, "init", "-q", "-b", "main")
        (tmp_path / "a.txt").write_text("a")
        TestCreateRunsInPlace._git(tmp_path, "add", "a.txt")
        TestCreateRunsInPlace._git(tmp_path, "commit", "-q", "-m", "init")
        return tmp_path

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        import factory.cli._ceo_helpers as h

        monkeypatch.setattr(h, "_IN_PLACE_START", None)
        monkeypatch.delenv("FACTORY_IN_PLACE", raising=False)
        yield

    def test_a_commit_during_the_run_is_reported_with_how_to_undo(self, tmp_path, capsys):
        import factory.cli._ceo_helpers as h

        repo = self._repo(tmp_path)
        h._begin_in_place_run(repo)
        import os
        assert os.environ.get("FACTORY_IN_PLACE") == "1"
        head0 = h._git_out(repo, "rev-parse", "HEAD")
        (repo / "b.txt").write_text("b")
        TestCreateRunsInPlace._git(repo, "add", "b.txt")
        TestCreateRunsInPlace._git(repo, "commit", "-q", "-m", "run commit")

        h._end_in_place_run(repo)

        err = capsys.readouterr().err
        assert "changed your checkout's git state" in err
        assert f"git reset --soft {head0}" in err
        assert os.environ.get("FACTORY_IN_PLACE") is None

    def test_a_branch_switch_is_reported(self, tmp_path, capsys):
        import factory.cli._ceo_helpers as h

        repo = self._repo(tmp_path)
        h._begin_in_place_run(repo)
        TestCreateRunsInPlace._git(repo, "switch", "-q", "-c", "run-branch")

        h._end_in_place_run(repo)

        assert "git switch main" in capsys.readouterr().err

    def test_an_untouched_checkout_is_silent(self, tmp_path, capsys):
        import factory.cli._ceo_helpers as h

        repo = self._repo(tmp_path)
        h._begin_in_place_run(repo)
        (repo / "a.txt").write_text("edited, not committed")

        h._end_in_place_run(repo)

        assert "changed your checkout" not in capsys.readouterr().err

    def test_existing_uncommitted_work_is_pointed_out(self, tmp_path, capsys):
        import factory.cli._ceo_helpers as h

        repo = self._repo(tmp_path)
        (repo / "a.txt").write_text("their work in progress")
        h._begin_in_place_run(repo)
        assert "already has uncommitted changes" in capsys.readouterr().err
        h._end_in_place_run(repo)

    def test_every_exit_path_runs_the_check(self):
        import inspect

        import factory.cli._ceo_helpers as h

        src = inspect.getsource(h)
        assert src.count("print_resume_hint(project_path)") == src.count(
            "_end_in_place_run(project_path)\n"
        ) == 3


class TestInPlaceBackupAndWiring:
    """An in-place run edits files git does not track, so `git diff` can't show or undo
    the edit: the workflows are copied first, and the launchers call begin and end."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        import factory.cli._ceo_helpers as h

        monkeypatch.setattr(h, "_IN_PLACE_START", None)
        monkeypatch.delenv("FACTORY_IN_PLACE", raising=False)
        yield

    def test_the_workflows_are_backed_up_and_the_end_says_how_to_compare(
        self, tmp_path, capsys
    ):
        import factory.cli._ceo_helpers as h

        repo = TestInPlaceRunGuard._repo(tmp_path)
        wf = _write_project_workflow(repo, "growth-curve")
        original = wf.read_text()
        h._begin_in_place_run(repo)
        assert "workflows-before-run" in capsys.readouterr().err
        wf.write_text("# edited by the run\n")
        assert (repo / h.IN_PLACE_BACKUP / wf.name).read_text() == original
        h._end_in_place_run(repo)
        err = capsys.readouterr().err
        assert f"diff -r {h.IN_PLACE_BACKUP} .factory/workflows" in err
        assert "WARNING" not in err

    def test_a_second_run_replaces_the_older_backup(self, tmp_path):
        import factory.cli._ceo_helpers as h

        repo = TestInPlaceRunGuard._repo(tmp_path)
        wf = _write_project_workflow(repo, "growth-curve")
        h._begin_in_place_run(repo)
        h._end_in_place_run(repo)
        wf.write_text("# second version\n")
        (repo / h.IN_PLACE_BACKUP / "stale.py").write_text("# stale\n")
        h._begin_in_place_run(repo)
        backup = repo / h.IN_PLACE_BACKUP
        assert (backup / wf.name).read_text() == "# second version\n"
        assert not (backup / "stale.py").exists()
        h._end_in_place_run(repo)

    def test_the_launcher_begins_and_ends_the_in_place_run(self, tmp_path):
        """Through `factory ceo` (create runs only in the foreground; --headless is refused)."""
        from tests.test_cli import _mock_foreground

        from factory.cli import main

        _write_project_workflow(tmp_path, "growth-curve")
        with _mock_foreground(), \
                patch("factory.worktree.create_worktree") as cw, \
                patch("factory.cli._ceo_helpers._create_runs_in_place", return_value=True), \
                patch("factory.cli._ceo_helpers._begin_in_place_run") as begin, \
                patch("factory.cli._ceo_helpers._end_in_place_run") as end:
            rc = main(["ceo", str(tmp_path), "--mode", "create",
                       "--focus", "growth-curve: add a doubling-time column"])
        assert rc == 0
        cw.assert_not_called()
        begin.assert_called_once()
        end.assert_called_once()


def test_a_package_mode_update_says_where_the_mode_is_composed(tmp_path: Path):
    from factory.workflow.registry import WorkflowRegistry

    entry = MagicMock(source="package", path="<package>")
    with patch.object(WorkflowRegistry, "discover", return_value={"build-composed": entry}):
        task = _build_ceo_task(
            tmp_path, "create", create_description="add a lint step",
            update_existing_mode="build-composed",
        )
    assert "composed in code from packages" in task
    assert "factory/workflow/packages.py" in task
