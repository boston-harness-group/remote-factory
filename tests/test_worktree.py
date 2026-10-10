"""Tests for factory/worktree.py — git worktree lifecycle management."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.worktree import (
    _SHARED_SYMLINK_ENTRIES,
    _bootstrap_unborn_repo,
    _finalize_greenfield,
    _has_active_sessions,
    _is_greenfield_run,
    _is_unborn_repo,
    _list_active_worktrees,
    _list_worktrees_with_branches,
    _preserve_telemetry,
    _seed_experiment_factory,
    _sync_backlog_to_main,
    _sync_bootstrap_to_main,
    create_experiment_worktree,
    create_worktree,
    detect_default_branch,
    prune_stale,
    remove_worktree,
)

pytestmark = pytest.mark.real_worktree


@pytest.fixture
def git_project(tmp_path: Path) -> Path:
    """Create a minimal git project with .factory/ directory."""
    project = tmp_path / "project"
    project.mkdir()

    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }

    subprocess.run(["git", "init", "-b", "main"], cwd=project, capture_output=True, check=True)
    (project / ".gitignore").write_text(".factory/\n")
    (project / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=project,
        capture_output=True,
        check=True,
        env=env,
    )

    factory_dir = project / ".factory"
    factory_dir.mkdir()
    (factory_dir / "config.json").write_text("{}")
    (factory_dir / "results.tsv").write_text("id\n")

    return project


class TestCreateWorktree:
    def test_creates_worktree_dir(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)

        assert wt_path.exists()
        assert wt_path.is_dir()
        assert branch.startswith("factory/run-")
        assert wt_path.parent == git_project / ".factory-worktrees"

    def test_worktree_has_selective_factory(self, git_project: Path) -> None:
        wt_path, _ = create_worktree(git_project)

        wt_factory = wt_path / ".factory"
        assert wt_factory.is_dir()
        assert not wt_factory.is_symlink()

        assert (wt_factory / "config.json").is_symlink()
        assert (wt_factory / "results.tsv").is_symlink()

        for subdir in ("strategy", "reviews", "state"):
            d = wt_factory / subdir
            assert d.is_dir()
            assert not d.is_symlink()

    def test_worktree_contains_project_files(self, git_project: Path) -> None:
        wt_path, _ = create_worktree(git_project)

        assert (wt_path / "README.md").exists()
        assert (wt_path / "README.md").read_text() == "hello"

    def test_worktree_branch_is_checked_out(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == branch

    def test_worktree_uses_custom_base_branch(self, git_project: Path) -> None:
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(git_project.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }
        subprocess.run(
            ["git", "checkout", "-b", "develop"],
            cwd=git_project,
            capture_output=True,
            check=True,
        )
        (git_project / "extra.txt").write_text("dev")
        subprocess.run(["git", "add", "."], cwd=git_project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "dev commit"],
            cwd=git_project,
            capture_output=True,
            check=True,
            env=env,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=git_project,
            capture_output=True,
            check=True,
        )

        wt_path, _ = create_worktree(git_project, base_branch="develop")
        assert (wt_path / "extra.txt").exists()

    def test_uses_provided_run_id(self, git_project: Path) -> None:
        uuid_str = "d854881a-800d-44ff-beb5-b9fd77cc3fb9"
        wt_path, branch = create_worktree(git_project, run_id=uuid_str)

        # First 8 chars of UUID should be used
        assert branch == "factory/run-d854881a"
        assert wt_path.name == "run-d854881a"

    def test_run_id_truncated_to_8_chars(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project, run_id="abcdef1234567890")

        assert branch == "factory/run-abcdef12"
        assert wt_path.name == "run-abcdef12"

    def test_short_run_id_used_as_is(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project, run_id="abc")

        assert branch == "factory/run-abc"
        assert wt_path.name == "run-abc"

    def test_multiple_worktrees_coexist(self, git_project: Path) -> None:
        wt1, br1 = create_worktree(git_project)
        wt2, br2 = create_worktree(git_project)

        assert wt1 != wt2
        assert br1 != br2
        assert wt1.exists()
        assert wt2.exists()


class TestRemoveWorktree:
    def test_removes_worktree_completely(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()

        result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert branch not in result.stdout

    def test_safe_on_already_removed_path(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)
        remove_worktree(git_project, wt_path, branch)
        remove_worktree(git_project, wt_path, branch)

    def test_removes_from_worktree_list(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)
        remove_worktree(git_project, wt_path, branch)

        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert str(wt_path) not in result.stdout


class TestTelemetryPreservation:
    def test_trace_id_preserved_on_removal(self, git_project: Path) -> None:
        """trace_id.txt in worktree's real .factory/ is copied to main at teardown."""
        wt_path, branch = create_worktree(git_project)

        trace_id = "test-trace-12345"
        (wt_path / ".factory" / "trace_id.txt").write_text(trace_id)

        main_trace = git_project / ".factory" / "trace_id.txt"
        assert not main_trace.exists()

        remove_worktree(git_project, wt_path, branch)

        assert main_trace.exists()
        assert main_trace.read_text() == trace_id

    def test_no_trace_id_no_error(self, git_project: Path) -> None:
        """Cleanup succeeds when trace_id.txt doesn't exist."""
        wt_path, branch = create_worktree(git_project)

        assert not (wt_path / ".factory" / "trace_id.txt").exists()

        remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()


class TestPruneStale:
    def test_no_op_without_factory_dir(self, tmp_path: Path) -> None:
        project = tmp_path / "no-factory"
        project.mkdir()
        subprocess.run(["git", "init"], cwd=project, capture_output=True, check=True)

        pruned = prune_stale(project)
        assert pruned == []

    def test_cleans_orphaned_directory(self, git_project: Path) -> None:
        wt_dir = git_project / ".factory-worktrees"
        wt_dir.mkdir(parents=True, exist_ok=True)
        orphan = wt_dir / "run-deadbeef"
        orphan.mkdir()
        (orphan / "some_file.txt").write_text("stale")

        pruned = prune_stale(git_project)
        assert len(pruned) >= 1
        assert not orphan.exists()

    def test_preserves_active_worktrees(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)

        pruned = prune_stale(git_project)
        assert wt_path.exists()
        for msg in pruned:
            assert wt_path.name not in msg

    def test_crash_recovery_cleans_all_artifacts(self, git_project: Path) -> None:
        """Simulate a crash: create worktree, delete dir manually, then prune."""
        wt_path, branch = create_worktree(git_project)
        import shutil

        shutil.rmtree(wt_path)

        pruned = prune_stale(git_project)
        assert len(pruned) >= 1

        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert str(wt_path) not in result.stdout


@pytest.fixture
def git_project_master(tmp_path: Path) -> Path:
    """Create a minimal git project with 'master' as the default branch."""
    project = tmp_path / "project"
    project.mkdir()

    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }

    subprocess.run(["git", "init", "-b", "master"], cwd=project, capture_output=True, check=True)
    (project / ".gitignore").write_text(".factory/\n")
    (project / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=project,
        capture_output=True,
        check=True,
        env=env,
    )

    factory_dir = project / ".factory"
    factory_dir.mkdir()
    (factory_dir / "config.json").write_text("{}")
    (factory_dir / "results.tsv").write_text("id\n")

    return project


class TestDetectDefaultBranch:
    def test_detects_main(self, git_project: Path) -> None:
        assert detect_default_branch(git_project) == "main"

    def test_detects_master(self, git_project_master: Path) -> None:
        assert detect_default_branch(git_project_master) == "master"

    def test_local_only_repo_no_origin(self, git_project: Path) -> None:
        result = detect_default_branch(git_project)
        assert result == "main"

    def test_fallback_to_current_branch(self, tmp_path: Path) -> None:
        """Repo with neither 'main' nor 'master' falls back to current HEAD."""
        project = tmp_path / "project"
        project.mkdir()

        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        subprocess.run(
            ["git", "init", "-b", "develop"],
            cwd=project,
            capture_output=True,
            check=True,
        )
        (project / "README.md").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=project,
            capture_output=True,
            check=True,
            env=env,
        )

        assert detect_default_branch(project) == "develop"

    def test_raises_when_no_branch_detectable(self, tmp_path: Path) -> None:
        """A detached HEAD with no main/master and no origin raises instead of
        returning a nonexistent 'main' branch."""
        project = tmp_path / "project"
        project.mkdir()

        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        subprocess.run(
            ["git", "init", "-b", "develop"],
            cwd=project,
            capture_output=True,
            check=True,
        )
        (project / "README.md").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=project,
            capture_output=True,
            check=True,
            env=env,
        )
        # Detach HEAD so abbrev-ref returns "HEAD" and symbolic-ref fails,
        # leaving no detectable default branch.
        subprocess.run(
            ["git", "checkout", "--detach", "HEAD"],
            cwd=project,
            capture_output=True,
            check=True,
            env=env,
        )

        with pytest.raises(RuntimeError, match="Could not determine a default branch"):
            detect_default_branch(project)


class TestCreateWorktreeWithMaster:
    def test_create_worktree_on_master_repo(self, git_project_master: Path) -> None:
        wt_path, branch = create_worktree(git_project_master, base_branch="master")
        try:
            assert wt_path.exists()
            assert branch.startswith("factory/run-")
            assert (wt_path / "README.md").exists()
        finally:
            remove_worktree(git_project_master, wt_path, branch)


class TestSHAResolution:
    def test_create_worktree_resolves_head(self, git_project: Path) -> None:
        """create_worktree('HEAD') resolves to the current commit SHA."""
        expected_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_worktree(git_project, "HEAD")

        wt_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        assert wt_sha == expected_sha

    def test_create_worktree_resolves_amended_head(self, git_project: Path) -> None:
        """After an amend, create_worktree('HEAD') branches from the new commit."""
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(git_project.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        (git_project / "new_file.txt").write_text("amended content")
        subprocess.run(["git", "add", "."], cwd=git_project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "--amend", "--no-edit"],
            cwd=git_project,
            capture_output=True,
            check=True,
            env=env,
        )
        amended_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_worktree(git_project, "HEAD")

        wt_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        assert wt_sha == amended_sha
        assert (wt_path / "new_file.txt").exists()


class TestSymlinkResolution:
    def test_shared_entries_resolve_to_main(self, git_project: Path) -> None:
        """Shared symlinked entries in worktree resolve to main .factory/."""
        wt_path, _ = create_worktree(git_project)
        main_factory = git_project / ".factory"

        for entry in ("config.json", "results.tsv"):
            wt_entry = wt_path / ".factory" / entry
            assert wt_entry.is_symlink()
            assert wt_entry.resolve() == (main_factory / entry).resolve()

    def test_config_readable_through_selective_symlink(self, git_project: Path) -> None:
        wt_path, _ = create_worktree(git_project)

        config_via_wt = (wt_path / ".factory" / "config.json").read_text()
        config_direct = (git_project / ".factory" / "config.json").read_text()
        assert config_via_wt == config_direct


class TestSessionGuard:
    """Tests for _has_active_sessions() and the remove_worktree() guard."""

    def test_active_session_detected(self, tmp_path: Path) -> None:
        sessions = [{"state": "working", "id": "abc"}]
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(sessions),
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is True

    def test_blocked_session_detected(self, tmp_path: Path) -> None:
        sessions = [{"state": "blocked", "id": "def"}]
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(sessions),
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is True

    def test_no_active_sessions(self, tmp_path: Path) -> None:
        sessions = [{"state": "completed", "id": "xyz"}]
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(sessions),
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is False

    def test_empty_session_list(self, tmp_path: Path) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="[]",
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is False

    def test_command_failure_returns_false(self, tmp_path: Path) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is False

    def test_timeout_returns_false(self, tmp_path: Path) -> None:
        with patch(
            "factory.worktree.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ):
            assert _has_active_sessions(tmp_path) is False

    def test_invalid_json_returns_false(self, tmp_path: Path) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not json",
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is False

    def test_non_list_json_returns_false(self, tmp_path: Path) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"state": "working"}',
            stderr="",
        )
        with patch("factory.worktree.subprocess.run", return_value=result):
            assert _has_active_sessions(tmp_path) is False

    def test_remove_worktree_skips_when_active_sessions(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=True):
            remove_worktree(git_project, wt_path, branch)

        assert wt_path.exists()

    def test_remove_worktree_proceeds_when_no_active_sessions(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()


class TestFilelockConcurrency:
    def test_filelock_prevents_concurrent_begin(self, git_project: Path) -> None:
        """Two stores targeting the same .factory/ get sequential IDs under real thread contention."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from factory.store import ExperimentStore

        (git_project / ".factory" / "experiments").mkdir(exist_ok=True)
        (git_project / ".factory" / "results.tsv").write_text(
            "id\ttimestamp\thypothesis\tchange_summary\tissue_number\tpr_number\t"
            "score_before\tscore_after\tdelta\tverdict\tcost_usd\tnotes\tresearch_citations\n"
        )

        def begin_in_thread(hypothesis: str) -> int:
            loop = asyncio.new_event_loop()
            try:
                store = ExperimentStore(git_project)
                return loop.run_until_complete(store.begin(hypothesis))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(begin_in_thread, "hypothesis A")
            fut_b = pool.submit(begin_in_thread, "hypothesis B")
            id_a = fut_a.result()
            id_b = fut_b.result()

        assert id_a != id_b
        assert {id_a, id_b} == {1, 2}


class TestCreateExperimentWorktree:
    def test_creates_experiment_worktree(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_experiment_worktree(git_project, 1, head_sha)

        assert wt_path.exists()
        assert wt_path.is_dir()
        assert branch == "factory/exp-1"
        assert wt_path.name == "exp-1"
        assert wt_path.parent == git_project / ".factory-worktrees"

    def test_experiment_worktree_has_independent_factory_dir(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, _ = create_experiment_worktree(git_project, 2, head_sha)

        wt_factory = wt_path / ".factory"
        assert wt_factory.is_dir()
        assert not wt_factory.is_symlink()
        assert (wt_factory / "config.json").read_text() == "{}"

    def test_experiment_worktree_has_project_files(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, _ = create_experiment_worktree(git_project, 3, head_sha)

        assert (wt_path / "README.md").exists()
        assert (wt_path / "README.md").read_text() == "hello"

    def test_experiment_branch_checked_out(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_experiment_worktree(git_project, 4, head_sha)

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == branch

    def test_multiple_experiment_worktrees_coexist(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt1, br1 = create_experiment_worktree(git_project, 5, head_sha)
        wt2, br2 = create_experiment_worktree(git_project, 6, head_sha)

        assert wt1 != wt2
        assert br1 != br2
        assert wt1.exists()
        assert wt2.exists()

    def test_experiment_worktrees_have_isolated_eval_state(self, git_project: Path) -> None:
        """Parallel experiment worktrees must not share last_eval.json."""
        import json

        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt1, _ = create_experiment_worktree(git_project, 10, head_sha)
        wt2, _ = create_experiment_worktree(git_project, 11, head_sha)

        (wt1 / ".factory" / "last_eval.json").write_text(json.dumps({"total": 0.9}))
        (wt2 / ".factory" / "last_eval.json").write_text(json.dumps({"total": 0.3}))

        score1 = json.loads((wt1 / ".factory" / "last_eval.json").read_text())["total"]
        score2 = json.loads((wt2 / ".factory" / "last_eval.json").read_text())["total"]
        assert score1 == 0.9
        assert score2 == 0.3

    def test_remove_experiment_worktree(self, git_project: Path) -> None:
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_experiment_worktree(git_project, 7, head_sha)
        assert wt_path.exists()

        remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert branch not in result.stdout


class TestPruneStaleExperimentWorktrees:
    def test_cleans_orphaned_exp_directory(self, git_project: Path) -> None:
        """prune_stale handles exp- prefixed directories with correct branch naming."""
        wt_dir = git_project / ".factory-worktrees"
        wt_dir.mkdir(parents=True, exist_ok=True)
        orphan = wt_dir / "exp-99"
        orphan.mkdir()
        (orphan / "some_file.txt").write_text("stale")

        pruned = prune_stale(git_project)
        assert len(pruned) >= 1
        assert not orphan.exists()
        assert any("exp-99" in msg for msg in pruned)


class TestSeedExperimentFactory:
    def test_copies_config_files(self, tmp_path: Path) -> None:
        source = tmp_path / ".factory"
        source.mkdir()
        (source / "config.json").write_text('{"key": "val"}')
        (source / "eval_profile.json").write_text('{"dims": []}')

        dest = tmp_path / "worktree" / ".factory"
        _seed_experiment_factory(source, dest)

        assert dest.is_dir()
        assert not dest.is_symlink()
        assert (dest / "config.json").read_text() == '{"key": "val"}'
        assert (dest / "eval_profile.json").read_text() == '{"dims": []}'

    def test_copies_strategy_directory(self, tmp_path: Path) -> None:
        source = tmp_path / ".factory"
        source.mkdir()
        (source / "strategy").mkdir()
        (source / "strategy" / "current.md").write_text("# strategy")

        dest = tmp_path / "worktree" / ".factory"
        _seed_experiment_factory(source, dest)

        assert (dest / "strategy" / "current.md").read_text() == "# strategy"

    def test_skips_mutable_state(self, tmp_path: Path) -> None:
        source = tmp_path / ".factory"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "results.tsv").write_text("id\n")
        (source / "last_eval.json").write_text('{"total": 0.5}')
        (source / "experiments").mkdir()
        (source / "experiments" / "001").mkdir()

        dest = tmp_path / "worktree" / ".factory"
        _seed_experiment_factory(source, dest)

        assert (dest / "config.json").exists()
        assert not (dest / "results.tsv").exists()
        assert not (dest / "last_eval.json").exists()
        assert not (dest / "experiments").exists()

    def test_replaces_existing_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / ".factory"
        source.mkdir()
        (source / "config.json").write_text("{}")

        dest = tmp_path / "worktree" / ".factory"
        dest.parent.mkdir(parents=True)
        dest.symlink_to(source)
        assert dest.is_symlink()

        _seed_experiment_factory(source, dest)

        assert dest.is_dir()
        assert not dest.is_symlink()

    def test_handles_missing_source(self, tmp_path: Path) -> None:
        source = tmp_path / ".factory"
        dest = tmp_path / "worktree" / ".factory"

        _seed_experiment_factory(source, dest)

        assert dest.is_dir()
        assert list(dest.iterdir()) == []


@pytest.fixture
def unborn_repo(tmp_path: Path) -> Path:
    """Create a git repo with no commits (unborn HEAD)."""
    project = tmp_path / "unborn"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, capture_output=True, check=True)
    factory_dir = project / ".factory"
    factory_dir.mkdir()
    (factory_dir / "config.json").write_text("{}")
    return project


class TestIsUnbornRepo:
    def test_unborn_repo_detected(self, unborn_repo: Path) -> None:
        assert _is_unborn_repo(unborn_repo) is True

    def test_repo_with_commits_not_unborn(self, git_project: Path) -> None:
        assert _is_unborn_repo(git_project) is False


class TestBootstrapUnbornRepo:
    def test_creates_initial_commit(self, unborn_repo: Path) -> None:
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(unborn_repo.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }
        with patch.dict("os.environ", env):
            _bootstrap_unborn_repo(unborn_repo)

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=unborn_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_commit_message_is_factory_bootstrap(self, unborn_repo: Path) -> None:
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(unborn_repo.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }
        with patch.dict("os.environ", env):
            _bootstrap_unborn_repo(unborn_repo)

        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=unborn_repo,
            capture_output=True,
            text=True,
        )
        assert "init (factory bootstrap)" in result.stdout


class TestCreateWorktreeUnbornRepo:
    def test_worktree_created_on_unborn_repo(self, unborn_repo: Path) -> None:
        """create_worktree bootstraps an unborn repo and creates the worktree."""
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(unborn_repo.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }
        with patch.dict("os.environ", env):
            wt_path, branch = create_worktree(unborn_repo)

        assert wt_path.exists()
        assert branch.startswith("factory/run-")

    def test_error_when_branch_missing_on_non_unborn_repo(self, git_project: Path) -> None:
        """Raises RuntimeError if the base branch doesn't exist and repo is not unborn."""
        with pytest.raises(RuntimeError, match="does not exist"):
            create_worktree(git_project, base_branch="nonexistent-branch")


class TestDetectDefaultBranchRemoteHead:
    def test_uses_remote_head_when_available(self, git_project: Path) -> None:
        """detect_default_branch returns the remote HEAD ref when origin is configured."""
        subprocess.run(
            ["git", "remote", "add", "origin", str(git_project)],
            cwd=git_project,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
            cwd=git_project,
            capture_output=True,
            check=True,
        )

        assert detect_default_branch(git_project) == "main"


class TestPruneStaleNonexistentPath:
    def test_returns_empty_for_nonexistent_path(self, tmp_path: Path) -> None:
        gone = tmp_path / "does-not-exist"
        assert prune_stale(gone) == []


class TestSeedExperimentFactoryExistingDir:
    def test_replaces_existing_directory(self, tmp_path: Path) -> None:
        """When dest is an existing directory (not a symlink), it is replaced."""
        source = tmp_path / ".factory"
        source.mkdir()
        (source / "config.json").write_text('{"new": true}')

        dest = tmp_path / "worktree" / ".factory"
        dest.mkdir(parents=True)
        (dest / "stale.txt").write_text("old data")

        _seed_experiment_factory(source, dest)

        assert dest.is_dir()
        assert not dest.is_symlink()
        assert (dest / "config.json").read_text() == '{"new": true}'
        assert not (dest / "stale.txt").exists()


class TestEventEmissionFailure:
    def test_event_error_does_not_propagate(self, git_project: Path) -> None:
        """create_experiment_worktree swallows event emission errors."""
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        with patch("factory.events.emit_event", side_effect=RuntimeError("event bus down")):
            wt_path, branch = create_experiment_worktree(git_project, 99, head_sha)

        assert wt_path.exists()
        assert branch == "factory/exp-99"


class TestCreateWorktreeEventFailure:
    def test_create_worktree_swallows_event_error(self, git_project: Path) -> None:
        with patch("factory.events.emit_event", side_effect=RuntimeError("boom")):
            wt_path, branch = create_worktree(git_project)

        assert wt_path.exists()
        assert branch.startswith("factory/run-")

    def test_remove_worktree_swallows_event_error(self, git_project: Path) -> None:
        wt_path, branch = create_worktree(git_project)

        with patch("factory.events.emit_event", side_effect=RuntimeError("boom")):
            remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()


class TestCreateWorktreeExistingFactory:
    def test_replaces_existing_factory_dir_with_selective_layout(self, tmp_path: Path) -> None:
        """When .factory/ is tracked in git, the worktree replaces it with selective layout."""
        project = tmp_path / "project"
        project.mkdir()

        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        subprocess.run(["git", "init", "-b", "main"], cwd=project, capture_output=True, check=True)
        factory_dir = project / ".factory"
        factory_dir.mkdir()
        (factory_dir / "config.json").write_text("{}")
        subprocess.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial with .factory"],
            cwd=project,
            capture_output=True,
            check=True,
            env=env,
        )

        wt_path, _ = create_worktree(project)

        wt_factory = wt_path / ".factory"
        assert wt_factory.is_dir()
        assert not wt_factory.is_symlink()
        assert (wt_factory / "config.json").is_symlink()


class TestPreserveTelemetryNoFactory:
    def test_no_factory_dir_is_noop(self, git_project: Path) -> None:
        """_preserve_telemetry returns early when worktree has no .factory/."""
        from factory.worktree import _preserve_telemetry

        fake_wt = git_project / "no-factory-here"
        fake_wt.mkdir()

        _preserve_telemetry(fake_wt, git_project)


class TestDetectDefaultBranchFallback:
    def test_fallback_when_all_detection_fails(self, tmp_path: Path) -> None:
        """When every detection method fails, returns 'main'."""
        project = tmp_path / "bare"
        project.mkdir()
        subprocess.run(["git", "init"], cwd=project, capture_output=True, check=True)

        with patch(
            "factory.worktree.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="",
            ),
        ):
            assert detect_default_branch(project) == "main"


class TestDetectDefaultBranchUnborn:
    def test_unborn_repo_returns_branch_via_symbolic_ref(self, unborn_repo: Path) -> None:
        """Unborn repo (no commits) still detects the branch name from symbolic HEAD."""
        result = detect_default_branch(unborn_repo)
        assert result == "main"

    def test_unborn_repo_with_custom_branch(self, tmp_path: Path) -> None:
        """Unborn repo initialized with a non-standard branch name."""
        project = tmp_path / "custom-branch"
        project.mkdir()
        subprocess.run(
            ["git", "init", "-b", "trunk"],
            cwd=project,
            capture_output=True,
            check=True,
        )

        result = detect_default_branch(project)
        assert result == "trunk"


class TestWorktreeRetention:
    """Tests for FACTORY_REMOVE_WORKTREE config and _should_remove_worktree()."""

    def test_remove_worktree_default_removes(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()

    def test_remove_worktree_false_retains(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "false")
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert wt_path.exists()

    def test_remove_worktree_zero_retains(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "0")
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert wt_path.exists()

    def test_remove_worktree_no_retains(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "no")
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with patch("factory.worktree._has_active_sessions", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert wt_path.exists()

    def test_experiment_worktree_always_removed(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "false")
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_project,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        wt_path, branch = create_experiment_worktree(git_project, 5, head_sha)
        assert wt_path.exists()
        assert branch == "factory/exp-5"

        remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()

    def test_retained_emits_event(self, git_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "false")
        wt_path, branch = create_worktree(git_project)
        run_id = branch.removeprefix("factory/run-")

        with (
            patch("factory.worktree._has_active_sessions", return_value=False),
            patch("factory.events.emit_event") as mock_emit,
        ):
            remove_worktree(git_project, wt_path, branch)

        mock_emit.assert_called_once_with(
            git_project,
            "worktree.retained",
            data={
                "run_id": run_id,
                "branch": branch,
                "worktree_path": str(wt_path),
            },
        )

    def test_prune_stale_respects_retention_for_run(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "false")
        wt_dir = git_project / ".factory-worktrees"
        wt_dir.mkdir(parents=True, exist_ok=True)
        orphan = wt_dir / "run-deadbeef"
        orphan.mkdir()
        (orphan / "some_file.txt").write_text("stale")

        pruned = prune_stale(git_project)

        assert orphan.exists()
        assert not any("run-deadbeef" in msg for msg in pruned)

    def test_prune_stale_always_cleans_exp(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACTORY_REMOVE_WORKTREE", "false")
        wt_dir = git_project / ".factory-worktrees"
        wt_dir.mkdir(parents=True, exist_ok=True)
        orphan = wt_dir / "exp-99"
        orphan.mkdir()
        (orphan / "some_file.txt").write_text("stale")

        pruned = prune_stale(git_project)

        assert not orphan.exists()
        assert any("exp-99" in msg for msg in pruned)


class TestSelectiveWorktreeIsolation:
    """Tests for selective symlink layout in CEO run worktrees (issue #1234)."""

    def test_shared_entries_are_symlinks_to_main(self, git_project: Path) -> None:
        factory_dir = git_project / ".factory"
        (factory_dir / "eval_profile.json").write_text("{}")
        (factory_dir / "experiments").mkdir(exist_ok=True)
        (factory_dir / "archive").mkdir(exist_ok=True)
        (factory_dir / "events.jsonl").write_text("")

        wt_path, _ = create_worktree(git_project)
        wt_factory = wt_path / ".factory"

        for entry in _SHARED_SYMLINK_ENTRIES:
            src = factory_dir / entry
            dst = wt_factory / entry
            if src.exists():
                assert dst.is_symlink(), f"{entry} should be a symlink"
                assert dst.resolve() == src.resolve(), f"{entry} should point to main"

    def test_copy_entries_are_independent(self, git_project: Path) -> None:
        agents_dir = git_project / ".factory" / "agents"
        agents_dir.mkdir(exist_ok=True)
        (agents_dir / "builder.md").write_text("# Builder")

        wt_path, _ = create_worktree(git_project)
        wt_agents = wt_path / ".factory" / "agents"

        assert wt_agents.is_dir()
        assert not wt_agents.is_symlink()
        assert (wt_agents / "builder.md").read_text() == "# Builder"

        (wt_agents / "builder.md").write_text("# Modified")
        assert (agents_dir / "builder.md").read_text() == "# Builder"

    def test_per_cycle_dirs_are_fresh_and_empty(self, git_project: Path) -> None:
        strategy_dir = git_project / ".factory" / "strategy"
        strategy_dir.mkdir(exist_ok=True)
        (strategy_dir / "current.md").write_text("# Old strategy")
        (strategy_dir / "observations.md").write_text("# Old obs")

        reviews_dir = git_project / ".factory" / "reviews"
        reviews_dir.mkdir(exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("# Old review")

        wt_path, _ = create_worktree(git_project)
        wt_factory = wt_path / ".factory"

        for subdir in ("strategy", "reviews", "state"):
            d = wt_factory / subdir
            assert d.is_dir()
            assert not d.is_symlink()

        assert not (wt_factory / "strategy" / "current.md").exists()
        assert not (wt_factory / "strategy" / "observations.md").exists()
        assert not (wt_factory / "reviews" / "researcher-latest.md").exists()
        assert list((wt_factory / "state").iterdir()) == []

    def test_backlog_copied_not_symlinked(self, git_project: Path) -> None:
        strategy_dir = git_project / ".factory" / "strategy"
        strategy_dir.mkdir(exist_ok=True)
        (strategy_dir / "backlog.md").write_text("- item 1\n- item 2\n")

        wt_path, _ = create_worktree(git_project)
        wt_backlog = wt_path / ".factory" / "strategy" / "backlog.md"

        assert wt_backlog.exists()
        assert not wt_backlog.is_symlink()
        assert wt_backlog.read_text() == "- item 1\n- item 2\n"

    def test_backlog_synced_back_on_removal(self, git_project: Path) -> None:
        strategy_dir = git_project / ".factory" / "strategy"
        strategy_dir.mkdir(exist_ok=True)
        (strategy_dir / "backlog.md").write_text("- item 1\n")

        wt_path, branch = create_worktree(git_project)
        wt_backlog = wt_path / ".factory" / "strategy" / "backlog.md"
        wt_backlog.write_text("- item 1\n- item 2\n- item 3\n")

        remove_worktree(git_project, wt_path, branch)

        main_backlog = git_project / ".factory" / "strategy" / "backlog.md"
        assert main_backlog.read_text() == "- item 1\n- item 2\n- item 3\n"

    def test_sync_backlog_to_main_skips_symlink(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        wt.mkdir()
        main = tmp_path / "main"
        main.mkdir()

        strategy_dir = wt / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        backlog = strategy_dir / "backlog.md"

        main_strategy = main / ".factory" / "strategy"
        main_strategy.mkdir(parents=True)
        main_backlog = main_strategy / "backlog.md"
        main_backlog.write_text("original")

        backlog.symlink_to(main_backlog)

        _sync_backlog_to_main(wt, main)

        assert main_backlog.read_text() == "original"

    def test_sync_bootstrap_copies_fresh_files(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        main = tmp_path / "main"
        wt_factory = wt / ".factory"
        wt_factory.mkdir(parents=True)
        main.mkdir()

        (wt_factory / "config.json").write_text('{"goal": "test"}')
        (wt_factory / "eval_profile.json").write_text('{"dims": []}')
        (wt / "factory.md").write_text("# Factory")

        _sync_bootstrap_to_main(wt, main)

        assert (main / ".factory" / "config.json").read_text() == '{"goal": "test"}'
        assert (main / ".factory" / "eval_profile.json").read_text() == '{"dims": []}'
        assert (main / "factory.md").read_text() == "# Factory"

    def test_sync_bootstrap_skips_symlinks(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        main = tmp_path / "main"
        wt_factory = wt / ".factory"
        wt_factory.mkdir(parents=True)
        main_factory = main / ".factory"
        main_factory.mkdir(parents=True)

        (main_factory / "config.json").write_text("original")
        (wt_factory / "config.json").symlink_to(main_factory / "config.json")

        _sync_bootstrap_to_main(wt, main)

        assert (main_factory / "config.json").read_text() == "original"

    def test_sync_bootstrap_skips_existing_main_files(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        main = tmp_path / "main"
        wt_factory = wt / ".factory"
        wt_factory.mkdir(parents=True)
        main_factory = main / ".factory"
        main_factory.mkdir(parents=True)

        (wt_factory / "config.json").write_text("new")
        (main_factory / "config.json").write_text("existing")

        _sync_bootstrap_to_main(wt, main)

        assert (main_factory / "config.json").read_text() == "existing"

    def test_sync_bootstrap_noop_without_factory_dir(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        main = tmp_path / "main"
        wt.mkdir()
        main.mkdir()

        _sync_bootstrap_to_main(wt, main)

        assert not (main / ".factory").exists()

    def test_two_worktrees_get_independent_dirs(self, git_project: Path) -> None:
        strategy_dir = git_project / ".factory" / "strategy"
        strategy_dir.mkdir(exist_ok=True)
        (strategy_dir / "backlog.md").write_text("- shared item\n")

        wt1, _ = create_worktree(git_project)
        wt2, _ = create_worktree(git_project)

        (wt1 / ".factory" / "strategy" / "current.md").write_text("# WT1 strategy")
        (wt1 / ".factory" / "reviews" / "researcher-latest.md").write_text("# WT1 review")

        assert not (wt2 / ".factory" / "strategy" / "current.md").exists()
        assert not (wt2 / ".factory" / "reviews" / "researcher-latest.md").exists()

        (wt2 / ".factory" / "strategy" / "current.md").write_text("# WT2 strategy")
        assert (wt1 / ".factory" / "strategy" / "current.md").read_text() == "# WT1 strategy"

    def test_shared_entries_write_to_main(self, git_project: Path) -> None:
        """Appending to symlinked results.tsv writes through to main."""
        wt_path, _ = create_worktree(git_project)

        wt_results = wt_path / ".factory" / "results.tsv"
        with open(wt_results, "a") as f:
            f.write("1\tdata\n")

        main_results = git_project / ".factory" / "results.tsv"
        assert "1\tdata\n" in main_results.read_text()

    def test_preserve_telemetry_works_with_selective_layout(self, git_project: Path) -> None:
        wt_path, _ = create_worktree(git_project)

        (wt_path / ".factory" / "trace_id.txt").write_text("trace-abc")

        main_trace = git_project / ".factory" / "trace_id.txt"
        assert not main_trace.exists()

        _preserve_telemetry(wt_path, git_project)

        assert main_trace.exists()
        assert main_trace.read_text() == "trace-abc"

    def test_missing_shared_entries_skipped(self, git_project: Path) -> None:
        """Shared entries that don't exist in main are silently skipped."""
        assert not (git_project / ".factory" / "archive").exists()
        assert not (git_project / ".factory" / "events.jsonl").exists()

        wt_path, _ = create_worktree(git_project)
        wt_factory = wt_path / ".factory"

        assert not (wt_factory / "archive").exists()
        assert not (wt_factory / "events.jsonl").exists()
        assert (wt_factory / "config.json").is_symlink()

    def test_no_backlog_no_error(self, git_project: Path) -> None:
        """Worktree creation succeeds when main has no backlog.md."""
        assert not (git_project / ".factory" / "strategy" / "backlog.md").exists()

        wt_path, _ = create_worktree(git_project)

        assert (wt_path / ".factory" / "strategy").is_dir()
        assert not (wt_path / ".factory" / "strategy" / "backlog.md").exists()


class TestIsGreenfieldRun:
    def test_greenfield_detected(self, git_project: Path) -> None:
        """Greenfield: main has no factory.md, worktree does."""
        wt_path, _ = create_worktree(git_project)
        assert not (git_project / "factory.md").exists()
        (wt_path / "factory.md").write_text("# Goal\nTest\n")
        assert _is_greenfield_run(wt_path, git_project) is True

    def test_not_greenfield_when_main_has_factory_md(self, git_project: Path) -> None:
        """Not greenfield when main already has factory.md."""
        (git_project / "factory.md").write_text("# Goal\nExisting\n")
        wt_path, _ = create_worktree(git_project)
        (wt_path / "factory.md").write_text("# Goal\nTest\n")
        assert _is_greenfield_run(wt_path, git_project) is False

    def test_not_greenfield_when_worktree_has_no_factory_md(self, git_project: Path) -> None:
        """Not greenfield when worktree has no factory.md."""
        wt_path, _ = create_worktree(git_project)
        assert _is_greenfield_run(wt_path, git_project) is False


class TestFinalizeGreenfield:
    def test_commits_untracked_factory_md(self, git_project: Path) -> None:
        """_finalize_greenfield commits factory.md when untracked."""
        wt_path, branch = create_worktree(git_project)
        (wt_path / "factory.md").write_text("# Goal\nTest project\n")

        result = _finalize_greenfield(wt_path, git_project, branch)

        assert result is True
        ls_files = subprocess.run(
            ["git", "ls-files", "factory.md"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        assert ls_files.stdout.strip() == "factory.md"

    def test_noop_when_already_tracked(self, git_project: Path) -> None:
        """_finalize_greenfield is a no-op when factory.md is already tracked."""
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(git_project.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }
        wt_path, branch = create_worktree(git_project)
        (wt_path / "factory.md").write_text("# Goal\nTest\n")
        subprocess.run(["git", "add", "factory.md"], cwd=wt_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add factory.md"],
            cwd=wt_path,
            capture_output=True,
            check=True,
            env=env,
        )

        result = _finalize_greenfield(wt_path, git_project, branch)
        assert result is True

    def test_returns_false_when_no_factory_md(self, git_project: Path) -> None:
        """_finalize_greenfield returns False when factory.md does not exist."""
        wt_path, branch = create_worktree(git_project)
        result = _finalize_greenfield(wt_path, git_project, branch)
        assert result is False


class TestRemoveWorktreeGreenfield:
    def test_retains_worktree_on_finalization_failure(self, git_project: Path) -> None:
        """remove_worktree retains worktree when greenfield finalization fails."""
        wt_path, branch = create_worktree(git_project)
        (wt_path / "factory.md").write_text("# Goal\nTest\n")

        with patch("factory.worktree._finalize_greenfield", return_value=False):
            remove_worktree(git_project, wt_path, branch)

        assert wt_path.exists()

    def test_proceeds_normally_for_non_greenfield(self, git_project: Path) -> None:
        """remove_worktree proceeds normally for non-greenfield runs (regression)."""
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()

    def test_greenfield_success_removes_worktree(self, git_project: Path) -> None:
        """remove_worktree removes worktree after successful greenfield finalization."""
        wt_path, branch = create_worktree(git_project)
        (wt_path / "factory.md").write_text("# Goal\nTest\n")

        with patch("factory.worktree._finalize_greenfield", return_value=True):
            remove_worktree(git_project, wt_path, branch)

        assert not wt_path.exists()

    def test_no_worktree_mode_skips_finalization(self, git_project: Path) -> None:
        """When worktree_path == project_path, skip greenfield finalization."""
        (git_project / "factory.md").write_text("# Goal\nTest\n")

        with patch("factory.worktree._finalize_greenfield") as mock_finalize:
            remove_worktree(git_project, git_project / "nonexistent", "factory/run-test")

        mock_finalize.assert_not_called()


class TestListWorktreesWithBranches:
    """Test _list_worktrees_with_branches porcelain parser."""

    def test_parses_standard_and_nonstandard_branches(self, tmp_path: Path) -> None:
        """Correctly maps path→branch including non-standard branch names."""
        porcelain = (
            "worktree /home/user/project\n"
            "HEAD abc123\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /home/user/project/.factory-worktrees/run-7935f857\n"
            "HEAD def456\n"
            "branch refs/heads/fix/readme-content-regression\n"
            "\n"
            "worktree /home/user/project/.factory-worktrees/run-55eccbe3\n"
            "HEAD aaa111\n"
            "branch refs/heads/factory/run-2e37f04d\n"
            "\n"
            "worktree /home/user/project/.factory-worktrees/exp-1\n"
            "HEAD bbb222\n"
            "branch refs/heads/factory/exp-1\n"
        )
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=porcelain, stderr=""
        )
        with patch("factory.worktree.subprocess.run", return_value=mock_result):
            mapping = _list_worktrees_with_branches(tmp_path)

        assert mapping == {
            "/home/user/project": "main",
            "/home/user/project/.factory-worktrees/run-7935f857": "fix/readme-content-regression",
            "/home/user/project/.factory-worktrees/run-55eccbe3": "factory/run-2e37f04d",
            "/home/user/project/.factory-worktrees/exp-1": "factory/exp-1",
        }

    def test_detached_head_omitted(self, tmp_path: Path) -> None:
        """Detached-HEAD worktrees (no branch line) are omitted from the result."""
        porcelain = (
            "worktree /home/user/project\n"
            "HEAD abc123\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /home/user/project/.factory-worktrees/run-detached\n"
            "HEAD ccc333\n"
            "detached\n"
            "\n"
            "worktree /home/user/project/.factory-worktrees/run-abcd1234\n"
            "HEAD ddd444\n"
            "branch refs/heads/factory/run-abcd1234\n"
        )
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=porcelain, stderr=""
        )
        with patch("factory.worktree.subprocess.run", return_value=mock_result):
            mapping = _list_worktrees_with_branches(tmp_path)

        assert "/home/user/project/.factory-worktrees/run-detached" not in mapping
        assert mapping["/home/user/project"] == "main"
        assert (
            mapping["/home/user/project/.factory-worktrees/run-abcd1234"]
            == "factory/run-abcd1234"
        )

    def test_empty_output(self, tmp_path: Path) -> None:
        """Empty git output returns empty dict."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("factory.worktree.subprocess.run", return_value=mock_result):
            mapping = _list_worktrees_with_branches(tmp_path)

        assert mapping == {}


class TestWorktreeGCIdleReclaim:
    """Tests for idle git-registered worktree GC in prune_stale()."""

    def test_idle_standard_branch_no_session_is_reclaimed(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Idle worktree on factory/run-* with no active session IS reclaimed."""
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        # Set a very short idle threshold (0 hours = reclaim everything idle)
        monkeypatch.setenv("FACTORY_WORKTREE_IDLE_RECLAIM_HOURS", "0")

        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with (
            patch("factory.worktree._has_active_sessions", return_value=False),
            patch("factory.events.emit_event") as mock_emit,
        ):
            pruned = prune_stale(git_project)

        # Directory should be removed
        assert not wt_path.exists()

        # Should have a GC reclaim message in pruned
        assert any("GC reclaimed" in msg and branch in msg for msg in pruned)

        # git branch -D should have been called for the real branch
        branch_result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert branch not in branch_result.stdout

        # worktree.gc_reclaimed event should be emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][1] == "worktree.gc_reclaimed"
        event_data = call_args[1]["data"] if "data" in call_args[1] else call_args[0][2]
        assert event_data["branch"] == branch
        assert event_data["reclaim_reason"] == "idle_standard_branch_no_session"
        assert "worktree_path" in event_data
        assert "idle_seconds" in event_data

    def test_idle_nonstandard_branch_not_touched(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Idle worktree on a non-standard branch (e.g. fix/...) is NOT touched."""
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        monkeypatch.setenv("FACTORY_WORKTREE_IDLE_RECLAIM_HOURS", "0")

        # Create a worktree, then manually associate it with a non-standard branch
        # in the porcelain mapping
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        # Create a fake mapping where this worktree has a non-standard branch
        fake_branch = "fix/readme-content-regression"
        real_mapping = {str(git_project.resolve()): "main"}
        real_mapping[str(wt_path.resolve())] = fake_branch

        with (
            patch("factory.worktree._list_worktrees_with_branches", return_value=real_mapping),
            patch("factory.worktree._has_active_sessions", return_value=False),
            patch("factory.events.emit_event") as mock_emit,
        ):
            pruned = prune_stale(git_project)

        # Directory should survive
        assert wt_path.exists()

        # No GC reclaim events
        mock_emit.assert_not_called()

        # No GC messages in pruned
        assert not any("GC reclaimed" in msg for msg in pruned)

    def test_active_session_worktree_not_touched(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worktree with active session is NOT touched regardless of age."""
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        monkeypatch.setenv("FACTORY_WORKTREE_IDLE_RECLAIM_HOURS", "0")

        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with (
            patch("factory.worktree._has_active_sessions", return_value=True),
            patch("factory.events.emit_event") as mock_emit,
        ):
            pruned = prune_stale(git_project)

        # Directory should survive
        assert wt_path.exists()

        # No GC events
        mock_emit.assert_not_called()
        assert not any("GC reclaimed" in msg for msg in pruned)

    def test_fresh_worktree_not_touched(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh (non-idle) worktree is NOT touched."""
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        # Use a very high threshold to ensure the worktree is considered fresh
        monkeypatch.setenv("FACTORY_WORKTREE_IDLE_RECLAIM_HOURS", "9999")

        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        with (
            patch("factory.worktree._has_active_sessions", return_value=False),
            patch("factory.events.emit_event") as mock_emit,
        ):
            pruned = prune_stale(git_project)

        # Directory should survive
        assert wt_path.exists()

        # No GC events
        mock_emit.assert_not_called()
        assert not any("GC reclaimed" in msg for msg in pruned)


class TestDetachedHeadWorktreeSurvives:
    """Regression test: a detached-HEAD worktree must NOT be deleted by prune_stale().

    Before this fix, _list_active_worktrees() derived from
    _list_worktrees_with_branches().keys(), which omits detached-HEAD worktrees.
    This caused Sweep 1 (orphan detection) to misclassify a git-registered
    detached-HEAD worktree as an orphan and destroy it — bypassing the idle-age
    and active-session guardrails entirely.
    """

    def test_detached_head_worktree_not_deleted(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A git-registered detached-HEAD worktree with uncommitted work survives prune_stale."""
        monkeypatch.delenv("FACTORY_REMOVE_WORKTREE", raising=False)
        # Conservative threshold: nothing should be reclaimed by idle-GC
        monkeypatch.setenv("FACTORY_WORKTREE_IDLE_RECLAIM_HOURS", "9999")

        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
            "HOME": str(git_project.parent),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        # 1. Create a real factory worktree
        wt_path, branch = create_worktree(git_project)
        assert wt_path.exists()

        # 2. Write an uncommitted file (precious work that must survive)
        precious_file = wt_path / "precious_uncommitted.txt"
        precious_file.write_text("important uncommitted work — must not be destroyed")

        # 3. Detach HEAD inside the worktree (simulates git checkout --detach,
        #    interrupted rebase, or bisect)
        subprocess.run(
            ["git", "checkout", "--detach"],
            cwd=wt_path,
            capture_output=True,
            check=True,
            env=env,
        )

        # Verify the worktree IS registered but has no branch (detached HEAD)
        porcelain = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=git_project,
            capture_output=True,
            text=True,
        ).stdout
        # The worktree path should appear in porcelain output
        assert str(wt_path.resolve()) in porcelain

        # Verify _list_active_worktrees includes the detached worktree
        active = _list_active_worktrees(git_project)
        assert str(wt_path.resolve()) in active

        # Verify _list_worktrees_with_branches does NOT include it (no branch)
        branch_map = _list_worktrees_with_branches(git_project)
        assert str(wt_path.resolve()) not in branch_map

        # 4. Run prune_stale with active sessions patched True (extra safety)
        with (
            patch("factory.worktree._has_active_sessions", return_value=True),
            patch("factory.events.emit_event") as mock_emit,
        ):
            pruned = prune_stale(git_project)

        # 5. Assert: the worktree directory STILL EXISTS
        assert wt_path.exists(), (
            f"Detached-HEAD worktree was destroyed by prune_stale! pruned={pruned}"
        )

        # 6. Assert: the uncommitted file STILL EXISTS
        assert precious_file.exists(), (
            "Uncommitted file in detached-HEAD worktree was destroyed!"
        )
        assert precious_file.read_text() == "important uncommitted work — must not be destroyed"

        # 7. Assert: the branch is NOT deleted
        branch_result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert branch in branch_result.stdout, (
            f"Branch {branch} was deleted even though the worktree is still registered!"
        )

        # 8. No orphan or GC messages for this worktree
        assert not any(wt_path.name in msg for msg in pruned), (
            f"Detached-HEAD worktree appeared in pruned messages: {pruned}"
        )
        mock_emit.assert_not_called()
