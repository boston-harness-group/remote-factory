"""`ls`, `attach`, `rm`, `sync` — and the label check that stands in front of all of them.

Two properties are load-bearing and both are easy to break silently. A command must refuse a name
the factory did not create, because a tool that acts on resources it did not make invites the user
to assume it manages them. And "the container is running" is not "the run is running" — the
container's PID 1 outlives the run on purpose, so the session is what answers the question a user
actually asked.

Every podman call is mocked. A leak here would be a live `podman ps` against the developer's engine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.contained import lifecycle
from factory.contained.lifecycle import (
    LifecycleError,
    Runtime,
    attach,
    dispatch_lifecycle,
    list_runtimes,
    local_runtimes,
    reap_stale,
    remove,
    render_table,
    sync,
    workspace_for,
)
from factory.podman import LABEL_CONTAINED, LABEL_NAME, LABEL_PROJECT, LABEL_SOURCE


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _entry(name: str = "rta-abc123", state: str = "running", **labels: str) -> dict[str, object]:
    return {
        "Names": [name],
        "State": state,
        "Created": 1_700_000_000,
        "Labels": {LABEL_CONTAINED: "true", LABEL_PROJECT: "abc123", **labels},
    }


@pytest.fixture()
def contained_root(tmp_path: Path):
    root = tmp_path / "contained-home"
    with patch.dict(os.environ, {"FACTORY_CONTAINED_HOME": str(root)}, clear=False):
        yield root


@pytest.fixture(autouse=True)
def _never_reach_a_cluster():
    """`ls` with no target consults the cluster only when the machine has used one; these tests
    must not depend on whether the developer's machine has."""
    with patch("factory.contained.usage.uses", return_value=False):
        yield  # type: ignore[misc]


def _args(**fields: object) -> argparse.Namespace:
    return argparse.Namespace(**fields)


# --------------------------------------------------------------------------------------------
# Listing: only ours, and the run's state rather than the container's
# --------------------------------------------------------------------------------------------


def test_a_container_without_the_factory_label_is_not_listed() -> None:
    """`build_ps_argv` already filters on the label; this is the second, independent filter site,
    and it is the one that survives someone loosening the first."""
    entries = [_entry(), {"Names": ["someone-elses"], "State": "running", "Labels": {}}]
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps(entries))), \
         patch("factory.contained.lifecycle._run_state", return_value="running"):
        names = [r.name for r in local_runtimes()]
    assert names == ["rta-abc123"]


def test_a_running_container_whose_panes_are_all_dead_reports_finished() -> None:
    """This is the case that tells a user a run is live and then gives them nothing to attach to."""
    with patch("factory.contained.lifecycle.subprocess.run",
               side_effect=[_completed(json.dumps([_entry()])), _completed("1\n")]):
        assert local_runtimes()[0].state == "finished"


def test_a_running_container_with_one_live_pane_reports_running() -> None:
    with patch("factory.contained.lifecycle.subprocess.run",
               side_effect=[_completed(json.dumps([_entry()])), _completed("0\n1\n")]):
        assert local_runtimes()[0].state == "running"


def test_a_container_that_is_not_running_is_reported_as_podman_saw_it() -> None:
    """No session probe is possible against a stopped container, and inventing one would report
    `finished` for a container that never started."""
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps([_entry(state="exited")]))) as run:
        assert local_runtimes()[0].state == "exited"
    assert run.call_count == 1


def test_a_session_probe_that_cannot_run_leaves_the_container_state_alone() -> None:
    """Degrading to podman's own answer is honest; guessing `finished` is not."""
    with patch("factory.contained.lifecycle.subprocess.run",
               side_effect=[_completed(json.dumps([_entry()])),
                            subprocess.TimeoutExpired(cmd="podman", timeout=10)]):
        assert local_runtimes()[0].state == "running"


def test_no_tmux_session_at_all_reports_finished() -> None:
    with patch("factory.contained.lifecycle.subprocess.run",
               side_effect=[_completed(json.dumps([_entry()])),
                            _completed("", returncode=1, stderr="no server")]):
        assert local_runtimes()[0].state == "finished"


def test_the_source_label_survives_into_the_listing() -> None:
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps([_entry(state="exited", **{LABEL_SOURCE: "/x"})]))):
        assert local_runtimes()[0].source == "/x"


def test_a_missing_podman_binary_names_the_fix_rather_than_raising_oserror() -> None:
    with patch("factory.contained.lifecycle.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(LifecycleError, match="not installed"):
            local_runtimes()


def test_an_unreachable_engine_reports_only_the_first_line_of_its_error() -> None:
    """podman's connection failure runs to five lines; a table with five lines of preamble in it
    is not a table."""
    stderr = "Error: unable to connect\nplease check\nthat the machine is running\n"
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=125, stderr=stderr)):
        with pytest.raises(LifecycleError) as excinfo:
            local_runtimes()
    assert "unable to connect" in str(excinfo.value)
    assert "please check" not in str(excinfo.value)


def test_a_leading_blank_line_in_podmans_error_is_skipped() -> None:
    """podman's connection failure routinely starts with a newline; reporting that as the error
    gives the user a table with a blank note under it."""
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=125, stderr="\n\nError: unable to connect")):
        with pytest.raises(LifecycleError, match="unable to connect"):
            local_runtimes()


def test_an_engine_failure_with_no_stderr_at_all_still_says_something() -> None:
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=125)):
        with pytest.raises(LifecycleError, match="no details given"):
            local_runtimes()


def test_output_that_is_not_json_is_reported_as_such() -> None:
    with patch("factory.contained.lifecycle.subprocess.run", return_value=_completed("not json")):
        with pytest.raises(LifecycleError, match="isn't JSON"):
            local_runtimes()


def test_a_json_object_instead_of_a_list_is_an_empty_listing_not_a_crash() -> None:
    with patch("factory.contained.lifecycle.subprocess.run", return_value=_completed("{}")):
        assert local_runtimes() == []


def test_an_explicit_local_target_surfaces_the_engine_failure() -> None:
    """Asked for `local` specifically, "your engine is down" is the answer — not an empty table."""
    with patch("factory.contained.lifecycle.local_runtimes",
               side_effect=LifecycleError("cannot reach podman")):
        with pytest.raises(LifecycleError):
            list_runtimes("local")


def test_an_unasked_for_local_failure_becomes_a_note_not_an_exception() -> None:
    """`ls` spans both targets, so one broken target must not hide the other's runtimes."""
    with patch("factory.contained.lifecycle.local_runtimes",
               side_effect=LifecycleError("cannot reach podman")):
        runtimes, notes, unconfigured = list_runtimes(None)
    assert runtimes == []
    assert notes and notes[0].startswith("local:")


def test_an_explicit_cluster_target_surfaces_its_failure() -> None:
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[]), \
         patch("factory.contained.k8s.cluster_runtimes",
               side_effect=LifecycleError("cluster unreachable")):
        with pytest.raises(LifecycleError):
            list_runtimes("k8s")


def test_a_cluster_the_user_has_used_but_cannot_reach_becomes_a_note() -> None:
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[]), \
         patch("factory.contained.usage.uses", return_value=True), \
         patch("factory.contained.k8s.has_cluster_context", return_value=True), \
         patch("factory.contained.k8s.cluster_runtimes",
               side_effect=LifecycleError("cluster unreachable")):
        runtimes, notes, unconfigured = list_runtimes(None)
    assert notes and notes[0].startswith("k8s:")
    assert unconfigured == []


def test_a_machine_with_no_kubeconfig_reports_the_cluster_unconfigured_not_broken() -> None:
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[]), \
         patch("factory.contained.usage.uses", return_value=True), \
         patch("factory.contained.k8s.has_cluster_context", return_value=False), \
         patch("factory.contained.lifecycle.openshell_runtimes", return_value=[]):
        runtimes, notes, unconfigured = list_runtimes(None)
    assert notes == []
    assert unconfigured == ["k8s"]


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def test_an_empty_fleet_suggests_how_to_start_one() -> None:
    assert "factory contained -- ceo" in render_table([])


def test_an_empty_fleet_with_a_note_does_not_claim_nothing_is_running() -> None:
    """Reporting "no runtimes" for "could not reach the engine" tells a user their fleet is empty
    when it is merely invisible."""
    body = render_table([], notes=["local: cannot reach podman"])
    assert "No contained runtimes" not in body
    assert "cannot reach podman" in body


def test_the_table_carries_name_target_project_age_and_state() -> None:
    created = datetime.now(timezone.utc) - timedelta(hours=3)
    table = render_table([
        Runtime(name="rta-abc123", target="local", project="abc123", state="running",
                created=created)
    ])
    assert "NAME" in table and "rta-abc123" in table and "3h" in table


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(minutes=7), "7m"),
        (timedelta(hours=5), "5h"),
        (timedelta(days=2), "2d"),
        (timedelta(seconds=-30), "?"),
    ],
)
def test_ages_are_rendered_at_one_significant_unit(delta: timedelta, expected: str) -> None:
    """A clock skewed into the future renders `?` rather than a negative age — subtracting into a
    negative would otherwise print something like `-1s`."""
    created = datetime.now(timezone.utc) - delta
    table = render_table([Runtime("n", "local", "p", "running", created=created)])
    assert expected in table


def test_an_age_under_a_minute_is_rendered_in_seconds() -> None:
    created = datetime.now(timezone.utc) - timedelta(seconds=5)
    table = render_table([Runtime("n", "local", "p", "running", created=created)])
    # Not the exact number: a scheduling stall between these two `now()` calls would move it.
    assert any(f"{n}s" in table for n in range(5, 15))


def test_a_runtime_with_no_creation_time_renders_a_question_mark() -> None:
    assert "?" in render_table([Runtime("n", "local", "p", "running")])


def test_a_naive_timestamp_is_read_as_utc_rather_than_crashing_the_table() -> None:
    """Subtracting a naive datetime from an aware one raises, and it would raise *inside* `ls` —
    taking the whole listing down over one badly-formatted field."""
    created = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=4)
    assert "4m" in render_table([Runtime("n", "local", "p", "running", created=created)])


# --------------------------------------------------------------------------------------------
# Which states count as active — the guard in front of every destructive operation
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["exited", "stopped", "created", "dead", "succeeded", "failed"])
def test_terminal_states_are_inactive(state: str) -> None:
    assert not Runtime("n", "local", "p", state).active


@pytest.mark.parametrize("state", ["running", "Pending", "ContainerCreating", "", "something-new"])
def test_anything_unrecognised_is_treated_as_active(state: str) -> None:
    """The safe default for a check guarding a delete: a state we have never seen must not be read
    as "nothing is happening"."""
    assert Runtime("n", "local", "p", state).active


# --------------------------------------------------------------------------------------------
# attach
# --------------------------------------------------------------------------------------------


def test_attaching_to_a_name_the_factory_did_not_create_is_refused(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([], [], [])):
        assert attach("someone-elses", "local") == 1
    assert "not a runtime" in capsys.readouterr().err


def test_attaching_to_a_stopped_container_points_at_the_workspace_instead(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """The work is not lost when the container is — and that is the first thing the user wants."""
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])):
        assert attach("rta-abc123", "local") == 1
    err = capsys.readouterr().err
    assert "sync rta-abc123" in err and "rm rta-abc123" in err


def test_attaching_to_a_finished_run_offers_the_shell_and_the_sync(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """The container is up but the session is gone. Raw tmux answers "no sessions", which is not
    something a user can act on."""
    runtime = Runtime("rta-abc123", "local", "abc123", "finished")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])):
        assert attach("rta-abc123", "local") == 1
    err = capsys.readouterr().err
    assert "podman exec -it rta-abc123" in err
    assert "no sessions" not in err


def test_attaching_locally_goes_through_tmux_in_the_container() -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "running")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.subprocess.call", return_value=0) as call:
        assert attach("rta-abc123", "local") == 0
    argv = call.call_args.args[0]
    assert argv[:2] == ["podman", "exec"] and "tmux attach" in " ".join(argv)


def test_attaching_to_a_pod_goes_through_the_cluster_exec() -> None:
    runtime = Runtime("rta-abc123", "k8s", "abc123", "running")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.k8s.build_pod_attach_argv", return_value=["oc", "exec"]), \
         patch("factory.contained.lifecycle.subprocess.call", return_value=0) as call:
        assert attach("rta-abc123", "k8s", "ns") == 0
    assert call.call_args.args[0] == ["oc", "exec"]


# --------------------------------------------------------------------------------------------
# rm
# --------------------------------------------------------------------------------------------


def test_removing_a_name_the_factory_did_not_create_is_refused() -> None:
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([], [], [])):
        assert remove("someone-elses", "local", assume_yes=True) == 1


def test_removing_an_active_runtime_non_interactively_refuses_rather_than_hanging(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """An unanswerable prompt in a CI job is a hang, and a hang is worse than a refusal that names
    the flag."""
    runtime = Runtime("rta-abc123", "local", "abc123", "running")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.subprocess.run") as run:
        assert remove("rta-abc123", "local", assume_yes=False, interactive=False) == 1
    run.assert_not_called()
    assert "--yes" in capsys.readouterr().err


def test_declining_the_prompt_leaves_the_runtime_alone(
    capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "running")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("builtins.input", return_value="n"), \
         patch("factory.contained.lifecycle.subprocess.run") as run:
        assert remove("rta-abc123", "local", assume_yes=False, interactive=True) == 1
    run.assert_not_called()
    assert "was not deleted" in capsys.readouterr().err


def test_confirming_the_prompt_removes_it(contained_root: Path) -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "running")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("builtins.input", return_value="yes"), \
         patch("factory.contained.lifecycle.subprocess.run", return_value=_completed()), \
         patch("factory.contained.division.stop_recorded", return_value=False):
        assert remove("rta-abc123", "local", assume_yes=False, interactive=True) == 0


def test_a_failed_removal_propagates_podmans_exit_code(
    contained_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=2, stderr="container is in use")):
        assert remove("rta-abc123", "local", assume_yes=True) == 2
    assert "container is in use" in capsys.readouterr().err


def test_removing_a_run_also_stops_the_host_side_division(
    contained_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The division server is a host process the run depends on and is deliberately detached from
    the command that started it, so nothing else ever ends it."""
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.subprocess.run", return_value=_completed()), \
         patch("factory.contained.division.stop_recorded", return_value=True) as stop:
        assert remove("rta-abc123", "local", assume_yes=True) == 0
    stop.assert_called_once_with("rta-abc123")
    assert "division endpoint stopped" in capsys.readouterr().out


def test_removing_a_run_says_the_work_survives_and_how_to_clean_up_the_repository(
    contained_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The copy is a git worktree registered in the user's own repo with its branch in their refs.
    Deleting only the directory leaves a stale registration that blocks the next run of that name."""
    from factory.contained.workspace import Workspace

    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    ws = Workspace(source=Path("/src/rta"), path=Path("/copy/rta"), kind="worktree",
                   branch="contained/rta-abc123")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.subprocess.run", return_value=_completed()), \
         patch("factory.contained.division.stop_recorded", return_value=False), \
         patch("factory.contained.lifecycle.workspace_for", return_value=ws):
        assert remove("rta-abc123", "local", assume_yes=True) == 0
    out = capsys.readouterr().out
    assert "Your work is kept" in out
    assert "worktree remove" in out and "branch -D contained/rta-abc123" in out


def test_removing_a_pod_goes_through_the_cluster_remover() -> None:
    runtime = Runtime("rta-abc123", "k8s", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.k8s.remove_cluster_runtime", return_value=0) as remover:
        assert remove("rta-abc123", "k8s", "ns", assume_yes=True) == 0
    remover.assert_called_once()


# --------------------------------------------------------------------------------------------
# reap_stale — the automatic path, which is allowed to be silent only when it is safe
# --------------------------------------------------------------------------------------------


def test_a_stale_container_is_reaped_so_the_next_run_of_that_name_is_not_blocked() -> None:
    """Otherwise every later invocation dies on a bare "name already in use" with nothing pointing
    at how to get unstuck."""
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[runtime]), \
         patch("factory.contained.lifecycle.subprocess.run", return_value=_completed()):
        reaped, detail = reap_stale("rta-abc123")
    assert reaped and "was exited" in detail


def test_a_running_container_is_never_reaped_automatically() -> None:
    """A name collision can equally mean "you meant to reattach"."""
    runtime = Runtime("rta-abc123", "local", "abc123", "running")
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[runtime]), \
         patch("factory.contained.lifecycle.subprocess.run") as run:
        reaped, detail = reap_stale("rta-abc123")
    run.assert_not_called()
    assert not reaped and "still active" in detail


def test_a_container_the_factory_did_not_create_is_never_reaped() -> None:
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[]):
        reaped, detail = reap_stale("someone-elses")
    assert not reaped and "not a runtime" in detail


def test_an_unreachable_engine_makes_reaping_report_rather_than_raise() -> None:
    """The caller is already handling a failure; a second exception out of the cleanup path buries
    the first."""
    with patch("factory.contained.lifecycle.local_runtimes",
               side_effect=LifecycleError("cannot reach podman")):
        reaped, detail = reap_stale("rta-abc123")
    assert not reaped and "cannot reach podman" in detail


def test_a_failed_reap_says_so_rather_than_claiming_success() -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.local_runtimes", return_value=[runtime]), \
         patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=2, stderr="in use")):
        reaped, detail = reap_stale("rta-abc123")
    assert not reaped and "in use" in detail


# --------------------------------------------------------------------------------------------
# workspace_for — recovering the source path from the copy, with no manifest
# --------------------------------------------------------------------------------------------


def _worktree_copy(contained_root: Path, pointer: str) -> Path:
    path = contained_root / "rta-abc123" / "rta"
    path.mkdir(parents=True)
    (path / ".git").write_text(pointer)
    return path


def test_a_worktree_copy_yields_the_source_repository_and_its_branch(
    contained_root: Path
) -> None:
    """Nothing persists a run-name-to-source-path manifest; the worktree's `.git` pointer is the
    only record on disk, which is what makes `rm`'s "your work is kept" message possible."""
    _worktree_copy(contained_root, "gitdir: /home/u/code/rta/.git/worktrees/rta-abc123\n")
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("contained/rta-abc123\n")):
        ws = workspace_for("rta-abc123")
    assert ws is not None
    assert ws.source == Path("/home/u/code/rta")
    assert ws.branch == "contained/rta-abc123"


def test_no_directory_for_the_run_yields_nothing(contained_root: Path) -> None:
    assert workspace_for("never-existed") is None


def test_more_than_one_child_directory_is_ambiguous_and_yields_nothing(
    contained_root: Path
) -> None:
    root = contained_root / "rta-abc123"
    (root / "rta").mkdir(parents=True)
    (root / "other").mkdir()
    assert workspace_for("rta-abc123") is None


def test_a_plain_copy_yields_nothing_because_no_source_path_is_recoverable(
    contained_root: Path
) -> None:
    """A non-git source carries no pointer, and guessing a source path would send a user's `rsync
    --merge` at the wrong tree."""
    (contained_root / "rta-abc123" / "rta").mkdir(parents=True)
    assert workspace_for("rta-abc123") is None


def test_a_git_pointer_that_is_not_a_worktree_pointer_yields_nothing(
    contained_root: Path
) -> None:
    _worktree_copy(contained_root, "gitdir: /home/u/code/rta/.git\n")
    assert workspace_for("rta-abc123") is None


def test_a_git_file_with_unexpected_contents_yields_nothing(contained_root: Path) -> None:
    _worktree_copy(contained_root, "not a gitdir pointer\n")
    assert workspace_for("rta-abc123") is None


def test_a_git_pointer_that_cannot_be_read_yields_nothing(contained_root: Path) -> None:
    _worktree_copy(contained_root, "gitdir: /home/u/code/rta/.git/worktrees/rta-abc123\n")
    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        assert workspace_for("rta-abc123") is None


def test_a_worktree_whose_branch_cannot_be_read_yields_nothing(contained_root: Path) -> None:
    """`merge_hint` treats a falsy branch as a plain copy and prints an rsync merge for what is
    actually a worktree — wrong guidance is worse than "not found"."""
    _worktree_copy(contained_root, "gitdir: /home/u/code/rta/.git/worktrees/rta-abc123\n")
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed("", returncode=128)):
        assert workspace_for("rta-abc123") is None


# --------------------------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------------------------


def test_syncing_a_name_the_factory_did_not_create_is_refused() -> None:
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([], [], [])):
        assert sync("someone-elses", "local") == 1


def test_syncing_a_local_run_says_the_work_is_already_here(
    contained_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bind mount, not a transfer — telling a user to "download" it would be a lie."""
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    from factory.contained.workspace import Workspace

    ws = Workspace(source=Path("/src/rta"), path=Path("/copy/rta"), kind="worktree",
                   branch="contained/rta-abc123")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.lifecycle.workspace_for", return_value=ws):
        assert sync("rta-abc123", "local") == 0
    out = capsys.readouterr().out
    assert "already on this machine" in out and "contained/rta-abc123" in out


def test_syncing_a_run_whose_copy_is_gone_says_where_it_looked(
    contained_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = Runtime("rta-abc123", "local", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])):
        assert sync("rta-abc123", "local") == 1
    assert str(contained_root) in capsys.readouterr().err


def test_syncing_a_pod_goes_through_the_cluster_sync() -> None:
    runtime = Runtime("rta-abc123", "k8s", "abc123", "exited")
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.k8s.sync_cluster_runtime", return_value=0) as syncer:
        assert sync("rta-abc123", "k8s", "ns") == 0
    syncer.assert_called_once()


# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------


def test_ls_covers_both_targets_in_one_table(capsys: pytest.CaptureFixture[str]) -> None:
    """A user asking "what is running?" should not have to ask it twice."""
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([], [], [])) as lister:
        assert dispatch_lifecycle(_args(subcommand="ls", target="k8s", namespace=None)) == 0
    assert lister.call_args.args[0] is None


def test_ls_exits_nonzero_when_a_target_could_not_be_listed() -> None:
    """A script wrapping `ls` must not read a dead engine as "nothing running"."""
    with patch("factory.contained.lifecycle.list_runtimes",
               return_value=([], ["local: cannot reach podman"], [])):
        assert dispatch_lifecycle(_args(subcommand="ls", target=None, namespace=None)) == 1


def test_a_lifecycle_error_during_dispatch_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("factory.contained.lifecycle.list_runtimes",
               side_effect=LifecycleError("cannot reach podman")):
        assert dispatch_lifecycle(_args(subcommand="ls", target=None, namespace=None)) == 1
    assert "cannot reach podman" in capsys.readouterr().err


@pytest.mark.parametrize("subcommand", ["attach", "rm", "sync"])
def test_a_subcommand_needing_a_name_and_given_none_exits_two(
    subcommand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(subcommand=subcommand, target="local", namespace=None, name=None)
    assert dispatch_lifecycle(args) == 2
    assert "needs a runtime name" in capsys.readouterr().err


def test_rm_carries_the_yes_flag_and_the_terminal_state_through() -> None:
    args = _args(subcommand="rm", target="local", namespace=None, name="rta-abc123", yes=True)
    with patch("factory.contained.lifecycle.remove", return_value=0) as remover:
        assert dispatch_lifecycle(args) == 0
    assert remover.call_args.kwargs["assume_yes"] is True
    assert "interactive" in remover.call_args.kwargs


def test_attach_and_sync_route_to_their_handlers() -> None:
    for subcommand, target in (("attach", "attach"), ("sync", "sync")):
        args = _args(subcommand=subcommand, target="local", namespace=None, name="rta-abc123")
        with patch(f"factory.contained.lifecycle.{target}", return_value=0) as handler:
            assert dispatch_lifecycle(args) == 0
        handler.assert_called_once_with("rta-abc123", "local", None, None)


def test_an_unrouted_subcommand_exits_two_rather_than_silently_succeeding(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert dispatch_lifecycle(_args(subcommand="teleport", target="local", namespace=None)) == 2
    assert "not implemented" in capsys.readouterr().err


def test_a_created_timestamp_as_an_rfc3339_string_is_accepted() -> None:
    """Older podman builds emit a string under the same key; anything unparseable degrades to `?`
    rather than raising inside a listing."""
    entry = _entry(state="exited")
    entry["Created"] = "2024-01-01T00:00:00Z"
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps([entry]))):
        assert local_runtimes()[0].created == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_an_unparseable_created_timestamp_degrades_to_none() -> None:
    entry = _entry(state="exited")
    entry["Created"] = "last tuesday"
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps([entry]))):
        assert local_runtimes()[0].created is None


def test_a_container_reported_under_name_rather_than_names_is_still_found() -> None:
    entry = {"Name": "rta-abc123", "State": "exited",
             "Labels": {LABEL_CONTAINED: "true", LABEL_NAME: "rta-abc123"}}
    with patch("factory.contained.lifecycle.subprocess.run",
               return_value=_completed(json.dumps([entry]))):
        assert local_runtimes()[0].name == "rta-abc123"


def test_the_lifecycle_module_never_composes_its_own_podman_arguments() -> None:
    """All podman knowledge lives in `factory.podman`, which is what makes the dry-run rendering
    and the real path provably the same commands."""
    source = Path(lifecycle.__file__).read_text()
    assert '"podman"' not in source
