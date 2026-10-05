"""The local setup wizard: what it automates, what it refuses to, and how it ends.

`verify` reports; `setup` fixes. Two properties are the whole contract and each is a thing a wizard
usually gets wrong: it must be idempotent (so it is also the way to repair a partial setup), and it
must never act silently. The one step deliberately left to the user is inference — the only step
that touches credential material.

Every podman call is mocked. `_start_machine` and `_image_present` shell out through the module's
`subprocess`, so a leak here would start the developer's podman machine.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.contained.prereq import Check
from factory.contained.setup import _image_present, _start_machine, run_setup


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.fixture(autouse=True)
def contained_root(tmp_path: Path):
    """`run_setup` records which target this machine uses, and that record is a real file under
    the user's home unless it is redirected."""
    root = tmp_path / "contained-home"
    with patch.dict(os.environ, {"FACTORY_CONTAINED_HOME": str(root)}, clear=False):
        yield root


@pytest.fixture(autouse=True)
def _no_engine_calls():
    """Default every seam to "already fine" so each test only patches what it is about."""
    with (
        patch("factory.contained.setup.subprocess.run", return_value=_completed()),
        patch("factory.contained.setup._image_present", return_value=True),
        patch(
            "factory.contained.setup.local_checks",
            return_value=[Check(name="container_engine", ok=True, detail="reachable")],
        ),
    ):
        yield  # type: ignore[misc]


# --------------------------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------------------------


def test_no_target_and_no_terminal_sets_up_the_local_runtime() -> None:
    """Non-interactive means nobody is there to answer, and `local` is the documented default."""
    with patch("factory.contained.k8s_setup.setup_k8s") as k8s:
        assert run_setup(None, interactive=False) == 0
    k8s.assert_not_called()


def test_the_chooser_is_skipped_when_a_target_was_named(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("builtins.input") as ask:
        run_setup("local", interactive=True)
    ask.assert_not_called()


@pytest.mark.parametrize(
    ("answer", "expect_k8s"),
    [("1", False), ("2", True), ("3", True), ("4", False)],
)
def test_the_chooser_maps_each_answer_to_a_target(answer: str, expect_k8s: bool) -> None:
    """`3` is `both` and `4` is `openshell`: openshell was appended rather than inserted, so
    anything scripted against this menu (a piped answer, a documented `echo 3`) keeps meaning
    what it always meant."""
    with (
        patch("builtins.input", return_value=answer),
        patch("factory.contained.k8s_setup.setup_k8s", return_value=0) as k8s,
        patch("factory.contained.setup._setup_openshell", return_value=None),
    ):
        run_setup(None, interactive=True)
    assert k8s.called is expect_k8s


def test_an_unrecognised_answer_falls_back_to_local_rather_than_asking_again() -> None:
    """A wizard that loops on a typo in a non-interactive-adjacent context is a hang."""
    with (
        patch("builtins.input", return_value="banana"),
        patch("factory.contained.k8s_setup.setup_k8s") as k8s,
    ):
        run_setup(None, interactive=True)
    k8s.assert_not_called()


def test_stdin_closed_at_the_prompt_takes_the_default_rather_than_erroring(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pipe, a CI job, or `< /dev/null`. An unanswered prompt must not become a bare `Error:`."""
    with (
        patch("builtins.input", side_effect=EOFError),
        patch("factory.contained.k8s_setup.setup_k8s") as k8s,
    ):
        assert run_setup(None, interactive=True) == 0
    k8s.assert_not_called()
    assert "the default" in capsys.readouterr().out


def test_both_labels_each_half_so_the_output_can_be_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("factory.contained.k8s_setup.setup_k8s", return_value=0):
        run_setup("both", interactive=False)
    out = capsys.readouterr().out
    assert "Local runtime" in out and "Cluster runtime" in out


def test_a_failing_cluster_setup_is_reported_even_when_local_succeeded() -> None:
    """`both` that returns 0 because one half worked would tell a script the setup is complete."""
    with patch("factory.contained.k8s_setup.setup_k8s", return_value=1):
        assert run_setup("both", interactive=False) == 1


def test_a_failing_local_setup_is_reported() -> None:
    with patch(
        "factory.contained.setup.local_checks",
        return_value=[Check(name="container_engine", ok=False, detail="not reachable")],
    ):
        assert run_setup("local", interactive=False) == 1


def test_setup_records_the_target_so_ls_knows_which_ones_to_consult(contained_root: Path) -> None:
    """`ls` only reaches for a cluster the machine has actually set up or used."""
    from factory.contained.usage import used_targets

    with patch("factory.contained.k8s_setup.setup_k8s", return_value=0):
        run_setup("k8s", interactive=False)
    assert used_targets() == ["k8s"]


# --------------------------------------------------------------------------------------------
# The three local steps
# --------------------------------------------------------------------------------------------


def test_every_step_is_numbered_so_working_can_be_told_from_finished(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_setup("local", interactive=False)
    out = capsys.readouterr().out
    assert "1/3" in out and "2/3" in out and "3/3" in out


def test_a_reachable_engine_is_left_alone(capsys: pytest.CaptureFixture[str]) -> None:
    """Idempotence: re-running must change nothing that is already correct."""
    with patch("factory.contained.setup._start_machine") as start:
        run_setup("local", interactive=False)
    start.assert_not_called()
    assert "nothing to do" in capsys.readouterr().out


def test_an_unreachable_engine_starts_the_machine() -> None:
    """On macOS the machine stops quietly and every later error blames podman instead."""
    with (
        patch(
            "factory.contained.setup.local_checks",
            return_value=[Check(name="container_engine", ok=False, detail="not reachable")],
        ),
        patch("factory.contained.setup._start_machine") as start,
    ):
        run_setup("local", interactive=False)
    start.assert_called_once()


def test_an_image_already_present_is_not_pulled_again(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("factory.contained.setup.subprocess.run") as run:
        run_setup("local", interactive=False)
    assert "already present" in capsys.readouterr().out
    assert not [c for c in run.call_args_list if "pull" in c.args[0]]


def test_a_missing_image_is_pulled(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("factory.contained.setup._image_present", return_value=False),
        patch("factory.contained.setup.subprocess.run", return_value=_completed()) as run,
    ):
        run_setup("local", interactive=False)
    assert any(c.args[0][:2] == ["podman", "pull"] for c in run.call_args_list)


def test_a_failed_pull_offers_both_ways_out_rather_than_just_failing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The image may simply not be published yet, and the Containerfile ships in the git repository
    rather than in the installed package — so "build it yourself" needs the clone step too."""
    with (
        patch("factory.contained.setup._image_present", return_value=False),
        patch(
            "factory.contained.setup.subprocess.run", return_value=_completed("", returncode=125)
        ),
    ):
        run_setup("local", interactive=False)
    err = capsys.readouterr().err
    assert "FACTORY_CONTAINED_IMAGE" in err
    assert "git clone" in err and "containers/factory/Containerfile" in err


# --------------------------------------------------------------------------------------------
# The two helpers that touch podman directly
# --------------------------------------------------------------------------------------------


def test_an_image_check_that_cannot_run_answers_no_rather_than_raising() -> None:
    """This runs before the engine has been proven reachable, so it has to tolerate no podman."""
    with patch("factory.contained.setup.subprocess.run", side_effect=FileNotFoundError):
        assert _image_present("img:latest") is False


def test_an_image_check_asks_podman_whether_the_reference_exists() -> None:
    with patch("factory.contained.setup.subprocess.run", return_value=_completed()) as run:
        assert _image_present("img:latest") is True
    assert run.call_args.args[0] == ["podman", "image", "exists", "img:latest"]


def test_with_no_machine_at_all_the_init_command_is_printed_not_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`podman machine init` downloads a VM image and picks resource limits — not something to do
    to someone's machine without asking."""
    with patch("factory.contained.setup.subprocess.run", return_value=_completed("")) as run:
        _start_machine()
    assert "podman machine init" in capsys.readouterr().out
    assert run.call_count == 1


def test_a_stopped_machine_is_started_because_it_mutates_nothing_durable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "factory.contained.setup.subprocess.run",
        side_effect=[_completed("podman-machine-default\n"), _completed()],
    ) as run:
        _start_machine()
    assert run.call_args.args[0] == ["podman", "machine", "start"]
    assert "Starting the podman machine" in capsys.readouterr().out


def test_a_machine_listing_that_fails_prints_the_init_command() -> None:
    with patch(
        "factory.contained.setup.subprocess.run", return_value=_completed("", returncode=125)
    ) as run:
        _start_machine()
    assert run.call_count == 1


def test_no_podman_binary_at_all_leaves_the_machine_step_silent() -> None:
    """The trailing `local_checks()` reports it; a second, weaker message here would just be
    noise ahead of the real one."""
    with patch("factory.contained.setup.subprocess.run", side_effect=FileNotFoundError):
        _start_machine()
