"""`factory contained --target openshell` — plan composition, policy, dry run, checks.

The discipline from the k8s tests carries over: never reach a real gateway. Client
construction, the SDK, and the CLI are patched at their seams (`openshell.connect`,
`openshell.run_cli`, `import_sdk`), the same way `test_contained_k8s.py` stubs
`list_contexts` rather than shelling out to a real `oc`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.cli import contained as cli
from factory.contained import openshell
from factory.contained.errors import ContainedError
from factory.contained.openshell_prereq import openshell_checks


def parse(argv: list[str]) -> argparse.Namespace:
    """Parse a `factory contained ...` command line the way the real CLI does."""
    parser = argparse.ArgumentParser(prog="factory")
    sub = parser.add_subparsers(dest="command")
    cli.build_contained_parser(sub)
    return parser.parse_args(["contained", *argv])


def interpret(argv: list[str]) -> argparse.Namespace:
    args = parse(argv)
    cli.interpret(cli._PARSER, args)
    return args


# A stand-in for the generated proto messages, so policy tests run without the optional SDK.
@dataclass(frozen=True)
class _FakeRule:
    name: str
    endpoints: list = field(default_factory=list)
    binaries: list = field(default_factory=list)


@dataclass
class _FakePolicy:
    version: int = 1
    filesystem: object = None
    process: object = None
    network_policies: dict = field(default_factory=dict)
    loaded: dict | None = None       # set by the json_format fake in the replacement test


class _FakePb2:
    """The surface `build_default_policy` touches, shaped like the real sandbox_pb2."""

    NetworkEndpoint = dict
    NetworkBinary = dict
    NetworkPolicyRule = _FakeRule
    FilesystemPolicy = dict
    ProcessPolicy = dict
    SandboxPolicy = _FakePolicy


def _fake_protos():
    class _OpenShellPb2:
        pass

    return _OpenShellPb2, _FakePb2


# --------------------------------------------------------------------------------------------
# Command surface
# --------------------------------------------------------------------------------------------


def test_the_target_is_a_choice() -> None:
    args = interpret(["--target", "openshell", "--", "study", "/tmp"])
    assert args.target == "openshell"
    assert args.policy is None
    assert args.gateway is None


def test_sandbox_names_fit_the_gateway_budget() -> None:
    """The gateway enforces names at create time as an INVALID_ARGUMENT; deriving a compliant
    name here turns that into something a user never sees. Long stems truncate, the hash never
    does, and illegal characters (underscore, dot, uppercase) collapse like podman's slugs."""
    from factory.contained.openshell import MAX_SANDBOX_NAME, sandbox_name

    short = sandbox_name(Path("/code/rta"))
    assert short.startswith("rta-") and len(short) <= MAX_SANDBOX_NAME

    long = sandbox_name(Path("/code/a-really-long-project-stem-here"))
    assert len(long) == MAX_SANDBOX_NAME
    # the hash suffix is never the truncated part: 6 hex chars survive at the end
    assert long[-7] == "-" and len(long.rsplit("-", 1)[-1]) == 6

    weird = sandbox_name(Path("/code/My_Project.v2"))
    assert len(weird) == MAX_SANDBOX_NAME
    assert weird.startswith("my-project-v-") and "._" not in weird

    # Non-ASCII letters are legal to `str.isalnum` and illegal to the gateway; they collapse.
    unicode_stem = sandbox_name(Path("/code/café"))
    assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in unicode_stem)


def test_user_supplied_names_are_validated_before_any_copy() -> None:
    """`--name` reaches the gateway verbatim; a name it would reject is a parse-time error,
    not a create-time traceback after the workspace was already copied."""
    from factory.contained.openshell import validate_sandbox_name

    assert validate_sandbox_name("fine-name") == "fine-name"
    assert validate_sandbox_name("run-1") == "run-1"      # digits are legal — a precedence bug
    # in an earlier formulation rejected them while the error message claimed otherwise
    assert validate_sandbox_name("run2") == "run2"
    assert validate_sandbox_name("") == ""
    with pytest.raises(ContainedError, match="at most 19"):
        validate_sandbox_name("a" * 20)
    with pytest.raises(ContainedError, match="lowercase alphanumeric"):
        validate_sandbox_name("Bad_Name")
    with pytest.raises(ContainedError, match="lowercase alphanumeric"):
        validate_sandbox_name("café")                     # non-ASCII is not in the gateway's set


def test_every_generated_name_passes_validation() -> None:
    """The invariant the review's precedence bug broke in spirit: `sandbox_name` output must
    always satisfy `validate_sandbox_name`, so a user can copy a name out of `ls` into `--name`
    (and a future change to either half of the alphabet cannot drift from the other)."""
    from factory.contained.openshell import sandbox_name, validate_sandbox_name

    for stem in ("rta", "My_Project.v2", "café", "a-very-long-project-stem", "digits-123"):
        generated = sandbox_name(Path(f"/code/{stem}"))
        assert validate_sandbox_name(generated) == generated


def test_policy_flag_is_rejected_outside_openshell() -> None:
    with pytest.raises(SystemExit):
        interpret(["--policy", "p.yaml", "--", "study", "/tmp"])


def test_gateway_flag_is_rejected_outside_openshell() -> None:
    with pytest.raises(SystemExit):
        interpret(["--gateway", "prod", "--", "study", "/tmp"])


def test_division_is_refused_on_openshell() -> None:
    """The build plane is egress this target's policy exists to deny; refused at parse time,
    not three provisioning steps in."""
    with pytest.raises(SystemExit):
        interpret(["--target", "openshell", "--division", "--", "study", "/tmp"])


def test_local_flags_are_rejected_on_openshell() -> None:
    with pytest.raises(SystemExit):
        interpret(["--target", "openshell", "--mount", "/tmp", "--", "study", "/tmp"])


# --------------------------------------------------------------------------------------------
# Interactive payloads (no terminal in a sandbox)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["ceo", "resume", "tmux"],
)
def test_interactive_payloads_are_rejected_without_an_opt_out(command: str) -> None:
    """An OpenShell run is a detached process writing a log — it has no terminal for an
    interactive CEO to sit at, and a design-mode gate would wait forever on an answer nobody
    can give. Refused at parse time, the way `--division` is, rather than three provisioning
    steps in. `ceo` (interactive by default), `run`/`agent` with --tmux-persist, and
    `resume`/`tmux` (which *are* a terminal) are each refused for their own reason."""
    with pytest.raises(SystemExit):
        interpret(["--target", "openshell", "--", command, "/tmp/project"])


@pytest.mark.parametrize("command", ["resume", "tmux"])
def test_terminal_commands_have_no_opt_out(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    """`resume` and `tmux` need a PTY by their nature, so no flag makes them acceptable —
    and the refusal must say so rather than suggesting a flag the command does not accept:
    advice like `re-run with --headless` would pass the check and then die inside the
    sandbox, because neither command takes --headless."""
    with pytest.raises(SystemExit):
        interpret(["--target", "openshell", "--", command, "/tmp/project", "--headless"])
    err = capsys.readouterr().err
    assert f"There is no headless form of `{command}`" in err
    # The only --headless in the message belongs to the suggested `ceo` replacement, never
    # to the refused command itself.
    assert f"-- {command} <path> --headless" not in err


@pytest.mark.parametrize(
    "payload",
    [
        # `run` is headless by default — refusing it was a false refusal, and worse, the
        # error's --headless advice would then die inside the sandbox: `run` accepts no
        # such flag.
        ["run", "/tmp/project"],
        ["agent", "researcher", "--task", "t", "--project", "/tmp/project"],
        # `ceo` opts out three ways, not one.
        ["ceo", "/tmp/project", "--headless"],
        ["ceo", "/tmp/project", "--bg"],
        ["ceo", "/tmp/project", "--mode", "design", "--auto-approve"],
    ],
)
def test_headless_forms_are_not_refused(payload: list[str]) -> None:
    """Every payload that runs without a terminal passes the check untouched — the false
    refusals (run, agent, ceo --bg, ceo --auto-approve) each burned a full provisioning
    cycle before failing, or never failed at all."""
    args = interpret(["--target", "openshell", "--", *payload])
    assert args.factory_args == payload


@pytest.mark.parametrize("command", ["run", "agent"])
def test_persist_flags_make_headless_commands_interactive(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run` and `agent` are headless by default; --tmux-persist is the one way to make
    them interactive, and the refusal must name dropping it — not a --headless these
    commands do not accept."""
    with pytest.raises(SystemExit):
        interpret(["--target", "openshell", "--", command, "/tmp/project", "--tmux-persist"])
    assert "Drop --tmux-persist" in capsys.readouterr().err


def test_non_interactive_payloads_pass_untouched() -> None:
    """Only the commands whose defining property is interactivity are looked at; everything
    else is verbatim by contract."""
    args = interpret(["--target", "openshell", "--", "study", "/tmp/project"])
    assert args.factory_args == ["study", "/tmp/project"]


def test_interactive_payloads_are_fine_on_the_other_targets() -> None:
    """local and k8s run in tmux, which supplies the terminal — the check is openshell's."""
    args = interpret(["--", "ceo", "/tmp/project"])
    assert args.factory_args[0] == "ceo"


# --------------------------------------------------------------------------------------------
# Gateway selection on every CLI invocation
# --------------------------------------------------------------------------------------------


def test_every_cli_composer_carries_the_selected_gateway() -> None:
    """`--gateway X` must reach the CLI calls, not just the SDK: with Y active, an upload
    without the flag lands in a same-named sandbox on Y (the name is deterministic per
    project) — worse than a clean failure. Omitted, not empty, when no gateway was named."""
    from factory.contained.openshell import (
        build_attach_argv,
        build_download_argv,
        build_profile_export_argv,
        build_provider_list_argv,
        build_upload_argv,
    )

    tarball = Path("/tmp/upload.tar.gz")
    for argv in (
        build_upload_argv("rta-abc123", tarball, gateway="gw"),
        build_download_argv("rta-abc123", ".", Path("/tmp/dest"), gateway="gw"),
        build_attach_argv("rta-abc123", gateway="gw"),
        build_provider_list_argv("gw"),
        build_profile_export_argv("claude-code", gateway="gw"),
    ):
        assert argv[:3] == ["openshell", "--gateway", "gw"], argv
    for argv in (
        build_upload_argv("rta-abc123", tarball),
        build_download_argv("rta-abc123", ".", Path("/tmp/dest")),
        build_attach_argv("rta-abc123"),
        build_provider_list_argv(),
    ):
        assert "--gateway" not in argv, argv


def test_the_plan_carries_the_gateway_for_the_run_path(tmp_path: Path) -> None:
    from factory.cli.contained_openshell import _build_plan

    project = tmp_path / "rta"
    project.mkdir()
    args = _plan_args(project, "--gateway", "gw")
    plan = _build_plan(args, _workspace(project), "rta-abc123", {}, {})
    assert plan.gateway == "gw"


# --------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------


def test_the_default_policy_grants_exactly_the_intended_egress() -> None:
    """The allowlist is the security core; a test that enumerates it is the review artifact.
    Anyone adding an endpoint here changes what untrusted code can reach, and this failing
    diff is where that fact becomes visible.

    Inference egress is granted by the *provider*, not this policy: the gateway synthesizes a
    `_provider_claude_code` rule when the provider attaches, and a second rule for the same
    endpoints is a hard create-time failure (ambiguity validation), so its absence here is
    itself a security property to pin."""
    with patch.object(openshell, "import_protos", _fake_protos):
        policy = openshell.build_default_policy()

    assert policy.version == 1
    rules = policy.network_policies
    assert set(rules) == {"python_packages"}

    pypi = rules["python_packages"]
    assert {e["host"] for e in pypi.endpoints} == {"pypi.org", "files.pythonhosted.org"}
    assert all(e["port"] == 443 for e in pypi.endpoints)
    # uv only, deliberately: binary matching resolves /proc/<pid>/exe, and pip runs as the
    # Python interpreter, so a pip path never matches — while allowing the *interpreter* would
    # grant PyPI egress to every script the agent writes.
    assert {b["path"] for b in pypi.binaries} == {"/usr/local/bin/uv", "/usr/bin/uv"}

    # No inference endpoints of our own: the attached provider is the only route to them,
    # and the binary restriction on that route comes from the provider profile.
    for rule in rules.values():
        assert all(
            e["host"] not in {"api.anthropic.com", "statsig.anthropic.com", "sentry.io"}
            for e in rule.endpoints
        )

    # The workspace is writable; the process runs as a stated non-root identity, because an
    # omitted one falls back to the image's OCI USER whose primary GID is 0 — and the gateway
    # hard-rejects any workload identity containing GID 0. The image's mode recipe (`o=u` on
    # the container home and /workspace only) keeps its writable paths open to this gid, so
    # the other targets are unaffected.
    assert policy.filesystem == {"include_workdir": True, "read_only": ["/opt/factory"]}
    assert policy.process == {"run_as_user": "1001", "run_as_group": "1001"}


def test_a_policy_file_replaces_the_default_never_merges() -> None:
    """WYSIWYG: the file is the whole policy. The one rename between the documented YAML
    schema and the proto is applied, so users write what the OpenShell docs show."""
    import yaml

    document = {
        "version": 1,
        "filesystem_policy": {"include_workdir": True, "read_only": ["/usr"]},
        "process": {"run_as_user": "1500"},
    }
    path = Path("/tmp/does-not-matter.yaml")

    import types

    class _JsonFormat:
        ParseError = ValueError

        @staticmethod
        def ParseDict(translated, policy):
            assert translated["filesystem"] == document["filesystem_policy"]
            assert "filesystem_policy" not in translated
            policy.loaded = translated
            return policy

    fake_json_format = _JsonFormat
    with patch.object(openshell, "import_protos", _fake_protos), \
         patch.dict(
             "sys.modules",
             {
                 "google": types.ModuleType("google"),
                 "google.protobuf": types.ModuleType("google.protobuf"),
                 "google.protobuf.json_format": fake_json_format,
             },
         ), \
         patch.object(Path, "read_text", return_value=yaml.safe_dump(document)), \
         patch.object(Path, "is_file", return_value=True):
        policy = openshell.load_policy(path)

    assert policy.loaded["process"] == {"run_as_user": "1500"}


def test_a_malformed_policy_file_is_an_error_naming_the_file() -> None:
    path = Path("/tmp/policy.yaml")
    with patch.object(openshell, "import_protos", _fake_protos), \
         patch.object(Path, "read_text", return_value="just a string"), \
         patch.object(Path, "is_file", return_value=True):
        with pytest.raises(ContainedError, match="policy file"):
            openshell.load_policy(path)


def test_the_builders_construct_real_proto_messages_when_the_sdk_is_installed() -> None:
    """The fakes above prove composition, not that the messages exist or accept these fields —
    every hard create-time rejection found in live verification (names, labels, endpoint
    ambiguity, GID 0) was exactly that class of drift. When the optional SDK is installed
    (locally via `uv sync --extra contained-openshell`, in CI via the opt-in job), build the
    real `SandboxSpec`/`SandboxPolicy` and assert the fields the gateway validates actually
    carry what the builders meant to send. Skipped, not failed, when the extra is absent."""
    pytest.importorskip("openshell", reason="contained-openshell extra not installed")

    policy = openshell.build_default_policy()
    assert policy.version == 1
    assert policy.filesystem.include_workdir is True
    assert list(policy.filesystem.read_only) == ["/opt/factory"]
    rule = policy.network_policies["python_packages"]
    assert {e.host for e in rule.endpoints} == {"pypi.org", "files.pythonhosted.org"}
    assert {b.path for b in rule.binaries} == {"/usr/local/bin/uv", "/usr/bin/uv"}

    plan = openshell.OpenShellPlan(
        name="rta-abc123", image="img", project_dir="/workspace/rta", env={},
        labels={"factory.contained": "true"}, provider="claude-code", policy_path=None,
        run_command="factory study /workspace/rta", factory_command="factory study /workspace/rta",
    )
    spec = openshell.build_spec(plan, policy)
    assert list(spec.providers) == ["claude-code"]
    assert list(spec.command) == list(openshell.IDLE_COMMAND)
    # Protobuf copies the sub-message on assignment, so identity is off the table — the
    # serialized forms are the honest comparison.
    assert spec.policy.SerializeToString() == policy.SerializeToString()


# --------------------------------------------------------------------------------------------
# Plan composition and dry run
# --------------------------------------------------------------------------------------------


def _plan_args(project: Path, *flags: str) -> argparse.Namespace:
    args = parse(["--target", "openshell", *flags, "--", "study", str(project)])
    cli.interpret(cli._PARSER, args)
    return args


def _workspace(project: Path, run_id: str = "rta-abc123"):
    from factory.contained.workspace import plan_workspace

    return plan_workspace(project, run_id, self_contained=True)


def test_dry_run_prints_the_real_steps_and_provisions_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same contract as the other targets: dry-run prints what the real path would execute —
    the SDK operations as described steps, the CLI parts as exact argv — and contacts
    nothing, not even a gateway."""
    from factory.cli.contained_openshell import run_openshell

    project = tmp_path / "rta"
    project.mkdir()
    args = _plan_args(project)

    with patch.dict(os.environ, {"FACTORY_CONTAINED_DRY_RUN": "1"}, clear=False):
        code = run_openshell(args)

    assert code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "nothing is provisioned" in out
    assert "factory default" in out                       # states which policy applies
    assert "openshell sandbox upload" in out              # exact CLI argv for the transfer
    for step in ("[create]", "[upload]", "[extract]", "[assert:workdir]", "[run]"):
        assert step in out, f"dry run does not show the {step} step"


def test_the_environment_carries_configuration_and_names_secret_smuggling(
    tmp_path: Path,
) -> None:
    """The provider is the only supported credential route; a secret-looking key in the plan
    env means --forward/--env was used to smuggle one past that boundary, and the plan warns
    rather than quietly forwarding it."""
    from factory.cli.contained_openshell import _build_plan

    project = tmp_path / "rta"
    project.mkdir()
    ws = _workspace(project)
    args = _plan_args(project, "--env", "ANTHROPIC_API_KEY=sk-ant-smuggled")

    plan = _build_plan(args, ws, "rta-abc123", {"ANTHROPIC_API_KEY": "sk-ant-smuggled"}, {})

    assert plan.env["ANTHROPIC_API_KEY"] == "sk-ant-smuggled"   # escape hatch still wins
    assert any("provider" in warning for warning in plan.warnings)


def test_a_secret_forwarded_into_the_plan_is_redacted_in_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run output is a log line and an evidence file as often as a terminal; a real key
    must never appear in it."""
    from factory.cli.contained_openshell import run_openshell

    project = tmp_path / "rta"
    project.mkdir()
    args = _plan_args(project, "--env", "ANTHROPIC_API_KEY=sk-ant-real-value")

    with patch.dict(os.environ, {"FACTORY_CONTAINED_DRY_RUN": "1"}, clear=False):
        run_openshell(args)

    out = capsys.readouterr().out
    assert "sk-ant-real-value" not in out
    assert "ANTHROPIC_API_KEY=<redacted>" in out


def test_policy_path_resolves_and_a_missing_file_fails_before_any_copy(tmp_path: Path) -> None:
    from factory.cli.contained_openshell import _build_plan

    project = tmp_path / "rta"
    project.mkdir()
    ws = _workspace(project)
    args = _plan_args(project, "--policy", str(tmp_path / "absent.yaml"))

    with pytest.raises(ContainedError, match="no such file"):
        _build_plan(args, ws, "rta-abc123", {}, {})


def test_labels_carry_the_factory_join_keys(tmp_path: Path) -> None:
    """`ls` selects on `factory.contained=true`; the project hash and source path make the
    listing useful without reaching into the gateway again."""
    from factory.cli.contained_openshell import _build_plan

    project = tmp_path / "rta"
    project.mkdir()
    ws = _workspace(project)
    args = _plan_args(project)

    plan = _build_plan(args, ws, "rta-abc123", {}, {})

    assert plan.labels["factory.contained"] == "true"
    assert plan.labels["factory.name"] == "rta-abc123"
    assert plan.labels["factory.project"]
    # A source path is not a legal label value on a gateway (alphanumeric/-/_/. only), so it is
    # deliberately absent — `workspace_for` recovers it from the local workspace copy instead.
    assert "factory.source" not in plan.labels
    assert plan.project_dir == f"{openshell.WORKSPACE_ROOT}/rta"


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def test_every_check_degrades_to_a_fix_carrying_failure(tmp_path: Path) -> None:
    """A machine with nothing installed gets a list of what is missing, not a traceback."""
    with patch("shutil.which", return_value=None), \
         patch("factory.contained.openshell.connect", side_effect=Exception("no gateway")):
        checks = openshell_checks()

    assert [c.name for c in checks] == [
        "openshell_cli", "openshell_sdk", "openshell_gateway", "runtime_image",
        "openshell_provider",
    ]
    for check in checks:
        if not check.ok:
            assert check.fix, f"{check.name} fails without a fix"


# --------------------------------------------------------------------------------------------
# The provider profile: shipped, and read back
# --------------------------------------------------------------------------------------------


def test_the_shipped_provider_profile_is_the_stock_profile_plus_two_edits() -> None:
    """Snapshot of openshell_claude_code.yaml: exactly one endpoint (api.anthropic.com —
    the stock profile's telemetry endpoints are inherited egress for everything the Bash
    tool runs, i.e. exfiltration channels for untrusted code), and the binary the runtime
    image actually executes (the kernel matches /proc/<pid>/exe, which follows the npm
    symlink, so /usr/local/bin/claude never matches). If the image's claude install moves,
    this snapshot is what forces the profile to move with it."""
    from factory.contained.openshell import load_provider_profile, provider_profile_path

    assert provider_profile_path().exists()
    profile = load_provider_profile()
    assert profile["id"] == "claude-code"
    endpoints = {(e["host"], e["port"]) for e in profile["endpoints"]}
    assert endpoints == {("api.anthropic.com", 443)}
    assert profile["binaries"] == [
        "/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"
    ]
    env_vars = [v for c in profile["credentials"] for v in c["env_vars"]]
    assert "ANTHROPIC_API_KEY" in env_vars


def test_the_provider_fix_names_the_shipped_profile_and_never_a_manual_edit() -> None:
    """PROVIDER_FIX is a single import of the file the factory ships — no curl, no $EDITOR.
    A fix that edits by hand is a mitigation that exists only on the machines that followed
    the instructions correctly."""
    from factory.contained.openshell import PROVIDER_FIX, provider_profile_path

    assert f"openshell profile import -f {provider_profile_path()}" in PROVIDER_FIX
    assert "$EDITOR" not in PROVIDER_FIX
    assert "curl" not in PROVIDER_FIX


@dataclass
class _ProcResult:
    returncode: int
    stdout: str = ""


class _NoopClient:
    def close(self) -> None:
        pass


def _provider_check_with(
    providers_json: str, profile_json: str | None, profile_rc: int = 0
):
    """Run `_provider_check` against a stubbed gateway: one provider listing, optionally one
    profile export. `None` for the profile means the export failed outright."""
    import factory.contained.openshell_prereq as prereq

    def fake_run(argv, *, timeout=30):
        if "provider" in argv and "list" in argv:
            return _ProcResult(0, providers_json)
        if "profile" in argv and "export" in argv:
            if profile_json is None:
                return _ProcResult(profile_rc, "")
            return _ProcResult(profile_rc, profile_json)
        raise AssertionError(f"unexpected argv: {argv}")

    with patch.object(prereq, "_run", side_effect=fake_run), \
         patch.object(openshell, "connect", return_value=_NoopClient()):
        return prereq._provider_check(None)


def test_a_stock_profile_import_does_not_pass_the_provider_check() -> None:
    """The check reads the gateway's profile back and compares it with the shipped one —
    importing the stock profile (telemetry endpoints still allowed) must fail, because the
    endpoint list is the security mitigation and a name-only check cannot see it."""
    stock = {
        "id": "claude-code",
        "endpoints": [
            {"host": "api.anthropic.com", "port": 443},
            {"host": "statsig.anthropic.com", "port": 443},
            {"host": "sentry.io", "port": 443},
        ],
        "binaries": ["/usr/bin/claude", "/usr/local/bin/claude"],
    }
    check = _provider_check_with(
        '{"providers": [{"name": "claude-code", "type": "claude-code"}]}',
        json.dumps(stock),
    )
    assert not check.ok
    assert "statsig.anthropic.com" in check.detail
    assert "sentry.io" in check.detail
    assert "binaries" in check.detail          # the unresolved symlink path is named too


def test_the_matching_profile_passes_and_reports_its_shape() -> None:
    """A profile that matches the shipped one — endpoints and binaries both — passes, with
    a detail that states what was verified, never the credential material."""
    from factory.contained.openshell import load_provider_profile

    live = load_provider_profile()
    check = _provider_check_with(
        '{"providers": [{"name": "claude-code", "type": "claude-code"}]}',
        json.dumps(live),
    )
    assert check.ok
    assert "api.anthropic.com only" in check.detail


def test_the_check_follows_the_provider_to_its_actual_profile() -> None:
    """A provider's type is the profile it was created from — the profile named
    'claude-code' is not necessarily the one in use, so the readback follows the provider's
    own reference rather than assuming the name."""
    from factory.contained.openshell import load_provider_profile

    live = load_provider_profile()
    check = _provider_check_with(
        '{"providers": [{"name": "claude-code", "type": "claude-code-vm2"}]}',
        json.dumps(live),
    )
    assert check.ok
    assert "claude-code-vm2" in check.detail


def test_an_unreadable_profile_fails_with_the_fix() -> None:
    """A provider whose profile cannot be read back is a failure carrying the import
    command, not a pass on the strength of the name alone."""
    check = _provider_check_with(
        '{"providers": [{"name": "claude-code", "type": "claude-code"}]}',
        None,
        profile_rc=1,
    )
    assert not check.ok
    assert check.fix


def test_a_missing_provider_still_fails_with_the_import_fix() -> None:
    check = _provider_check_with('{"providers": []}', None)
    assert not check.ok
    assert "no provider named 'claude-code'" in check.detail
    assert check.fix


# --------------------------------------------------------------------------------------------
# Listing and lifecycle seams
# --------------------------------------------------------------------------------------------


def test_listing_selects_on_the_factory_label_and_maps_phases() -> None:
    @dataclass
    class _Status:
        phase: int = 2
        exit_code: int | None = None

    @dataclass
    class _Ref:
        name: str
        workspace: str = "default"
        status: _Status = field(default_factory=_Status)
        labels: dict = field(default_factory=dict)

        @property
        def phase(self) -> int:
            return self.status.phase

    class _Pager:
        def __init__(self, items):
            self._items = items

        def all(self):
            return self._items

    class _Client:
        def list(self, **kwargs):
            assert kwargs["label_selector"] == "factory.contained=true"
            return _Pager([
                _Ref(name="rta-abc123", labels={"factory.project": "abc", "factory.source": "/p"}),
                _Ref(name="other", labels={"factory.project": "x"}),
            ])

    with patch.object(openshell, "connect", return_value=_Client()):
        entries = openshell.list_runtimes()

    assert entries[0]["name"] == "rta-abc123"
    assert entries[0]["phase"] == "running"
    assert entries[0]["project"] == "abc"

    # The full mapping, phase number -> word. 3 (error) once vanished into a trailing
    # comment on the `running` entry, so an errored sandbox listed as "unknown" — every
    # phase the factory can observe is asserted here to keep that from regressing.
    for number, word in (
        (1, "provisioning"),
        (2, "running"),
        (3, "error"),
        (4, "deleting"),
        (6, "stopping"),
        (7, "stopped"),
        (8, "starting"),
        (9, "completed"),
        (99, "unknown"),
    ):
        assert openshell._phase_name(number) == word, f"phase {number}"


def test_a_launch_failure_is_distinguished_from_a_command_failure() -> None:
    """`exec` reports -1 when the command did not run at all; the caller's message must not
    claim the probe executed and failed."""
    from factory.contained.openshell import exec_argv

    @dataclass
    class _Result:
        exit_code: int
        stdout: str = ""
        stderr: str = ""

    class _Session:
        def exec(self, argv, **kwargs):
            return _Result(exit_code=-1)

    with pytest.raises(ContainedError, match="did not execute at all"):
        exec_argv(_Session(), ["pwd"])


# --------------------------------------------------------------------------------------------
# Run model: detached launch, log/pid/exit artifacts, liveness
# --------------------------------------------------------------------------------------------


def test_the_run_launch_detaches_logs_and_records_its_own_lifecycle() -> None:
    """The launch script is the target's `build_tmux_launch` replacement, so its properties are
    the contract: detached (nohup + background), output to the run log, pid to the pidfile, exit
    code to the exit file — and *no tmux anywhere*, because a sandbox cannot allocate a PTY.
    This test also executes the script, because a composed-but-unparsed shell string is how the
    `&`-vs-`cd` precedence bug shipped: the pidfile landed in the launcher's cwd, not the run's.
    """
    import subprocess as sp

    from factory.contained.openshell import build_run_launch

    script = build_run_launch("/workspace/rta", "factory study /workspace/rta")
    assert "nohup" in script and "&" in script          # detached: exec returns immediately
    for token in (".factory/run.log", ".factory/run.pid", ".factory/run.exit"):
        assert token in script, f"the run's {token} artifact is missing from the launch"
    assert "rm -f" in script                             # a stale prior run's artifacts go first
    assert "tmux" not in script                          # cannot work in a sandbox (#749)
    assert sp.run(["sh", "-n"], input=script, text=True).returncode == 0

    # Execute it for real: the three artifacts must exist, in the *run's* directory.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "proj"
        project.mkdir()
        run = build_run_launch(str(project), "echo hi-from-run; sleep 0.1")
        sp.run(["sh", "-c", run], check=True, cwd=tmp, capture_output=True)
        deadline = time.monotonic() + 5
        while not (project / ".factory/run.exit").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        artifacts = sorted(p.name for p in (project / ".factory").iterdir())
        assert artifacts == ["run.exit", "run.log", "run.pid"]
        assert (project / ".factory/run.log").read_text().strip() == "hi-from-run"
        assert (project / ".factory/run.exit").read_text().strip() == "0"


def test_the_run_launch_deletes_a_prior_runs_stale_artifacts() -> None:
    """`sync` writes a finished run's artifacts into the local workspace copy, and the copy is
    what the next run uploads — so without the `rm -f`, `attach` on the new run opens with a
    phantom `[factory exited N]` from the old one."""
    import subprocess as sp
    import tempfile

    from factory.contained.openshell import build_run_launch

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "proj"
        factory = project / ".factory"
        factory.mkdir(parents=True)
        (factory / "run.exit").write_text("7")
        (factory / "run.pid").write_text("999999")
        (factory / "run.log").write_text("a finished run's log")

        run = build_run_launch(str(project), "echo fresh-run")
        sp.run(["sh", "-c", run], check=True, cwd=tmp, capture_output=True)
        deadline = time.monotonic() + 5
        while not (factory / "run.exit").exists() or "fresh-run" not in (factory / "run.log").read_text():
            if time.monotonic() > deadline:
                raise AssertionError("the fresh run never started")
            time.sleep(0.05)
        assert (factory / "run.exit").read_text().strip() == "0"
        assert "finished run" not in (factory / "run.log").read_text()


def test_the_pack_excludes_a_prior_runs_artifacts(tmp_path: Path) -> None:
    """The upload must not carry a previous run's interface files (see the launch test above
    for why) — but a project file that happens to share a name outside `.factory` still packs."""
    import tarfile

    from factory.cli.contained_openshell import _pack
    from factory.contained.workspace import Workspace

    project = tmp_path / "rta"
    (project / ".factory").mkdir(parents=True)
    (project / ".factory" / "run.exit").write_text("0")
    (project / ".factory" / "run.log").write_text("old")
    (project / "docs").mkdir()
    (project / "docs" / "run.log").write_text("a project file, not a run artifact")

    ws = Workspace(source=project, path=project, kind="copy", branch="")
    tarball = _pack(ws, "rta-abc123")

    with tarfile.open(tarball) as archive:
        names = archive.getnames()
    assert "rta/.factory/run.exit" not in names
    assert "rta/.factory/run.log" not in names
    assert "rta/docs/run.log" in names


def test_the_attach_argv_follows_the_log_and_never_touches_tmux() -> None:
    """Attach is a read-only log follow through the CLI's outer PTY (which the supervisor
    allocates *before* the Landlock boundary, so it works where a nested one cannot)."""
    from factory.contained.openshell import build_attach_argv

    argv = build_attach_argv("rta-abc123")
    assert argv[:6] == ["openshell", "sandbox", "exec", "-n", "rta-abc123", "--tty"]
    script = argv[-1]
    assert "tail -n 200 -f .factory/run.log" in script
    assert "run.exit" in script                            # the finished run's stamp
    assert "tmux" not in script


def test_the_run_environment_is_unbuffered() -> None:
    """The run's stdout is a file, where Python block-buffers; unbuffered output is what keeps
    `attach`'s log tail distinguishable from a hang."""
    from factory.cli.contained_openshell import _build_plan

    project = tmp_project()
    args = _plan_args(project)
    plan = _build_plan(args, _workspace(project), "rta-abc123", {}, {})
    assert plan.env["PYTHONUNBUFFERED"] == "1"
    # setdefault semantics: an explicit --env PYTHONUNBUFFERED=0 is the user's to make
    plan2 = _build_plan(
        _plan_args(project, "--env", "PYTHONUNBUFFERED=0"), _workspace(project), "rta-abc123",
        {}, {"PYTHONUNBUFFERED": "0"},
    )
    assert plan2.env["PYTHONUNBUFFERED"] == "0"


def test_the_run_environment_disables_claude_nonessential_traffic() -> None:
    """Claude's telemetry/statsig/sentry traffic is egress the provider profile deliberately
    does not allow — Claude honours this switch, so its own non-essential traffic stops rather
    than being denied connection-by-connection. setdefault semantics, like PYTHONUNBUFFERED."""
    from factory.cli.contained_openshell import _build_plan

    project = tmp_project()
    args = _plan_args(project)
    plan = _build_plan(args, _workspace(project), "rta-abc123", {}, {})
    assert plan.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    plan2 = _build_plan(
        _plan_args(project, "--env", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=0"),
        _workspace(project), "rta-abc123",
        {}, {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0"},
    )
    assert plan2.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "0"


def tmp_project() -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "rta"
    path.mkdir()
    return path


# --------------------------------------------------------------------------------------------
# Lifecycle routing
# --------------------------------------------------------------------------------------------


def test_lifecycle_subcommands_route_to_the_openshell_handlers() -> None:
    """attach/rm/sync take their target from --target; the handlers are reached with the
    gateway and confirmation flags passed through."""
    import factory.contained.lifecycle as lifecycle_mod
    from factory.contained.lifecycle import dispatch_lifecycle

    for subcommand, handler_name in (
        ("attach", "attach"), ("rm", "remove"), ("sync", "sync")
    ):
        args = argparse.Namespace(
            subcommand=subcommand,
            target="openshell",
            name="rta-abc123",
            namespace=None,
            gateway="gw",
            yes=True,
        )
        with patch.object(lifecycle_mod, handler_name, return_value=0) as handler:
            assert dispatch_lifecycle(args) == 0
        handler.assert_called_once()
        # The gateway rides along either positionally (attach/sync) or as a keyword (remove,
        # whose signature keeps `assume_yes`/`interactive` keyword-only after it).
        call = handler.call_args
        assert call.args[:3] == ("rta-abc123", "openshell", None)
        assert call.kwargs.get("gateway", call.args[3] if len(call.args) > 3 else None) == "gw"


def test_rm_deletes_the_sandbox_and_keeps_the_workspace_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deleting the sandbox must not delete the local workspace copy — the work is kept, and
    the merge hint says how to get it back."""
    from factory.contained.lifecycle import remove
    from factory.contained.runtimes import Runtime

    runtime = Runtime(
        name="rta-abc123", target="openshell", project="abc", state="finished"
    )
    with patch("factory.contained.lifecycle.list_runtimes", return_value=([runtime], [], [])), \
         patch("factory.contained.openshell.remove_runtime") as deleter, \
         patch("factory.contained.lifecycle.workspace_for", return_value=None):
        assert remove("rta-abc123", "openshell", assume_yes=True) == 0
    deleter.assert_called_once_with("rta-abc123", gateway=None)
    assert "sandbox deleted" in capsys.readouterr().out

