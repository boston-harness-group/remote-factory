"""End-to-end guardrail tests through the real ``WorkflowExecutor`` (INT-002).

Drives the ``toy-guard`` contributed mode with a deterministic stub agent: the
only subprocesses are the local aggregator CLI and ``python -m factory workflow
validate``. No model binary, API key, network or paid service is involved.

See spec §10.2. The stub is a module-level ``fake_agent`` bound to per-test
state via ``functools.partial`` — do **not** use ``FakeAgent.behavior`` keyed on
``node_id``: that hook never receives ``node_id``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from factory.workflow.cli import cmd_workflow
from factory.workflow.contributed.toy_guard.workflow import (
    GUARD_MAX_ITERATIONS,
    GUARDS,
    TOY_AGENT_ID,
    workflow,
)
from factory.workflow.executor import ExecutionResult, WorkflowExecutor
from factory.workflow.guardrails import (
    GUARDRAILS_HEADER,
    guard_dir,
    guard_rules_payload,
    guard_slug,
)
from factory.workflow.primitives import AgentNode
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.skill_export import validate_skill
from factory.workflow.validation import validate_workflow

_STAGE_DIR = guard_dir(TOY_AGENT_ID)
_SEGMENT = _STAGE_DIR.name
_SLUGS = tuple(guard_slug(guard.rule) for guard in GUARDS)
_CHECKER_IDS = tuple(f"guard_{_SEGMENT}_{slug}" for slug in _SLUGS)
_GATE_ID = f"guard_{_SEGMENT}_gate"
_CHECKER_PREFIX = f"guard_{_SEGMENT}_"
_RELOOP_FIX = "adjust the draft accordingly"
_WORKTREE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StubState:
    """Per-run state recorded by the stub agent."""

    scenario: str
    calls: list[str] = field(default_factory=list)
    tasks: dict[str, list[str]] = field(default_factory=dict)
    draft_calls: int = 0
    violations_at_rerun: str | None = None


def _write_draft(project: Path, stub: StubState) -> tuple[str, int]:
    stub.draft_calls += 1
    (project / "draft.md").write_text("Toy draft.\nStatus: ready\n", encoding="utf-8")
    if stub.draft_calls == 2 and stub.violations_at_rerun is None:
        violations = project / _STAGE_DIR / "violations.md"
        if violations.exists():
            stub.violations_at_rerun = violations.read_text(encoding="utf-8")
    return ("draft written", 0)


def _write_verdict(project: Path, node_id: str, stub: StubState) -> tuple[str, int]:
    slug = node_id.removeprefix(_CHECKER_PREFIX)
    if stub.scenario == "crash":
        raise RuntimeError(f"checker {slug} crashed")
    if stub.scenario == "missing":
        # No verdict file: auto_write_outputs persists this prose, and the
        # aggregator classifies the resulting non-JSON file as malformed.
        return ("checker produced no verdict", 0)
    stage = project / _STAGE_DIR
    token = (stage / "token").read_text(encoding="utf-8").strip()
    if stub.scenario == "stale":
        token = "stale-token"
    violated = stub.scenario == "always_violated" or (
        stub.scenario == "reloop_once" and stub.draft_calls == 1 and slug == _SLUGS[0]
    )
    verdict = {
        "guard": slug,
        "token": token,
        "violated": violated,
        "evidence": f"evidence for {slug}",
        "fix": _RELOOP_FIX if violated else "",
    }
    (stage / f"{slug}.json").write_text(json.dumps(verdict), encoding="utf-8")
    return (f"verdict written for {slug}", 0)


async def fake_agent(
    role: str,
    task: str,
    project_path: Path,
    *,
    stub: StubState,
    model: str | None = None,
    timeout: float = 600.0,
    node_id: str | None = None,
    **_extra: Any,
) -> tuple[str, int]:
    """Stub agent implementing every §10.2 scenario for the toy-guard graph."""
    del role, model, timeout
    assert node_id is not None, "the executor always passes node_id"
    project = Path(project_path)
    stub.calls.append(node_id)
    stub.tasks.setdefault(node_id, []).append(task)

    if node_id == TOY_AGENT_ID:
        return _write_draft(project, stub)
    if node_id == "publish":
        (project / "published.md").write_text(
            (project / "draft.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        return ("published draft.md", 0)
    if node_id in _CHECKER_IDS:
        return _write_verdict(project, node_id, stub)
    raise AssertionError(f"unexpected agent node id {node_id!r}")


@dataclass
class Run:
    """Everything a test needs from one executor run."""

    result: ExecutionResult
    stub: StubState
    executor: WorkflowExecutor


async def _run(project: Path, scenario: str) -> Run:
    stub = StubState(scenario=scenario)
    executor = WorkflowExecutor(
        workflow(),
        project,
        agent_fn=partial(fake_agent, stub=stub),
    )
    result = await executor.execute()
    return Run(result=result, stub=stub, executor=executor)


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Isolate registry state for the in-process CLI tests (and restore it after)."""
    WorkflowRegistry.reset()
    yield
    WorkflowRegistry.reset()


class TestToyModeGraph:
    """R-011 — the registered mode expands through ``Package.compile()``."""

    def test_registered_and_expanded(self) -> None:
        WorkflowRegistry.reset()
        wf = WorkflowRegistry.get_workflow("toy-guard")
        assert wf is not None
        assert _GATE_ID in wf.nodes
        for checker_id in _CHECKER_IDS:
            assert checker_id in wf.nodes
        assert validate_workflow(wf) == []
        assert len(wf.nodes) == 8
        assert len(wf.edges) == 6

    def test_guarded_agent_carries_block(self) -> None:
        wf = workflow()
        draft = wf.nodes[TOY_AGENT_ID]
        assert isinstance(draft, AgentNode)
        assert draft.prompt_template.count(GUARDRAILS_HEADER) == 1
        for guard in GUARDS:
            assert guard.rule in draft.prompt_template
        assert draft.guard_max_iterations == GUARD_MAX_ITERATIONS
        assert len(draft.guards) == len(GUARDS)


class TestToyModeExecution:
    """UC-001/002/009/010/012/013 through the real executor with stub agents."""

    async def test_pass(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "pass")
        assert run.result.success, run.result.halt_reason
        assert run.stub.calls == [TOY_AGENT_ID, *_CHECKER_IDS, "publish"]
        assert run.result.nodes_executed == 8
        assert not (tmp_path / _STAGE_DIR / "violations.md").exists()
        assert (tmp_path / _STAGE_DIR / "token").exists()

    async def test_reloop_then_pass(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "reloop_once")
        assert run.result.success, run.result.halt_reason
        assert run.stub.draft_calls == 2
        assert run.executor.iteration_counts == {(_GATE_ID, TOY_AGENT_ID): 1}

        tasks = run.stub.tasks[TOY_AGENT_ID]
        assert len(tasks) == 2
        feedback = tasks[1].split("[Feedback iteration 1]: ", 1)[1]
        assert _SLUGS[0] in feedback
        assert _RELOOP_FIX in feedback
        for guard in GUARDS:
            assert guard.rule not in feedback
        # The prompt itself still carries the binding block by design.
        assert GUARDRAILS_HEADER in tasks[1]

    async def test_violations_md_carries_rule(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "reloop_once")
        assert run.result.success, run.result.halt_reason
        content = run.stub.violations_at_rerun
        assert content is not None, "violations.md was absent at the reloop re-run"
        assert f"## {_SLUGS[0]}" in content
        assert f"- rule: {GUARDS[0].rule.strip()}" in content
        # Last-iteration-only: the successful re-run's reset deletes it.
        assert not (tmp_path / _STAGE_DIR / "violations.md").exists()

    async def test_exhaustion_halts(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "always_violated")
        assert run.result.halted
        assert not run.result.success
        assert run.stub.draft_calls == 3
        assert (
            f"max iterations ({GUARD_MAX_ITERATIONS}) exhausted" in run.result.halt_reason
        )

    async def test_missing_verdict_halts(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "missing")
        assert run.result.halted
        assert run.result.halt_reason.startswith("precheck failed: fail:")
        assert "malformed" in run.result.halt_reason
        assert run.stub.draft_calls == 1

    async def test_stale_token_halts(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "stale")
        assert run.result.halted
        assert "fail:" in run.result.halt_reason
        assert "stale" in run.result.halt_reason
        assert run.stub.draft_calls == 1

    async def test_checker_crash_halts(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "crash")
        assert run.result.halted
        assert run.result.halt_reason.startswith("fork branch 'guard_draft_")
        assert "fork branch" in run.result.halt_reason
        # A checker that raises halts in the fork branch before the aggregator,
        # so there is no `fail:` line (R-012).
        assert "fail:" not in run.result.halt_reason


class TestToyModeCli:
    """UC-015 — the CLI-observable acceptance for the registered toy mode."""

    def test_validate_file(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "factory",
                "workflow",
                "validate",
                "--file",
                "factory/workflow/contributed/toy_guard/workflow.py",
            ],
            cwd=_WORKTREE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "VALID" in proc.stdout
        assert "(8 nodes, 6 edges)" in proc.stdout

    def test_show_prints_stage(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        WorkflowRegistry.reset()
        args = argparse.Namespace(
            workflow_command="show", name="toy-guard", project_path=str(tmp_path)
        )
        assert cmd_workflow(args) == 0
        out = capsys.readouterr().out
        for node_id in (
            f"guard_{_SEGMENT}_reset",
            f"guard_{_SEGMENT}_fork",
            f"guard_{_SEGMENT}_join",
            _GATE_ID,
            *_CHECKER_IDS,
        ):
            assert node_id in out
        reloop_rows = [
            line
            for line in out.splitlines()
            if _GATE_ID in line and TOY_AGENT_ID in line and "reloop" in line
        ]
        assert len(reloop_rows) == 1

    def test_export_skills_renders_rules(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        WorkflowRegistry.reset()
        output_dir = tmp_path / "skills"
        args = argparse.Namespace(
            workflow_command="export-skills",
            output_dir=str(output_dir),
            verify=True,
            project_path=str(tmp_path),
        )
        assert cmd_workflow(args) == 0
        skill_path = output_dir / "workflow-toy-guard" / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")

        assert GUARDRAILS_HEADER in content
        for guard in GUARDS:
            assert guard.rule in content
        assert "factory.workflow.guardrails_aggregate" in content
        assert guard_rules_payload(_SLUGS, GUARDS) in content
        assert validate_skill(content) == []
        # N-010: the rendered budget comes from the RELOOP target's
        # AgentNode.max_iterations (default 1 -> hardcoded 3), not the gate.
        assert "max 3 iterations" in content
