"""End-to-end guardrail tests through the real executor with stub agents."""

from __future__ import annotations

import argparse
import json
import os
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


@pytest.fixture(autouse=True)
def _repo_on_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the spawned ``factory`` console script import this checkout."""
    existing = os.environ.get("PYTHONPATH")
    entries = [str(_WORKTREE_ROOT), *([existing] if existing else [])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(entries))


@dataclass
class StubState:
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
    WorkflowRegistry.reset()
    yield
    WorkflowRegistry.reset()


class TestToyModeGraph:
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
        assert GUARDRAILS_HEADER in tasks[1]

    async def test_violations_md_carries_rule(self, tmp_path: Path) -> None:
        run = await _run(tmp_path, "reloop_once")
        assert run.result.success, run.result.halt_reason
        content = run.stub.violations_at_rerun
        assert content is not None, "violations.md was absent at the reloop re-run"
        assert f"## {_SLUGS[0]}" in content
        assert f"- rule: {GUARDS[0].rule.strip()}" in content
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
        assert "fail:" not in run.result.halt_reason


class TestToyModeCli:
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
        assert "factory workflow guard" in content
        assert guard_rules_payload(_SLUGS, GUARDS) in content
        assert validate_skill(content) == []
        assert f"max {GUARD_MAX_ITERATIONS} iterations" in content
