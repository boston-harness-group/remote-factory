"""Tests for data_runtime bugs.

- Fix 1: verify exception → status=failed (not errored), score=0
- Fix 2: halted/crashed branches still report cost
- Bug 3 (moved from test_five_bugs): ItemResult.passed field
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


from factory.models import ItemResult, ItemStatus
from factory.task import Task, TaskDefinition, TaskInstance, VerifyResult
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow,
)


# ── Helpers ──────────────────────────────────────────────────────


class VerifyExplodingTask(Task):
    """Task whose verify() raises RuntimeError."""

    def __init__(self) -> None:
        super().__init__(TaskDefinition(name="verify-exploding"))

    def _raw_instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(id="boom")

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        pass

    def prompt(self, instance: TaskInstance) -> str:
        return "do something"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        raise RuntimeError("verify kaboom")


class AlwaysPassTask(Task):
    """Task that always passes verification."""

    def __init__(self) -> None:
        super().__init__(TaskDefinition(name="always-pass"))

    def _raw_instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(id="ok-item")

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        pass

    def prompt(self, instance: TaskInstance) -> str:
        return "do it"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        return VerifyResult(passed=True, score=1.0)


def _noop_fn(project_dir: str, **kw: Any) -> None:
    pass


def _make_fn_workflow() -> Workflow:
    return Workflow(
        name="fn-test",
        nodes={
            "data": DataNode(id="data"),
            "work": FnNode(
                id="work",
                command="echo work",
                callable_name="tests.test_data_runtime_bugs:_noop_fn",
            ),
            "join": JoinNode(id="join", sources=["work"]),
        },
        edges=[
            Edge(source="data", target="work"),
            Edge(source="work", target="join"),
        ],
        start_node="data",
    )


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True, capture_output=True,
    )


def _bootstrap(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    _git(project, "init")
    _git(project, "config", "user.email", "t@t")
    _git(project, "config", "user.name", "t")
    (project / ".gitignore").write_text(".factory/\n")
    (project / "README.md").write_text("x\n")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "init")
    return project


# ── Fix 1: verify exception → failed not errored ────────────────


class TestVerifyExceptionBecomesFailed:
    """When Task.verify() throws, status must be 'failed' (scored 0),
    NOT 'errored' (excluded from scoring)."""

    def test_verify_exception_sets_status_failed(self, tmp_path: Path) -> None:
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        task = VerifyExplodingTask()
        wf = _make_fn_workflow()

        results = asyncio.run(run_fork(
            workflow=wf,
            data_node=wf.nodes["data"],
            data_node_id="data",
            project_path=project,
            task=task,
            run_id="test-verify-exc",
        ))

        assert len(results) == 1
        r = results[0]
        # Must be 'failed', NOT 'errored'
        assert r["status"] == "failed", (
            f"verify exception should produce status=failed, got {r['status']}"
        )
        assert r["score"] == 0.0
        assert "verify_failed" in r.get("verify_details", {}).get("error", "")


# ── Fix 2: crashed branches still report cost ───────────────────


class TestCrashedBranchReportsCost:
    """When a branch crashes, the except block must still read agent costs."""

    def test_branch_crash_reports_nonzero_cost(self, tmp_path: Path) -> None:
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        task = AlwaysPassTask()

        # Build workflow with an agent node that will crash
        wf = Workflow(
            name="crash-cost-test",
            nodes={
                "data": DataNode(id="data"),
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build it",
                    reads=set(),
                    writes={".factory/reviews/builder-latest.md"},
                ),
                "join": JoinNode(id="join", sources=["builder"]),
            },
            edges=[
                Edge(source="data", target="builder"),
                Edge(source="builder", target="join"),
            ],
            start_node="data",
        )

        async def crashing_agent(
            role: str,
            task_prompt: str,
            project_path: Path | str,
            **kw: Any,
        ) -> tuple[str, int]:
            # Write an agent.completed event with cost BEFORE crashing
            pp = Path(project_path)
            factory_dir = pp / ".factory"
            factory_dir.mkdir(parents=True, exist_ok=True)
            event = {
                "type": "agent.completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"total_cost_usd": 0.42},
            }
            event_file = factory_dir / "events.jsonl"
            with open(event_file, "a") as f:
                f.write(json.dumps(event) + "\n")
            raise RuntimeError("agent crashed after spending money")

        results = asyncio.run(run_fork(
            workflow=wf,
            data_node=wf.nodes["data"],
            data_node_id="data",
            project_path=project,
            task=task,
            run_id="test-crash-cost",
            agent_fn=crashing_agent,
        ))

        assert len(results) == 1
        r = results[0]
        assert r["status"] == "failed"
        assert r.get("cost", 0.0) > 0, (
            f"crashed branch should report cost > 0, got {r.get('cost')}"
        )


# ── Bug 3 (moved from test_five_bugs): ItemResult.passed ────────


class TestItemResultPassed:
    """ItemResult must have a `passed` field."""

    def test_item_result_has_passed_field(self) -> None:
        ir = ItemResult(
            item_id="x",
            status=ItemStatus.ok,
            score=1.0,
            passed=True,
        )
        assert ir.passed is True

        ir2 = ItemResult(
            item_id="y",
            status=ItemStatus.failed,
            score=0.0,
            passed=False,
        )
        assert ir2.passed is False

    def test_passed_reaches_cycle_record(self) -> None:
        from factory.cycle_analyzer import CycleRecord

        items = [
            {"item_id": "a", "status": "ok", "score": 1.0, "passed": True},
            {"item_id": "b", "status": "failed", "score": 0.0, "passed": False},
        ]
        record = CycleRecord.from_run(items, aggregate="mean")
        assert record.instance_results is not None
        first = record.instance_results[0]
        assert first["passed"] is True, "passed=True must survive into CycleRecord"

    def test_data_runtime_populates_passed(self) -> None:
        ir = ItemResult(
            item_id="test",
            status=ItemStatus.ok,
            score=1.0,
            passed=True,
        )
        dumped = ir.model_dump()
        assert "passed" in dumped
        assert dumped["passed"] is True
