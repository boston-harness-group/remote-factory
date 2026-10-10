"""Unit tests for factory/workflow/data_runtime.py.

Covers split labeling, per-item cost, item store (items.jsonl),
and split filtering via task.instances(split).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator

import pytest

from factory.cycle_analyzer import CycleRecord
from factory.models import ItemResult, ItemStatus
from factory.task import Task, TaskDefinition, TaskInstance, VerifyResult
from factory.workflow.primitives import (
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow,
)


# ── Minimal task for tests ──────────────────────────────────────────


class _SplitTask(Task):
    """Task with train/val split support."""

    def __init__(self) -> None:
        from factory.task import InstancesConfig

        defn = TaskDefinition(
            name="split-test-task",
            instances_config=InstancesConfig(holdout_ids=["v1", "v2"]),
        )
        super().__init__(definition=defn)

    def _raw_instances(self) -> Iterator[TaskInstance]:
        for iid in ["t1", "t2", "t3", "v1", "v2"]:
            yield TaskInstance(id=iid)

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        pass

    def prompt(self, instance: TaskInstance) -> str:
        return f"Do {instance.id}"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        scores = {"t1": 0.8, "t2": 0.6, "t3": 1.0, "v1": 0.9, "v2": 0.7}
        s = scores.get(instance.id, 0.5)
        return VerifyResult(passed=s > 0.0, score=s, details={"item": instance.id})


def _noop_fn(project_dir: str, **kw: Any) -> None:
    pass


def _make_test_workflow() -> Workflow:
    return Workflow(
        name="split-test",
        nodes={
            "data": DataNode(id="data", parallelism=2),
            "work": FnNode(
                id="work",
                command="echo work",
                callable_name="tests.test_outer_loop.test_data_runtime_bugs:_noop_fn",
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
        check=True,
        capture_output=True,
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


# ── Bug 1: split label ──────────────────────────────────────────────


class TestBug1SplitLabel:
    """ItemResult.split must match the requested split."""

    def test_val_split_propagates(self, tmp_path: Path) -> None:
        """run_fork with split='val' → every ItemResult has split='val'."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="val-test",
            split="val",
        ))

        assert len(results) > 0, "Expected val items"
        for r in results:
            assert r["split"] == "val", (
                f"Item {r['item_id']} has split={r['split']!r}, expected 'val'"
            )

    def test_train_split_propagates(self, tmp_path: Path) -> None:
        """run_fork with split='train' → every ItemResult has split='train'."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="train-test",
            split="train",
        ))

        assert len(results) > 0
        for r in results:
            assert r["split"] == "train"


# ── Bug 2: per-item cost ────────────────────────────────────────────


COST_PER_AGENT_CALL = 0.01


async def _cost_emitting_agent(
    role: str,
    task: str,
    project_path: Any,
    *,
    model: Any = None,
    timeout: float = 600.0,
    node_id: str | None = None,
    **kw: Any,
) -> tuple[str, int]:
    """Mock agent_fn that writes an agent.completed event with known cost."""
    from factory.events import emit_event

    proj = Path(project_path)
    emit_event(
        proj,
        "agent.completed",
        agent=role,
        data={"total_cost_usd": COST_PER_AGENT_CALL, "return_code": 0},
    )
    return "ok", 0


def _make_agent_workflow() -> Workflow:
    """Workflow with DataNode → AgentNode → JoinNode for cost testing."""
    from factory.workflow.primitives import AgentNode, AgentRole

    return Workflow(
        name="cost-test",
        nodes={
            "data": DataNode(id="data", parallelism=2),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build {project_path}",
            ),
            "join": JoinNode(id="join", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="join"),
        ],
        start_node="data",
    )


class TestBug2PerItemCost:
    """ItemResult.cost equals real agent cost from events — never fabricated."""

    def test_item_cost_from_agent_events(self, tmp_path: Path) -> None:
        """Each item's cost equals the sum of agent costs emitted during its branch."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_agent_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=False,
            task=task,
            run_id="cost-exact",
            split="train",
            agent_fn=_cost_emitting_agent,
        ))

        # Train split: t1, t2, t3 → 3 items, each gets 1 agent call
        assert len(results) == 3
        for r in results:
            assert r["cost"] == pytest.approx(COST_PER_AGENT_CALL), (
                f"Item {r['item_id']} cost={r['cost']}, "
                f"expected {COST_PER_AGENT_CALL}"
            )

    def test_no_agent_branch_cost_is_zero(self, tmp_path: Path) -> None:
        """Branch with only FnNodes → cost stays 0.0 (no fabrication)."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()  # FnNode only, no agent
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=False,
            task=task,
            run_id="zero-cost",
            split="train",
        ))

        assert len(results) > 0
        for r in results:
            assert r["cost"] == 0.0, (
                f"Item {r['item_id']} cost={r['cost']}, "
                f"expected 0.0 for FnNode-only branch"
            )

    def test_cycle_record_sums_exact_costs(self) -> None:
        """CycleRecord.from_run() sums per-item costs into total_cost_usd exactly."""
        items = [
            {"item_id": "a", "status": "ok", "score": 0.8, "cost": 0.01},
            {"item_id": "b", "status": "ok", "score": 0.6, "cost": 0.01},
            {"item_id": "c", "status": "errored", "score": 0.0, "cost": 0.01},
        ]
        record = CycleRecord.from_run(items, aggregate="mean")
        assert record.total_cost_usd == pytest.approx(0.03), (
            f"Expected total_cost_usd=0.03, got {record.total_cost_usd}"
        )


# ── Bug 3: item store ───────────────────────────────────────────────


class TestBug3ItemStore:
    """run_fork writes .factory/runs/<run>/items.jsonl."""

    def test_items_jsonl_written(self, tmp_path: Path) -> None:
        """items.jsonl exists after run_fork and contains valid ItemResults."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()
        run_id = "store-test"

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id=run_id,
            split="train",
        ))

        items_path = project / ".factory" / "runs" / run_id / "items.jsonl"
        assert items_path.exists(), f"items.jsonl not found at {items_path}"

        lines = [
            line for line in items_path.read_text().strip().splitlines()
            if line.strip()
        ]
        assert len(lines) == len(results), (
            f"Expected {len(results)} lines in items.jsonl, got {len(lines)}"
        )

        # Each line should deserialize to a valid ItemResult
        for i, line in enumerate(lines):
            data = json.loads(line)
            ir = ItemResult(**data)
            assert ir.item_id, f"Line {i} has empty item_id"
            assert ir.status in (ItemStatus.ok, ItemStatus.failed, ItemStatus.errored)


# ── Bug 4: split filtering via task.instances(split) ─────────────────


class TestBug4SplitFiltering:
    """Data runtime uses task.instances(split) — not allowed_instance_ids alone."""

    def test_val_items_only(self, tmp_path: Path) -> None:
        """split='val' returns only val instances."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="val-filter",
            split="val",
        ))

        item_ids = {r["item_id"] for r in results}
        assert item_ids == {"v1", "v2"}, (
            f"Expected val items {{v1, v2}}, got {item_ids}"
        )

    def test_train_items_only(self, tmp_path: Path) -> None:
        """split='train' returns only train instances."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="train-filter",
            split="train",
        ))

        item_ids = {r["item_id"] for r in results}
        assert item_ids == {"t1", "t2", "t3"}, (
            f"Expected train items {{t1, t2, t3}}, got {item_ids}"
        )

    def test_subset_with_invalid_ids_raises(self, tmp_path: Path) -> None:
        """If subset contains IDs not in the split, raise ValueError."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        with pytest.raises(ValueError, match="not in split"):
            asyncio.run(run_fork(
                wf,
                wf.nodes["data"],  # type: ignore[arg-type]
                "data",
                project,
                dry_run=True,
                task=task,
                run_id="bad-subset",
                split="train",
                allowed_instance_ids={"t1", "v1"},  # v1 is val, not train
            ))

    def test_valid_subset_filters(self, tmp_path: Path) -> None:
        """Valid subset IDs within the split work correctly."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _SplitTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="good-subset",
            split="train",
            allowed_instance_ids={"t1", "t2"},
        ))

        item_ids = {r["item_id"] for r in results}
        assert item_ids == {"t1", "t2"}, (
            f"Expected {{t1, t2}}, got {item_ids}"
        )


# ── Bug 5: branch crash → failed (not errored) ───────────────────


async def _crashing_agent(
    role: str,
    task: str,
    project_path: Any,
    *,
    model: Any = None,
    timeout: float = 600.0,
    node_id: str | None = None,
    **kw: Any,
) -> tuple[str, int]:
    """Agent function that always raises during branch execution."""
    raise RuntimeError("simulated branch crash")


def _make_crashing_workflow() -> Workflow:
    """Workflow with DataNode → AgentNode (crashes) → JoinNode."""
    from factory.workflow.primitives import AgentNode, AgentRole

    return Workflow(
        name="crash-test",
        nodes={
            "data": DataNode(id="data", parallelism=2),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build {project_path}",
            ),
            "join": JoinNode(id="join", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="join"),
        ],
        start_node="data",
    )


class TestBug5BranchCrashProducesFailed:
    """A branch execution crash must produce status=failed (score=0),
    NOT status=errored (which would exclude the item from scoring).

    Errored is reserved for setup/infrastructure failures.
    """

    def test_branch_crash_is_failed_not_errored(self, tmp_path: Path) -> None:
        """Agent that raises during execution → failed with score=0.0."""
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        wf = _make_crashing_workflow()
        task = _SplitTask()

        # dry_run=False so the executor actually invokes the agent_fn
        # (dry_run skips execution entirely)
        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=False,
            task=task,
            run_id="crash-test",
            split="train",
            agent_fn=_crashing_agent,
        ))

        assert len(results) == 3, f"Expected 3 train items, got {len(results)}"
        for r in results:
            assert r["status"] == "failed", (
                f"Item {r['item_id']} has status={r['status']!r}, "
                f"expected 'failed' (not 'errored') for branch crash"
            )
            assert r["score"] == 0.0, (
                f"Item {r['item_id']} has score={r['score']}, expected 0.0"
            )
            assert r["error"] is not None, (
                f"Item {r['item_id']} should have error message"
            )
            assert "branch_failed" in r["error"], (
                f"Error should indicate branch failure: {r['error']}"
            )

    def test_setup_failure_is_still_errored(self, tmp_path: Path) -> None:
        """Task.setup() failure → errored (infrastructure), not failed."""
        from factory.workflow.data_runtime import run_fork

        class _FailSetupTask(_SplitTask):
            def setup(self, instance: TaskInstance, workspace: Path) -> None:
                raise OSError("disk full")

        project = _bootstrap(tmp_path)
        wf = _make_test_workflow()
        task = _FailSetupTask()

        results = asyncio.run(run_fork(
            wf,
            wf.nodes["data"],  # type: ignore[arg-type]
            "data",
            project,
            dry_run=True,
            task=task,
            run_id="setup-fail",
            split="train",
        ))

        assert len(results) == 3
        for r in results:
            assert r["status"] == "errored", (
                f"Item {r['item_id']} has status={r['status']!r}, "
                f"expected 'errored' for setup failure"
            )
            assert "setup_failed" in r["error"]


# ── Moved from test_data_runtime_bugs.py ──────────────────────────────


class _VerifyExplodingTask(Task):
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


class _AlwaysPassTask(Task):
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


def _make_fn_workflow() -> Workflow:
    return Workflow(
        name="fn-test",
        nodes={
            "data": DataNode(id="data"),
            "work": FnNode(
                id="work",
                command="echo work",
                callable_name="tests.test_outer_loop.test_data_runtime:_noop_fn",
            ),
            "join": JoinNode(id="join", sources=["work"]),
        },
        edges=[
            Edge(source="data", target="work"),
            Edge(source="work", target="join"),
        ],
        start_node="data",
    )


class TestItemStatus:
    """When Task.verify() throws, status must be 'failed' (scored 0),
    NOT 'errored' (excluded from scoring)."""

    def test_verify_exception_sets_status_failed(self, tmp_path: Path) -> None:
        from factory.workflow.data_runtime import run_fork

        project = _bootstrap(tmp_path)
        task = _VerifyExplodingTask()
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
        assert r["status"] == "failed", (
            f"verify exception should produce status=failed, got {r['status']}"
        )
        assert r["score"] == 0.0
        assert "verify_failed" in r.get("verify_details", {}).get("error", "")


class TestItemCost:
    """When a branch crashes, the except block must still read agent costs."""

    def test_branch_crash_reports_nonzero_cost(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        from factory.workflow.data_runtime import run_fork
        from factory.workflow.primitives import AgentNode, AgentRole

        project = _bootstrap(tmp_path)
        task = _AlwaysPassTask()

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


class TestItemResultPassedField:
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
