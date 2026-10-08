"""Tier 1 E2E fork pipeline tests — frozen contract for #1576.

Exercises the DataNode fork pipeline: plan → data → work → check → join → summarize.
All subgraph nodes are FnNode (NO AgentNode, NO claude binary).

The DataNode is constructed WITHOUT subgraph_entry/subgraph_exit to test the
edge-inferred subgraph path.  Until #1576 lands, DataNode validation rejects
this and all tests surface a ValidationError.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

# ── Make tests/fixtures importable ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from fork_task import ForkTask  # noqa: E402

from factory.cycle_analyzer import CycleRecord
from factory.inner_loop import InnerLoop
from factory.models import AggregateMethod, InnerLoopConfig
from factory.outer_loop.reflector import OuterLoopReflector
from factory.workflow.primitives import (
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental real claude calls in all tests."""
    monkeypatch.setenv("FACTORY_CLAUDE_BIN", "false")


# ── Helpers ────────────────────────────────────────────────────────


@dataclass
class _WorkflowOrError:
    """Holds either a constructed Workflow or the ValidationError that prevented it."""

    workflow: Workflow | None = None
    error: ValidationError | None = None


# Inline python commands for subgraph FnNodes.
# Each command reads .factory/current_item.json and appends structured
# JSON lines to $FORK_TEST_LOG so tests can inspect execution sequence,
# concurrency, isolation, and prompt forwarding.

_PLAN_CMD = textwrap.dedent("""\
    python3 -c "
import json, os, time
t = time.time()
with open('plan.md', 'w') as f:
    f.write('# Test Plan\\n')
log = os.environ.get('FORK_TEST_LOG')
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'node':'plan','cwd':os.getcwd(),'start':t,'end':time.time()}) + '\\n')
"
""").strip()

_WORK_CMD = textwrap.dedent("""\
    python3 -c "
import json, os, time, hashlib
t = time.time()
item = json.load(open('.factory/current_item.json'))
item_id = item['id']
prompt = item.get('prompt', '')
plan_hash = hashlib.md5(open('plan.md','rb').read()).hexdigest()[:8]
with open('answer.txt', 'w') as f:
    f.write(item_id + ':' + plan_hash + '\\n')
time.sleep(0.2)
log = os.environ.get('FORK_TEST_LOG')
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'node':'work','item':item_id,'prompt':prompt,'cwd':os.getcwd(),'start':t,'end':time.time()}) + '\\n')
"
""").strip()

_CHECK_CMD = textwrap.dedent("""\
    python3 -c "
import json, os, time
t = time.time()
item = json.load(open('.factory/current_item.json'))
item_id = item['id']
log = os.environ.get('FORK_TEST_LOG')
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'node':'check','item':item_id,'cwd':os.getcwd(),'start':t,'end':time.time()}) + '\\n')
"
""").strip()

_SUMMARIZE_CMD = textwrap.dedent("""\
    python3 -c "
import json, os, time
t = time.time()
log = os.environ.get('FORK_TEST_LOG')
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'node':'summarize','cwd':os.getcwd(),'start':t,'end':time.time()}) + '\\n')
"
""").strip()


def _make_fork_workflow() -> _WorkflowOrError:
    """Try to build the fork pipeline workflow.

    ALL nodes are FnNode (NO AgentNode).  The DataNode is constructed
    WITHOUT subgraph_entry / subgraph_exit / split — the subgraph
    should be inferred from edges once #1576 lands.

    On ValidationError (DataNode missing required fields), catch and
    store in _WorkflowOrError so tests surface a clear diagnostic.
    """
    try:
        plan = FnNode(id="plan", command=_PLAN_CMD, writes={"plan.md"})
        data = DataNode(
            id="data",
            task_ref="fork_task:ForkTask",
            parallelism=2,
        )
        work = FnNode(
            id="work",
            command=_WORK_CMD,
            reads={".factory/current_item.json", "plan.md"},
            writes={"answer.txt"},
        )
        check = FnNode(
            id="check",
            command=_CHECK_CMD,
            reads={".factory/current_item.json"},
        )
        summarize = FnNode(id="summarize", command=_SUMMARIZE_CMD)
        join = JoinNode(id="join", sources=["data"])

        wf = Workflow(
            name="fork-e2e",
            nodes={
                "plan": plan,
                "data": data,
                "work": work,
                "check": check,
                "summarize": summarize,
                "join": join,
            },
            edges=[
                Edge(source="plan", target="data"),
                Edge(source="data", target="work"),
                Edge(source="work", target="check"),
                Edge(source="check", target="join"),
                Edge(source="join", target="summarize"),
            ],
            start_node="plan",
        )
        return _WorkflowOrError(workflow=wf)
    except ValidationError as exc:
        return _WorkflowOrError(error=exc)


def _git(project: Path, *args: str) -> None:
    """Run a git command in ``project``, raising on failure."""
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
    )


def _bootstrap_git_project(project: Path) -> None:
    """Turn *project* into a minimal git repo with .factory/ set up."""
    project.mkdir(parents=True, exist_ok=True)
    _git(project, "init")
    _git(project, "config", "user.email", "test@test")
    _git(project, "config", "user.name", "test")

    (project / ".gitignore").write_text(".factory/\n")

    factory_dir = project / ".factory"
    factory_dir.mkdir(exist_ok=True)
    config = {"inner_loop": {"aggregate": "mean"}}
    (factory_dir / "config.json").write_text(json.dumps(config))

    tasks_dir = factory_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    import shutil
    src = Path(__file__).resolve().parents[1] / "fixtures" / "fork_task.py"
    shutil.copy(src, tasks_dir / "fork_task.py")

    (project / "README.md").write_text("test project\n")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "init")

    # Uncommitted marker — tests assert it propagates to worktrees
    (project / "pre_fork_marker.txt").write_text("marker\n")


def _make_project(root: Path) -> Path:
    """Bootstrap a git project at *root* (used by the parity test)."""
    root.mkdir(parents=True, exist_ok=True)
    _bootstrap_git_project(root)
    return root


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def fork_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp file for FORK_TEST_LOG env var."""
    log_path = tmp_path / "fork_test.log"
    log_path.touch()
    monkeypatch.setenv("FORK_TEST_LOG", str(log_path))
    return log_path


@pytest.fixture
def fork_task() -> ForkTask:
    return ForkTask()


@pytest.fixture
def fork_workflow() -> _WorkflowOrError:
    return _make_fork_workflow()


@pytest.fixture
def fork_project(tmp_path: Path) -> Path:
    project = tmp_path / "fork_project"
    _bootstrap_git_project(project)
    return project


# ── Run helper ─────────────────────────────────────────────────────


def _run_fork(
    engine: str,
    task: ForkTask,
    wf_or_err: _WorkflowOrError,
    project: Path,
    config: InnerLoopConfig | None = None,
    split: str = "train",
) -> CycleRecord:
    """Build an InnerLoop and run one step.

    Raises the stored ValidationError if workflow construction failed.
    """
    if wf_or_err.error is not None:
        raise wf_or_err.error

    assert wf_or_err.workflow is not None

    loop = InnerLoop(
        project_dir=project,
        workflow=wf_or_err.workflow,
        task=task,
        execution_strategy=engine,
        inner_loop_config=config or InnerLoopConfig(aggregate=AggregateMethod.mean),
    )

    # Set split so that the executor filters to train or val instances
    if split == "val":
        # Use a subset selector that picks holdout instances
        class _ValSelector:
            def select(self, ids: list[str]) -> list[str]:
                holdout = {"i5", "i6"}
                return [i for i in ids if i in holdout]

        loop._subset_selector = _ValSelector()  # type: ignore[attr-defined]

    return loop.step()


# ── Parametrize helpers ────────────────────────────────────────────

ENGINE_EXEC = pytest.param("executor", id="executor")
ENGINE_CEO = pytest.param(
    "ceo-skill",
    id="ceo-skill",
    marks=pytest.mark.xfail(strict=True, reason="Phase 6"),
)


# ── Test class ─────────────────────────────────────────────────────


class TestForkPipeline:

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_statuses(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        record = _run_fork(engine, fork_task, fork_workflow, fork_project)
        results = {r["instance_id"]: r for r in record.instance_results}
        assert results["i1"]["status"] == "ok"
        assert results["i2"]["status"] == "ok"
        assert results["i3"]["status"] == "errored"
        assert results["i4"]["status"] == "failed"

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_scoring_mean(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        config = InnerLoopConfig(aggregate=AggregateMethod.mean)
        record = _run_fork(engine, fork_task, fork_workflow, fork_project, config=config)
        assert abs(record.score_end - (0.8 + 0.6 + 0) / 3) < 0.01

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_scoring_all_pass(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        config = InnerLoopConfig(aggregate=AggregateMethod.all_pass)
        record = _run_fork(engine, fork_task, fork_workflow, fork_project, config=config)
        assert record.score_end == 0.0

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_train_split(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError,
        fork_project: Path, fork_log: Path,
    ) -> None:
        record = _run_fork(engine, fork_task, fork_workflow, fork_project)
        ids = {r["instance_id"] for r in record.instance_results}
        assert ids == {"i1", "i2", "i3", "i4"}
        entries = [json.loads(line) for line in fork_log.read_text().splitlines()]
        logged_items = {e["item"] for e in entries if "item" in e}
        assert "i5" not in logged_items
        assert "i6" not in logged_items

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_val_split(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        record = _run_fork(engine, fork_task, fork_workflow, fork_project, split="val")
        ids = {r["instance_id"] for r in record.instance_results}
        assert ids == {"i5", "i6"}

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_execution_sequence(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError,
        fork_project: Path, fork_log: Path,
    ) -> None:
        _run_fork(engine, fork_task, fork_workflow, fork_project)
        entries = [json.loads(line) for line in fork_log.read_text().splitlines()]
        nodes_seq = [e["node"] for e in entries]
        # plan first, summarize last
        assert nodes_seq[0] == "plan"
        assert nodes_seq[-1] == "summarize"
        # work+check for each of i1,i2,i4 (i3 errored at setup, no work)
        work_items = {e["item"] for e in entries if e["node"] == "work"}
        assert work_items == {"i1", "i2", "i4"}
        check_items = {e["item"] for e in entries if e["node"] == "check"}
        assert check_items == {"i1", "i2", "i4"}
        # exactly 1 plan, 3 work, 3 check, 1 summarize
        assert nodes_seq.count("plan") == 1
        assert nodes_seq.count("work") == 3
        assert nodes_seq.count("check") == 3
        assert nodes_seq.count("summarize") == 1

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_branch_prompt(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError,
        fork_project: Path, fork_log: Path,
    ) -> None:
        _run_fork(engine, fork_task, fork_workflow, fork_project)
        entries = [json.loads(line) for line in fork_log.read_text().splitlines()]
        for e in entries:
            if e["node"] == "work":
                assert e["prompt"] == f"Solve {e['item']}"

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_isolation(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError,
        fork_project: Path, fork_log: Path,
    ) -> None:
        _run_fork(engine, fork_task, fork_workflow, fork_project)
        entries = [json.loads(line) for line in fork_log.read_text().splitlines()]
        work_cwds = [e["cwd"] for e in entries if e["node"] == "work"]
        for cwd in work_cwds:
            assert (Path(cwd) / "pre_fork_marker.txt").exists()
        assert not (fork_project / "answer.txt").exists()

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_parallelism(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError,
        fork_project: Path, fork_log: Path,
    ) -> None:
        _run_fork(engine, fork_task, fork_workflow, fork_project)
        entries = [json.loads(line) for line in fork_log.read_text().splitlines()]
        work_entries = [e for e in entries if e["node"] == "work"]
        events: list[tuple[float, int]] = []
        for e in work_entries:
            events.append((e["start"], +1))
            events.append((e["end"], -1))
        events.sort()
        concurrent = max_c = 0
        for _, d in events:
            concurrent += d
            max_c = max(max_c, concurrent)
        assert max_c == 2  # parallelism=2, should use it

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_reflector_details(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        record = _run_fork(engine, fork_task, fork_workflow, fork_project)
        details = OuterLoopReflector._collect_individual_details("test-id", 0.5, record)
        assert "item_id" in details
        assert "check" in details

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_cleanup(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        _run_fork(engine, fork_task, fork_workflow, fork_project)
        import subprocess as sp

        r = sp.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=fork_project,
            capture_output=True,
            text=True,
        )
        data_wts = [
            line for line in r.stdout.splitlines()
            if line.startswith("worktree ") and "data-" in line
        ]
        assert len(data_wts) == 0

    @pytest.mark.xfail(strict=True, reason="ceo-skill not implemented until Phase 6")
    def test_parity(
        self, fork_task: ForkTask, fork_workflow: _WorkflowOrError, tmp_path: Path,
    ) -> None:
        p1 = _make_project(tmp_path / "exec")
        p2 = _make_project(tmp_path / "ceo")
        r1 = _run_fork("executor", fork_task, fork_workflow, p1)
        r2 = _run_fork("ceo-skill", fork_task, fork_workflow, p2)
        assert {r["instance_id"] for r in r1.instance_results} == {
            r["instance_id"] for r in r2.instance_results
        }
        s1 = {r["instance_id"]: r["status"] for r in r1.instance_results}
        s2 = {r["instance_id"]: r["status"] for r in r2.instance_results}
        assert s1 == s2
        assert r1.score_end == r2.score_end

    @pytest.mark.parametrize("engine", [ENGINE_EXEC, ENGINE_CEO])
    def test_item_result_fields(
        self, engine: str, fork_task: ForkTask, fork_workflow: _WorkflowOrError, fork_project: Path,
    ) -> None:
        record = _run_fork(engine, fork_task, fork_workflow, fork_project)
        for r in record.instance_results:
            assert isinstance(r["split"], str)
            assert isinstance(r["verify_details"], dict)
            assert isinstance(r["cost"], (int, float))
