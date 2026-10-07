"""Tier 1 E2E outer-loop wiring tests — REAL pipeline path.

Tests 1-6 share a single engine run via a class-level fixture that exercises
the REAL pipeline: SwarmEngine → SwarmEvaluator._evaluate_via_inner_loop →
git worktree → compose() → InnerLoop._step_with_data_node →
WorkflowExecutor._execute_data → task.setup/verify.

No AgentNodes. No claude binary. The DataNode subgraph is a single FnNode
("echo ok"), so the only process spawned is /bin/echo.

Tests 7-10 are standalone unit tests (no engine needed).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ── Make tests/fixtures importable ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from wiring_task import WiringTask  # noqa: E402

from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.population import Population
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_datanode_workflow(task_ref: str = "wiring_task:WiringTask") -> Workflow:
    """Minimal DataNode + FnNode subgraph. No AgentNodes → no claude."""
    return Workflow(
        name="wiring-e2e",
        nodes={
            "data": DataNode(
                id="data",
                task_ref=task_ref,
                subgraph_entry="process",
                subgraph_exit="process",
            ),
            "process": FnNode(id="process", command="echo ok"),
        },
        edges=[],
        start_node="data",
    )


def _git(project: Path, *args: str) -> None:
    """Run a git command in ``project``, raising on failure."""
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
    )


def _bootstrap_git_project(project: Path) -> None:
    """Turn *project* into a minimal git repo with .factory/ set up."""
    # 1. git init + config
    _git(project, "init")
    _git(project, "config", "user.email", "test@test")
    _git(project, "config", "user.name", "test")

    # 2. .gitignore (keep .factory/ out of tracked tree)
    (project / ".gitignore").write_text(".factory/\n")

    # 3. .factory/config.json — aggregate = all_pass
    factory_dir = project / ".factory"
    factory_dir.mkdir(exist_ok=True)
    config = {"inner_loop": {"aggregate": "all_pass"}}
    (factory_dir / "config.json").write_text(json.dumps(config))

    # 4. Copy wiring_task.py into .factory/tasks/
    tasks_dir = factory_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    src = Path(__file__).resolve().parents[1] / "fixtures" / "wiring_task.py"
    shutil.copy(src, tasks_dir / "wiring_task.py")

    # 5. Seed file + first commit (worktree needs ≥ 1 commit)
    (project / "README.md").write_text("test project\n")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "init")


# ── Module-level shared fixture (real pipeline path) ──────────────


class _SharedState:
    """Lazily populated cache for the shared engine run (tests 1-6).

    Runs the REAL path: SwarmEngine → SwarmEvaluator._evaluate_via_inner_loop
    → worktree → compose() → InnerLoop._step_with_data_node →
    WorkflowExecutor._execute_data → task.setup/verify.
    """

    _populated = False
    summary: Any = None
    # Collected from CycleRecord.instance_results via inner loop
    instance_results: list[dict[str, Any]] = []
    # All individual eval results
    eval_results: list[EvalResult] = []
    # Training instances the engine selected
    training_ids_used: list[str] = []

    @classmethod
    def ensure(cls, tmp_path_factory: pytest.TempPathFactory) -> None:
        if cls._populated:
            return
        cls._populated = True

        project = tmp_path_factory.mktemp("wiring_project")
        _bootstrap_git_project(project)

        # ── Build workflow ──────────────────────────────────────
        wf = _make_datanode_workflow()

        # ── Build SwarmConfig ───────────────────────────────────
        swarm_config = SwarmConfig(
            benchmark="wiring",
            budget=6,
            population_size=2,
            tournament_size=2,
            mutation_rate=0.0,  # no mutations → evaluate only the seed
            training_instances=["i1", "i2", "i3", "i4"],
            holdout_instances=["i5", "i6"],
            frozen_node_ids=["data"],
            designer_count=0,
        )
        task = WiringTask(str(project))
        swarm_config.set_task(task)

        # ── Build evaluator (inner_loop path) ───────────────────
        # inner_loop_factory=True triggers _evaluate_via_inner_loop;
        # with task set, compose() wires the real InnerLoop.
        evaluator = SwarmEvaluator(
            swarm_config,
            inner_loop_factory=True,  # triggers real inner-loop path
            project_dir=project,
        )

        # ── Build engine and seed population ────────────────────
        engine = SwarmEngine(
            swarm_config,
            evaluator,
            project_dir=project,
        )
        pop = Population()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        # ── Run one generation (the REAL path) ──────────────────
        cls.summary = engine.evolve_generation(pop, 0, str(project))

        # ── Harvest results from the population ─────────────────
        for individual in pop.individuals:
            if individual.score is not None:
                # The inner-loop cycle record is stored in the evaluator
                record = evaluator.get_cycle_record(individual.id)
                if record is not None and record.instance_results:
                    cls.instance_results.extend(record.instance_results)

        # Also harvest from cycle_cache to ensure we have results
        if not cls.instance_results:
            # Fall back: read from cycle summary on disk
            for d in (project / ".factory" / "outer_loop" / "runs").rglob(
                "cycle_summary.json"
            ):
                try:
                    data = json.loads(d.read_text())
                    if "instance_results" in data:
                        cls.instance_results.extend(data["instance_results"])
                except (json.JSONDecodeError, OSError):
                    pass


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> type[_SharedState]:
    _SharedState.ensure(tmp_path_factory)
    return _SharedState


# ── Tests 1-6: shared engine run (real pipeline path) ─────────────


class TestRealPipelineRun:
    """Tests 1-6 inspect the single shared evolve_generation() result.

    The entire pipeline runs for real: worktree isolation, compose(),
    InnerLoop._step_with_data_node, WorkflowExecutor._execute_data,
    task.setup/verify.  Only process spawned is `echo ok` (FnNode).
    """

    def test_holdout_never_reaches_training(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 1: i5/i6 do NOT appear in instance_results from training eval."""
        holdout = {"i5", "i6"}
        seen_ids = {
            r.get("instance_id") or r.get("item_id", "")
            for r in shared.instance_results
            if isinstance(r, dict)
        }
        assert not (seen_ids & holdout), (
            f"Holdout instances leaked into training eval: {seen_ids & holdout}"
        )

    def test_scores_are_fractional(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 2: At least one instance score is fractional (not 0 or 1)."""
        scores = [
            r.get("score", 0.0)
            for r in shared.instance_results
            if isinstance(r, dict) and "score" in r
        ]
        has_fractional = any(0.0 < s < 1.0 for s in scores)
        assert has_fractional, f"No fractional scores found: {scores}"

    def test_all_pass_vs_mean(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 3: all_pass → 0.0 (i3/i4 fail). mean would give ~0.39."""
        scores = [
            r.get("score", 0.0)
            for r in shared.instance_results
            if isinstance(r, dict) and "score" in r
        ]
        if not scores:
            pytest.skip("No scores collected — cannot verify aggregate")
        all_pass = 1.0 if all(s >= 1.0 for s in scores) else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0

        assert all_pass == 0.0, f"Expected all_pass=0.0, got {all_pass}"
        assert mean_score != all_pass, (
            f"mean ({mean_score}) should differ from all_pass ({all_pass})"
        )

    def test_setup_failure_skips_instance(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 4: i3 setup fails → scored as 0.0, passed=False.

        The real pipeline path: executor._execute_data catches setup
        exceptions and records {error: "setup_failed", score: 0.0}.
        Inner loop's _step_with_data_node copies score/passed but may
        strip the error field, so we check score + passed.
        """
        i3_results = [
            r for r in shared.instance_results
            if isinstance(r, dict)
            and (r.get("instance_id") == "i3" or r.get("item_id") == "i3")
        ]
        assert len(i3_results) >= 1, (
            f"i3 not found in instance_results. Got IDs: "
            f"{[r.get('instance_id') or r.get('item_id') for r in shared.instance_results]}"
        )
        for r in i3_results:
            assert r.get("score", -1) == 0.0, f"i3 score should be 0.0, got {r.get('score')}"
            assert r.get("passed") is False, f"i3 should not pass, got {r.get('passed')}"

    def test_halt_reason_in_eval_details(self, tmp_path: Path) -> None:
        """Test 5: DataNode with 0 items after split-filter → halted."""
        data = DataNode(
            id="data",
            inline_items=[DataItem(id="x1", metadata={"split": "val"})],
            subgraph_entry="process",
            subgraph_exit="process",
            split="train",  # x1 is val → 0 items after filter
        )
        process = FnNode(id="process", command="echo ok")
        wf = Workflow(
            name="empty-data",
            nodes={"data": data, "process": process},
            edges=[],
            start_node="data",
        )

        from factory.workflow.executor import WorkflowExecutor

        ex = WorkflowExecutor(wf, tmp_path, dry_run=True, validate=False)
        result = asyncio.run(ex.execute())

        assert result.halted
        assert "0 items" in result.halt_reason

    def test_training_instances_limits_items(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 6: Only i1-i4 appear in instance_results (train split)."""
        train_ids = {"i1", "i2", "i3", "i4"}
        holdout_ids = {"i5", "i6"}
        seen_ids = {
            r.get("instance_id") or r.get("item_id", "")
            for r in shared.instance_results
            if isinstance(r, dict)
        }
        assert seen_ids <= train_ids, f"Non-train IDs appeared: {seen_ids - train_ids}"
        assert not (seen_ids & holdout_ids), f"Holdout IDs leaked: {seen_ids & holdout_ids}"


# ── Tests 7-10: standalone ─────────────────────────────────────────


class TestStandalone:
    """Standalone tests (no shared engine state)."""

    def test_empty_intersection_warns(self, tmp_path: Path) -> None:
        """Test 7: training_instances with no task overlap → warning, no crash."""
        task = WiringTask(str(tmp_path))
        config = SwarmConfig(
            benchmark="wiring",
            budget=4,
            population_size=2,
            tournament_size=2,
            training_instances=["x", "y"],  # Don't match any task instance
            holdout_instances=[],
        )
        config.set_task(task)

        call_log: list[list[str]] = []

        def track_eval(
            wf: Workflow, project_dir: str, instances: list[str],
        ) -> EvalResult:
            call_log.append(list(instances))
            return EvalResult(score=0.5, benchmark_score=0.5)

        evaluator = SwarmEvaluator(config, evaluator_fn=track_eval)
        engine = SwarmEngine(config, evaluator)
        wf = Workflow(
            name="simple",
            nodes={"a": FnNode(id="a", command="echo a")},
            edges=[],
            start_node="a",
        )
        pop = engine.seed(wf)

        engine.evolve_generation(pop, 0, str(tmp_path))

        assert len(call_log) > 0
        for call in call_log:
            assert len(call) > 0, "Evaluator received empty instance list"

    def test_designer_reloop_gate_not_broken(self) -> None:
        """Test 8: GateNode with only RELOOP edge — no spurious PROCEED error."""
        gate = GateNode(
            id="gate",
            evaluator_type="agent",
            evaluator_role=AgentRole.CEO,
        )
        builder = AgentNode(
            id="builder",
            role=AgentRole.BUILDER,
            prompt_template="build something",
        )
        wf = Workflow(
            name="reloop-only",
            nodes={"gate": gate, "builder": builder},
            edges=[
                Edge(source="builder", target="gate"),
                Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
            ],
            start_node="builder",
        )

        from factory.workflow.validation import validate_workflow

        issues = validate_workflow(wf)
        gate_issues = [i for i in issues if "PROCEED" in i and "gate" in i]
        assert len(gate_issues) == 0, (
            f"Validator incorrectly flagged RELOOP-only gate: {gate_issues}"
        )

    def test_inline_items_skip_id_filter(self, tmp_path: Path) -> None:
        """Test 9: DataNode with inline_items ignores allowed_instance_ids."""
        data = DataNode(
            id="data",
            inline_items=[
                DataItem(id="0", metadata={}),
                DataItem(id="1", metadata={}),
            ],
            subgraph_entry="process",
            subgraph_exit="process",
        )
        process = FnNode(id="process", command="echo ok")
        wf = Workflow(
            name="inline-test",
            nodes={"data": data, "process": process},
            edges=[],
            start_node="data",
        )

        from factory.workflow.executor import WorkflowExecutor

        ex = WorkflowExecutor(
            wf, tmp_path, dry_run=True,
            allowed_instance_ids={"t1"},
            validate=False,
        )
        result = asyncio.run(ex.execute())

        assert result.success
        executed_ids = {
            item["item_id"]
            for item in result.item_results
            if isinstance(item, dict)
        }
        assert executed_ids == {"0", "1"}, (
            f"Expected both inline items, got {executed_ids}"
        )

    def test_holdout_runs_val_instances(self, tmp_path: Path) -> None:
        """Test 10: Evaluator with holdout_instances processes i5/i6."""
        task = WiringTask(str(tmp_path))
        config = SwarmConfig(
            benchmark="wiring",
            budget=4,
            population_size=2,
            tournament_size=2,
            training_instances=["i1", "i2", "i3", "i4"],
            holdout_instances=["i5", "i6"],
        )
        config.set_task(task)

        holdout_calls: list[list[str]] = []

        def track_eval(
            wf: Workflow, project_dir: str, instances: list[str],
        ) -> EvalResult:
            holdout_calls.append(list(instances))
            result_details: dict[str, Any] = {"instance_results": []}
            for iid in instances:
                for inst in task.instances(split="all"):
                    if inst.id == iid:
                        try:
                            task.setup(inst, Path(project_dir))
                            vr = task.verify(inst, Path(project_dir))
                            result_details["instance_results"].append({
                                "instance_id": iid,
                                "passed": vr.passed,
                                "score": vr.score,
                            })
                        except Exception:
                            result_details["instance_results"].append({
                                "instance_id": iid,
                                "passed": False,
                                "score": 0.0,
                            })
            return EvalResult(score=0.9, benchmark_score=0.9, details=result_details)

        evaluator = SwarmEvaluator(config, evaluator_fn=track_eval)

        wf = Workflow(
            name="simple",
            nodes={"a": FnNode(id="a", command="echo a")},
            edges=[],
            start_node="a",
        )
        result = evaluator.evaluate(wf, str(tmp_path), ["i5", "i6"])

        assert len(holdout_calls) == 1
        assert set(holdout_calls[0]) == {"i5", "i6"}

        irs = result.details.get("instance_results", [])
        evaluated_ids = {r["instance_id"] for r in irs if isinstance(r, dict)}
        assert "i5" in evaluated_ids
        assert "i6" in evaluated_ids
