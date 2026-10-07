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

from wiring_task_wt import WiringTask  # noqa: E402

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

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental real claude calls in all tests."""
    monkeypatch.setenv("FACTORY_CLAUDE_BIN", "false")


# ── Helpers ────────────────────────────────────────────────────────


def _make_datanode_workflow(task_ref: str = "wiring_task_copied:WiringTask") -> Workflow:
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

    # 4. Copy wiring_task_wt.py into .factory/tasks/ as wiring_task_copied.py
    #    The module name "wiring_task_copied" is never imported at the top of this
    #    test file.  If TaskRef.resolve() finds it, that proves the evaluator
    #    copied .factory/tasks/ into the worktree AND the sys.path fix works.
    tasks_dir = factory_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    src = Path(__file__).resolve().parents[1] / "fixtures" / "wiring_task_wt.py"
    shutil.copy(src, tasks_dir / "wiring_task_copied.py")

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
    # Actual pipeline score from the evaluated individual
    pipeline_score: float | None = None

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
            mutation_rate=0.0,  # no mutations — offspring produced by crossover/structural changes only
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
        seed_id = ind.id
        pop.add(ind)

        # ── Run one generation (the REAL path) ──────────────────
        cls.summary = engine.evolve_generation(pop, 0, str(project))

        # ── Harvest results from the seed individual only ───────
        # Use ONLY the seed individual — offspring may fail validation and score 0.0
        record = evaluator.get_cycle_record(seed_id)
        if record is not None:
            cls.pipeline_score = record.score_end
            if record.instance_results:
                cls.instance_results = list(record.instance_results)

        # Fallback: if get_cycle_record didn't work, get score from the individual
        if cls.pipeline_score is None:
            seed = next((i for i in pop.individuals if i.id == seed_id), None)
            if seed is not None and seed.score is not None:
                cls.pipeline_score = seed.score


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
        assert shared.instance_results
        assert len(shared.instance_results) == 4  # i1, i2, i3 (setup_failed), i4
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
        """Test 3: pipeline score reflects all_pass (0.0), not mean.

        The pipeline ran with aggregate=all_pass.  i3 setup fails and i4
        scores 0.0, so all_pass → 0.0 (not all instances passed).
        Mean of individual scores would be >0.0.  We verify the pipeline
        actually applied all_pass by checking the score it assigned.
        """
        # a) Pipeline produced a score
        assert shared.pipeline_score is not None, (
            "pipeline_score was not captured — no evaluated individual"
        )

        # b) all_pass → 0.0 (accounting for parsimony: max(0.0, 0.0 - penalty) == 0.0)
        assert shared.pipeline_score == pytest.approx(0.0), (
            f"Expected pipeline_score ≈ 0.0 (all_pass), got {shared.pipeline_score}"
        )

        # c) Compute what mean would give from instance_results
        scores = [
            r.get("score", 0.0)
            for r in shared.instance_results
            if isinstance(r, dict) and "score" in r
        ]
        assert scores, "No instance scores collected"
        mean_score = sum(scores) / len(scores)

        # d) Mean is > 0.0 (some instances passed with nonzero scores)
        assert mean_score > 0.0, (
            f"Expected mean > 0.0 (partial passes), got {mean_score}"
        )

        # e) Pipeline used all_pass, not mean
        assert shared.pipeline_score != pytest.approx(mean_score), (
            f"pipeline_score ({shared.pipeline_score}) should differ from "
            f"mean ({mean_score}) — proves all_pass is in effect"
        )

    def test_setup_failure_skips_instance(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 4: i3 setup fails → scored as 0.0, passed=False, error preserved.

        The real pipeline path: executor._execute_data catches setup
        exceptions and records {error: "setup_failed", score: 0.0}.
        Inner loop preserves the error field in instance_results (item 1 fix).
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
            assert r.get("error") == "setup_failed", (
                f"i3 should have error='setup_failed', got {r.get('error')!r}"
            )

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
        assert shared.instance_results
        assert len(shared.instance_results) == 4  # i1, i2, i3 (setup_failed), i4
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
            mutation_rate=0.0,
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

    def test_designer_validate_and_fix(self) -> None:
        """Test 8: _validate_and_fix fills empty prompts but respects RELOOP gates.

        Create a workflow with BOTH an empty-prompt AgentNode AND a
        RELOOP-only GateNode.  After _validate_and_fix:
        - The AgentNode gets a default prompt (fix applied).
        - The GateNode does NOT get a spurious PROCEED edge
          (RELOOP counts as valid outgoing flow).
        """
        from factory.outer_loop.designer import _validate_and_fix

        gate = GateNode(
            id="gate",
            evaluator_type="agent",
            evaluator_role=AgentRole.CEO,
        )
        builder = AgentNode(
            id="builder",
            role=AgentRole.BUILDER,
            prompt_template="",  # empty — should be fixed
        )
        wf = Workflow(
            name="fix-test",
            nodes={"gate": gate, "builder": builder},
            edges=[
                Edge(source="builder", target="gate"),
                Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
            ],
            start_node="builder",
        )

        seed_wf = Workflow(
            name="seed",
            nodes={"builder": AgentNode(
                id="builder", role=AgentRole.BUILDER,
                prompt_template="seed prompt",
            )},
            edges=[],
            start_node="builder",
        )

        fixed = _validate_and_fix(wf, seed_wf)

        # Fix 1 applied: empty prompt filled
        fixed_builder = fixed.nodes["builder"]
        assert isinstance(fixed_builder, AgentNode)
        assert fixed_builder.prompt_template.strip(), (
            "Empty prompt should have been filled by _validate_and_fix"
        )

        # Fix 2 NOT applied: RELOOP-only gate should NOT get a PROCEED edge
        proceed_edges = [
            e for e in fixed.edges
            if e.source == "gate" and e.condition == VerdictType.PROCEED
        ]
        assert len(proceed_edges) == 0, (
            f"RELOOP-only gate should not get a PROCEED edge, got: {proceed_edges}"
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
        """Test 10: Real pipeline evaluator processes holdout instances i5/i6.

        Uses inner_loop_factory=True (the real pipeline path) with a
        DataNode workflow.  Evaluates on holdout instances ["i5", "i6"]
        and asserts real scores (0.90, 0.95) from WiringTask.verify().
        """
        project = tmp_path / "holdout_project"
        project.mkdir()
        _bootstrap_git_project(project)

        wf = _make_datanode_workflow()

        config = SwarmConfig(
            benchmark="wiring",
            budget=4,
            population_size=2,
            tournament_size=2,
            training_instances=["i1", "i2", "i3", "i4"],
            holdout_instances=["i5", "i6"],
            frozen_node_ids=["data"],
        )
        task = WiringTask(str(project))
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config,
            inner_loop_factory=True,
            project_dir=project,
        )

        result = evaluator.evaluate(wf, str(project), ["i5", "i6"])

        irs = result.details.get("instance_results", [])
        evaluated_ids = {r["instance_id"] for r in irs if isinstance(r, dict)}
        assert "i5" in evaluated_ids, f"i5 missing from holdout results: {irs}"
        assert "i6" in evaluated_ids, f"i6 missing from holdout results: {irs}"

        # Check real scores from WiringTask
        score_map = {r["instance_id"]: r["score"] for r in irs if isinstance(r, dict)}
        assert score_map.get("i5") == pytest.approx(0.90), (
            f"i5 score should be 0.90, got {score_map.get('i5')}"
        )
        assert score_map.get("i6") == pytest.approx(0.95), (
            f"i6 score should be 0.95, got {score_map.get('i6')}"
        )


# ── Tests 11-13: high-value E2E tests ────────────────────────────


class TestEngineRunE2E:
    """Tests 11-13: full engine.run, evaluator halt, and xfail NODE_REMOVE."""

    def test_engine_run_holdout_evaluation(self, tmp_path: Path) -> None:
        """Test 11: engine.run() with real pipeline produces holdout scores.

        Uses aggregate=mean so that partial passes yield a non-zero score
        (unlike all_pass which demands every instance passes).  Verifies
        that the OuterLoopResult carries val_score ≈ 0.905, completes at
        least one generation, and that holdout eval sees exactly i5/i6
        while no earlier eval call includes those holdout instances.
        """
        project = tmp_path / "holdout_run_project"
        project.mkdir()
        _bootstrap_git_project(project)

        # Override aggregate to mean so holdout scores reflect partial success
        config_path = project / ".factory" / "config.json"
        config_path.write_text(json.dumps({"inner_loop": {"aggregate": "mean"}}))

        wf = _make_datanode_workflow()

        swarm_config = SwarmConfig(
            benchmark="wiring",
            budget=2,
            population_size=1,
            tournament_size=1,
            mutation_rate=0.0,
            designer_count=0,
            training_instances=["i1", "i2", "i3", "i4"],
            holdout_instances=["i5", "i6"],
            frozen_node_ids=["data"],
        )
        task = WiringTask(str(project))
        swarm_config.set_task(task)

        evaluator = SwarmEvaluator(
            swarm_config,
            inner_loop_factory=True,
            project_dir=project,
        )

        # Wrap evaluator.evaluate to record which instances each call receives
        eval_log: list[set[str]] = []
        original_eval = evaluator.evaluate

        def wrapped(wf: Workflow, pd: str, instances: list[str], **kw: Any) -> EvalResult:
            eval_log.append(set(instances))
            return original_eval(wf, pd, instances, **kw)

        evaluator.evaluate = wrapped  # type: ignore[assignment]

        engine = SwarmEngine(
            swarm_config,
            evaluator,
            project_dir=project,
        )

        result = engine.run(wf, project_dir=str(project))

        # val_score should reflect holdout instances i5 (0.90) and i6 (0.95)
        # minus parsimony penalty (0.01 * 2 nodes = 0.02), mean(0.90, 0.95)=0.925 - 0.02=0.905
        assert result.val_score == pytest.approx(0.905, abs=0.01), (
            f"Expected val_score ≈ 0.905 from holdout eval, got {result.val_score}"
        )
        assert result.generations_completed >= 1, (
            f"Expected at least 1 generation, got {result.generations_completed}"
        )

        # Find the first eval call that saw holdout instances
        holdout_ids = {"i5", "i6"}
        holdout_idx = next(
            (i for i, c in enumerate(eval_log) if c & holdout_ids), None,
        )
        assert holdout_idx is not None, (
            f"No eval call saw holdout instances. Calls: {eval_log}"
        )
        # No call before holdout should have seen i5 or i6
        for call_instances in eval_log[:holdout_idx]:
            assert not (call_instances & holdout_ids), (
                f"Holdout instances leaked into non-holdout eval call: {call_instances}"
            )
        # The holdout eval call saw exactly i5 and i6
        assert eval_log[holdout_idx] == holdout_ids, (
            f"Expected holdout eval to see exactly {holdout_ids}, "
            f"got {eval_log[holdout_idx]}"
        )

    def test_halt_reason_through_evaluator(self, tmp_path: Path) -> None:
        """Test 12: DataNode with split='test' → 0 items → score 0.0.

        Exercises the evaluator pipeline with a DataNode whose split filter
        matches zero task instances (WiringTask has only 'train' and 'val'
        splits, never 'test').  The executor raises ValueError("0 items"),
        which the inner loop catches, and the evaluator surfaces score=0.0
        with error details.
        """
        project = tmp_path / "halt_project"
        project.mkdir()
        _bootstrap_git_project(project)

        # Build workflow with split='test' → 0 items after filtering
        # (WiringTask instances are split into train/val only, no 'test')
        wf = Workflow(
            name="halt-e2e",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="wiring_task_copied:WiringTask",
                    subgraph_entry="process",
                    subgraph_exit="process",
                    split="test",
                ),
                "process": FnNode(id="process", command="echo ok"),
            },
            edges=[],
            start_node="data",
        )

        swarm_config = SwarmConfig(
            benchmark="wiring",
            budget=4,
            population_size=2,
            tournament_size=2,
            mutation_rate=0.0,
            training_instances=["i1", "i2"],
            holdout_instances=[],
            frozen_node_ids=["data"],
        )
        task = WiringTask(str(project))
        swarm_config.set_task(task)

        evaluator = SwarmEvaluator(
            swarm_config,
            inner_loop_factory=True,
            project_dir=project,
        )

        result = evaluator.evaluate(wf, str(project), ["i1", "i2"])

        assert result.score == 0.0, (
            f"Expected score 0.0 for 0-item DataNode, got {result.score}"
        )
        assert "0 items" in result.details.get("halt_reason", ""), (
            f"Expected halt_reason with 0 items, got {result.details}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="NODE_REMOVE deletes subgraph nodes, issue 1574",
    )
    def test_no_validation_rejection_xfail(self) -> None:
        """Test 13: remove_node on a DataNode subgraph node produces a valid workflow.

        Either refusing the removal (returning None) or rewiring to a valid
        graph satisfies issue 1574.  Currently remove_node produces an
        invalid graph.
        """
        from factory.outer_loop.mutations import remove_node

        wf = _make_datanode_workflow()
        result = remove_node(wf, "process", frozen_nodes={"data"})
        assert result is None or not result[0].validate_graph()
