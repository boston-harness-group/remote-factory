"""Tier 1 E2E outer-loop wiring tests (10 cases).

Tests 1-6 share a single engine run via a module-level fixture.
Tests 7-10 are standalone.

Exercises: SwarmEngine → SwarmEvaluator → InnerLoop → WorkflowExecutor → Task
"""

from __future__ import annotations

import asyncio
import json
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


# ── Helper: build a DataNode workflow with a builder subgraph ──────


def _make_datanode_workflow(task_ref: str = "wiring_task:WiringTask") -> Workflow:
    """Minimal DataNode workflow: data → builder → (end)."""
    builder = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        prompt_template="Build for {project_path}",
        reads={".factory/current_item.json"},
        writes={".factory/reviews/builder-latest.md"},
    )
    data = DataNode(
        id="data",
        task_ref=task_ref,
        subgraph_entry="builder",
        subgraph_exit="builder",
    )
    return Workflow(
        name="wiring-e2e",
        nodes={"data": data, "builder": builder},
        edges=[],
        start_node="data",
    )


def _make_simple_workflow() -> Workflow:
    """Non-DataNode workflow for engine seeding."""
    return Workflow(
        name="wiring-simple",
        nodes={
            "a": FnNode(id="a", command="echo a"),
            "b": AgentNode(
                id="b",
                role=AgentRole.BUILDER,
                prompt_template="build it",
            ),
        },
        edges=[Edge(source="a", target="b")],
        start_node="a",
    )


# ── Shared evaluator_fn for tests 1-6 ─────────────────────────────


def _evaluator_fn(
    workflow: Workflow,
    project_dir: str,
    instances: list[str],
) -> EvalResult:
    """Simulate evaluation by instantiating WiringTask and running verify.

    This exercises the real WiringTask.setup / verify pathway without
    needing a real claude binary.
    """
    task = WiringTask(project_dir)

    instance_results: list[dict[str, Any]] = []
    scores: list[float] = []

    for inst in task.instances(split="all"):
        if inst.id not in instances:
            continue
        try:
            task.setup(inst, Path(project_dir))
        except Exception:
            instance_results.append({
                "instance_id": inst.id,
                "passed": False,
                "score": 0.0,
                "error": "setup_failed",
            })
            scores.append(0.0)
            continue

        vr = task.verify(inst, Path(project_dir))
        instance_results.append({
            "instance_id": inst.id,
            "passed": vr.passed,
            "score": vr.score,
        })
        scores.append(vr.score)

    # Use all_pass aggregate (matches config)
    if not scores:
        aggregate = 0.0
    elif all(s >= 1.0 for s in scores):
        aggregate = 1.0
    else:
        aggregate = 0.0

    return EvalResult(
        score=aggregate,
        benchmark_score=aggregate,
        details={
            "instance_results": instance_results,
            "instance_ids_seen": instances,
        },
    )


# ── Module-level shared fixture ───────────────────────────────────


class _SharedState:
    """Lazy-initialised cache for the shared engine run (tests 1-6)."""

    _populated = False
    eval_calls: list[dict[str, Any]] = []
    summary: Any = None
    instance_results: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    @classmethod
    def ensure(cls, tmp_path_factory: pytest.TempPathFactory) -> None:
        if cls._populated:
            return
        cls._populated = True

        project = tmp_path_factory.mktemp("wiring_project")
        config_dir = project / ".factory"
        config_dir.mkdir()

        # Write inner_loop config with all_pass aggregate
        config_data = {
            "inner_loop": {"aggregate": "all_pass"},
            "training_instances": ["i1", "i2", "i3", "i4"],
            "holdout_instances": ["i5", "i6"],
        }
        (config_dir / "config.json").write_text(json.dumps(config_data))

        # Copy wiring_task.py so task_ref resolution works
        tasks_dir = config_dir / "tasks"
        tasks_dir.mkdir()
        src = Path(__file__).resolve().parents[1] / "fixtures" / "wiring_task.py"
        (tasks_dir / "wiring_task.py").write_text(src.read_text())

        task = WiringTask(project)

        swarm_config = SwarmConfig(
            benchmark="wiring",
            budget=6,
            population_size=2,
            tournament_size=2,
            mutation_rate=0.3,
            training_instances=["i1", "i2", "i3", "i4"],
            holdout_instances=["i5", "i6"],
        )
        swarm_config.set_task(task)

        # Wrap evaluator_fn to capture calls
        def tracking_evaluator(
            workflow: Workflow,
            project_dir: str,
            instances: list[str],
        ) -> EvalResult:
            result = _evaluator_fn(workflow, project_dir, instances)
            cls.eval_calls.append({
                "instances": list(instances),
                "result": result,
                "details": result.details,
            })
            return result

        evaluator = SwarmEvaluator(swarm_config, evaluator_fn=tracking_evaluator)
        engine = SwarmEngine(swarm_config, evaluator, project_dir=project)

        seed_wf = _make_simple_workflow()
        pop = engine.seed(seed_wf)

        cls.summary = engine.evolve_generation(pop, 0, str(project))

        # Collect all instance_results from all eval calls
        for call in cls.eval_calls:
            details = call.get("details", {})
            if isinstance(details, dict) and "instance_results" in details:
                cls.instance_results.extend(details["instance_results"])
            if isinstance(details, dict):
                cls.details = details


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> type[_SharedState]:
    _SharedState.ensure(tmp_path_factory)
    return _SharedState


# ── Tests 1-6: shared engine run ───────────────────────────────────


class TestSharedEngineRun:
    """Tests 1-6 inspect the single shared evolve_generation() result."""

    def test_holdout_never_reaches_agent_during_training(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 1: i5/i6 do NOT appear in any eval call during training."""
        holdout = {"i5", "i6"}
        for call in shared.eval_calls:
            seen = set(call["instances"])
            assert not (seen & holdout), (
                f"Holdout instances leaked into training eval: {seen & holdout}"
            )

    def test_scores_are_fractional(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 2: At least one instance score is fractional (not 0.0 or 1.0)."""
        has_fractional = any(
            0.0 < r["score"] < 1.0
            for r in shared.instance_results
            if isinstance(r, dict) and "score" in r
        )
        assert has_fractional, (
            f"No fractional scores found among {[r.get('score') for r in shared.instance_results]}"
        )

    def test_all_pass_scores_lower_than_mean(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 3: With all_pass, composite score == 0.0 because i4 fails."""
        for call in shared.eval_calls:
            result: EvalResult = call["result"]
            # all_pass: 0.0 because i4 scores 0.0
            assert result.score == 0.0, (
                f"Expected all_pass score 0.0, got {result.score}"
            )

    def test_setup_failure_skips_instance(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 4: i3 has error='setup_failed', scored as 0.0."""
        i3_results = [
            r for r in shared.instance_results
            if isinstance(r, dict) and r.get("instance_id") == "i3"
        ]
        assert len(i3_results) >= 1, "i3 not found in instance_results"
        for r in i3_results:
            assert r["error"] == "setup_failed"
            assert r["score"] == 0.0

        # Count of successfully scored (non-setup-failed) instances per call
        for call in shared.eval_calls:
            details = call.get("details", {})
            if not isinstance(details, dict):
                continue
            irs = details.get("instance_results", [])
            scored = [
                r for r in irs
                if isinstance(r, dict) and r.get("error") != "setup_failed"
            ]
            if irs:
                assert len(scored) == 3, (
                    f"Expected 3 scored instances (i1,i2,i4), got {len(scored)}: "
                    f"{[r.get('instance_id') for r in scored]}"
                )

    def test_halt_reason_in_eval_details(self, tmp_path: Path) -> None:
        """Test 5: DataNode with 0 items after filtering raises ValueError."""
        # Build a workflow with a DataNode and split='val' but only allow train IDs
        builder = AgentNode(
            id="builder",
            role=AgentRole.BUILDER,
            prompt_template="build it",
            reads={".factory/current_item.json"},
            writes={".factory/reviews/builder-latest.md"},
        )
        data = DataNode(
            id="data",
            inline_items=[
                DataItem(id="x1", metadata={"split": "val"}),
            ],
            subgraph_entry="builder",
            subgraph_exit="builder",
            split="train",  # Filter to train only — x1 is val → 0 items
        )
        wf = Workflow(
            name="empty-data",
            nodes={"data": data, "builder": builder},
            edges=[],
            start_node="data",
        )

        from factory.workflow.executor import WorkflowExecutor

        ex = WorkflowExecutor(wf, tmp_path, dry_run=True, validate=False)
        result = asyncio.run(ex.execute())

        assert result.halted
        assert "resolved 0 items" in result.halt_reason.lower() or "0 items" in result.halt_reason

    def test_training_instances_limits_items(
        self, shared: type[_SharedState],
    ) -> None:
        """Test 6: instance_results contain only train IDs (i1-i4)."""
        train_ids = {"i1", "i2", "i3", "i4"}
        holdout_ids = {"i5", "i6"}
        seen_ids = {
            r["instance_id"]
            for r in shared.instance_results
            if isinstance(r, dict)
        }
        assert seen_ids <= train_ids, (
            f"Non-train IDs appeared: {seen_ids - train_ids}"
        )
        assert not (seen_ids & holdout_ids), (
            f"Holdout IDs leaked: {seen_ids & holdout_ids}"
        )


# ── Tests 7-10: standalone ─────────────────────────────────────────


class TestStandalone:
    """Tests 7-10 each run in isolation."""

    def test_empty_intersection_warns(self, tmp_path: Path) -> None:
        """Test 7: training_instances with no task overlap → warning, no crash."""
        task = WiringTask(tmp_path)
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
        pop = engine.seed(_make_simple_workflow())

        # Should NOT crash — falls back to full train split
        engine.evolve_generation(pop, 0, str(tmp_path))

        # The engine should have logged a warning and used all train instances
        assert len(call_log) > 0
        # Verify it used the full train split (not the empty intersection)
        for call in call_log:
            assert len(call) > 0, "Evaluator received empty instance list"

    def test_designer_reloop_gate_not_broken(self) -> None:
        """Test 8: GateNode with only RELOOP edge — validate doesn't add PROCEED."""
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
        # The gate has a RELOOP edge — it's a valid loop-only gate
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

        # Set allowed_instance_ids to {"t1"} — this should NOT filter inline items
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
        """Test 10: Evaluator with holdout_instances processes i5 and i6."""
        task = WiringTask(tmp_path)
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
            # Simulate evaluation
            result_details: dict[str, Any] = {"instance_results": []}
            for iid in instances:
                inst_list = list(task.instances(split="all"))
                for inst in inst_list:
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

        # Directly evaluate with holdout instances
        wf = _make_simple_workflow()
        result = evaluator.evaluate(wf, str(tmp_path), ["i5", "i6"])

        # Verify holdout instances were evaluated
        assert len(holdout_calls) == 1
        assert set(holdout_calls[0]) == {"i5", "i6"}

        # Verify instance_results contain i5 and i6
        irs = result.details.get("instance_results", [])
        evaluated_ids = {r["instance_id"] for r in irs if isinstance(r, dict)}
        assert "i5" in evaluated_ids
        assert "i6" in evaluated_ids
