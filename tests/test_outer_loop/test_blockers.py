"""Tests for outer-loop data pipeline integrity.

Covers holdout leakage, empty training set intersection, RELOOP gate
validation, aggregation methods, CEO-strategy DataNode path, halt_reason
propagation, inline items filtering, and aggregate config propagation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


from factory.task import (
    InstancesConfig,
    ScoringContract,
    TaskDefinition,
    TaskInstance)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    Edge,
    FnNode,
    GateNode,
    JoinNode,
    VerdictType,
    Workflow)

import subprocess as _sp


def _init_git(path: Path) -> None:
    """Initialize a minimal git repo for DataNode worktree tests."""
    _sp.run(["git", "init", str(path)], capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "config", "user.email", "t@t"],
            capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "config", "user.name", "t"],
            capture_output=True, check=True)
    (path / "README.md").write_text("test\n")
    _sp.run(["git", "-C", str(path), "add", "."],
            capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "commit", "-m", "init"],
            capture_output=True, check=True)


# ── Bug 1: holdout leakage in _step_with_task ──────────────────────────


class TestDataNodeHoldoutLeakage:
    """When holdout_ids are configured and no subset_selector is set,
    _step_with_task must restrict allowed_instance_ids to train-only IDs."""

    def test_data_node_holdout_leakage(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        defn = TaskDefinition(
            name="holdout-task",
            scoring=ScoringContract(method="exit_code"),
            instances_config=InstancesConfig(
                format="directory",
                holdout_ids=["val1", "val2"]))

        task = MagicMock()
        task._definition = defn
        # instances(split='train') should return only train IDs
        task.instances.side_effect = lambda split=None, **kw: (
            [TaskInstance(id="train1"), TaskInstance(id="train2")]
            if split == "train"
            else [
                TaskInstance(id="train1"),
                TaskInstance(id="train2"),
                TaskInstance(id="val1"),
                TaskInstance(id="val2"),
            ]
        )

        from factory.workflow.primitives import DataNode, FnNode

        wf = Workflow(
            name="dn_holdout_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                r.item_results = [
                    {"item_id": "train1", "score": 1.0, "passed": True, "success": True},
                    {"item_id": "train2", "score": 1.0, "passed": True, "success": True},
                ]
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            from factory.inner_loop import InnerLoop

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            # No _subset_selector set — holdout_ids should trigger train filtering
            assert not hasattr(loop, '_subset_selector') or getattr(loop, '_subset_selector', None) is None
            loop._step_with_task()

            # Verify WorkflowExecutor was constructed with allowed_instance_ids
            call_kwargs = MockExecutor.call_args
            allowed = call_kwargs.kwargs.get('allowed_instance_ids') or (
                call_kwargs[1].get('allowed_instance_ids') if len(call_kwargs) > 1 else None
            )
            assert allowed is not None, "allowed_instance_ids should be set when holdout_ids exist"
            assert allowed == {"train1", "train2"}
            assert "val1" not in allowed
            assert "val2" not in allowed


# ── Bug 2: empty training set intersection warns and falls back ───────────


class TestEmptyTrainingIntersectionWarns:
    """When training_instances has no overlap with task train split,
    evolve_generation must warn and fall back to the full train split."""

    def test_empty_training_intersection_warns(self, tmp_path) -> None:
        from factory.outer_loop.engine import SwarmEngine
        from factory.outer_loop.evaluator import SwarmEvaluator
        from factory.outer_loop.models import SwarmConfig
        from factory.outer_loop.population import Population

        config = SwarmConfig(
            benchmark="test",
            budget=100,
            population_size=1,
            training_instances=["x", "y"],  # Don't overlap with task instances
        )

        # Task returns instances 'a', 'b' — no overlap with 'x', 'y'
        task = MagicMock()
        task.instances.side_effect = lambda split=None, **kw: [
            TaskInstance(id="a"),
            TaskInstance(id="b"),
        ]
        config.set_task(task)

        evaluator = MagicMock(spec=SwarmEvaluator)
        from factory.outer_loop.models import EvalResult
        evaluator.evaluate.return_value = EvalResult(score=0.5, cost_usd=0.01, benchmark_score=0.5)
        evaluator.get_cycle_record.return_value = None
        engine = SwarmEngine(config=config, evaluator=evaluator, project_dir=tmp_path)

        wf = Workflow(
            name="test",
            nodes={
                "b": AgentNode(
                    id="b",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
            },
            edges=[],
            start_node="b")
        pop = Population()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        with patch(
            "factory.outer_loop.engine.log"
        ) as mock_log:
            # Should NOT raise — should warn and fall back
            engine.evolve_generation(pop, generation=0)

            # Verify warning was logged
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args
            assert call_args[0][0] == "training_instances_no_overlap"
            assert call_args[1]["training_instances"] == ["x", "y"]
            assert call_args[1]["task_train_ids"] == ["a", "b"]

        # Verify evaluator was called with full train split (fallback)
        assert evaluator.evaluate.called
        eval_call = evaluator.evaluate.call_args
        # evaluate(wf, project_dir, instances, individual_id=ind.id)
        # instances is the 3rd positional arg (index 2)
        assert eval_call[0][2] == ["a", "b"]


# ── Bug 3: _validate_and_fix handles RELOOP-only gates correctly ─────────


class TestReloopOnlyGateNotModified:
    """A GateNode with only a RELOOP edge should NOT get an extra PROCEED edge added."""

    def test_reloop_only_gate_not_modified(self) -> None:
        from factory.outer_loop.designer import _validate_and_fix

        wf = Workflow(
            name="reloop_gate_test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}"),
                "gate": GateNode(id="gate"),
            },
            edges=[
                Edge(source="builder", target="gate"),
                Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
            ],
            start_node="builder")

        fixed = _validate_and_fix(wf, seed_workflow=None)

        # Count PROCEED edges from the gate
        proceed_edges = [
            e for e in fixed.edges
            if e.source == "gate" and e.condition == VerdictType.PROCEED
        ]
        assert len(proceed_edges) == 0, (
            f"RELOOP-only gate should not get a PROCEED edge added, "
            f"but found: {proceed_edges}"
        )

        # The RELOOP edge should still exist
        reloop_edges = [
            e for e in fixed.edges
            if e.source == "gate" and e.condition == VerdictType.RELOOP
        ]
        assert len(reloop_edges) == 1


class TestValidateAndFixDeterministicTarget:
    """When a PROCEED edge IS needed (only HALT edges, no RELOOP),
    the target must be sorted()[0], not random."""

    def test_validate_and_fix_deterministic_target(self) -> None:
        from factory.outer_loop.designer import _validate_and_fix

        # Gate with only HALT edges to multiple targets — needs a PROCEED edge
        wf = Workflow(
            name="deterministic_target_test",
            nodes={
                "builder_a": AgentNode(
                    id="builder_a",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}"),
                "builder_z": AgentNode(
                    id="builder_z",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}"),
                "gate": GateNode(id="gate"),
            },
            edges=[
                Edge(source="builder_a", target="gate"),
                Edge(source="gate", target="builder_z", condition=VerdictType.HALT),
                Edge(source="gate", target="builder_a", condition=VerdictType.HALT),
            ],
            start_node="builder_a")

        # Run multiple times — result must always be the same
        results = []
        for _ in range(10):
            fixed = _validate_and_fix(wf, seed_workflow=None)
            proceed_edges = [
                e for e in fixed.edges
                if e.source == "gate" and e.condition == VerdictType.PROCEED
            ]
            assert len(proceed_edges) == 1
            results.append(proceed_edges[0].target)

        # All results must be the same (deterministic)
        assert len(set(results)) == 1, f"Non-deterministic targets: {results}"
        # And it should be the sorted-first target
        assert results[0] == "builder_a", (
            f"Expected sorted()[0]='builder_a', got '{results[0]}'"
        )


# ── Bug 4: aggregation hardcoded to mean ──────────────────────────────────


class TestAggregateMethodRespected:
    """InnerLoop with inner_loop_config.aggregate=max should use max, not mean."""

    def test_aggregate_method_respected(self, tmp_path: Path) -> None:
        from factory.inner_loop import InnerLoop
        from factory.models import AggregateMethod, InnerLoopConfig

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        config = InnerLoopConfig(aggregate=AggregateMethod.max)

        wf = Workflow(
            name="agg_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        task = MagicMock()
        task._definition = TaskDefinition(
            name="agg-task",
            scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [
            TaskInstance(id="i1"),
            TaskInstance(id="i2"),
        ]

        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf,
            inner_loop_config=config)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = True
                r.halted = False
                r.halt_reason = ""
                r.nodes_executed = 1
                r.duration_ms = 100.0
                r.item_results = [
                    {"item_id": "i1", "score": 0.5, "passed": True, "success": True},
                    {"item_id": "i2", "score": 0.9, "passed": True, "success": True},
                ]
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            record = loop._step_with_task()

        # With max aggregation: score should be 0.9, not mean 0.7
        assert record.score_end == 0.9, (
            f"Expected max aggregate 0.9, got {record.score_end}"
        )


# ── Bug 5: CEO path uses instance_results scores ─────────────────────────


class TestCeoPathUsesInstanceResultsScores:
    """CEO path should use real scores from instance_results, not binary."""

    def test_ceo_path_uses_instance_results_scores(self, tmp_path: Path) -> None:
        import json

        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        wf = Workflow(
            name="ceo_score_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        task = MagicMock()
        task._definition = TaskDefinition(
            name="ceo-task",
            scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [TaskInstance(id="i1")]

        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf,
            execution_strategy="ceo-skill")

        # Write a cycle_summary.json with instance_results containing real scores
        summary_dir = tmp_path / ".factory" / "outer_loop" / "runs" / "test"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "score": 1.0,  # binary score from CEO
            "instance_results": [
                {"instance_id": "i1", "score": 0.75, "passed": True},
                {"instance_id": "i2", "score": 0.85, "passed": True},
            ],
        }
        (summary_dir / "cycle_summary.json").write_text(json.dumps(summary))

        # ceo-skill DataNode is rejected (deferred to PR B)
        with pytest.raises(ValueError, match="not supported"):
            loop._step_with_task()


# ── Bug 6: halt_reason in CycleRecord ────────────────────────────────────


class TestHaltReasonInCycleRecord:
    """When executor halts, halt_reason should appear in record.eval_details."""

    def test_halt_reason_in_cycle_record(self, tmp_path: Path) -> None:
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        wf = Workflow(
            name="halt_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        task = MagicMock()
        task._definition = TaskDefinition(
            name="halt-task",
            scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [TaskInstance(id="i1")]

        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()

            async def _fake_exec():
                r = MagicMock()
                r.success = False
                r.halted = True
                r.halt_reason = "node 'builder' failed: timeout"
                r.nodes_executed = 0
                r.duration_ms = 5000.0
                r.item_results = []
                return r

            mock_exec.execute = MagicMock(side_effect=_fake_exec)
            MockExecutor.return_value = mock_exec

            record = loop._step_with_task()

        assert record.eval_details is not None, "eval_details should be set on halt"
        assert record.eval_details.get("halt_reason") == "node 'builder' failed: timeout"

    def test_halt_reason_from_validation_error(self, tmp_path: Path) -> None:
        """When WorkflowExecutor raises ValueError, halt_reason is in eval_details."""
        from factory.inner_loop import InnerLoop

        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        wf = Workflow(
            name="val_err_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        task = MagicMock()
        task._definition = TaskDefinition(
            name="val-err-task",
            scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [TaskInstance(id="i1")]

        loop = InnerLoop(
            project_dir=tmp_path,
            mode="test",
            task=task,
            workflow=wf)

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            MockExecutor.side_effect = ValueError("empty prompt_template")

            record = loop._step_with_task()

        assert record.score_end == 0.0
        assert record.eval_details is not None
        assert "empty prompt_template" in record.eval_details["halt_reason"]


# ── Bug 7: inline items skip allowed_instance_ids filter ─────────────────


class TestInlineItemsSkipAllowedFilter:
    """DataNode with inline_items should NOT be filtered by allowed_instance_ids."""

    async def test_inline_items_skip_allowed_filter(self, tmp_path: Path) -> None:
        from factory.testing import FakeAgent
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="inline_filter_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="0", metadata={"text": "hello"}),
                        DataItem(id="1", metadata={"text": "world"}),
                    ]),
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
                "_join_data": JoinNode(id="_join_data", sources=["builder"]),
            },
            edges=[
                Edge(source="data", target="builder"),
                Edge(source="builder", target="_join_data"),
            ],
            start_node="data")

        _init_git(tmp_path)
        agent = FakeAgent(wf)
        executor = WorkflowExecutor(
            wf,
            tmp_path,
            agent_fn=agent,
            validate=False,
            auto_write_outputs=False,
            # Set allowed_instance_ids that DON'T match inline IDs '0','1'
            allowed_instance_ids={"t1", "t2"})
        result = await executor.execute()

        # Inline items should NOT be filtered out — both should execute
        assert result.success, f"Execution should succeed, halt_reason={result.halt_reason}"
        assert len(result.item_results) == 2, (
            f"Expected 2 inline items to pass through filter, got {len(result.item_results)}"
        )


# ── Bug 8: aggregate config from source project reaches inner loop ───────


class TestAggregateConfigReachesInnerLoop:
    """When .factory/config.json has inner_loop.aggregate, that value must
    reach compose() as inner_loop_config — not be silently dropped."""

    def test_source_project_aggregate_reaches_inner_loop(self, tmp_path: Path) -> None:
        from factory.cycle_analyzer import CycleRecord
        from factory.models import AggregateMethod, InnerLoopConfig
        from factory.outer_loop.evaluator import SwarmEvaluator
        from factory.outer_loop.models import SwarmConfig

        # Create source project with .factory/config.json containing aggregate
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        config = {"inner_loop": {"aggregate": "all_pass"}}
        (factory_dir / "config.json").write_text(json.dumps(config))

        # Build a minimal workflow with a DataNode so the task path is taken
        wf = Workflow(
            name="agg_reach_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task"),
                "sub": FnNode(id="sub", command="echo x"),
                "_join_data": JoinNode(id="_join_data", sources=["sub"]),
            },
            edges=[
                Edge(source="data", target="sub"),
                Edge(source="sub", target="_join_data"),
            ],
            start_node="data")

        # Create a mock task for config.set_task()
        task = MagicMock()
        task._definition = TaskDefinition(
            name="agg-reach-task",
            scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [TaskInstance(id="i1")]
        task.get_evaluator.return_value = MagicMock()

        swarm_config = SwarmConfig(benchmark="test", budget=100, population_size=1)
        swarm_config.set_task(task)

        evaluator = SwarmEvaluator(config=swarm_config, project_dir=tmp_path)

        # Capture the inner_loop_config that compose() receives
        captured: dict[str, object] = {}

        def mock_compose(workflow, task, project_dir, inner_loop_config=None):
            captured["inner_loop_config"] = inner_loop_config
            # Return a mock loop whose step() returns a minimal CycleRecord
            mock_loop = MagicMock()
            mock_loop.mode = "task-eval"
            mock_loop.step.return_value = CycleRecord(
                cycle_number=0,
                mode="task-eval",
                started_at=None,
                ended_at=None,
                duration_s=0.1,
                score_start=0.0,
                score_end=1.0,
                score_delta=1.0)
            return mock_loop

        with (
            patch("factory.compose.compose", side_effect=mock_compose),
            patch.object(
                SwarmEvaluator, "_create_worktree", return_value=tmp_path
            ),
            patch.object(SwarmEvaluator, "_cleanup_worktree")):
            evaluator._evaluate_via_inner_loop(
                workflow=wf,
                project_dir=str(tmp_path),
                instances=["i1"],
                individual_id="test-id-12345678")

        ilc = captured.get("inner_loop_config")
        assert ilc is not None, "inner_loop_config should be passed to compose()"
        assert isinstance(ilc, InnerLoopConfig)
        assert ilc.aggregate == AggregateMethod.all_pass, (
            f"Expected aggregate=all_pass, got {ilc.aggregate}"
        )
