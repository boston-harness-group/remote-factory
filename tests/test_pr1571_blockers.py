"""Tests for PR #1571 code review blockers.

Bug 1: holdout leakage in _step_with_data_node
Bug 2: empty training set intersection silently becomes 'all'
Bug 3: _validate_and_fix breaks RELOOP-only gates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from factory.task import (
    InstancesConfig,
    ScoringContract,
    TaskDefinition,
    TaskInstance,
    VerifyResult,
)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    GateNode,
    VerdictType,
    Workflow,
)


# ── Bug 1: holdout leakage in _step_with_data_node ──────────────────────────


class TestDataNodeHoldoutLeakage:
    """When holdout_ids are configured and no subset_selector is set,
    _step_with_data_node must restrict allowed_instance_ids to train-only IDs."""

    def test_data_node_holdout_leakage(self, tmp_path: Path) -> None:
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        defn = TaskDefinition(
            name="holdout-task",
            scoring=ScoringContract(method="exit_code"),
            instances_config=InstancesConfig(
                format="directory",
                holdout_ids=["val1", "val2"],
            ),
        )

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

        from factory.workflow.primitives import DataItem, DataNode, FnNode

        wf = Workflow(
            name="dn_holdout_test",
            nodes={
                "data": DataNode(
                    id="data",
                    task_ref="fake:Task",
                    subgraph_entry="sub",
                    subgraph_exit="sub",
                ),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[],
            start_node="data",
        )

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
            loop._step_with_data_node()

            # Verify WorkflowExecutor was constructed with allowed_instance_ids
            call_kwargs = MockExecutor.call_args
            allowed = call_kwargs.kwargs.get('allowed_instance_ids') or (
                call_kwargs[1].get('allowed_instance_ids') if len(call_kwargs) > 1 else None
            )
            assert allowed is not None, "allowed_instance_ids should be set when holdout_ids exist"
            assert allowed == {"train1", "train2"}
            assert "val1" not in allowed
            assert "val2" not in allowed


# ── Bug 2: empty training set intersection raises ValueError ─────────────


class TestEmptyTrainingIntersectionRaises:
    """When training_instances has no overlap with task train split,
    evolve_generation must raise ValueError instead of silently using empty list."""

    def test_empty_training_intersection_raises(self) -> None:
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
        task.instances.return_value = [
            TaskInstance(id="a"),
            TaskInstance(id="b"),
        ]
        config.set_task(task)

        evaluator = MagicMock(spec=SwarmEvaluator)
        engine = SwarmEngine(config=config, evaluator=evaluator)

        wf = Workflow(
            name="test",
            nodes={
                "b": AgentNode(
                    id="b",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                ),
            },
            edges=[],
            start_node="b",
        )
        pop = Population()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        with pytest.raises(ValueError, match="no overlap"):
            engine.evolve_generation(pop, generation=0)


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
                    prompt_template="build {project_path}",
                ),
                "gate": GateNode(id="gate"),
            },
            edges=[
                Edge(source="builder", target="gate"),
                Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
            ],
            start_node="builder",
        )

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
                    prompt_template="build {project_path}",
                ),
                "builder_z": AgentNode(
                    id="builder_z",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}",
                ),
                "gate": GateNode(id="gate"),
            },
            edges=[
                Edge(source="builder_a", target="gate"),
                Edge(source="gate", target="builder_z", condition=VerdictType.HALT),
                Edge(source="gate", target="builder_a", condition=VerdictType.HALT),
            ],
            start_node="builder_a",
        )

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
