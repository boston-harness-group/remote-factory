"""Tests for SwarmEvaluator._evaluate_via_inner_loop compose/fallback paths (lines 324-350)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import SwarmConfig
from factory.workflow.primitives import AgentNode, AgentRole, Workflow


def _make_simple_workflow(name: str = "test-wf") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="build",
            ),
        },
        edges=[],
        start_node="builder",
    )


class TestEvaluateViaInnerLoopCompose:
    """When get_task() returns a Task, _evaluate_via_inner_loop uses compose()."""

    def test_compose_path_called_when_task_available(self, tmp_path: Path):
        """When SwarmConfig has a task, compose() is called."""
        from factory.cycle_analyzer import CycleRecord

        config = SwarmConfig(benchmark="test", budget=10)
        mock_task = MagicMock()
        config.set_task(mock_task)

        wf = _make_simple_workflow()
        mock_record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=1.0,
            score_start=None,
            score_end=0.75,
            score_delta=None,
        )

        evaluator = SwarmEvaluator(
            config=config,
            inner_loop_factory=lambda w: "test-mode",
        )

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = mock_record
            mock_loop.mode = "test-mode"
            mock_compose.return_value = mock_loop

            result = evaluator.evaluate(wf, str(tmp_path), ["inst1"])

        mock_compose.assert_called_once_with(wf, mock_task, tmp_path, inner_loop_config=None, split='train')
        assert result.score >= 0.0

    def test_compose_path_sets_loop_attributes(self, tmp_path: Path):
        """Compose path sets mode, frozen_nodes, test_command, etc. on the loop."""
        from factory.cycle_analyzer import CycleRecord

        config = SwarmConfig(
            benchmark="test",
            budget=10,
            test_command="pytest -v",
            test_format="exit_code",
            metric_path="results.score",
            frozen_node_ids=["builder"],
        )
        mock_task = MagicMock()
        config.set_task(mock_task)

        wf = _make_simple_workflow()
        mock_record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=1.0,
            score_start=None,
            score_end=0.5,
            score_delta=None,
        )

        evaluator = SwarmEvaluator(
            config=config,
            inner_loop_factory=lambda w: "my-mode",
        )

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = mock_record
            mock_loop.mode = "my-mode"
            mock_compose.return_value = mock_loop

            evaluator.evaluate(wf, str(tmp_path), ["inst1"])

        assert mock_loop.mode == "task-eval"
        assert mock_loop.frozen_nodes == frozenset(["builder"])
        assert mock_loop.test_command == "pytest -v"
        assert mock_loop.test_format == "exit_code"
        assert mock_loop.metric_path == "results.score"


    def test_inner_loop_factory_not_called_when_task_set(self, tmp_path: Path):
        """When task is available, _inner_loop_factory must NOT be called."""
        from factory.cycle_analyzer import CycleRecord

        config = SwarmConfig(benchmark="test", budget=10)
        mock_task = MagicMock()
        config.set_task(mock_task)

        wf = _make_simple_workflow()
        mock_record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=1.0,
            score_start=None,
            score_end=0.8,
            score_delta=None,
        )

        factory_fn = MagicMock(return_value="evolve-gen0-abc12345")
        evaluator = SwarmEvaluator(
            config=config,
            inner_loop_factory=factory_fn,
        )

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = mock_record
            mock_loop.mode = "task-eval"
            mock_compose.return_value = mock_loop

            evaluator.evaluate(wf, str(tmp_path), ["inst1"])

        factory_fn.assert_not_called()
        mock_compose.assert_called_once()


class TestEvaluateViaInnerLoopFallback:
    """When get_task() returns None, _evaluate_via_inner_loop constructs InnerLoop directly."""

    def test_fallback_path_when_no_task(self, tmp_path: Path):
        """Without a task, InnerLoop is constructed directly (no compose)."""
        from factory.cycle_analyzer import CycleRecord

        config = MagicMock(spec=SwarmConfig)
        config.get_task.return_value = None
        config.frozen_node_ids = []
        config.mandatory_node_roles = []
        config.test_command = "pytest"
        config.test_format = "pytest"
        config.metric_path = "score"

        wf = _make_simple_workflow()
        mock_record = CycleRecord(
            cycle_number=1,
            mode="test",
            started_at=None,
            ended_at=None,
            duration_s=1.0,
            score_start=None,
            score_end=0.6,
            score_delta=None,
        )

        evaluator = SwarmEvaluator(
            config=config,
            inner_loop_factory=lambda w: "evolve-mode",
        )

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
            patch("factory.inner_loop.InnerLoop") as mock_inner_loop_cls,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = mock_record
            mock_loop.mode = "evolve-mode"
            mock_inner_loop_cls.return_value = mock_loop

            result = evaluator.evaluate(wf, str(tmp_path), ["inst1"])

        mock_compose.assert_not_called()
        mock_inner_loop_cls.assert_called_once()
        call_kwargs = mock_inner_loop_cls.call_args[1]
        assert call_kwargs["project_dir"] == tmp_path
        assert call_kwargs["mode"] == "evolve-mode"
        assert call_kwargs["workflow"] is wf
        assert result.score >= 0.0
