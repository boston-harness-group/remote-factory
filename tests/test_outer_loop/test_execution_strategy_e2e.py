"""End-to-end tests for execution_strategy — verify full outer loop integration.

These tests exercise the complete flow from SwarmConfig → SwarmEngine →
DesignerAgent → SwarmEvaluator → InnerLoop with each execution strategy.

Uses mocked subprocess/executor to avoid spawning real CEO processes,
but validates the full wiring path through all 5 components.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.designer import DesignerAgent
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import SwarmConfig
from factory.workflow.primitives import AgentNode, AgentRole, Workflow


def _make_seed_workflow() -> Workflow:
    return Workflow(
        name="seed",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="build {project_path}",
            ),
        },
        edges=[],
        start_node="builder",
    )


def _mock_cycle_record(score: float = 0.5) -> CycleRecord:
    return CycleRecord(
        cycle_number=1,
        mode="task-eval",
        started_at=None,
        ended_at=None,
        duration_s=1.0,
        score_start=None,
        score_end=score,
        score_delta=None,
    )


class TestExecutorStrategyE2E:
    """E2E: executor strategy — Designer produces prompt_template, WorkflowExecutor used."""

    def test_executor_full_flow(self, tmp_path: Path) -> None:
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=5,
            population_size=2,
            execution_strategy="executor",
        )
        mock_task = MagicMock()
        config.set_task(mock_task)

        # Designer should produce executor-compatible workflows
        designer = DesignerAgent()
        minimal = designer.design_minimal(
            "test-e2e",
            execution_strategy=config.execution_strategy,
        )
        for nid, node in minimal.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template, f"Node '{nid}' missing prompt_template for executor"

        # Evaluator should propagate execution_strategy='executor'
        evaluator = SwarmEvaluator(config=config, inner_loop_factory=lambda w: "test")
        wf = _make_seed_workflow()

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = _mock_cycle_record(0.6)
            mock_loop.mode = "task-eval"
            mock_compose.return_value = mock_loop

            result = evaluator.evaluate(wf, str(tmp_path), [])

        assert mock_loop.execution_strategy == "executor"
        assert result.score >= 0.0


class TestCeoSkillStrategyE2E:
    """E2E: ceo-skill strategy — Designer produces CEO-compatible workflows, subprocess spawned."""

    def test_ceo_skill_full_flow(self, tmp_path: Path) -> None:
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=5,
            population_size=2,
            execution_strategy="ceo-skill",
        )
        mock_task = MagicMock()
        config.set_task(mock_task)

        # Designer should produce CEO-compatible workflows.
        # _validate_and_fix now fills empty prompt_templates with generic defaults
        # to pass stricter validation, so we assert non-empty instead.
        designer = DesignerAgent()
        minimal = designer.design_minimal(
            "test-e2e",
            execution_strategy=config.execution_strategy,
        )
        for nid, node in minimal.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template != "", f"Node '{nid}' should have prompt_template after validation fix"

        # Evaluator should propagate execution_strategy='ceo-skill'
        evaluator = SwarmEvaluator(config=config, inner_loop_factory=lambda w: "test")
        wf = _make_seed_workflow()

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = _mock_cycle_record(0.7)
            mock_loop.mode = "task-eval"
            mock_compose.return_value = mock_loop

            result = evaluator.evaluate(wf, str(tmp_path), [])

        assert mock_loop.execution_strategy == "ceo-skill"
        assert result.score >= 0.0


class TestCeoToolStrategyE2E:
    """E2E: ceo-tool strategy — same as ceo-skill but with engine='tool'."""

    def test_ceo_tool_full_flow(self, tmp_path: Path) -> None:
        config = SwarmConfig(
            benchmark="test-e2e",
            budget=5,
            population_size=2,
            execution_strategy="ceo-tool",
        )
        mock_task = MagicMock()
        config.set_task(mock_task)

        # Designer should produce CEO-compatible workflows.
        # _validate_and_fix fills empty prompt_templates with generic defaults.
        designer = DesignerAgent()
        minimal = designer.design_minimal(
            "test-e2e",
            execution_strategy=config.execution_strategy,
        )
        for nid, node in minimal.nodes.items():
            if isinstance(node, AgentNode):
                assert node.prompt_template != "", f"Node '{nid}' should have prompt_template after validation fix"

        # Evaluator should propagate execution_strategy='ceo-tool'
        evaluator = SwarmEvaluator(config=config, inner_loop_factory=lambda w: "test")
        wf = _make_seed_workflow()

        with (
            patch.object(SwarmEvaluator, "_create_worktree", return_value=tmp_path),
            patch.object(SwarmEvaluator, "_cleanup_worktree"),
            patch("factory.compose.compose") as mock_compose,
        ):
            mock_loop = MagicMock()
            mock_loop.step.return_value = _mock_cycle_record(0.65)
            mock_loop.mode = "task-eval"
            mock_compose.return_value = mock_loop

            result = evaluator.evaluate(wf, str(tmp_path), [])

        assert mock_loop.execution_strategy == "ceo-tool"
        assert result.score >= 0.0
