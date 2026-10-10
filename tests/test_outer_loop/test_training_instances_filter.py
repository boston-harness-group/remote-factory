"""Tests for SwarmConfig.training_instances intersecting with Task split support.

Regression tests for the bug where engine.py line 291 used ALL instances from
task.instances(split='train'), completely ignoring config.training_instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.population import Population
from factory.task import Task, TaskDefinition, TaskInstance
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    Workflow,
)


# ── Helpers ──────────────────────────────────────────────────────


class _SplitAwareTask(Task):
    """Task that returns four train instances: a, b, c, d."""

    def __init__(self) -> None:
        super().__init__(TaskDefinition(name="split-aware"))

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[TaskInstance]:
        all_instances = [
            TaskInstance(id="a", split="train"),
            TaskInstance(id="b", split="train"),
            TaskInstance(id="c", split="train"),
            TaskInstance(id="d", split="train"),
        ]
        for inst in all_instances:
            if split == "all" or inst.split == split:
                yield inst


def _make_config(**overrides: object) -> SwarmConfig:
    defaults: dict[str, object] = {
        "benchmark": "test",
        "budget": 30,
        "population_size": 2,
        "tournament_size": 2,
        "mutation_rate": 0.3,
        "designer_count": 0,
    }
    defaults.update(overrides)
    return SwarmConfig(**defaults)  # type: ignore[arg-type]


def _make_workflow() -> Workflow:
    return Workflow(
        name="test_filter",
        nodes={
            "study": FnNode(
                id="study", command="factory study", writes={".factory/obs.md"},
            ),
            "researcher": AgentNode(
                id="researcher", role=AgentRole.RESEARCHER,
                prompt_template="Research {project_path}",
                reads={".factory/obs.md"}, writes={".factory/research.md"},
            ),
            "builder": AgentNode(
                id="builder", role=AgentRole.BUILDER,
                prompt_template="Build for {project_path}",
                reads={".factory/research.md"}, writes={".factory/build.md"},
            ),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="builder"),
        ],
        start_node="study",
    )


class TestTrainingInstancesFilter:
    """Verify that config.training_instances filters task.instances(split='train')."""

    def test_training_instances_limits_task_split(self, tmp_path: Path) -> None:
        """Only instances in BOTH task split AND config.training_instances are used."""
        seen_instances: list[list[str]] = []

        def tracking_eval(
            wf: Workflow, project_dir: str, instances: list[str],
        ) -> EvalResult:
            seen_instances.append(list(instances))
            return EvalResult(
                score=0.0, benchmark_score=0.5, hygiene_score=0.5,
                cost_usd=0.01, complexity=1.0,
            )

        config = _make_config(training_instances=["a", "c"])
        config.set_task(_SplitAwareTask())

        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        engine = SwarmEngine(config, evaluator, project_dir=tmp_path)

        pop = Population()
        wf = _make_workflow()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        engine.evolve_generation(pop, generation=0)

        # Every evaluator call should have received only ["a", "c"]
        assert len(seen_instances) >= 1
        for call_instances in seen_instances:
            assert set(call_instances) == {"a", "c"}, (
                f"Expected only {{a, c}} but got {call_instances}"
            )
            # Order should be preserved from task.instances()
            assert call_instances == ["a", "c"]

    def test_training_instances_empty_uses_all(self, tmp_path: Path) -> None:
        """When training_instances is empty, all task split instances are used."""
        seen_instances: list[list[str]] = []

        def tracking_eval(
            wf: Workflow, project_dir: str, instances: list[str],
        ) -> EvalResult:
            seen_instances.append(list(instances))
            return EvalResult(
                score=0.0, benchmark_score=0.5, hygiene_score=0.5,
                cost_usd=0.01, complexity=1.0,
            )

        config = _make_config(training_instances=[])
        config.set_task(_SplitAwareTask())

        evaluator = SwarmEvaluator(config, evaluator_fn=tracking_eval)
        engine = SwarmEngine(config, evaluator, project_dir=tmp_path)

        pop = Population()
        wf = _make_workflow()
        ind = Population.make_individual(wf, generation=0)
        pop.add(ind)

        engine.evolve_generation(pop, generation=0)

        # All four train instances should be passed through
        assert len(seen_instances) >= 1
        for call_instances in seen_instances:
            assert set(call_instances) == {"a", "b", "c", "d"}, (
                f"Expected all four instances but got {call_instances}"
            )
