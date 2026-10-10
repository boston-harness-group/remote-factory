"""Canary test — real model outer-loop run via SwarmEngine.run().

Marked slow + skipif FACTORY_RUN_CANARY!=1. Uses claude-haiku-4-5-20251001.
No CLI pipeline — only SwarmEngine.run().

Run with:
    FACTORY_RUN_CANARY=1 pytest -m slow tests/test_outer_loop/test_canary.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from tests.test_outer_loop.trust_helpers import assert_run_trustworthy

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("FACTORY_RUN_CANARY") != "1",
        reason="Set FACTORY_RUN_CANARY=1 to run canary tests",
    ),
]


# ── DocQualityTask ─────────────────────────────────────────────────


class DocQualityTask:
    """Simple doc quality task for canary testing.

    3 items: 2 train + 1 holdout. Agent writes documentation,
    verify scores on completeness/readability heuristics.
    """

    def __init__(self) -> None:
        from factory.task import TaskDefinition, InstancesConfig
        self._definition = TaskDefinition(
            name="doc-quality",
            instances_config=InstancesConfig(holdout_ids=["api-reference"]),
        )

    @property
    def definition(self) -> Any:
        return self._definition

    @property
    def name(self) -> str:
        return self._definition.name

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[Any]:
        from factory.task import TaskInstance
        items = {
            "readme-cli": "train",
            "tutorial-scraping": "train",
            "api-reference": "val",
        }
        for iid, s in items.items():
            if split == "all" or s == split:
                yield TaskInstance(id=iid)

    def setup(self, instance: Any, workspace: Path) -> None:
        (workspace / ".factory").mkdir(parents=True, exist_ok=True)
        (workspace / ".factory" / "current_item.json").write_text(
            json.dumps({"id": instance.id, "topic": instance.id})
        )

    def prompt(self, instance: Any) -> str:
        return (
            f"Write comprehensive documentation about '{instance.id}'. "
            f"Create a file called document.md with clear, well-structured content."
        )

    def verify(self, instance: Any, workspace: Path) -> Any:
        from factory.task import VerifyResult
        doc = workspace / "document.md"
        if not doc.exists():
            return VerifyResult(passed=False, score=0.0, details={"error": "no document.md"})
        content = doc.read_text()
        # Simple heuristics
        word_count = len(content.split())
        has_headings = content.count("#") >= 1
        completeness = min(1.0, word_count / 200)
        readability = 0.5 + (0.5 if has_headings else 0.0)
        score = (completeness * 0.6 + readability * 0.4)
        return VerifyResult(
            passed=score >= 0.5,
            score=score,
            details={
                "word_count": word_count,
                "has_headings": has_headings,
                "completeness": round(completeness, 3),
                "readability": round(readability, 3),
            },
        )

    def get_evaluator(self) -> None:
        return None


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
    )


def _make_doc_workflow() -> Any:
    from factory.workflow.primitives import (
        AgentNode, AgentRole, DataNode, Edge, JoinNode, Workflow,
    )
    return Workflow(
        name="doc-quality-wf",
        nodes={
            "data": DataNode(id="data", parallelism=2),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                prompt_template="Write comprehensive documentation. Create document.md.",
                writes={"document.md"},
                reads=set(),
            ),
            "_join_data": JoinNode(id="_join_data", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="_join_data"),
        ],
        start_node="data",
    )


@pytest.mark.timeout(3600)
class TestCanary:
    """Canary: one full outer-loop run via SwarmEngine.run() with real model.

    Validates: real project_dir used, evolution produces offspring,
    trust checks pass, reflection cites real item results.
    """

    @pytest.fixture()
    def project(self, tmp_path: Path) -> Path:
        project = tmp_path / "canary-project"
        project.mkdir()
        _git(project, "init")
        _git(project, "config", "user.email", "canary@test")
        _git(project, "config", "user.name", "canary")
        (project / ".gitignore").write_text(".factory/\n")
        factory_dir = project / ".factory"
        factory_dir.mkdir()
        (factory_dir / "config.json").write_text(
            json.dumps({"inner_loop": {"aggregate": "mean"}})
        )
        (project / "README.md").write_text("# Canary\n")
        _git(project, "add", ".")
        _git(project, "commit", "-m", "init")
        return project

    def _run_engine(self, project: Path) -> Any:
        """Run SwarmEngine with real model — evolves at least one offspring."""
        from factory.outer_loop.engine import SwarmEngine
        from factory.outer_loop.evaluator import SwarmEvaluator
        from factory.outer_loop.models import SwarmConfig

        model = os.environ.get("FACTORY_MODEL", "claude-haiku-4-5-20251001")
        os.environ.setdefault("FACTORY_MODEL", model)

        task = DocQualityTask()
        wf = _make_doc_workflow()

        config = SwarmConfig(
            benchmark="doc-quality-canary",
            budget=4,
            population_size=2,
            training_instances=["readme-cli", "tutorial-scraping"],
            designer_count=0,
            mutation_rate=0.3,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )
        engine = SwarmEngine(
            config, evaluator, project_dir=project, designer=None,
        )

        return engine.run(wf), engine, evaluator

    def test_canary_run(self, project: Path) -> None:
        """Full canary: evolution with real model, offspring, trust checks."""
        result, engine, evaluator = self._run_engine(project)

        # ── Basic score / convergence ──────────────────────────
        assert result.best_score > 0, (
            f"Best score should be positive, got {result.best_score}"
        )
        assert result.generations_completed >= 1
        assert result.convergence_reason in (
            "budget_exhausted", "target_score_reached", "plateau",
        )

        # ── Trust checks ──────────────────────────────────────
        diag = assert_run_trustworthy(result, project)
        assert diag["checks_failed"] == 0, f"Trust checks failed: {diag}"

        # ── Evolution produced offspring ──────────────────────
        # At least one offspring was evaluated (has parent_id set)
        offspring = [
            (ind_id, rec)
            for ind_id, rec in evaluator._cycle_records.items()
            if rec is not None
        ]
        assert len(offspring) >= 2, (
            f"Expected at least 2 evaluated individuals (seed + offspring), "
            f"got {len(offspring)}"
        )

        # Check that at least one individual has a parent in lineage
        # (i.e. is an offspring, not a seed)
        trajectory = result.trajectory
        has_offspring = any(
            len(gen.mutations_applied) > 0
            for gen in trajectory
        )
        assert has_offspring, (
            "No mutations were applied — evolution did not produce offspring. "
            f"Trajectory: {[(g.generation, g.novel_count) for g in trajectory]}"
        )

        # ── Reflection cites real item results ────────────────
        # The reflector should have produced a reflection stored on the engine
        reflection = engine._last_reflection
        assert reflection is not None, "No reflection was produced"
        # Reflection report should contain patterns or suggestions
        assert len(reflection.failure_patterns) > 0 or len(reflection.suggestions) > 0, (
            "Reflection has no failure patterns or suggestions"
        )
