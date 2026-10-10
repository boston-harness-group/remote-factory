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
            "data": DataNode(id="data"),
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


class TestCanary:
    """Canary: two full outer-loop runs via SwarmEngine.run() with real model."""

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

    def _run_engine(self, project: Path, budget: int = 2) -> Any:
        """Run SwarmEngine with real model."""
        from factory.outer_loop.engine import SwarmEngine
        from factory.outer_loop.evaluator import SwarmEvaluator
        from factory.outer_loop.models import SwarmConfig

        model = os.environ.get("FACTORY_MODEL", "claude-haiku-4-5-20251001")
        os.environ.setdefault("FACTORY_MODEL", model)

        task = DocQualityTask()
        wf = _make_doc_workflow()

        config = SwarmConfig(
            benchmark="doc-quality-canary",
            budget=budget,
            population_size=1,
            training_instances=["readme-cli", "tutorial-scraping"],
            designer_count=0,
            mutation_rate=0.0,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )
        engine = SwarmEngine(
            config, evaluator, project_dir=project, designer=None,
        )

        return engine.run(wf)

    def test_canary_run_1(self, project: Path) -> None:
        """First canary run — verify basic trust properties."""
        result = self._run_engine(project, budget=2)

        assert result.best_score > 0, f"Best score should be positive, got {result.best_score}"
        assert result.generations_completed >= 1

        # Run trust checks
        diag = assert_run_trustworthy(result, project)
        assert diag["checks_failed"] == 0, f"Trust checks failed: {diag}"

    def test_canary_run_2(self, project: Path) -> None:
        """Second canary run — fresh project, independent verification."""
        result = self._run_engine(project, budget=2)

        assert result.best_score > 0
        assert result.convergence_reason in (
            "budget_exhausted", "target_score_reached", "plateau",
        )

        diag = assert_run_trustworthy(result, project)
        assert diag["checks_failed"] == 0
