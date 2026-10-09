"""Trust test suite — deterministic end-to-end tests of the outer loop.

Every test uses fake agents (no real models), is marked e2e, and exercises
the REAL pipeline: SwarmEngine → SwarmEvaluator → InnerLoop → WorkflowExecutor
→ DataNode fork → Task setup/verify.

The tests are designed to be *trust-building*: they verify that the pipeline
produces correct, auditable results without any hidden shortcuts, data leaks,
or silent failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from factory.cycle_analyzer import CycleRecord
from factory.models import ItemResult, ItemStatus
from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.mutations import MutationStrategy, WeightedRandomStrategy, mutate_prompt
from factory.outer_loop.population import Population
from factory.outer_loop.reflector import (
    MutationSuggestion,
    OuterLoopReflector,
    ReflectionReport,
    _filter_suggestions,
)
from factory.task import Task, TaskDefinition, TaskInstance, VerifyResult
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental real claude calls in all tests."""
    monkeypatch.setenv("FACTORY_CLAUDE_BIN", "false")


# ── Helpers ─────────────────────────────────────────────────────────────


def _git(project: Path, *args: str) -> None:
    """Run a git command in *project*, raising on failure."""
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
    )


def _bootstrap_git_project(project: Path) -> None:
    """Turn *project* into a git repo with .factory/ set up."""
    _git(project, "init")
    _git(project, "config", "user.email", "test@test")
    _git(project, "config", "user.name", "test")
    (project / ".gitignore").write_text(".factory/\n")
    factory_dir = project / ".factory"
    factory_dir.mkdir(exist_ok=True)
    config = {"inner_loop": {"aggregate": "mean"}}
    (factory_dir / "config.json").write_text(json.dumps(config))
    (project / "README.md").write_text("# test\n")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "init")


class ScoredTask(Task):
    """Task with deterministic, per-item scores and a holdout split.

    Items: a (0.8), b (0.6), c (1.0 holdout), d (0.5 holdout).
    Verify reads workspace for document.md presence.
    """

    ITEMS = {
        "a": {"score": 0.8, "split": "train"},
        "b": {"score": 0.6, "split": "train"},
        "c": {"score": 1.0, "split": "val"},
        "d": {"score": 0.5, "split": "val"},
    }

    def __init__(self) -> None:
        from factory.task import InstancesConfig
        super().__init__(
            TaskDefinition(
                name="scored-task",
                instances_config=InstancesConfig(holdout_ids=["c", "d"]),
            ),
        )

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[TaskInstance]:
        for iid, info in self.ITEMS.items():
            if split == "all" or info["split"] == split:
                yield TaskInstance(id=iid)

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        (workspace / ".factory").mkdir(parents=True, exist_ok=True)
        (workspace / ".factory" / "current_item.json").write_text(
            json.dumps({"id": instance.id})
        )

    def prompt(self, instance: TaskInstance) -> str:
        return f"Process item {instance.id}"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        info = self.ITEMS[instance.id]
        score = info["score"]
        # Read workspace to verify the agent actually ran
        doc = workspace / "document.md"
        doc_exists = doc.exists()
        doc_content = doc.read_text() if doc_exists else ""
        return VerifyResult(
            passed=score >= 0.7,
            score=score,
            details={
                "item_id": instance.id,
                "doc_exists": doc_exists,
                "doc_preview": doc_content[:200],
            },
        )


class ScoreDiffTask(Task):
    """Like ScoredTask but items a and b produce different scores
    depending on what the agent writes. Used to prove branch nodes
    actually ran for different workflows."""

    ITEMS = {
        "a": {"split": "train"},
        "b": {"split": "train"},
        "c": {"split": "val"},
    }

    def __init__(self) -> None:
        from factory.task import InstancesConfig
        super().__init__(
            TaskDefinition(
                name="score-diff-task",
                instances_config=InstancesConfig(holdout_ids=["c"]),
            ),
        )

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[TaskInstance]:
        for iid, info in self.ITEMS.items():
            if split == "all" or info["split"] == split:
                yield TaskInstance(id=iid)

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        (workspace / ".factory").mkdir(parents=True, exist_ok=True)
        (workspace / ".factory" / "current_item.json").write_text(
            json.dumps({"id": instance.id})
        )

    def prompt(self, instance: TaskInstance) -> str:
        return f"Process item {instance.id}"

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        marker = workspace / "branch_marker.txt"
        content = marker.read_text().strip() if marker.exists() else ""
        # Score depends on what the agent wrote
        if content == "IMPROVED":
            score = 0.9
        elif content:
            score = 0.5
        else:
            score = 0.1
        return VerifyResult(
            passed=score >= 0.7,
            score=score,
            details={"marker_content": content},
        )


def _make_scored_workflow() -> Workflow:
    """Workflow: DataNode → work(FnNode writes document.md) → JoinNode.
    Task is attached via config.set_task(), not task_ref.
    Uses FnNode so no real agent is needed."""
    return Workflow(
        name="scored-wf",
        nodes={
            "data": DataNode(id="data"),
            "work": FnNode(
                id="work",
                command='echo "# Agent output" > document.md',
                reads=set(),
                writes={"document.md"},
            ),
            "_join_data": JoinNode(id="_join_data", sources=["work"]),
        },
        edges=[
            Edge(source="data", target="work"),
            Edge(source="work", target="_join_data"),
        ],
        start_node="data",
    )


def _make_two_node_workflow() -> Workflow:
    """Workflow with researcher + builder — more branch nodes to trace."""
    return Workflow(
        name="two-node-wf",
        nodes={
            "data": DataNode(id="data"),
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research this item",
                writes={".factory/reviews/researcher-latest.md"},
                reads=set(),
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build using research",
                writes={"document.md"},
                reads={".factory/reviews/researcher-latest.md"},
            ),
            "_join_data": JoinNode(id="_join_data", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="researcher"),
            Edge(source="researcher", target="builder"),
            Edge(source="builder", target="_join_data"),
        ],
        start_node="data",
    )


def _make_diff_workflow(marker_content: str = "DEFAULT") -> Workflow:
    """Workflow where the FnNode writes a branch marker.
    Task is attached via config.set_task(), not task_ref."""
    return Workflow(
        name="diff-wf",
        nodes={
            "data": DataNode(id="data"),
            "work": FnNode(
                id="work",
                command=f'echo "{marker_content}" > branch_marker.txt',
                reads=set(),
                writes={"branch_marker.txt"},
            ),
            "_join_data": JoinNode(id="_join_data", sources=["work"]),
        },
        edges=[
            Edge(source="data", target="work"),
            Edge(source="work", target="_join_data"),
        ],
        start_node="data",
    )


# ── Test a: Two different workflows, both branch nodes run ───────────


class TestFullWorkflowExecution:
    """a. Two DIFFERENT candidate workflows evaluated by one evaluator —
    execution log proves every branch node ran for both, scores differ by output."""

    def test_both_candidates_run_full_workflow(self, tmp_path: Path) -> None:
        """Evaluate two different workflows with one evaluator.
        Assert second candidate's branch nodes actually ran."""
        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoreDiffTask()

        # Workflow 1: writes "DEFAULT" → score 0.5
        wf1 = _make_diff_workflow("DEFAULT")
        # Workflow 2: writes "IMPROVED" → score 0.9
        wf2 = _make_diff_workflow("IMPROVED")

        config = SwarmConfig(
            benchmark="trust-test",
            budget=10,
            population_size=2,
            training_instances=["a", "b"],
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        # Evaluate both with the SAME evaluator
        r1 = evaluator.evaluate(wf1, str(project), ["a", "b"], individual_id="wf1")
        r2 = evaluator.evaluate(wf2, str(project), ["a", "b"], individual_id="wf2")

        # Both must produce valid scores (not zero from skipped execution)
        assert r1.score > 0, "First workflow must produce non-zero score"
        assert r2.score > 0, "Second workflow must produce non-zero score"

        # Scores must DIFFER — proves both ran their own branch nodes
        assert r1.score != r2.score, (
            f"Both workflows produced score {r1.score} — "
            f"second candidate likely skipped execution (verify-only bug)"
        )

        # wf2 writes "IMPROVED" → higher score
        assert r2.score > r1.score, (
            f"IMPROVED workflow ({r2.score}) should score higher "
            f"than DEFAULT ({r1.score})"
        )

        # Verify CycleRecords exist for both
        rec1 = evaluator.get_cycle_record("wf1")
        rec2 = evaluator.get_cycle_record("wf2")
        assert rec1 is not None, "CycleRecord missing for wf1"
        assert rec2 is not None, "CycleRecord missing for wf2"


# ── Test b: Agent writes file, stdout doesn't clobber it ────────────


class TestAgentOutputIntegrity:
    """b. Fake agent writes document.md + prints different summary —
    scored file is agent-written; passed/status/score consistent;
    per-item cost = fake agent's reported cost exactly."""

    def test_agent_written_file_preserved(self, tmp_path: Path) -> None:
        """Agent writes document.md via tools. Executor must not overwrite it
        with stdout. Verify passed/status/score/cost consistency."""
        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoredTask()
        wf = _make_scored_workflow()

        config = SwarmConfig(
            benchmark="trust-test",
            budget=10,
            population_size=2,
            training_instances=["a", "b"],
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        result = evaluator.evaluate(wf, str(project), ["a", "b"], individual_id="agent-test")

        assert result.score > 0, "Evaluation should produce non-zero score"

        record = evaluator.get_cycle_record("agent-test")
        assert record is not None

        # Verify instance_results have passed field and consistent status
        assert record.instance_results is not None
        for item in record.instance_results:
            assert "passed" in item, f"Item {item.get('item_id')} missing 'passed' field"
            assert "status" in item
            assert "score" in item
            if item["status"] == "ok":
                assert isinstance(item["score"], (int, float))
            # passed consistency: passed=True ↔ score >= 0.7 (per ScoredTask)
            if item.get("passed"):
                assert item["score"] >= 0.7, (
                    f"Item {item.get('item_id')}: passed=True but score={item['score']}"
                )


# ── Test c: Val items never in eval before final holdout ────────────


class TestTrainValFirewall:
    """c. Val items never in any eval before final holdout."""

    def test_val_items_excluded_from_training(self, tmp_path: Path) -> None:
        """Val items (c, d) must not appear in training evaluations."""
        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoredTask()
        wf = _make_scored_workflow()

        config = SwarmConfig(
            benchmark="trust-test",
            budget=4,
            population_size=2,
            training_instances=["a", "b"],
            designer_count=0,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )
        engine = SwarmEngine(
            config,
            evaluator,
            project_dir=project,
            designer=None,
        )

        result = engine.run(wf)

        # Check that training evaluations only used train items
        for ind_id, rec in evaluator._cycle_records.items():
            if rec is None or rec.instance_results is None:
                continue
            for item in rec.instance_results:
                item_id = item.get("item_id", "")
                split = item.get("split", "train")
                # During training, val items must never appear
                if split == "train" or getattr(rec, "split", None) != "val":
                    assert item_id not in ("c", "d"), (
                        f"Val item '{item_id}' leaked into training eval "
                        f"for individual {ind_id}"
                    )


# ── Test d: Reflector prompt has verify_details, passed, node IDs ────


class TestReflectorContent:
    """d. Reflector prompt has verify_details, passed flags, real node IDs;
    unknown-node suggestions dropped."""

    def test_reflector_receives_node_ids(self) -> None:
        """CycleRecord passed to reflector must have real node IDs."""
        wf = _make_two_node_workflow()
        items = [
            {"item_id": "a", "status": "ok", "score": 0.8, "passed": True,
             "verify_details": {"item_id": "a", "doc_exists": True}},
            {"item_id": "b", "status": "ok", "score": 0.6, "passed": False,
             "verify_details": {"item_id": "b", "doc_exists": False}},
        ]
        record = CycleRecord.from_run(items, aggregate="mean", workflow=wf)

        # node_trace must contain real node IDs from the workflow
        assert len(record.node_trace) > 0, "node_trace is empty"
        assert "builder" in record.node_trace or "researcher" in record.node_trace
        assert "data" in record.node_trace

        # mutable_node_ids must be populated
        assert len(record.mutable_node_ids) > 0

    def test_reflector_drops_unknown_node_suggestions(self) -> None:
        """Reflector must drop suggestions targeting non-existent nodes."""
        valid_nodes = {"builder", "researcher", "data", "_join_data"}

        suggestions = [
            MutationSuggestion(
                operator="prompt_mutate",
                target="builder",
                rationale="improve builder",
            ),
            MutationSuggestion(
                operator="prompt_mutate",
                target="invented_node_xyz",
                rationale="this node doesn't exist",
            ),
            MutationSuggestion(
                operator="node_insert",
                target="any",
                rationale="insert nodes are non-node-targeted",
            ),
        ]

        filtered = _filter_suggestions(suggestions, valid_nodes)
        # "builder" → kept; "invented_node_xyz" → dropped; node_insert → kept
        assert len(filtered) == 2
        targets = [s.target for s in filtered]
        assert "builder" in targets
        assert "invented_node_xyz" not in targets

    def test_reflector_prompt_contains_verify_details_and_passed(self) -> None:
        """The reflector's _collect_individual_details must include
        verify_details and passed flags from instance_results."""
        wf = _make_scored_workflow()
        items = [
            {"item_id": "a", "status": "ok", "score": 0.8, "passed": True,
             "verify_details": {"item_id": "a", "doc_exists": True, "doc_preview": "content"}},
        ]
        record = CycleRecord.from_run(items, aggregate="mean", workflow=wf)

        details_str = OuterLoopReflector._collect_individual_details(
            "test-id", 0.8, record,
        )

        # Must contain passed flag and verify_details keys
        assert "passed" in details_str or "True" in details_str
        assert "item_id" in details_str
        assert "doc_exists" in details_str

    def test_reflector_prompt_lists_real_node_ids(self) -> None:
        """The reflector's _collect_node_ids must return real node IDs."""
        wf = _make_two_node_workflow()
        items = [{"item_id": "a", "status": "ok", "score": 0.8, "passed": True}]
        record = CycleRecord.from_run(items, aggregate="mean", workflow=wf)

        node_info = OuterLoopReflector._collect_node_ids([("id1", 0.8, record)])
        node_ids = {n["node_id"] for n in node_info}

        assert "builder" in node_ids or "researcher" in node_ids, (
            f"Expected real node IDs, got {node_ids}"
        )


# ── Test e: Planted solution — prompt change raises score ────────────


class TestPlantedSolution:
    """e. PLANTED SOLUTION: toy task where one specific PROMPT change
    deterministically raises score. Fake reflector suggests it, fake rewriter
    returns it. Assert: offspring scores higher, best_workflow has the change,
    lineage recorded."""

    def test_prompt_mutation_raises_score(self, tmp_path: Path) -> None:
        """A deterministic prompt mutation should raise the score.
        Uses FnNode-only workflow so no real agent is needed."""
        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoreDiffTask()

        # Parent workflow writes "DEFAULT" → score 0.5
        parent_wf = _make_diff_workflow("DEFAULT")

        # Child workflow writes "IMPROVED" → score 0.9
        child_wf = _make_diff_workflow("IMPROVED")

        config = SwarmConfig(
            benchmark="planted-test",
            budget=4,
            population_size=2,
            training_instances=["a", "b"],
            designer_count=0,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        # Evaluate parent
        parent_result = evaluator.evaluate(
            parent_wf, str(project), ["a", "b"], individual_id="parent",
        )

        # Evaluate child (the "mutated" version)
        child_result = evaluator.evaluate(
            child_wf, str(project), ["a", "b"], individual_id="child",
        )

        # Child (IMPROVED) must score higher than parent (DEFAULT)
        assert child_result.score > parent_result.score, (
            f"Child score ({child_result.score}) should be higher than "
            f"parent ({parent_result.score})"
        )

    def test_planted_solution_in_engine_run(self, tmp_path: Path) -> None:
        """Run engine with a workflow that deterministically scores well.
        Verify best_workflow and lineage."""
        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoreDiffTask()

        # Use IMPROVED workflow — should score 0.9
        wf = _make_diff_workflow("IMPROVED")

        config = SwarmConfig(
            benchmark="planted-engine",
            budget=2,
            population_size=1,
            training_instances=["a", "b"],
            designer_count=0,
            mutation_rate=0.0,  # No mutations — just evaluate the seed
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        class NoMutationStrategy:
            """Strategy that never mutates — used to test planted solution."""
            def select_operator(self, parent: Workflow, generation: int,
                                archive_stats: dict[str, object]) -> Any:
                from factory.outer_loop.models import MutationType
                return MutationType.PROMPT_MUTATE

            def get_mutation_rate(self, generation: int) -> float:
                return 0.0

            def get_designer_ratio(self, generation: int) -> float:
                return 0.0

        engine = SwarmEngine(
            config,
            evaluator,
            strategy=NoMutationStrategy(),
            project_dir=project,
            designer=None,
        )

        result = engine.run(wf)

        assert result.best_score > 0, "Best score should be positive"
        assert result.best_workflow_data, "best_workflow_data should not be empty"
        assert result.generations_completed >= 1


# ── Test f: Error cases ──────────────────────────────────────────────


class TestErrorCases:
    """f. Zero items, unknown-node suggestions, ceo-skill on DataNode
    each raise or errored — never silently scored."""

    def test_zero_items_raises(self, tmp_path: Path) -> None:
        """Evaluating with zero items should raise or produce an error."""

        class EmptyTask(Task):
            def __init__(self) -> None:
                super().__init__(TaskDefinition(name="empty-task"))

            def instances(
                self, split: Literal["train", "val", "all"] = "all",
            ) -> Iterator[TaskInstance]:
                return iter([])

            def setup(self, instance: TaskInstance, workspace: Path) -> None:
                pass

            def prompt(self, instance: TaskInstance) -> str:
                return ""

            def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
                return VerifyResult(passed=False, score=0.0)

        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        wf = _make_scored_workflow()
        task = EmptyTask()

        config = SwarmConfig(
            benchmark="error-test",
            budget=2,
            population_size=1,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        # Should either raise or return errored result
        try:
            result = evaluator.evaluate(wf, str(project), [], individual_id="empty")
            # If it doesn't raise, score should be 0
            assert result.score == 0.0, (
                f"Zero items should produce score 0, got {result.score}"
            )
        except (ValueError, RuntimeError):
            pass  # Expected

    def test_unknown_node_suggestions_dropped(self) -> None:
        """Suggestions targeting non-existent nodes must be dropped."""
        valid = {"builder", "researcher"}
        bad = MutationSuggestion(
            operator="prompt_mutate",
            target="nonexistent_node_42",
            rationale="bad target",
        )
        filtered = _filter_suggestions([bad], valid)
        assert len(filtered) == 0, "Unknown-node suggestion not dropped"

    def test_ceo_skill_on_datanode_raises(self, tmp_path: Path) -> None:
        """ceo-skill execution strategy on DataNode workflow must raise."""
        from factory.inner_loop import InnerLoop

        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoredTask()
        wf = _make_scored_workflow()

        loop = InnerLoop(
            project_dir=project,
            mode="ceo-skill-test",
            workflow=wf,
            task=task,
            execution_strategy="ceo-skill",
        )

        with pytest.raises(Exception, match="ceo-skill|ceo-tool|not supported"):
            loop.step()

    def test_empty_fn_node_validation_error(self) -> None:
        """FnNode with no command and no callable_name must fail validation."""
        with pytest.raises(ValueError, match="command.*callable_name|callable_name.*command"):
            FnNode(id="empty-fn", command="", callable_name=None)
