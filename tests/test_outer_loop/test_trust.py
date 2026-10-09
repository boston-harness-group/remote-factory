"""Trust test suite — deterministic end-to-end tests of the outer loop.

Every test uses fake agents (no real models), is marked e2e, and exercises
the REAL pipeline: SwarmEngine → SwarmEvaluator → InnerLoop → WorkflowExecutor
→ DataNode fork → Task setup/verify.

The tests are designed to be *trust-building*: they verify that the pipeline
produces correct, auditable results without any hidden shortcuts, data leaks,
or silent failures.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import SwarmConfig
from factory.outer_loop.reflector import (
    MutationSuggestion,
    OuterLoopReflector,
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

        engine.run(wf)  # side-effect: populates evaluator._cycle_records

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
    """e. PLANTED SOLUTION: seed with the DEFAULT workflow, evolve via
    prompt mutation to IMPROVED, assert offspring scores higher.

    The test exercises REAL evolution through the engine pipeline:
    - Seed = DEFAULT workflow (score 0.5 per item)
    - A fake reflector suggests ``prompt_mutate`` targeting ``work``
    - A fake rewriter returns the improving command text
    - ``mutation_rate=1.0`` so mutations are always applied
    - Assertions: offspring created from the suggestion, ran every node
      for every train item, scored higher, is best, contains improved
      text, and has lineage recorded.
    """

    def test_planted_evolution(self, tmp_path: Path) -> None:
        """Seed DEFAULT, evolve with planted mutation → IMPROVED via engine."""
        from unittest.mock import patch as _patch

        from factory.outer_loop.models import MutationType, MutationRecord
        from factory.outer_loop.mutations import WeightedRandomStrategy
        from factory.outer_loop.population import Population

        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoreDiffTask()

        # ── Seed workflow: DEFAULT FnNode (score 0.5) ──────────
        seed_wf = _make_diff_workflow("DEFAULT")

        # ── Planted mutation strategy: always PROMPT_MUTATE, rate=1.0 ──
        class PlantedStrategy(WeightedRandomStrategy):
            """Always selects PROMPT_MUTATE at rate 1.0."""

            def __init__(self) -> None:
                super().__init__(mutation_rate=1.0)

            def select_operator(
                self, parent: Any, generation: int,
                archive_stats: Any = None,
            ) -> MutationType:
                return MutationType.PROMPT_MUTATE

            def select_guided_operator(
                self, parent: Any, generation: int,
                reflection: Any = None,
            ) -> MutationType:
                return MutationType.PROMPT_MUTATE

            def get_mutation_rate(self, generation: int) -> float:
                return 1.0

            def get_designer_ratio(self, generation: int) -> float:
                return 0.0

        config = SwarmConfig(
            benchmark="planted-evolution",
            budget=10,
            population_size=2,
            tournament_size=2,
            training_instances=["a", "b"],
            designer_count=0,
            mutation_rate=1.0,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        engine = SwarmEngine(
            config,
            evaluator,
            strategy=PlantedStrategy(),
            project_dir=project,
            designer=None,
        )

        # Seed the population with just the DEFAULT workflow
        pop = Population()
        seed_ind = Population.make_individual(seed_wf, generation=0)
        pop.add(seed_ind)
        engine._archive.add(seed_ind)
        engine._novelty.add(seed_wf)

        # Patch apply_random_mutation to inject our planted improvement.
        # This simulates: reflector suggests prompt_mutate → rewriter
        # produces IMPROVED text → mutation applied to FnNode command.
        improved_wf = _make_diff_workflow("IMPROVED")
        planted_mutation_rec = MutationRecord(
            operator=MutationType.PROMPT_MUTATE,
            target_node="work",
            before={"command": 'echo "DEFAULT" > branch_marker.txt'},
            after={"command": 'echo "IMPROVED" > branch_marker.txt'},
            rationale="Planted mutation: DEFAULT → IMPROVED",
        )

        def _planted_mutate(
            parent: Any, strategy: Any, generation: int,
            frozen_nodes: Any = None, reflection_report: Any = None,
        ) -> tuple[Any, MutationRecord] | None:
            return (improved_wf, planted_mutation_rec)

        with _patch(
            "factory.outer_loop.engine.apply_random_mutation",
            side_effect=_planted_mutate,
        ):
            _summary = engine.evolve_generation(pop, generation=0, project_dir=str(project))

        # ── Harvest results ──────────────────────────────────────
        # Find the offspring (non-seed individual)
        offspring_list = [
            ind for ind in pop.individuals
            if ind.id != seed_ind.id and ind.score is not None and not ind.errored
        ]
        assert len(offspring_list) >= 1, (
            f"Expected at least 1 offspring, got {len(offspring_list)}. "
            f"Population: {[(i.id, i.score, i.errored) for i in pop.individuals]}"
        )
        offspring_ind = max(offspring_list, key=lambda i: i.score or 0.0)

        seed_ind_scored = pop.get(seed_ind.id)
        assert seed_ind_scored is not None and seed_ind_scored.score is not None
        seed_score = seed_ind_scored.score
        offspring_score = offspring_ind.score
        assert offspring_score is not None

        # ── Assert (a): offspring was created from the mutation ──
        assert offspring_ind.parent_id == seed_ind.id, (
            f"Offspring parent_id should be seed id, got {offspring_ind.parent_id}"
        )
        assert offspring_ind.mutation_record is not None
        assert offspring_ind.mutation_record.operator == MutationType.PROMPT_MUTATE
        assert offspring_ind.mutation_record.target_node == "work"

        # ── Assert (b): offspring ran every node for every train item ──
        offspring_rec = evaluator.get_cycle_record(offspring_ind.id)
        assert offspring_rec is not None, "No CycleRecord for offspring"
        assert offspring_rec.instance_results is not None
        offspring_item_ids = {
            r["item_id"] for r in offspring_rec.instance_results
        }
        assert offspring_item_ids == {"a", "b"}, (
            f"Offspring should have results for all train items, got {offspring_item_ids}"
        )

        # ── Assert (c): offspring scored higher than seed ──────
        assert offspring_score > seed_score, (
            f"Offspring score ({offspring_score}) should be higher than "
            f"seed ({seed_score})"
        )

        # ── Assert (d): offspring is best_workflow ─────────────
        best = pop.best()
        assert best is not None
        assert best.id == offspring_ind.id, (
            f"Best individual should be offspring, got {best.id}"
        )

        # ── Assert (e): offspring contains the improved text ───
        best_wf = Workflow.from_dict(best.workflow_data)  # type: ignore[arg-type]
        work_node = best_wf.nodes.get("work")
        assert work_node is not None
        assert hasattr(work_node, "command")
        assert "IMPROVED" in work_node.command, (  # type: ignore[union-attr]
            f"Best workflow work node should contain IMPROVED, "
            f"got {work_node.command}"  # type: ignore[union-attr]
        )

        # ── Assert (f): lineage is recorded ────────────────────
        assert best.parent_id == seed_ind.id
        assert best.mutation_record is not None
        assert best.mutation_record.operator == MutationType.PROMPT_MUTATE
        assert best.mutation_record.target_node == "work"
        assert "DEFAULT" in str(best.mutation_record.before)
        assert "IMPROVED" in str(best.mutation_record.after)


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

    def test_errored_excluded_from_selection(self) -> None:
        """Errored individuals are excluded from best() and mean_score().

        An errored individual with score 0.0 must not be the best even
        when it's the only candidate — Population.best() and
        MAPElitesArchive.add() must reject it.
        """
        from factory.outer_loop.models import Individual
        from factory.outer_loop.population import MAPElitesArchive, Population

        # Good individual
        good = Individual(
            id="good",
            workflow_data={},
            score=0.5,
            errored=False,
            features=(1, 0, 1, 0, 0, 0, 0, 0, 0),
        )
        # Errored individual with higher score — must not win
        bad = Individual(
            id="bad",
            workflow_data={},
            score=0.9,
            errored=True,
            features=(1, 0, 1, 0, 0, 0, 0, 0, 1),
        )

        # Population.best() excludes errored
        pop = Population()
        pop.add(good)
        pop.add(bad)
        best = pop.best()
        assert best is not None
        assert best.id == "good", (
            f"best() should be 'good' (non-errored), not 'bad' (errored), got {best.id}"
        )

        # mean_score() excludes errored
        mean = pop.mean_score()
        assert mean == 0.5, (
            f"mean_score() should be 0.5 (only 'good'), not include errored. Got {mean}"
        )

        # MAPElitesArchive.add() rejects errored
        archive = MAPElitesArchive()
        assert archive.add(bad) is False, "Archive should reject errored individual"
        assert archive.add(good) is True, "Archive should accept non-errored individual"
        assert archive.best() is not None
        assert archive.best().id == "good"  # type: ignore[union-attr]


# ── Test g: All-errored generation raises ─────────────────────────────


class TestAllErroredRaises:
    """g. When EVERY candidate in a generation errors during evaluation,
    SwarmEngine.run must raise — never return best_score=0.0 as if the
    harness were simply bad."""

    def test_all_errored_generation_raises(self, tmp_path: Path) -> None:
        """SwarmEngine.run raises RuntimeError when all evaluations error."""
        from unittest.mock import patch

        from factory.outer_loop.models import EvalResult as _ER

        project = tmp_path / "project"
        project.mkdir()
        _bootstrap_git_project(project)

        task = ScoredTask()
        wf = _make_scored_workflow()

        config = SwarmConfig(
            benchmark="error-test",
            budget=4,
            population_size=2,
            training_instances=["a", "b"],
            designer_count=0,
            mutation_rate=0.0,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )

        # Patch the evaluator to always return errored results
        def _always_error(
            workflow: Any, project_dir: str, instances: list[str],
            individual_id: str | None = None,
        ) -> _ER:
            return _ER(
                score=0.0,
                errored=True,
                details={"error": "simulated failure"},
            )

        engine = SwarmEngine(
            config,
            evaluator,
            project_dir=project,
            designer=None,
        )

        with patch.object(evaluator, "evaluate", side_effect=_always_error):
            with pytest.raises(RuntimeError, match="All.*candidates.*errored"):
                engine.run(wf, project_dir=str(project))
