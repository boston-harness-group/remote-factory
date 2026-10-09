"""Integration tests for inner/outer loop infrastructure.

Covers:
- factory.md -> config.json round-trip with Multi-Run and Surface Scoping
- execute_multi_run() with deterministic commands and all aggregation methods
- CheckpointState with new plateau_count and loop_level fields
- Model validation for InnerLoopConfig, OuterLoopConfig, FactoryConfig
- Parser tests for _parse_inner_loop, _parse_outer_loop
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from factory.checkpoint import (
    CheckpointState,
    format_checkpoint,
    load_checkpoint,
    save_checkpoint)
from factory.models import (
    AggregateMethod,
    FactoryConfig,
    InnerLoopConfig,
    OuterLoopConfig)
from factory.research.runner import aggregate_metric
from factory.store import ExperimentStore, _parse_inner_loop, _parse_outer_loop
from factory.workflow.primitives import Workflow


# ── Model validation ────────────────────────────────────────────


class TestInnerLoopConfig:
    def test_defaults(self) -> None:
        config = InnerLoopConfig()
        assert config.runs_per_cycle == 1
        assert config.aggregate == AggregateMethod.mean
        assert config.plateau_threshold == 3
        assert config.max_inner_runs_per_cycle is None

    def test_custom_values(self) -> None:
        config = InnerLoopConfig(
            runs_per_cycle=5,
            aggregate=AggregateMethod.median,
            plateau_threshold=5,
            max_inner_runs_per_cycle=10)
        assert config.runs_per_cycle == 5
        assert config.aggregate == AggregateMethod.median
        assert config.plateau_threshold == 5
        assert config.max_inner_runs_per_cycle == 10

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            InnerLoopConfig(runs_per_cycle=1, unknown_field="x")  # type: ignore[call-arg]

    def test_all_aggregate_methods(self) -> None:
        for method in AggregateMethod:
            config = InnerLoopConfig(aggregate=method)
            assert config.aggregate == method


class TestOuterLoopConfig:
    def test_defaults(self) -> None:
        config = OuterLoopConfig()
        assert config.max_outer_cycles is None
        assert config.inner_surfaces == []
        assert config.outer_surfaces == []

    def test_custom_values(self) -> None:
        config = OuterLoopConfig(
            max_outer_cycles=5,
            inner_surfaces=["prompts/*.md"],
            outer_surfaces=["src/**/*.py"])
        assert config.max_outer_cycles == 5
        assert config.inner_surfaces == ["prompts/*.md"]
        assert config.outer_surfaces == ["src/**/*.py"]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            OuterLoopConfig(bad="field")  # type: ignore[call-arg]


class TestFactoryConfigWithLoops:
    def test_inner_loop_none_by_default(self) -> None:
        config = FactoryConfig(
            goal="test",
            scope=[],
            guards=[],
            eval_command="echo ok",
            eval_threshold=0.8,
            constraints=[])
        assert config.inner_loop is None
        assert config.outer_loop is None

    def test_with_inner_loop(self) -> None:
        config = FactoryConfig(
            goal="test",
            scope=[],
            guards=[],
            eval_command="echo ok",
            eval_threshold=0.8,
            constraints=[],
            inner_loop=InnerLoopConfig(runs_per_cycle=3))
        assert config.inner_loop is not None
        assert config.inner_loop.runs_per_cycle == 3

    def test_with_outer_loop(self) -> None:
        config = FactoryConfig(
            goal="test",
            scope=[],
            guards=[],
            eval_command="echo ok",
            eval_threshold=0.8,
            constraints=[],
            outer_loop=OuterLoopConfig(inner_surfaces=["a.md"], outer_surfaces=["b.py"]))
        assert config.outer_loop is not None
        assert config.outer_loop.inner_surfaces == ["a.md"]


# ── Parser tests ────────────────────────────────────────────────


class TestParseInnerLoop:
    def test_empty_returns_none(self) -> None:
        assert _parse_inner_loop([]) is None
        assert _parse_inner_loop("") is None
        assert _parse_inner_loop(0.0) is None

    def test_basic_parsing(self) -> None:
        items = ["runs_per_cycle: 5", "aggregate: median", "plateau_threshold: 4"]
        result = _parse_inner_loop(items)
        assert result is not None
        assert result.runs_per_cycle == 5
        assert result.aggregate == AggregateMethod.median
        assert result.plateau_threshold == 4
        assert result.max_inner_runs_per_cycle is None

    def test_with_max_inner_runs(self) -> None:
        items = ["runs_per_cycle: 3", "aggregate: max", "max_inner_runs_per_cycle: 10"]
        result = _parse_inner_loop(items)
        assert result is not None
        assert result.runs_per_cycle == 3
        assert result.aggregate == AggregateMethod.max
        assert result.max_inner_runs_per_cycle == 10

    def test_defaults_when_partial(self) -> None:
        items = ["runs_per_cycle: 2"]
        result = _parse_inner_loop(items)
        assert result is not None
        assert result.runs_per_cycle == 2
        assert result.aggregate == AggregateMethod.mean
        assert result.plateau_threshold == 3


class TestParseOuterLoop:
    def test_empty_returns_none(self) -> None:
        assert _parse_outer_loop([]) is None
        assert _parse_outer_loop("") is None
        assert _parse_outer_loop(0.0) is None

    def test_basic_parsing(self) -> None:
        items = [
            "max_outer_cycles: 5",
            "inner: prompts/*.md",
            "inner: config/*.yaml",
            "outer: src/**/*.py",
        ]
        result = _parse_outer_loop(items)
        assert result is not None
        assert result.max_outer_cycles == 5
        assert result.inner_surfaces == ["prompts/*.md", "config/*.yaml"]
        assert result.outer_surfaces == ["src/**/*.py"]

    def test_surfaces_only(self) -> None:
        items = ["inner: a.md", "outer: b.py"]
        result = _parse_outer_loop(items)
        assert result is not None
        assert result.max_outer_cycles is None
        assert result.inner_surfaces == ["a.md"]
        assert result.outer_surfaces == ["b.py"]

    def test_no_relevant_items_returns_none(self) -> None:
        items = ["something: else", "random text"]
        result = _parse_outer_loop(items)
        assert result is None


# ── Aggregation tests ───────────────────────────────────────────


class TestAggregateMetric:
    def test_mean(self) -> None:
        assert aggregate_metric([0.2, 0.4, 0.6], AggregateMethod.mean) == pytest.approx(0.4)

    def test_median_odd(self) -> None:
        assert aggregate_metric([0.1, 0.5, 0.9], AggregateMethod.median) == pytest.approx(0.5)

    def test_median_even(self) -> None:
        assert aggregate_metric([0.2, 0.4, 0.6, 0.8], AggregateMethod.median) == pytest.approx(0.5)

    def test_max(self) -> None:
        assert aggregate_metric([0.1, 0.3, 0.9], AggregateMethod.max) == pytest.approx(0.9)

    def test_all_pass(self) -> None:
        assert aggregate_metric([0.5, 0.3, 0.7], AggregateMethod.all_pass) == pytest.approx(0.3)

    def test_empty_returns_zero(self) -> None:
        assert aggregate_metric([], AggregateMethod.mean) == 0.0

    def test_single_value(self) -> None:
        for method in AggregateMethod:
            assert aggregate_metric([0.42], method) == pytest.approx(0.42)


# ── Checkpoint extension tests ──────────────────────────────────


class TestCheckpointStateExtension:
    def test_default_values(self) -> None:
        state = CheckpointState(
            mode="research",
            active_experiment_id=None,
            completed_agents=[],
            pending_agents=[],
            last_eval_scores={},
            current_hypothesis=None,
            timestamp="2026-01-01T00:00:00Z")
        assert state.plateau_count == 0
        assert state.loop_level == "inner"

    def test_custom_values(self) -> None:
        state = CheckpointState(
            mode="research",
            active_experiment_id=1,
            completed_agents=["baseline"],
            pending_agents=["builder"],
            last_eval_scores={"score": 0.5},
            current_hypothesis="test",
            timestamp="2026-01-01T00:00:00Z",
            plateau_count=2,
            loop_level="outer")
        assert state.plateau_count == 2
        assert state.loop_level == "outer"

    def test_serialization_roundtrip(self) -> None:
        state = CheckpointState(
            mode="research",
            active_experiment_id=None,
            completed_agents=[],
            pending_agents=[],
            last_eval_scores={},
            current_hypothesis=None,
            timestamp="2026-01-01T00:00:00Z",
            plateau_count=3,
            loop_level="outer")
        data = json.loads(state.model_dump_json())
        restored = CheckpointState.model_validate(data)
        assert restored.plateau_count == 3
        assert restored.loop_level == "outer"


class TestCheckpointSaveLoadFormat:
    """Tests for save/load roundtrip, backward compat, and format_checkpoint."""

    def test_save_load_roundtrip_with_new_fields(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (project / ".factory").mkdir()

        state = CheckpointState(
            mode="research",
            active_experiment_id=10,
            completed_agents=["researcher", "strategist"],
            pending_agents=["builder"],
            last_eval_scores={"acc": 0.9},
            current_hypothesis="Scale up",
            completed_hypotheses=[1, 2, 3],
            plateau_count=2,
            loop_level="outer",
            timestamp="2026-05-24T15:00:00")
        save_checkpoint(project, state)
        loaded = load_checkpoint(project)
        assert loaded is not None
        assert loaded.plateau_count == 2
        assert loaded.loop_level == "outer"
        assert loaded == state

    def test_backward_compat_without_new_fields(self, tmp_path: Path) -> None:
        """Old checkpoint JSON without plateau_count/loop_level should use defaults."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".factory").mkdir()

        old_data = {
            "mode": "design",
            "active_experiment_id": None,
            "completed_agents": [],
            "pending_agents": [],
            "last_eval_scores": {},
            "current_hypothesis": None,
            "timestamp": "2026-01-01T00:00:00",
        }
        (project / ".factory" / "checkpoint.json").write_text(json.dumps(old_data))

        loaded = load_checkpoint(project)
        assert loaded is not None
        assert loaded.plateau_count == 0
        assert loaded.loop_level == "inner"

    def test_format_checkpoint_includes_loop_info(self) -> None:
        state = CheckpointState(
            mode="research",
            active_experiment_id=5,
            completed_agents=[],
            pending_agents=[],
            last_eval_scores={},
            current_hypothesis=None,
            plateau_count=2,
            loop_level="outer",
            timestamp="2026-05-24T00:00:00")
        output = format_checkpoint(state)
        assert "research" in output
        assert "Loop level:    outer" in output
        assert "Plateau count: 2" in output

    def test_format_checkpoint_omits_zero_plateau(self) -> None:
        state = CheckpointState(
            mode="research",
            active_experiment_id=None,
            completed_agents=[],
            pending_agents=[],
            last_eval_scores={},
            current_hypothesis=None,
            plateau_count=0,
            loop_level="inner",
            timestamp="2026-05-24T00:00:00")
        output = format_checkpoint(state)
        assert "Loop level:    inner" in output
        assert "Plateau count" not in output

    def test_loop_level_rejects_invalid(self) -> None:
        """loop_level only accepts 'inner' or 'outer'."""
        with pytest.raises(Exception):
            CheckpointState(
                mode="research",
                active_experiment_id=None,
                completed_agents=[],
                pending_agents=[],
                last_eval_scores={},
                current_hypothesis=None,
                loop_level="invalid",
                timestamp="2026-05-24T00:00:00")


# ── Integration: factory.md -> config.json -> FactoryConfig ──────


class TestFactoryMdRoundTrip:
    async def test_inner_loop_roundtrip(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".factory").mkdir()

        factory_md = project / "factory.md"
        factory_md.write_text(
            "## Goal\nTest project\n\n"
            "## Scope\n### Modifiable\n- src/*.py\n\n"
            "## Guards\n- no secrets\n\n"
            "## Eval\n### Command\n```bash\necho ok\n```\n\n"
            "### Threshold\n0.8\n\n"
            "## Constraints\n- be careful\n\n"
            "## Inner Loop\n"
            "- runs_per_cycle: 5\n"
            "- aggregate: median\n"
            "- plateau_threshold: 4\n\n"
            "## Outer Loop Surfaces\n"
            "- max_outer_cycles: 3\n"
            "- inner: prompts/*.md\n"
            "- outer: src/**/*.py\n"
            "- outer: agents/**/*.md\n"
        )

        store = ExperimentStore(project)
        config = await store.reparse_config()

        assert config.inner_loop is not None
        assert config.inner_loop.runs_per_cycle == 5
        assert config.inner_loop.aggregate == AggregateMethod.median
        assert config.inner_loop.plateau_threshold == 4

        assert config.outer_loop is not None
        assert config.outer_loop.max_outer_cycles == 3
        assert config.outer_loop.inner_surfaces == ["prompts/*.md"]
        assert config.outer_loop.outer_surfaces == ["src/**/*.py", "agents/**/*.md"]

        config_json = json.loads((project / ".factory" / "config.json").read_text())
        assert config_json["inner_loop"]["runs_per_cycle"] == 5
        assert config_json["outer_loop"]["outer_surfaces"] == ["src/**/*.py", "agents/**/*.md"]

        restored = FactoryConfig(**config_json)
        assert restored.inner_loop is not None
        assert restored.inner_loop.aggregate == AggregateMethod.median
        assert restored.outer_loop is not None
        assert restored.outer_loop.max_outer_cycles == 3

    async def test_no_loop_config_roundtrip(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".factory").mkdir()

        factory_md = project / "factory.md"
        factory_md.write_text(
            "## Goal\nTest project\n\n"
            "## Scope\n### Modifiable\n- src/*.py\n\n"
            "## Guards\n- no secrets\n\n"
            "## Eval\n### Command\n```bash\necho ok\n```\n\n"
            "### Threshold\n0.8\n\n"
            "## Constraints\n- be careful\n"
        )

        store = ExperimentStore(project)
        config = await store.reparse_config()

        assert config.inner_loop is None
        assert config.outer_loop is None


# ── Spec format E2E round-trip ────────────────────────────


class TestSpecFormatRoundTrip:
    """Prove that factory.md written in the spec template format
    parses correctly through reparse_config() and round-trips via JSON."""

    @pytest.fixture()
    def math_benchmark_project(self, tmp_path: Path) -> Path:
        project = tmp_path / "math-bench"
        project.mkdir()
        (project / ".factory").mkdir()

        (project / "prompts").mkdir()
        (project / "prompts" / "solve.md").write_text(
            "Solve the following math problem step by step.\n"
        )
        (project / "config").mkdir()
        (project / "config" / "model.yaml").write_text(
            "model: llama-3.3-70b\ntemperature: 0.7\nmax_tokens: 1024\n"
        )
        (project / "src").mkdir()
        (project / "src" / "solver.py").write_text(
            "def solve(problem: str) -> str:\n    return 'answer'\n"
        )
        (project / "eval.py").write_text(
            "import json, random\n"
            "print(json.dumps({'score': round(random.uniform(0.3, 0.9), 2)}))\n"
        )

        (project / "factory.md").write_text(
            "## Goal\nImprove accuracy of math benchmark solver\n\n"
            "## Scope\n### Modifiable\n- prompts/*.md\n- config/*.yaml\n- src/solver.py\n\n"
            "## Guards\n- no secrets\n- preserve existing test coverage\n\n"
            "## Eval\n### Command\n```bash\npython eval.py\n```\n\n"
            "### Threshold\n0.85\n\n"
            "## Constraints\n- keep inference under 30s\n\n"
            "### Multi-Run\n"
            "- runs_per_cycle: 5\n"
            "- aggregate: median\n"
            "- max_inner_runs_per_cycle: 10\n"
            "- plateau_threshold: 3\n\n"
            "### Surface Scoping\n"
            "- max_outer_cycles: 4\n"
            "- inner: prompts/*.md\n"
            "- inner: config/*.yaml\n"
            "- outer: src/**/*.py\n"
        )
        return project

    def test_project_files_exist(self, math_benchmark_project: Path) -> None:
        project = math_benchmark_project
        assert (project / "prompts" / "solve.md").exists()
        assert (project / "config" / "model.yaml").exists()
        assert (project / "src" / "solver.py").exists()
        assert (project / "eval.py").exists()
        assert (project / "factory.md").exists()
        assert (project / ".factory").is_dir()

    async def test_spec_format_parses_inner_loop(self, math_benchmark_project: Path) -> None:
        store = ExperimentStore(math_benchmark_project)
        config = await store.reparse_config()

        assert config.inner_loop is not None
        assert config.inner_loop.runs_per_cycle == 5
        assert config.inner_loop.aggregate == AggregateMethod.median
        assert config.inner_loop.max_inner_runs_per_cycle == 10
        assert config.inner_loop.plateau_threshold == 3

    async def test_spec_format_parses_outer_loop(self, math_benchmark_project: Path) -> None:
        store = ExperimentStore(math_benchmark_project)
        config = await store.reparse_config()

        assert config.outer_loop is not None
        assert config.outer_loop.max_outer_cycles == 4
        assert config.outer_loop.inner_surfaces == ["prompts/*.md", "config/*.yaml"]
        assert config.outer_loop.outer_surfaces == ["src/**/*.py"]

    def test_multi_run_aggregation(self) -> None:
        scores = [0.72, 0.85, 0.64, 0.91, 0.78]
        result = aggregate_metric(scores, AggregateMethod.median)
        assert result == pytest.approx(0.78)

    async def test_eval_harness_multi_run(self, math_benchmark_project: Path) -> None:
        store = ExperimentStore(math_benchmark_project)
        config = await store.reparse_config()

        assert config.inner_loop is not None
        runs = config.inner_loop.runs_per_cycle
        assert runs == 5

        scores: list[float] = []
        for _ in range(runs):
            result = subprocess.run(
                ["python", "eval.py"],
                cwd=math_benchmark_project,
                capture_output=True,
                text=True,
                check=True)
            output = json.loads(result.stdout)
            scores.append(output["score"])

        assert len(scores) == runs

        aggregated = aggregate_metric(scores, config.inner_loop.aggregate)
        assert isinstance(aggregated, float)
        assert 0 < aggregated < 1

    async def test_config_json_roundtrip(self, math_benchmark_project: Path) -> None:
        store = ExperimentStore(math_benchmark_project)
        await store.reparse_config()

        config_json = json.loads((math_benchmark_project / ".factory" / "config.json").read_text())
        restored = FactoryConfig(**config_json)

        assert restored.inner_loop is not None
        assert restored.inner_loop.runs_per_cycle == 5
        assert restored.inner_loop.aggregate == AggregateMethod.median
        assert restored.inner_loop.max_inner_runs_per_cycle == 10
        assert restored.inner_loop.plateau_threshold == 3

        assert restored.outer_loop is not None
        assert restored.outer_loop.max_outer_cycles == 4
        assert restored.outer_loop.inner_surfaces == ["prompts/*.md", "config/*.yaml"]
        assert restored.outer_loop.outer_surfaces == ["src/**/*.py"]


# ── detect_research_plateau tests ────────────────────────────────


class TestDetectResearchPlateau:
    """Tests for detect_research_plateau in factory.strategy."""

    def test_not_enough_data(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [{"metric_value": 0.5}, {"metric_value": 0.6}]
        assert detect_research_plateau(summaries, threshold=3) is False

    def test_plateau_detected(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [
            {"metric_value": 0.8},
            {"metric_value": 0.7},
            {"metric_value": 0.6},
            {"metric_value": 0.75},
        ]
        assert detect_research_plateau(summaries, threshold=3) is True

    def test_no_plateau_with_improvement(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [
            {"metric_value": 0.5},
            {"metric_value": 0.6},
            {"metric_value": 0.7},
            {"metric_value": 0.9},
        ]
        assert detect_research_plateau(summaries, threshold=3) is False

    def test_custom_threshold(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [
            {"metric_value": 0.8},
            {"metric_value": 0.7},
            {"metric_value": 0.75},
        ]
        assert detect_research_plateau(summaries, threshold=2) is True

    def test_improvement_in_window_breaks_plateau(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [
            {"metric_value": 0.5},
            {"metric_value": 0.4},
            {"metric_value": 0.3},
            {"metric_value": 0.6},
        ]
        assert detect_research_plateau(summaries, threshold=3) is False

    def test_zero_threshold_returns_false(self) -> None:
        from factory.strategy import detect_research_plateau

        summaries = [{"metric_value": 0.5}]
        assert detect_research_plateau(summaries, threshold=0) is False


# ── DataNode integration: Task → InnerLoop → WorkflowExecutor → outer loop ──


class TestDataNodeIntegration:
    """Integration tests proving the full DataNode pipeline works end-to-end:
    Task → InnerLoop → WorkflowExecutor → DataNode → outer loop feature extraction.
    """

    @staticmethod
    def _async_return(val: object) -> MagicMock:
        async def _coro(*a: object, **kw: object) -> object:
            return val
        m = MagicMock(side_effect=_coro)
        return m

    @staticmethod
    def _make_data_workflow() -> Workflow:
        from factory.workflow.primitives import (
            DataItem, DataNode, Edge, FnNode, JoinNode, Workflow,
        )

        return Workflow(
            name="data_integration",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[
                        DataItem(id="item-1", prompt="solve 1"),
                        DataItem(id="item-2", prompt="solve 2"),
                    ],
                    parallelism=1),
                "sub_start": FnNode(id="sub_start", command="echo start"),
                "sub_end": FnNode(id="sub_end", command="echo end"),
                "join": JoinNode(id="join", sources=["sub_end"]),
            },
            edges=[
                Edge(source="data", target="sub_start"),
                Edge(source="sub_start", target="sub_end"),
                Edge(source="sub_end", target="join"),
            ],
            start_node="data")

    @staticmethod
    def _make_plain_workflow() -> Workflow:
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        return Workflow(
            name="plain",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build {project_path}"),
            },
            edges=[],
            start_node="builder")

    @staticmethod
    def _make_exec_result(success: bool = True) -> MagicMock:
        r = MagicMock()
        r.success = success
        r.halted = not success
        r.halt_reason = "" if success else "halted"
        r.nodes_executed = 1
        r.duration_ms = 100.0
        return r

    def test_data_node_routes_through_step_with_task(self, tmp_path: Path) -> None:
        """DataNode workflow routes through InnerLoop._step_with_task()."""
        from unittest.mock import AsyncMock, patch

        from factory.inner_loop import InnerLoop
        from factory.workflow.executor import ExecutionResult

        wf = self._make_data_workflow()
        task = MagicMock()

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "inst-1", "score": 1.0, "status": "ok"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop.step()

        assert record.score_end == 1.0
        assert record.cycle_number == 1

    def test_data_node_delegates_to_executor_not_per_instance(self, tmp_path: Path) -> None:
        """DataNode workflow calls WorkflowExecutor.execute() once, not per instance."""
        from unittest.mock import AsyncMock, patch

        from factory.inner_loop import InnerLoop
        from factory.workflow.executor import ExecutionResult

        wf = self._make_data_workflow()
        task = MagicMock()
        task._definition = None  # prevent holdout check from cascading through MagicMock attrs

        mock_result = ExecutionResult()
        mock_result.success = True

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result) as mock_execute:
            loop.step()

        mock_execute.assert_called_once()
        task.instances.assert_not_called()
        task.setup.assert_not_called()
        task.verify.assert_not_called()

    def test_no_data_node_gets_auto_wrapped(self, tmp_path: Path) -> None:
        """Without DataNode, InnerLoop auto-wraps with DataNode+JoinNode."""
        from unittest.mock import patch

        from factory.inner_loop import InnerLoop
        from factory.task import ScoringContract, TaskDefinition, TaskInstance, VerifyResult
        from factory.workflow.executor import ExecutionResult
        from factory.workflow.primitives import DataNode

        wf = self._make_plain_workflow()
        task = MagicMock()
        task._definition = TaskDefinition(name="mock", scoring=ScoringContract(method="exit_code"))
        task.instances.return_value = [TaskInstance(id="inst-1")]
        task.setup.return_value = None
        task.prompt.return_value = "test prompt"
        task.verify.return_value = VerifyResult(passed=True, score=0.75)

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "inst-1", "score": 0.75, "status": "ok"},
        ]

        with patch("factory.workflow.executor.WorkflowExecutor") as MockExecutor:
            mock_exec = MagicMock()
            mock_exec.execute = self._async_return(mock_result)
            MockExecutor.return_value = mock_exec

            loop = InnerLoop(project_dir=tmp_path, mode="test", task=task, workflow=wf)
            record = loop.step()

        # The workflow should now have a DataNode (auto-wrapped)
        assert any(isinstance(n, DataNode) for n in loop.workflow.nodes.values())
        assert record.score_end == pytest.approx(0.75)

    def test_compute_features_detects_data_node(self) -> None:
        """compute_features returns has_data_node=1 at index 8 for DataNode workflows."""
        from factory.outer_loop.similarity import compute_features

        wf_with = self._make_data_workflow()
        features_with = compute_features(wf_with)
        assert len(features_with) == 9
        assert features_with[8] == 1

        wf_without = self._make_plain_workflow()
        features_without = compute_features(wf_without)
        assert len(features_without) == 9
        assert features_without[8] == 0

    def test_full_pipeline_data_node_through_inner_loop_with_features(self, tmp_path: Path) -> None:
        """Full pipeline: DataNode workflow → InnerLoop.step() → CycleRecord + compute_features."""
        from unittest.mock import AsyncMock, patch

        from factory.inner_loop import InnerLoop
        from factory.outer_loop.similarity import compute_features
        from factory.workflow.executor import ExecutionResult

        wf = self._make_data_workflow()
        task = MagicMock()
        task._definition = None

        mock_result = ExecutionResult()
        mock_result.success = True
        mock_result.item_results = [
            {"item_id": "inst-1", "score": 1.0, "status": "ok"},
        ]

        loop = InnerLoop(project_dir=tmp_path, workflow=wf, task=task)

        with patch(
            "factory.workflow.executor.WorkflowExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result):
            record = loop.step()

        assert record.score_end is not None
        assert record.score_end == 1.0

        features = compute_features(wf)
        assert features[8] == 1


# ── DataNode compose() validation (PR #1483 fix 8) ───────────────


class _ComposeTestTask:
    """Task satisfying TaskProtocol for compose() tests."""

    def __init__(self) -> None:
        from factory.task import ScoringContract, TaskDefinition

        self.definition = TaskDefinition(
            name="mock", scoring=ScoringContract(method="exit_code"))
        self.scoring = self.definition.scoring
        self.constraints = None

    def instances(self):
        from factory.task import TaskInstance
        return [TaskInstance(id="inst-1")]

    def setup(self, instance, workspace):
        pass

    def prompt(self, instance):
        return "test"

    def verify(self, instance, workspace):
        from factory.task import VerifyResult
        return VerifyResult(passed=True, score=1.0)

    def get_evaluator(self):
        return None


class TestComposeDataNodeWorkflowIntegration:
    def test_compose_datanode_workflow_succeeds(self, tmp_path: Path) -> None:
        """compose() does not raise IncompatibleCompositionError for DataNode workflows."""
        from factory.compose import compose
        from factory.workflow.primitives import (
            AgentNode,
            AgentRole,
            DataItem,
            DataNode,
            Edge,
            FnNode)

        wf = Workflow(
            name="eval_only",
            nodes={
                "gen": AgentNode(
                    id="gen",
                    role=AgentRole.RESEARCHER,
                    prompt_template="research"),
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="i1", prompt="test")]),
                "sub": FnNode(id="sub", command="echo x"),
            },
            edges=[Edge(source="gen", target="data")],
            start_node="gen")

        task = _ComposeTestTask()

        loop = compose(wf, task, tmp_path)
        assert loop is not None
        assert loop.workflow is wf
