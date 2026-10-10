"""InnerLoop — model-like wrapper for mode + evaluator that an outer-loop optimizer calls.

CycleAnalyzer handles execution tracing (what agents ran, costs, verdicts).
Evaluator handles score interpretation (parses evaluator-specific output artifacts).
InnerLoop composes both.

Usage:
    evaluator = CirclePackingEvaluator()
    loop = InnerLoop(project_dir, mode="evolve", evaluator=evaluator)

    for i in range(budget):
        result = loop.step()
        if result.score_end > target:
            break
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from factory.compose import IncompatibleCompositionError
from factory.cycle_analyzer import CycleAnalyzer, CycleRecord
from factory.workflow.primitives import DataNode, Workflow

log = structlog.get_logger()


@dataclass
class EvalResult:
    """Structured evaluator output."""

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    artifacts: list[str] = field(default_factory=list)


@dataclass
class _SubprocessExecutionResult:
    """Duck-types ExecutionResult for CEO subprocess evaluation.

    Provides the same fields that _step_with_task() reads from
    the real ExecutionResult after execution.
    """

    success: bool = False
    halted: bool = False
    halt_reason: str = ""
    nodes_executed: int = 0
    duration_ms: int = 0
    node_outputs: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Interface for parsing evaluator-specific output artifacts.

    Each implementation knows the output format of one evaluator.
    It reads artifact files that the inner loop already produced —
    it doesn't run the evaluator itself.
    """

    def parse(self, artifact_path: Path) -> EvalResult:
        """Parse an evaluator output artifact into a structured EvalResult."""
        ...

    def parse_many(self, artifact_paths: list[Path]) -> EvalResult:
        """Parse multiple artifacts, returning the most recent/best result."""
        ...

    def get_info(self) -> dict:
        """Return static info about this evaluator (name, target, etc.)."""
        ...


class CirclePackingEvaluator:
    """Parses output artifacts from skydiscover's circle packing evaluator.

    Knows how to read JSON files with the schema:
        {sum_radii, target_ratio, validity, eval_time, combined_score}
    """

    def __init__(self, target: float = 2.635) -> None:
        self.target = target

    def parse(self, artifact_path: Path) -> EvalResult:
        try:
            data = json.loads(Path(artifact_path).read_text())
        except (json.JSONDecodeError, OSError):
            return EvalResult(score=0.0, valid=False)
        return EvalResult(
            score=float(data.get("combined_score", 0.0)),
            metrics={k: float(v) for k, v in data.items() if isinstance(v, (int, float))},
            valid=data.get("validity", 0.0) == 1.0,
            artifacts=[str(artifact_path)],
        )

    def parse_many(self, artifact_paths: list[Path]) -> EvalResult:
        best = EvalResult(score=0.0, valid=False)
        for p in artifact_paths:
            result = self.parse(p)
            if result.score > best.score:
                best = result
        return best

    def get_info(self) -> dict:
        return {
            "benchmark": "circle_packing",
            "target": self.target,
            "metrics": ["sum_radii", "target_ratio", "validity", "eval_time", "combined_score"],
        }


class InnerLoop:
    """Wraps a factory mode + evaluator. Optimizer calls loop.step().

    frozen_nodes declares which workflow nodes are immutable during outer-loop
    optimization. Node-only: edges remain mutable. Orthogonal to file-level
    mutable_surfaces/fixed_surfaces in FactoryConfig. The outer loop is
    responsible for checking is_mutable() before modifying nodes.
    """

    def __init__(
        self,
        project_dir: Path,
        mode: str = "evolve",
        evaluator: Evaluator | None = None,
        workflow: Workflow | None = None,
        frozen_nodes: frozenset[str] = frozenset(),
        test_command: str = "",
        test_format: str | None = None,
        metric_path: str = "score",
        task: Any | None = None,
        instance: Any | None = None,
        execution_strategy: str = "executor",
        inner_loop_config: Any | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.factory_dir = self.project_dir / ".factory"
        self.mode = mode
        self.evaluator = evaluator
        self.workflow = workflow
        self.frozen_nodes = frozenset(frozen_nodes)
        self.test_command = test_command
        self.test_format = test_format or "pytest"
        self.metric_path = metric_path
        self.task = task
        self.instance = instance
        self.execution_strategy = execution_strategy
        self._inner_loop_config = inner_loop_config
        self._step_count = 0
        self._history: list[CycleRecord] = []
        self._has_data_node: bool | None = None
        self._ceo_cost_warned = False
        self._validate_frozen_nodes()

        # When task is set, derive flat fields from it for backward compat
        if self.task is not None:
            defn = getattr(self.task, "definition", None)
            if defn is not None:
                if not self.test_command and hasattr(defn, "verify_config"):
                    self.test_command = defn.verify_config.command
                if hasattr(defn, "scoring"):
                    scoring = defn.scoring
                    method = getattr(scoring, "method", "pytest")
                    if test_format is None:
                        self.test_format = method

    def _validate_frozen_nodes(self) -> None:
        if not self.frozen_nodes or self.workflow is None:
            return
        invalid = self.frozen_nodes - self.workflow.nodes.keys()
        if invalid:
            raise ValueError(
                f"frozen_nodes contains IDs not in workflow.nodes: {sorted(invalid)}"
            )
        if len(self.frozen_nodes) == len(self.workflow.nodes):
            warnings.warn(
                "All nodes are frozen — outer loop has no mutable surface",
                stacklevel=3,
            )

    def is_mutable(self, node_id: str) -> bool:
        """Return True if node can be modified by the outer loop."""
        if self.workflow is None:
            return True
        if node_id not in self.workflow.nodes:
            raise ValueError(f"Unknown node ID: {node_id!r}")
        return node_id not in self.frozen_nodes

    def mutable_nodes(self) -> set[str]:
        """Return the set of node IDs the outer loop may modify."""
        if self.workflow is None:
            return set()
        return set(self.workflow.nodes.keys()) - self.frozen_nodes

    def immutable_nodes(self) -> set[str]:
        """Return the set of frozen node IDs."""
        return set(self.frozen_nodes)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        return len(path.read_text().splitlines())

    @staticmethod
    def _count_tsv_data_rows(path: Path) -> int:
        if not path.exists():
            return 0
        lines = path.read_text().splitlines()
        return max(0, len(lines) - 1)

    def step(self, directives: dict[str, Any] | None = None) -> CycleRecord:
        """Run one inner-loop cycle and return structured results.

        When self.task is None (default): runs the factory mode via subprocess
        (existing behavior, byte-for-byte backward compat).

        When self.task is set: iterates task.instances(), runs
        setup → WorkflowExecutor → verify per instance, aggregates scores
        via AggregateMethod from InnerLoopConfig.
        """
        if self.task is not None:
            return self._step_with_task(directives)
        return self._step_subprocess(directives)

    def _step_subprocess(self, directives: dict[str, Any] | None = None) -> CycleRecord:
        """Original subprocess-based step (task is None path)."""
        if directives:
            self._write_directives(directives)

        event_offset = self._count_lines(self.factory_dir / "events.jsonl")
        tsv_offset = self._count_tsv_data_rows(self.factory_dir / "results.tsv")

        head_before = self._get_git_head()
        t0 = time.monotonic()

        result = subprocess.run(
            [sys.executable, "-m", "factory", "ceo", str(self.project_dir),
             "--mode", self.mode, "--headless", "--no-worktree"],
            cwd=self.project_dir,
        )

        duration_ms = int((time.monotonic() - t0) * 1000)
        head_after = self._get_git_head()
        builder_committed = (
            head_before is not None
            and head_after is not None
            and head_before != head_after
        )

        record = self._collect_results(
            event_offset=event_offset, tsv_offset=tsv_offset,
        )

        test_score, test_details = self._run_test_command() if self.test_command else (None, None)

        self._write_cycle_summary(
            returncode=result.returncode,
            event_offset=event_offset,
            duration_ms=duration_ms,
            builder_committed=builder_committed,
            experiments=len(record.experiments),
            test_score=test_score,
            test_details=test_details,
        )

        if result.returncode != 0:
            record.errored = (record.errored or 0) + 1
        record.cycle_number = self._step_count + 1
        self._step_count += 1
        self._history.append(record)
        return record

    def _workflow_has_data_node(self) -> bool:
        if getattr(self, '_has_data_node', None) is None:
            self._has_data_node = (
                self.workflow is not None
                and any(isinstance(n, DataNode) for n in self.workflow.nodes.values())
            )
        return bool(self._has_data_node)

    def _ensure_ephemeral_mode(self) -> str:
        """Register self.workflow as an ephemeral mode for CEO subprocess discovery.

        Returns the registered mode name.
        """
        import hashlib

        assert self.workflow is not None

        # Use a content-based mode name to avoid collisions
        wf_hash = hashlib.sha256(
            self.workflow.model_dump_json().encode()
        ).hexdigest()[:8]
        mode_name = f"eval-{self.mode}-{wf_hash}"

        # Write mode JSON so WorkflowRegistry can load it
        modes_dir = self.factory_dir / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True, exist_ok=True)

        wf_data = self.workflow.to_dict()
        wf_data["name"] = mode_name
        mode_path = modes_dir / f"{mode_name}.json"
        mode_path.write_text(json.dumps(wf_data, indent=2))

        # Write workflow wrapper for WorkflowRegistry discovery
        workflows_dir = self.factory_dir / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        wrapper = (
            "import json\n"
            "from pathlib import Path\n"
            "from factory.workflow.primitives import Workflow\n"
            "\n"
            f"meta = {{'name': '{mode_name}', 'description': 'Ephemeral eval candidate'}}\n"
            "\n"
            "def workflow():\n"
            f"    data_path = Path(__file__).parent.parent / 'outer_loop' / 'modes' / '{mode_name}.json'\n"
            "    data = json.loads(data_path.read_text())\n"
            "    return Workflow.from_dict(data)\n"
        )
        (workflows_dir / f"{mode_name}.py").write_text(wrapper)

        log.debug("ephemeral_eval_mode_registered", mode=mode_name)
        return mode_name

    def _cleanup_ephemeral_mode(self, mode_name: str) -> None:
        """Remove ephemeral mode files after CEO subprocess completes."""
        mode_json = self.factory_dir / "outer_loop" / "modes" / f"{mode_name}.json"
        wrapper = self.factory_dir / "workflows" / f"{mode_name}.py"
        if mode_json.exists():
            mode_json.unlink()
        if wrapper.exists():
            wrapper.unlink()

    def _run_ceo_subprocess(self, prompt_text: str, engine: str) -> _SubprocessExecutionResult:
        """Spawn a headless CEO subprocess for workflow evaluation.

        Writes the per-instance prompt to a temp file, registers the
        workflow as an ephemeral mode, spawns
        ``factory ceo --headless --no-worktree --engine <engine>``,
        and recovers metrics from cycle_summary.json.
        """
        prompt_path = self.factory_dir / "current_prompt.md"
        self.factory_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text)

        # Register workflow as ephemeral mode so CEO can discover it
        mode_name = self._ensure_ephemeral_mode()

        cmd = [
            sys.executable, "-m", "factory", "ceo", str(self.project_dir),
            "--headless", "--no-worktree",
            "--engine", engine,
            "--mode", mode_name,
            "--prompt", str(prompt_path),
        ]

        env = dict(__import__("os").environ)
        env["FACTORY_CEO_RESPAWN_DISABLED"] = "1"

        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, cwd=self.project_dir, env=env)
            duration_ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:
            log.error("ceo_subprocess_failed", error=str(exc))
            return _SubprocessExecutionResult(
                success=False,
                halted=True,
                halt_reason=str(exc),
            )
        finally:
            if prompt_path.exists():
                prompt_path.unlink()
            # Cleanup ephemeral mode files
            self._cleanup_ephemeral_mode(mode_name)

        # Recover supplementary metrics from cycle_summary.json
        # Check registered mode_name path first, fall back to self.mode
        summary_path = (
            self.factory_dir / "outer_loop" / "runs" / mode_name / "cycle_summary.json"
        )
        if not summary_path.exists():
            summary_path = (
                self.factory_dir / "outer_loop" / "runs" / self.mode / "cycle_summary.json"
            )
        nodes_executed = 0
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text())
                nodes_executed = int(summary_data.get("agents_spawned", 0))
            except (json.JSONDecodeError, OSError):
                pass

        return _SubprocessExecutionResult(
            success=result.returncode == 0,
            halted=result.returncode != 0,
            halt_reason=f"exit code {result.returncode}" if result.returncode != 0 else "",
            nodes_executed=nodes_executed,
            duration_ms=duration_ms,
        )

    def _step_with_task(self, directives: dict[str, Any] | None = None) -> CycleRecord:
        """Task-driven step: setup → WorkflowExecutor → verify per instance."""
        assert self.task is not None
        assert self.workflow is not None
        import asyncio
        import statistics

        from factory.compose import validate_composition
        from factory.models import AggregateMethod, InnerLoopConfig
        from factory.workflow.executor import ExecutionResult, WorkflowExecutor

        if directives:
            self._write_directives(directives)

        if self._workflow_has_data_node():
            return self._step_with_data_node(directives)

        # Belt-and-suspenders: catch post-mutation composition failures
        # (e.g. NODE_REMOVE stripping the Builder after initial composition)
        try:
            validate_composition(self.workflow, self.task)
        except IncompatibleCompositionError as exc:
            log.warning(
                "composition_incompatible",
                workflow=getattr(self.workflow, "name", "unknown"),
                task=getattr(self.task, "name", "unknown"),
                error=str(exc),
            )
            self._step_count += 1
            record = CycleRecord(
                cycle_number=self._step_count,
                mode=self.mode,
                started_at=None,
                ended_at=None,
                duration_s=0.0,
                score_start=None,
                score_end=0.0,
                score_delta=None,
            )
            self._history.append(record)
            return record

        # One-time cost warning for CEO subprocess strategies
        if self.execution_strategy in ("ceo-skill", "ceo-tool") and not self._ceo_cost_warned:
            log.warning(
                "ceo_subprocess_cost_warning",
                execution_strategy=self.execution_strategy,
                note="CEO subprocess evaluation is 5-10x more expensive per evaluation than executor",
            )
            self._ceo_cost_warned = True

        event_offset = self._count_lines(self.factory_dir / "events.jsonl")

        t0 = time.monotonic()
        workflow = self.workflow

        # Default to train split when holdout_ids are configured and no
        # subset_selector is set, to prevent holdout leakage into training.
        subset_selector = getattr(self, "_subset_selector", None)
        _defn = getattr(self.task, "_definition", None)
        _holdout_ids = (
            getattr(getattr(_defn, "instances_config", None), "holdout_ids", None)
            if _defn is not None
            else None
        )
        if subset_selector is None and _holdout_ids:
            try:
                all_instances = list(self.task.instances(split="train"))
            except TypeError:
                log.warning('task_instances_no_split', task=type(self.task).__name__)
                all_instances = list(self.task.instances())
        else:
            all_instances = list(self.task.instances())
        if subset_selector is not None:
            selected_ids = subset_selector.select(
                [inst.id for inst in all_instances]
            )
            if selected_ids:
                selected_set = set(selected_ids)
                all_instances = [i for i in all_instances if i.id in selected_set]
            # If selected_ids is empty, skip filtering (allow all instances)

        instance_results: list[dict[str, Any]] = []
        scores: list[float] = []

        # Pre-validate workflow so evolved candidates with empty prompts
        # get score=0 instead of crashing on every instance.
        if self.execution_strategy not in ("ceo-skill", "ceo-tool"):
            try:
                from factory.workflow.validation import validate_workflow

                issues = validate_workflow(workflow)
                if issues:
                    raise ValueError(
                        f"Workflow '{workflow.name}' has validation errors:\n"
                        + "\n".join(f"  - {i}" for i in issues)
                    )
            except ValueError as exc:
                log.warning(
                    "workflow_validation_failed",
                    error=str(exc),
                    workflow=getattr(workflow, "name", "unknown"),
                )
                duration_s = time.monotonic() - t0
                record = CycleRecord(
                    cycle_number=self._step_count + 1,
                    mode=self.mode,
                    started_at=None,
                    ended_at=None,
                    duration_s=duration_s,
                    score_start=None,
                    score_end=0.0,
                    score_delta=None,
                )
                record.frozen_nodes = sorted(self.frozen_nodes)
                record.mutable_node_ids = sorted(self.mutable_nodes())
                self._step_count += 1
                self._history.append(record)
                return record

        for inst in all_instances:
            try:
                try:
                    self.task.setup(inst, self.project_dir)
                except Exception as setup_exc:
                    log.warning(
                        "instance_setup_failed",
                        instance_id=inst.id,
                        error=str(setup_exc),
                    )
                    instance_results.append({
                        "instance_id": inst.id,
                        "passed": False,
                        "score": 0.0,
                        "error": "setup_failed",
                    })
                    scores.append(0.0)
                    continue

                prompt_text = self.task.prompt(inst)

                exec_result: _SubprocessExecutionResult | ExecutionResult
                if self.execution_strategy in ("ceo-skill", "ceo-tool"):
                    engine = "tool" if self.execution_strategy == "ceo-tool" else "skill"
                    exec_result = self._run_ceo_subprocess(prompt_text, engine=engine)
                else:
                    executor = WorkflowExecutor(
                        workflow,
                        self.project_dir,
                        initial_context=prompt_text,
                        validate=False,
                    )
                    exec_result = asyncio.run(executor.execute())

                vr = self.task.verify(inst, self.project_dir)

                vr_details = dict(vr.details)
                vr_details["executor_success"] = exec_result.success
                vr_details["executor_nodes_executed"] = exec_result.nodes_executed
                vr_details["executor_duration_ms"] = exec_result.duration_ms
                if exec_result.halted:
                    vr_details["executor_halt_reason"] = exec_result.halt_reason

                instance_results.append({
                    "instance_id": inst.id,
                    "passed": vr.passed,
                    "score": vr.score,
                    "details": vr_details,
                })
                scores.append(vr.score)
            except Exception as exc:
                instance_results.append({
                    "instance_id": inst.id,
                    "passed": False,
                    "score": 0.0,
                    "error": str(exc),
                })
                scores.append(0.0)

        aggregate_method = (
            self._inner_loop_config.aggregate
            if self._inner_loop_config
            else InnerLoopConfig().aggregate
        )

        if not scores:
            aggregate_score = 0.0
        elif aggregate_method == AggregateMethod.mean:
            aggregate_score = statistics.mean(scores)
        elif aggregate_method == AggregateMethod.median:
            aggregate_score = statistics.median(scores)
        elif aggregate_method == AggregateMethod.max:
            aggregate_score = max(scores)
        elif aggregate_method == AggregateMethod.all_pass:
            aggregate_score = 1.0 if all(s >= 1.0 for s in scores) else 0.0
        else:
            aggregate_score = statistics.mean(scores)

        duration_s = time.monotonic() - t0

        analyzer = CycleAnalyzer(
            self.factory_dir,
            workflow=self.workflow,
            event_offset=event_offset,
        )
        cost_record = analyzer.latest()
        total_cost_usd = cost_record.total_cost_usd if cost_record else 0.0
        cost_by_agent = cost_record.cost_by_agent if cost_record else {}

        record = CycleRecord(
            cycle_number=self._step_count + 1,
            mode=self.mode,
            started_at=None,
            ended_at=None,
            duration_s=duration_s,
            score_start=None,
            score_end=aggregate_score,
            score_delta=None,
            instance_results=instance_results,
            total_cost_usd=total_cost_usd,
            cost_by_agent=cost_by_agent,
        )
        record.frozen_nodes = sorted(self.frozen_nodes)
        record.mutable_node_ids = sorted(self.mutable_nodes())

        self._write_cycle_summary(
            returncode=0,
            event_offset=event_offset,
            duration_ms=int(duration_s * 1000),
            builder_committed=False,
            experiments=0,
            test_score=aggregate_score,
            instance_results=instance_results,
        )

        self._step_count += 1
        self._history.append(record)
        return record

    def _step_with_data_node(self, directives: dict[str, Any] | None = None) -> CycleRecord:
        """Execute DataNode workflows, respecting execution_strategy.

        executor: WorkflowExecutor handles DataNode iteration internally via _execute_data().
        ceo-skill: CEO subprocess reads SKILL.md with concrete factory task CLI steps
            for per-item lifecycle.
        ceo-tool: Same as ceo-skill but CEO uses tool-based execution engine.
        """
        import asyncio
        import json

        t0 = time.monotonic()
        assert self.workflow is not None

        if self.execution_strategy in ('ceo-skill', 'ceo-tool'):
            import statistics

            from factory.models import AggregateMethod, InnerLoopConfig

            # CEO subprocess path — same as non-DataNode workflows
            engine = 'tool' if self.execution_strategy == 'ceo-tool' else 'skill'

            # Holdout firewall: warn if holdout_ids are configured (CEO subprocess
            # doesn't support instance filtering)
            _defn = getattr(self.task, '_definition', None) if self.task is not None else None
            _holdout = (
                getattr(getattr(_defn, 'instances_config', None), 'holdout_ids', None)
                if _defn else None
            )
            if _holdout and self.task is not None:
                log.warning('ceo_strategy_no_holdout_firewall', holdout_ids=_holdout)

            # Build a prompt that tells the CEO about the DataNode workflow
            prompt_text = (
                'Execute this workflow which contains a DataNode for data iteration. '
                'Follow the SKILL.md instructions to iterate over items using factory task CLI commands.'
            )
            exec_result = self._run_ceo_subprocess(prompt_text, engine=engine)

            duration_s = time.monotonic() - t0
            score = 1.0 if exec_result.success else 0.0

            # Try to recover per-item scores from cycle_summary.json
            instance_results: list[dict[str, Any]] | None = None
            summary_path = self.factory_dir / 'outer_loop' / 'runs' / self.mode / 'cycle_summary.json'
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    if 'instance_results' in summary:
                        instance_results = summary['instance_results']
                    if 'score' in summary:
                        score = float(summary['score'])
                except (json.JSONDecodeError, OSError):
                    pass

            # Use real scores from instance_results when available
            if instance_results:
                item_scores = [
                    r.get('score', 0.0) for r in instance_results
                    if isinstance(r, dict)
                ]
                if item_scores:
                    aggregate_method = (
                        self._inner_loop_config.aggregate
                        if self._inner_loop_config
                        else InnerLoopConfig().aggregate
                    )
                    if aggregate_method == AggregateMethod.mean:
                        score = statistics.mean(item_scores)
                    elif aggregate_method == AggregateMethod.median:
                        score = statistics.median(item_scores)
                    elif aggregate_method == AggregateMethod.max:
                        score = max(item_scores)
                    elif aggregate_method == AggregateMethod.all_pass:
                        score = 1.0 if all(s >= 1.0 for s in item_scores) else 0.0
                    else:
                        score = statistics.mean(item_scores)

            record = CycleRecord(
                cycle_number=self._step_count + 1,
                mode=self.mode,
                started_at=None,
                ended_at=None,
                duration_s=duration_s,
                score_start=None,
                score_end=score,
                score_delta=None,
                instance_results=instance_results,
            )
            record.frozen_nodes = sorted(self.frozen_nodes)
            record.mutable_node_ids = sorted(self.mutable_nodes())

            self._write_cycle_summary(
                returncode=0 if exec_result.success else 1,
                event_offset=0,
                duration_ms=int(duration_s * 1000),
                builder_committed=False,
                experiments=0,
                test_score=score,
                instance_results=instance_results,
            )

            self._step_count += 1
            self._history.append(record)
            return record

        # executor path — WorkflowExecutor handles DataNode iteration
        from factory.workflow.executor import WorkflowExecutor

        # Get allowed instance IDs from subset selector (train/val firewall)
        subset_selector = getattr(self, '_subset_selector', None)
        allowed_instance_ids: set[str] | None = None

        # Default to train split when holdout_ids are configured and no
        # subset_selector is set, to prevent holdout leakage into training.
        _defn = getattr(self.task, '_definition', None) if self.task is not None else None
        _holdout_ids = (
            getattr(getattr(_defn, 'instances_config', None), 'holdout_ids', None)
            if _defn is not None
            else None
        )
        if subset_selector is None and _holdout_ids and self.task is not None:
            try:
                train_ids = [inst.id for inst in self.task.instances(split='train')]
            except TypeError:
                log.warning('task_instances_no_split', task=type(self.task).__name__)
                train_ids = [inst.id for inst in self.task.instances()]
            allowed_instance_ids = set(train_ids)

        if subset_selector is not None and self.task is not None:
            all_ids = [inst.id for inst in self.task.instances()]
            selected = subset_selector.select(all_ids)
            if selected:
                allowed_instance_ids = set(selected)
            # If selected is empty, leave allowed_instance_ids = None (allow everything)
            # This handles the case where no split is configured and training_instances is empty

        try:
            executor = WorkflowExecutor(
                self.workflow,
                self.project_dir,
                allowed_instance_ids=allowed_instance_ids,
                task=self.task,  # Pass task for verify() on non-task_ref DataNodes
            )
            exec_result_wf = asyncio.run(executor.execute())
        except ValueError as exc:
            log.warning(
                "workflow_validation_failed",
                error=str(exc),
                workflow=getattr(self.workflow, "name", "unknown"),
            )
            duration_s = time.monotonic() - t0
            record = CycleRecord(
                cycle_number=self._step_count + 1,
                mode=self.mode,
                started_at=None,
                ended_at=None,
                duration_s=duration_s,
                score_start=None,
                score_end=0.0,
                score_delta=None,
            )
            record.frozen_nodes = sorted(self.frozen_nodes)
            record.mutable_node_ids = sorted(self.mutable_nodes())
            record.eval_details = {'halt_reason': str(exc)}
            self._step_count += 1
            self._history.append(record)
            return record

        duration_s = time.monotonic() - t0
        import statistics

        from factory.models import AggregateMethod, InnerLoopConfig

        raw_items = exec_result_wf.item_results
        instance_results = None
        scores: list[float] = []

        if raw_items:
            instance_results = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                item_score = float(item.get("score", 0.0))
                scores.append(item_score)
                entry = {
                    "instance_id": item.get("item_id", ""),
                    "score": item_score,
                    "passed": item.get("passed", False),
                    "details": item.get("verify_details", {}),
                }
                if item.get("error"):
                    entry["error"] = item["error"]
                instance_results.append(entry)

        # Aggregate using the same configurable method as _step_with_task
        aggregate_method = (
            self._inner_loop_config.aggregate
            if self._inner_loop_config
            else InnerLoopConfig().aggregate
        )

        if not scores:
            # No item results (executor crashed before reaching DataNode)
            score = 0.0
        elif aggregate_method == AggregateMethod.mean:
            score = statistics.mean(scores)
        elif aggregate_method == AggregateMethod.median:
            score = statistics.median(scores)
        elif aggregate_method == AggregateMethod.max:
            score = max(scores)
        elif aggregate_method == AggregateMethod.all_pass:
            score = 1.0 if all(s >= 1.0 for s in scores) else 0.0
        else:
            score = statistics.mean(scores)

        record = CycleRecord(
            cycle_number=self._step_count + 1,
            mode=self.mode,
            started_at=None,
            ended_at=None,
            duration_s=duration_s,
            score_start=None,
            score_end=score,
            score_delta=None,
            instance_results=instance_results,
        )
        record.frozen_nodes = sorted(self.frozen_nodes)
        record.mutable_node_ids = sorted(self.mutable_nodes())

        # Propagate halt_reason from executor to CycleRecord
        if exec_result_wf.halt_reason:
            record.eval_details = {'halt_reason': exec_result_wf.halt_reason}

        self._write_cycle_summary(
            returncode=0 if exec_result_wf.success else 1,
            event_offset=0,
            duration_ms=int(duration_s * 1000),
            builder_committed=False,
            experiments=0,
            test_score=score,
            instance_results=instance_results,
        )

        self._step_count += 1
        self._history.append(record)
        return record

    def collect(self) -> CycleRecord:
        """Collect results without running a cycle. Useful after manual runs."""
        return self._collect_results()

    def score_trajectory(self) -> list[float]:
        """Score history across all steps."""
        if self._history:
            return [r.score_end for r in self._history if r.score_end is not None]
        analyzer = CycleAnalyzer(self.factory_dir, workflow=self.workflow)
        return analyzer.trajectory()

    def total_cost(self) -> float:
        """Cumulative cost across all steps."""
        return sum(r.total_cost_usd for r in self._history)

    def history(self) -> list[CycleRecord]:
        """All cycle records from this session."""
        return list(self._history)

    def _collect_results(
        self,
        event_offset: int = 0,
        tsv_offset: int = 0,
    ) -> CycleRecord:
        """Read execution artifacts + eval artifacts, compose into CycleRecord."""
        analyzer = CycleAnalyzer(
            self.factory_dir,
            workflow=self.workflow,
            event_offset=event_offset,
            tsv_offset=tsv_offset,
        )
        record = analyzer.latest()
        if record is None:
            record = CycleRecord(
                cycle_number=0,
                mode=self.mode,
                started_at=None,
                ended_at=None,
                duration_s=0,
                score_start=None,
                score_end=None,
                score_delta=None,
            )

        if record.mode is None:
            record.mode = self.mode

        record.frozen_nodes = sorted(self.frozen_nodes)
        record.mutable_node_ids = sorted(self.mutable_nodes())

        if self.evaluator and record.experiments:
            for exp in record.experiments:
                eval_files = [
                    Path(a) for a in exp.eval_artifacts
                    if a.endswith(".json") and "eval" in Path(a).name
                ]
                if eval_files:
                    eval_result = self.evaluator.parse_many(eval_files)
                    if eval_result.valid:
                        exp.score_after = eval_result.score

            last_eval_files = [
                Path(a) for exp in record.experiments
                for a in exp.eval_artifacts
                if a.endswith(".json") and "eval" in Path(a).name
            ]
            if last_eval_files:
                final = self.evaluator.parse(last_eval_files[-1])
                record.score_end = final.score

        return record

    def _get_git_head(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _run_test_command(self) -> tuple[float | None, dict[str, Any] | None]:
        """Run the configured test command and return (score, details).

        Dispatches output parsing based on self.test_format:
        - pytest: parse stdout for pass/fail counts
        - exit_code: binary pass/fail from returncode
        - json: parse stdout as JSON, extract metric
        - exact_match: compare output to expected answer
        """
        try:
            result = subprocess.run(
                shlex.split(self.test_command),
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            return self._parse_test_output(result)
        except subprocess.TimeoutExpired:
            return 0.0, {"error": "test_command_timeout"}
        except Exception as exc:
            return None, {"error": str(exc)}

    def _parse_test_output(
        self, result: subprocess.CompletedProcess[str],
    ) -> tuple[float, dict[str, Any]]:
        """Parse test command output based on test_format."""
        if self.test_format == "exit_code":
            score = 1.0 if result.returncode == 0 else 0.0
            return score, {
                "returncode": result.returncode,
                "passed": score,
                "test_format": "exit_code",
            }

        if self.test_format == "json":
            try:
                data = json.loads(result.stdout)
                obj: Any = data
                for key in self.metric_path.split("."):
                    obj = obj[key]
                score = float(obj)
                return score, {
                    "score": score,
                    "test_format": "json",
                    "test_returncode": result.returncode,
                    **{k: v for k, v in data.items() if isinstance(v, (int, float))},
                }
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                return 0.0, {"error": "json_parse_failed", "test_format": "json"}

        if self.test_format == "exact_match":
            output = result.stdout.strip()
            expected_path = self.project_dir / "expected_answer.txt"
            if not expected_path.exists():
                expected_path = self.project_dir / "expected.txt"
            if not expected_path.exists():
                return 0.0, {"error": "expected_answer_file_missing", "test_format": "exact_match"}
            expected = expected_path.read_text(errors="replace").strip()
            score = 1.0 if output == expected else 0.0
            return score, {
                "match": score,
                "test_format": "exact_match",
                "test_returncode": result.returncode,
            }

        from factory.outer_loop.benchmark_evaluator import parse_pytest_stdout
        metrics = parse_pytest_stdout(result.stdout)
        pass_rate = metrics.get("pass_rate", 0.0)
        return pass_rate, {
            "tests_passed": metrics.get("tests_passed", 0.0),
            "tests_total": metrics.get("tests_total", 0.0),
            "pass_rate": pass_rate,
            "test_returncode": result.returncode,
            "test_format": "pytest",
        }

    def _write_cycle_summary(
        self,
        returncode: int,
        event_offset: int,
        duration_ms: int,
        builder_committed: bool,
        experiments: int,
        test_score: float | None = None,
        test_details: dict[str, Any] | None = None,
        instance_results: list[dict[str, Any]] | None = None,
        kept: int = 0,
        reverted: int = 0,
    ) -> Path:
        """Write a structured summary of observable outcomes from this cycle."""
        events_path = self.factory_dir / "events.jsonl"

        agents_spawned = 0
        agents_succeeded = 0
        agents_failed = 0
        total_cost = 0.0

        if events_path.exists():
            for idx, line in enumerate(events_path.read_text().splitlines()):
                if idx < event_offset:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                etype = e.get("type", "")
                if etype == "agent.started":
                    agents_spawned += 1
                elif etype == "agent.completed":
                    agents_succeeded += 1
                    total_cost += e.get("data", {}).get("total_cost_usd", 0) or 0
                elif etype == "agent.failed":
                    agents_failed += 1

        heuristic_score = 0.0
        if agents_spawned > 0:
            heuristic_score += 0.2
        if builder_committed:
            heuristic_score += 0.2
        if returncode == 0:
            heuristic_score += 0.2
        if agents_failed == 0 and agents_spawned > 0:
            heuristic_score += 0.2
        if experiments > 0:
            heuristic_score += 0.2

        score = test_score if test_score is not None else heuristic_score

        errors: list[str] = []
        if returncode != 0:
            errors.append(f"subprocess exited with code {returncode}")

        summary: dict[str, Any] = {
            "mode": self.mode,
            "score": round(score, 4),
            "scoring_method": ("task_verify" if self._workflow_has_data_node() else "pytest_pass_rate") if test_score is not None else "heuristic",
            "heuristic_score": round(heuristic_score, 2),
            "cost_usd": round(total_cost, 2),
            "agents_spawned": agents_spawned,
            "agents_succeeded": agents_succeeded,
            "agents_failed": agents_failed,
            "builder_committed": builder_committed,
            "tests_passed": returncode == 0,
            "experiments": experiments,
            "duration_ms": duration_ms,
            "errors": errors,
            "kept": kept,
            "reverted": reverted,
        }
        if test_details:
            summary["test_details"] = test_details

        if instance_results:
            summary["instance_results"] = instance_results
            from factory.outer_loop.verify_adapter import eval_result_from_verify_results
            from factory.task import VerifyResult

            verify_results = [
                VerifyResult(
                    passed=bool(ir.get("passed", False)),
                    score=float(ir.get("score", 0.0)),
                    details=ir.get("details") or {},
                )
                for ir in instance_results
                if isinstance(ir, dict)
            ]
            adapted = eval_result_from_verify_results(verify_results)
            summary["verify"] = adapted.details

        summary_dir = self.factory_dir / "outer_loop" / "runs" / self.mode
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "cycle_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        return summary_path

    def _write_directives(self, directives: dict[str, Any]) -> None:
        """Write outer-loop directives as a factory message."""
        if self.frozen_nodes:
            directives['frozen_nodes'] = sorted(self.frozen_nodes)
        msg_dir = self.factory_dir / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)
        msg_id = f"outer-loop-{self._step_count:04d}"
        msg_path = msg_dir / f"{msg_id}.md"

        lines = ["# Outer Loop Directives\n"]
        for key, value in directives.items():
            if isinstance(value, list):
                lines.append(f"- **{key}:** {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- **{key}:** {value}")

        msg_path.write_text("\n".join(lines) + "\n")
