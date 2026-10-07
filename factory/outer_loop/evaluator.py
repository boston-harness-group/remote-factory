"""Fitness evaluation for workflow candidates in the evolutionary search.

Supports both legacy EvaluatorFn protocol (DirectFeatureBenchEvaluator) and
InnerLoop-based evaluation via compose(). CycleRecordCache provides
content-addressable caching keyed by workflow hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.similarity import structural_hash
from factory.workflow.primitives import Workflow

log = structlog.get_logger()


class FitnessCache:
    """Cache evaluation results keyed by (structural_hash, frozenset(instances))."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, frozenset[str]], tuple[float, float, float]] = {}

    def get(
        self, workflow: Workflow, instances: list[str]
    ) -> tuple[float, float, float] | None:
        key = (structural_hash(workflow), frozenset(instances))
        return self._cache.get(key)

    def put(
        self, workflow: Workflow, instances: list[str], score: float, cost: float
    ) -> None:
        key = (structural_hash(workflow), frozenset(instances))
        self._cache[key] = (score, cost, time.time())

    @property
    def size(self) -> int:
        return len(self._cache)


class CycleRecordCache:
    """Cache CycleRecords keyed by workflow content hash.

    Content-addressable via sha256(workflow.to_dict()).
    Supports JSONL persistence for crash-resilient resume.
    """

    def __init__(self) -> None:
        self._cache: dict[str, CycleRecord] = {}

    @staticmethod
    def workflow_hash(workflow: Workflow) -> str:
        blob = json.dumps(workflow.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _cache_key(workflow: Workflow, instances: list[str] | None = None) -> str:
        """Compute composite cache key: hash(workflow) + ':' + hash(sorted(instances)).

        Instance-aware keying prevents cache poisoning where a workflow
        evaluated on search instances returns cached results for holdout queries.
        """
        wf_hash = CycleRecordCache.workflow_hash(workflow)
        if instances is not None:
            inst_blob = json.dumps(sorted(instances), separators=(",", ":"))
            inst_hash = hashlib.sha256(inst_blob.encode()).hexdigest()
            return f"{wf_hash}:{inst_hash}"
        return wf_hash

    def get(
        self, workflow: Workflow, instances: list[str] | None = None,
    ) -> CycleRecord | None:
        key = self._cache_key(workflow, instances)
        return self._cache.get(key)

    def put(
        self, workflow: Workflow, record: CycleRecord,
        instances: list[str] | None = None,
    ) -> None:
        key = self._cache_key(workflow, instances)
        self._cache[key] = record

    @property
    def size(self) -> int:
        return len(self._cache)

    def save_cache(self, path: Path) -> None:
        """Append all cached entries to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_hashes: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    existing_hashes.add(existing.get("workflow_hash", ""))
                except json.JSONDecodeError:
                    continue

        new_entries: list[str] = []
        for wf_hash, record in self._cache.items():
            if wf_hash in existing_hashes:
                continue
            entry: dict[str, Any] = {
                "workflow_hash": wf_hash,
                "score": record.score_end,
                "cost": record.total_cost_usd,
                "kept": record.kept,
                "reverted": record.reverted,
                "timestamp": record.ended_at or record.started_at,
            }
            if record.eval_details is not None:
                entry["eval_details"] = record.eval_details
            if record.instance_results is not None:
                entry["instance_results"] = record.instance_results
            new_entries.append(json.dumps(entry, separators=(",", ":")))

        if new_entries:
            with path.open("a") as f:
                for line in new_entries:
                    f.write(line + "\n")
            log.info("cycle_cache_saved", path=str(path), new_entries=len(new_entries))

    def load_cache(self, path: Path) -> int:
        """Load cached entries from a JSONL file. Returns number of entries loaded."""
        if not path.exists():
            return 0

        loaded = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                log.warning("cycle_cache_corrupt_line", line=line[:80])
                continue

            wf_hash = entry.get("workflow_hash")
            if not wf_hash or wf_hash in self._cache:
                continue

            record = CycleRecord(
                cycle_number=0,
                mode=None,
                started_at=entry.get("timestamp"),
                ended_at=entry.get("timestamp"),
                duration_s=0.0,
                score_start=None,
                score_end=entry.get("score"),
                score_delta=None,
                kept=entry.get("kept", 0),
                reverted=entry.get("reverted", 0),
                total_cost_usd=entry.get("cost", 0.0),
                instance_results=entry.get("instance_results"),
                eval_details=entry.get("eval_details"),
            )
            self._cache[wf_hash] = record
            loaded += 1

        if loaded:
            log.info("cycle_cache_loaded", path=str(path), entries=loaded)
        return loaded


@runtime_checkable
class EvaluatorFn(Protocol):
    """Protocol for pluggable evaluation functions."""

    def __call__(
        self, workflow: Workflow, project_dir: str, instances: list[str]
    ) -> EvalResult: ...


class SwarmEvaluator:
    """Evaluates workflow candidates against benchmark instances.

    Supports both legacy EvaluatorFn and InnerLoop-based evaluation.
    When inner_loop_factory is provided, it takes precedence.
    """

    def __init__(
        self,
        config: SwarmConfig,
        evaluator_fn: EvaluatorFn | None = None,
        inner_loop_factory: Any | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._evaluator_fn = evaluator_fn
        self._inner_loop_factory = inner_loop_factory
        self._cache = FitnessCache()
        self._cycle_cache = CycleRecordCache()
        self._cycle_records: dict[str, CycleRecord] = {}
        self._cache_path: Path | None = None

        if project_dir is not None:
            self._cache_path = Path(project_dir) / ".factory" / "outer_loop" / "eval_cache.jsonl"
            self._cycle_cache.load_cache(self._cache_path)

    def checkpoint_cache(self) -> None:
        """Persist the cycle record cache to disk."""
        if self._cache_path is not None:
            self._cycle_cache.save_cache(self._cache_path)

    @property
    def cache(self) -> FitnessCache:
        return self._cache

    @property
    def cycle_cache(self) -> CycleRecordCache:
        return self._cycle_cache

    def get_cycle_record(self, individual_id: str) -> CycleRecord | None:
        return self._cycle_records.get(individual_id)

    def evaluate(
        self,
        workflow: Workflow,
        project_dir: str,
        instances: list[str],
        individual_id: str | None = None,
    ) -> EvalResult:
        """Evaluate a workflow on the given instances, using cache if available."""
        cached = self._cache.get(workflow, instances)
        if cached is not None:
            score, cost, _ = cached
            log.info("fitness_cache_hit", score=score)
            return EvalResult(score=score, cost_usd=cost, benchmark_score=score)

        if not self._check_mandatory_components(workflow):
            log.warning("mandatory_component_missing", workflow=workflow.name)
            return EvalResult(score=0.0, details={"rejected": "mandatory_component_missing"})

        if not self._check_frozen_nodes(workflow):
            log.warning("frozen_node_violated", workflow=workflow.name)
            return EvalResult(score=0.0, details={"rejected": "frozen_node_violated"})

        # CHECK 4: graph validation — reject structurally/semantically broken workflows
        validation_issues = workflow.validate_graph()
        if validation_issues:
            log.warning(
                "graph_validation_failed",
                workflow=workflow.name,
                issues=validation_issues,
            )
            return EvalResult(
                score=0.0,
                details={
                    "rejected": "graph_validation_failed",
                    "validation_errors": validation_issues,
                },
            )

        if self._inner_loop_factory is not None:
            return self._evaluate_via_inner_loop(
                workflow, project_dir, instances, individual_id
            )

        if self._evaluator_fn is not None:
            result = self._evaluator_fn(workflow, project_dir, instances)
        else:
            result = EvalResult(score=0.0, details={"note": "no_evaluator_fn_configured"})

        composite = self._compute_composite(result)
        result = result.model_copy(update={"score": composite})

        self._cache.put(workflow, instances, composite, result.cost_usd)
        return result

    @staticmethod
    def _create_worktree(project_dir: str, label: str) -> Path:
        """Create an isolated git worktree from the target project."""
        src = Path(project_dir)
        wt_base = src.parent / ".eval-worktrees"
        wt_base.mkdir(parents=True, exist_ok=True)
        wt_path = wt_base / f"wt-{label}-{uuid.uuid4().hex[:8]}"

        result = subprocess.run(
            ["git", "-C", str(src), "worktree", "add", "--detach", str(wt_path), "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {result.stderr}")

        for subdir in ["outer_loop/modes", "workflows", "tasks"]:
            src_dir = src / ".factory" / subdir
            dst_dir = wt_path / ".factory" / subdir
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in src_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst_dir / f.name)

        log.info("worktree_created", path=str(wt_path), source=str(src))
        return wt_path

    @staticmethod
    def _cleanup_worktree(project_dir: str, wt_path: Path) -> None:
        """Remove a git worktree."""
        try:
            subprocess.run(
                ["git", "-C", str(project_dir), "worktree", "remove", "--force", str(wt_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            shutil.rmtree(wt_path, ignore_errors=True)
            try:
                subprocess.run(
                    ["git", "-C", str(project_dir), "worktree", "prune"],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
        log.info("worktree_cleaned", path=str(wt_path))

    def _evaluate_via_inner_loop(
        self,
        workflow: Workflow,
        project_dir: str,
        instances: list[str],
        individual_id: str | None = None,
    ) -> EvalResult:
        """Evaluate using InnerLoop.step() in an isolated worktree.

        When a Task is available via self._config.get_task(), uses compose()
        to construct a task-attached InnerLoop. Falls back to InnerLoop()
        directly for backward compat with legacy flat-field configs.
        """
        from factory.inner_loop import InnerLoop

        cached_record = self._cycle_cache.get(workflow, instances)
        if cached_record is not None:
            score = cached_record.score_end or 0.0
            cost = cached_record.total_cost_usd
            log.info("cycle_record_cache_hit", score=score)
            if individual_id:
                self._cycle_records[individual_id] = cached_record
            return EvalResult(score=score, cost_usd=cost, benchmark_score=score)

        wt_path: Path | None = None
        try:
            task = self._config.get_task() if hasattr(self._config, "get_task") else None

            if task is not None:
                mode_name = "task-eval"
            else:
                mode_name = self._inner_loop_factory(workflow) if callable(self._inner_loop_factory) else "evolve"

            label = individual_id[:8] if individual_id else mode_name[:12]
            wt_path = self._create_worktree(project_dir, label)

            if task is not None:
                from factory.compose import compose

                loop = compose(workflow, task, wt_path)
                loop.mode = mode_name
                loop.frozen_nodes = frozenset(self._config.frozen_node_ids)
                loop.test_command = self._config.test_command
                loop.test_format = self._config.test_format or "pytest"
                loop.metric_path = self._config.metric_path
                loop.execution_strategy = getattr(
                    self._config, "execution_strategy", "executor"
                )
                if instances:
                    from factory.outer_loop.subset import FixedSubsetSelector
                    loop._subset_selector = FixedSubsetSelector(instances)
            else:
                loop = InnerLoop(
                    project_dir=wt_path,
                    mode=mode_name,
                    workflow=workflow,
                    frozen_nodes=frozenset(self._config.frozen_node_ids),
                    test_command=self._config.test_command,
                    test_format=self._config.test_format,
                    metric_path=self._config.metric_path,
                    execution_strategy=getattr(
                        self._config, "execution_strategy", "executor"
                    ),
                )
            record = loop.step()

            summary_data = self._read_cycle_summary(wt_path, loop.mode)
            summary_score = float(summary_data.get("score", 0.0)) if summary_data else None
            score = summary_score if summary_score is not None else (record.score_end or 0.0)
            cost = record.total_cost_usd

            record.split = "train"
            self._cycle_cache.put(workflow, record, instances)
            if individual_id:
                self._cycle_records[individual_id] = record

            num_nodes = len(workflow.nodes)
            parsimony = 0.01 * num_nodes
            composite = max(0.0, score - parsimony)

            self._cache.put(workflow, instances, composite, cost)

            details: dict[str, object] = {
                "experiments": len(record.experiments),
                "steps": len(record.steps),
                "kept": record.kept,
                "reverted": record.reverted,
                "parsimony_penalty": parsimony,
            }
            if summary_data:
                details["scoring_method"] = summary_data.get("scoring_method", "unknown")
                if "test_details" in summary_data:
                    details["test_details"] = summary_data["test_details"]

            if isinstance(record.instance_results, list) and record.instance_results:
                from factory.outer_loop.verify_adapter import (
                    eval_result_from_verify_results,
                )
                from factory.task import VerifyResult

                verify_results: list[VerifyResult] = []
                for ir in record.instance_results:
                    if isinstance(ir, dict):
                        verify_results.append(VerifyResult(
                            passed=bool(ir.get("passed", False)),
                            score=float(ir.get("score", 0.0)),
                            details=ir.get("details", {}),
                        ))
                adapted = eval_result_from_verify_results(verify_results)
                details["verify"] = adapted.details

            record.eval_details = details

            return EvalResult(
                score=composite,
                benchmark_score=score,
                cost_usd=cost,
                complexity=float(num_nodes),
                details=details,
            )
        except Exception as exc:
            log.error("inner_loop_eval_failed", error=str(exc), exc_info=True)
            return EvalResult(
                score=0.0,
                details={"error": str(exc), "evaluation_method": "inner_loop"},
            )
        finally:
            if wt_path is not None:
                self._cleanup_worktree(project_dir, wt_path)

    def evaluate_batch(
        self,
        workflows: list[Workflow],
        project_dir: str,
        instances: list[str],
        parallelism: int = 1,
    ) -> list[EvalResult]:
        """Evaluate multiple workflows, optionally in parallel with worktree isolation."""
        if parallelism <= 1 or len(workflows) <= 1:
            return [self.evaluate(wf, project_dir, instances) for wf in workflows]

        results: list[EvalResult | None] = [None] * len(workflows)
        with ThreadPoolExecutor(max_workers=min(parallelism, len(workflows))) as pool:
            futures = {
                pool.submit(self.evaluate, wf, project_dir, instances): idx
                for idx, wf in enumerate(workflows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log.error("batch_eval_failed", index=idx, error=str(exc))
                    results[idx] = EvalResult(score=0.0, details={"error": str(exc)})

        return [r or EvalResult(score=0.0) for r in results]

    def _compute_composite(self, result: EvalResult) -> float:
        norm_cost = min(result.cost_usd / 10.0, 1.0) if result.cost_usd > 0 else 0.0
        norm_complexity = min(result.complexity / 20.0, 1.0) if result.complexity > 0 else 0.0
        return (
            0.6 * result.benchmark_score
            + 0.2 * result.hygiene_score
            + 0.1 * (1.0 - norm_cost)
            + 0.1 * (1.0 - norm_complexity)
        )

    @staticmethod
    def _read_cycle_summary(project_dir: Path, mode: str) -> dict | None:
        summary_path = (
            project_dir / ".factory" / "outer_loop" / "runs" / mode / "cycle_summary.json"
        )
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _check_mandatory_components(self, workflow: Workflow) -> bool:
        if not self._config.mandatory_node_roles:
            return True
        present_roles: set[str] = set()
        for node in workflow.nodes.values():
            if hasattr(node, "role"):
                present_roles.add(node.role.value if hasattr(node.role, "value") else str(node.role))
        for role in self._config.mandatory_node_roles:
            if role not in present_roles:
                return False
        return True

    def _check_frozen_nodes(self, workflow: Workflow) -> bool:
        for fid in self._config.frozen_node_ids:
            if fid not in workflow.nodes:
                return False
        return True
