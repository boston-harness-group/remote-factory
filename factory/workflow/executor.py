"""Deterministic async graph walker implementing formal execution semantics."""

from __future__ import annotations

import asyncio
import builtins
from collections import deque
import json
import re
import shlex
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import structlog

from factory.workflow.events import (
    GateVerdictEvent,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    WorkflowCompleted,
    WorkflowHalted,
    WorkflowStarted,
    emit_workflow_event,
)
from factory.workflow.primitives import (
    AgentConfig,
    AgentNode,
    DataNode,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    LLMNode,
    NodeType,
    SelectionNode,
    Study,
    SubgraphForkNode,
    Verdict,
    VerdictType,
    Workflow,
)

log = structlog.get_logger()

CEO_GATE_PROMPT = """\
You are reviewing the output of the {step_name} step in the {workflow_name} workflow.
The output is at: {output_file}
Previous context: {previous_context}

Read the output and decide:
- **Proceed**: the output is satisfactory, continue to the next step
- **Reloop(target, feedback)**: the output needs improvement. Reloop targets: {reloop_targets}. Specify which step to return to and what feedback to provide.
- **Halt(reason)**: something is fundamentally wrong, stop the workflow.

Respond with exactly one of:
PROCEED
RELOOP target="<node_id>" feedback="<your feedback>"
HALT reason="<your reason>"
"""

# Words skipped when parsing an explicit reloop target from user input ("reloop the builder").
_USER_GATE_RELOOP_FILLERS: frozenset[str] = frozenset({"the", "to", "back", "node", "a", "an", "and"})


class ExecutionResult:
    """Result of a workflow execution."""

    def __init__(self) -> None:
        self.success: bool = False
        self.halted: bool = False
        self.halt_reason: str = ""
        self.nodes_executed: int = 0
        self.events: list[dict[str, Any]] = []
        self.completed_files: set[str] = set()
        self.node_outputs: dict[str, str] = {}
        self.duration_ms: float = 0.0
        self.cost: float = 0.0
        self.item_results: list[dict[str, Any]] = []


class WorkflowExecutor:
    """Deterministic async graph walker for workflow execution."""

    def __init__(
        self,
        workflow: Workflow,
        project_path: Path,
        agent_pool: dict[str, AgentConfig] | None = None,
        *,
        dry_run: bool = False,
        auto_approve: bool = False,
        initial_context: str | None = None,
        agent_fn: Callable[..., Any] | None = None,
        input_fn: Callable[[str], str] | None = None,
        validate: bool = True,
        auto_write_outputs: bool = True,
        task: Any = None,  # factory.task.Task — use Any to avoid circular import
        split: Literal["train", "val", "all"] = "train",
        allowed_instance_ids: set[str] | None = None,
    ) -> None:
        if validate:
            from factory.workflow.validation import validate_workflow

            issues = validate_workflow(workflow)
            if issues:
                raise ValueError(
                    f"Workflow '{workflow.name}' has validation errors:\n"
                    + "\n".join(f"  - {i}" for i in issues)
                )

        self.workflow = workflow
        self.project_path = Path(project_path) if not isinstance(project_path, Path) else project_path
        self.agent_pool = agent_pool or {}
        self.dry_run = dry_run
        self.auto_approve = auto_approve
        self._allowed_instance_ids = allowed_instance_ids
        self._split = split
        self._task = task
        if agent_fn is not None:
            self._agent_fn = agent_fn
        else:
            from factory.agents.runner import invoke_agent

            self._agent_fn = invoke_agent
        self._input_fn: Callable[[str], str] = input_fn if input_fn is not None else builtins.input
        self.auto_write_outputs = auto_write_outputs
        self.run_id = uuid.uuid4().hex[:12]
        self.completed_files: set[str] = set()
        self.node_context: dict[str, str] = {}
        self.iteration_counts: dict[tuple[str, str], int] = {}
        self.background_tasks: list[asyncio.Task[Any]] = []
        self.result = ExecutionResult()
        self._edge_index: dict[str, list[Edge]] = {}
        for edge in workflow.edges:
            self._edge_index.setdefault(edge.source, []).append(edge)

        if initial_context is not None:
            start_node = workflow.nodes.get(workflow.start_node)
            if isinstance(start_node, AgentNode):
                self.node_context[workflow.start_node] = initial_context
            else:
                log.warning(
                    "initial_context_ignored",
                    start_node=workflow.start_node,
                    node_type=type(start_node).__name__ if start_node else "missing",
                    reason="initial_context only applies to AgentNode start nodes",
                )

    def _actual_writes(self, node: NodeType) -> set[str]:
        """Return the subset of *node.writes* that actually exist on disk.

        In ``dry_run`` mode every declared write is trusted because no files
        are created.  Otherwise only paths that resolve to an existing file
        under ``self.project_path`` are returned.  This keeps
        ``completed_files`` in sync with reality so that downstream
        ``_wait_for_reads`` checks fail when an upstream node neglects to
        produce a declared output.
        """
        if self.dry_run:
            return set(node.writes)
        actual: set[str] = set()
        for path_str in node.writes:
            if (self.project_path / path_str).exists():
                actual.add(path_str)
        return actual

    async def execute(self) -> ExecutionResult:
        """Run the workflow from start to completion."""
        start_time = time.monotonic()

        self._emit(
            "workflow.started",
            WorkflowStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                start_node=self.workflow.start_node,
            ),
        )

        try:
            await self._execute_from(self.workflow.start_node)
            self.result.success = not self.result.halted
        except Exception as exc:
            self.result.success = False
            self.result.halted = True
            self.result.halt_reason = str(exc)
            log.error("workflow.exception", error=str(exc), workflow=self.workflow.name)

        if self.background_tasks:
            done, pending = await asyncio.wait(
                self.background_tasks,
                timeout=30.0,
            )
            for task in pending:
                task.cancel()

        elapsed = (time.monotonic() - start_time) * 1000
        self.result.duration_ms = elapsed
        self.result.completed_files = set(self.completed_files)

        # Timing summary: extract per-node durations from completed events
        node_timings: list[dict[str, Any]] = []
        for ev in self.result.events:
            if ev.get("type") == "node.completed" and "duration_ms" in ev:
                node_timings.append({
                    "id": ev.get("node_id", ""),
                    "type": ev.get("node_type", ""),
                    "duration_ms": round(ev["duration_ms"], 1),
                })
        node_timings.sort(key=lambda n: n["duration_ms"], reverse=True)
        node_total_ms = sum(n["duration_ms"] for n in node_timings)
        log.info(
            "workflow.timing_summary",
            workflow=self.workflow.name,
            run_id=self.run_id,
            total_ms=round(elapsed, 1),
            node_count=len(node_timings),
            nodes=node_timings,
            overhead_ms=round(elapsed - node_total_ms, 1),
        )

        if self.result.halted:
            self._emit(
                "workflow.halted",
                WorkflowHalted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    reason=self.result.halt_reason,
                    halted_at_node="unknown",
                ),
            )
        else:
            self._emit(
                "workflow.completed",
                WorkflowCompleted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    nodes_executed=self.result.nodes_executed,
                    duration_ms=elapsed,
                ),
            )

        return self.result

    async def _execute_from(self, node_id: str) -> None:
        """Execute starting from the given node, following edges."""
        if self.result.halted:
            return

        node = self.workflow.nodes.get(node_id)
        if not node:
            self.result.halted = True
            self.result.halt_reason = f"node '{node_id}' not found"
            return

        await self._wait_for_reads(node)
        if self.result.halted:
            return

        if isinstance(node, SubgraphForkNode):
            await self._execute_subgraph_fork(node)
            return

        if isinstance(node, ForkNode):
            await self._execute_fork(node)
            return

        if isinstance(node, SelectionNode):
            await self._execute_selection(node)
            return

        if isinstance(node, JoinNode):
            pending = [
                s for s in node.sources
                if s not in self.result.node_outputs
                and getattr(self.workflow.nodes.get(s), "blocking", True)
            ]
            if pending:
                self.result.halted = True
                self.result.halt_reason = (
                    f"JoinNode '{node_id}' reached before sources completed: {pending}"
                )
                log.error(
                    "joinnode.sources_incomplete",
                    join_id=node_id,
                    pending=pending,
                    completed=list(self.result.node_outputs.keys()),
                )
                self._emit(
                    "node.failed",
                    NodeFailed(
                        workflow_name=self.workflow.name,
                        run_id=self.run_id,
                        node_id=node_id,
                        node_type="JoinNode",
                        error=self.result.halt_reason,
                    ),
                )
                return
            self.result.nodes_executed += 1
            self.completed_files |= self._actual_writes(node)
            self.result.node_outputs[node_id] = ""
            next_id = self._next_unconditional(node_id)
            if next_id:
                await self._execute_from(next_id)
            return

        if isinstance(node, GateNode):
            await self._execute_gate(node)
            return

        if isinstance(node, DataNode):
            await self._execute_data_fork(node_id, node)
            return

        await self._execute_action_node(node)

    async def _execute_action_node(self, node: NodeType) -> None:
        """Execute an AgentNode, FnNode, or Study node."""
        node_id = node.id
        node_type = type(node).__name__

        if not node.blocking:
            task = asyncio.create_task(self._run_node_background(node))
            self.background_tasks.append(task)
            next_id = self._next_unconditional(node_id)
            if next_id:
                await self._execute_from(next_id)
            return

        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                node_type=node_type,
            ),
        )

        start = time.monotonic()
        try:
            output = await self._run_node(node)
            elapsed = (time.monotonic() - start) * 1000

            self.result.node_outputs[node_id] = output
            self.completed_files |= self._actual_writes(node)

            # SPEC §7.3: enforce post_checks on AgentNodes (skip in dry-run).
            # Runs after output is recorded so files written by _run_agent
            # are available for validation.
            if isinstance(node, AgentNode) and node.post_checks and not self.dry_run:
                self._validate_post_checks(node)

            # Increment only after post_checks pass — a failed check should
            # not count as a successfully executed node.
            self.result.nodes_executed += 1

            self._emit(
                "node.completed",
                NodeCompleted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type=node_type,
                    files_written=sorted(node.writes),
                    duration_ms=elapsed,
                ),
            )

        except Exception as exc:
            self._emit(
                "node.failed",
                NodeFailed(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type=node_type,
                    error=str(exc),
                ),
            )
            self.result.halted = True
            self.result.halt_reason = f"node '{node_id}' failed: {exc}"
            return

        next_id = self._next_unconditional(node_id)
        if next_id:
            await self._execute_from(next_id)

    async def _run_node_background(self, node: NodeType) -> None:
        """Run a non-blocking node as a background task."""
        node_id = node.id
        node_type = type(node).__name__
        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                node_type=node_type,
            ),
        )
        start = time.monotonic()
        try:
            output = await self._run_node(node)
            elapsed = (time.monotonic() - start) * 1000
            self.result.node_outputs[node_id] = output
            self.completed_files |= self._actual_writes(node)

            # Enforce post_checks on background nodes (same as blocking path)
            if isinstance(node, AgentNode) and node.post_checks and not self.dry_run:
                self._validate_post_checks(node)

            self.result.nodes_executed += 1
            self._emit(
                "node.completed",
                NodeCompleted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type=node_type,
                    files_written=sorted(node.writes),
                    duration_ms=elapsed,
                ),
            )
        except Exception as exc:
            self._emit(
                "node.failed",
                NodeFailed(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type=node_type,
                    error=str(exc),
                ),
            )
            log.warning("background_node_failed", node=node_id, error=str(exc))

    async def _execute_gate(self, node: GateNode) -> None:
        """Execute a gate node, parse verdict, follow the matching edge."""
        node_id = node.id
        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                node_type="GateNode",
            ),
        )

        try:
            verdict = await self._evaluate_gate(node)
        except Exception as exc:
            self._emit(
                "node.failed",
                NodeFailed(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type="GateNode",
                    error=str(exc),
                ),
            )
            self.result.halted = True
            self.result.halt_reason = f"gate '{node_id}' failed: {exc}"
            return

        self.result.nodes_executed += 1
        self.result.node_outputs[node_id] = verdict.type.value

        self._emit(
            "gate.verdict",
            GateVerdictEvent(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                verdict_type=verdict.type,
                target=verdict.target,
                feedback=verdict.feedback,
                reason=verdict.reason,
            ),
        )

        if verdict.type == VerdictType.HALT:
            self.result.halted = True
            self.result.halt_reason = verdict.reason or "gate halted"
            return

        if verdict.type == VerdictType.RELOOP:
            target = verdict.target
            if not target:
                self.result.halted = True
                self.result.halt_reason = "reloop verdict missing target"
                return

            key = (node_id, target)
            count = self.iteration_counts.get(key, 0) + 1
            self.iteration_counts[key] = count

            if count > verdict.max_iterations:
                self.result.halted = True
                self.result.halt_reason = (
                    f"max iterations ({verdict.max_iterations}) exhausted "
                    f"for gate '{node_id}' -> '{target}'"
                )
                return

            if verdict.feedback:
                existing = self.node_context.get(target, "")
                self.node_context[target] = (
                    f"{existing}\n\n[Feedback iteration {count}]: {verdict.feedback}"
                    if existing
                    else f"[Feedback iteration {count}]: {verdict.feedback}"
                )

            await self._execute_from(target)
            return

        target_id = self._next_conditional(node_id, VerdictType.PROCEED)
        if target_id is None:
            target_id = self._next_unconditional(node_id)

        if target_id is None:
            log.warning(
                "gate_proceed_edge_missing",
                gate_id=node_id,
                workflow=self.workflow.name,
            )

        if target_id:
            await self._execute_from(target_id)

    async def _execute_fork(self, node: ForkNode) -> None:
        """Execute all fork targets concurrently via asyncio.gather.

        Branches are run in isolation — they do NOT follow outgoing edges.
        After all branches complete, the fork's own unconditional edge is followed.
        """
        self.result.nodes_executed += 1

        async def run_branch(target_id: str) -> None:
            target = self.workflow.nodes.get(target_id)
            if not target:
                return
            node_type = type(target).__name__
            self._emit(
                "node.started",
                NodeStarted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=target_id,
                    node_type=node_type,
                ),
            )
            start = time.monotonic()
            try:
                output = await self._run_node(target)
                elapsed = (time.monotonic() - start) * 1000
                self.result.node_outputs[target_id] = output
                self.completed_files |= self._actual_writes(target)

                # Enforce post_checks on fork branches (same as blocking path)
                if isinstance(target, AgentNode) and target.post_checks and not self.dry_run:
                    self._validate_post_checks(target)

                self.result.nodes_executed += 1
                self._emit(
                    "node.completed",
                    NodeCompleted(
                        workflow_name=self.workflow.name,
                        run_id=self.run_id,
                        node_id=target_id,
                        node_type=node_type,
                        files_written=sorted(target.writes),
                        duration_ms=elapsed,
                    ),
                )
            except Exception as exc:
                self._emit(
                    "node.failed",
                    NodeFailed(
                        workflow_name=self.workflow.name,
                        run_id=self.run_id,
                        node_id=target_id,
                        node_type=node_type,
                        error=str(exc),
                    ),
                )
                if not self.result.halted:
                    self.result.halt_reason = f"fork branch '{target_id}' failed: {exc}"
                self.result.halted = True

        await asyncio.gather(*(run_branch(t) for t in node.targets))

        if self.result.halted:
            return

        self.result.node_outputs[node.id] = ""
        branch_set = set(node.targets)
        next_id: str | None = None
        for edge in self._edge_index.get(node.id, []):
            if edge.condition is None and edge.target not in branch_set:
                next_id = edge.target
                break
        if next_id is None and node.targets:
            next_id = self._next_unconditional(node.targets[0])
        if next_id:
            await self._execute_from(next_id)

    async def _execute_subgraph_fork(self, node: SubgraphForkNode) -> None:
        """Execute N copies of a subgraph in parallel, each in an isolated worktree.

        Each branch gets an independent WorkflowExecutor with its own state,
        running against a separate git worktree branching from the same commit.
        """
        import subprocess as sp

        from factory.worktree import create_experiment_worktree

        self.result.nodes_executed += 1

        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node.id,
                node_type="SubgraphForkNode",
            ),
        )

        start = time.monotonic()

        # Resolve base commit for all branches
        if self.dry_run:
            base_commit = "0" * 40
        else:
            result = sp.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=True,
            )
            base_commit = result.stdout.strip()

        # Parse hypotheses from strategist output to determine branch count
        strategy_file = self.project_path / ".factory" / "strategy" / "current.md"
        hypotheses = _parse_hypotheses(strategy_file) if strategy_file.exists() else []
        branch_count = min(len(hypotheses), node.parallelism) if hypotheses else node.parallelism

        if branch_count < 1:
            branch_count = 1

        # Collect subgraph node IDs by walking edges from entry to exit
        subgraph_ids = _collect_subgraph_nodes(
            self.workflow, node.subgraph_entry, node.subgraph_exit,
        )
        sub_workflow = self.workflow.subgraph(
            subgraph_ids, name=f"{self.workflow.name}__branch", start_node=node.subgraph_entry,
        )

        branch_results: list[dict[str, Any]] = []
        worktrees: list[tuple[Path, str, int]] = []

        async def run_branch(idx: int) -> dict[str, Any]:
            from factory.store import ExperimentStore

            hypothesis = hypotheses[idx] if idx < len(hypotheses) else f"Hypothesis {idx + 1}"

            if self.dry_run:
                wt_path = self.project_path / ".factory-worktrees" / f"exp-dry-{idx}"
                branch_name = f"factory/exp-dry-{idx}"
                exp_id = idx + 1
            else:
                store = ExperimentStore(self.project_path)
                exp_id = await store.begin(hypothesis)
                wt_path, branch_name = create_experiment_worktree(
                    self.project_path, exp_id, base_commit,
                )
                worktrees.append((wt_path, branch_name, exp_id))

            branch_executor = WorkflowExecutor(
                sub_workflow.model_copy(deep=True),
                wt_path if not self.dry_run else self.project_path,
                agent_pool=self.agent_pool,
                dry_run=self.dry_run,
                agent_fn=self._agent_fn,
                auto_write_outputs=self.auto_write_outputs,
            )
            branch_result = await branch_executor.execute()

            return {
                "exp_id": exp_id,
                "hypothesis": hypothesis,
                "worktree_path": str(wt_path),
                "branch": branch_name,
                "success": branch_result.success,
                "halted": branch_result.halted,
                "halt_reason": branch_result.halt_reason,
                "nodes_executed": branch_result.nodes_executed,
                "node_outputs": branch_result.node_outputs,
            }

        sem = asyncio.Semaphore(node.parallelism)

        async def throttled_branch(idx: int) -> dict[str, Any]:
            async with sem:
                return await run_branch(idx)

        tasks = [throttled_branch(i) for i in range(branch_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, BaseException):
                log.warning("subgraph_branch_failed", error=str(r))
                branch_results.append({
                    "success": False, "halted": True, "halt_reason": str(r),
                })
            else:
                branch_results.append(r)  # type: ignore[arg-type]

        elapsed = (time.monotonic() - start) * 1000
        self.result.node_outputs[node.id] = json.dumps(branch_results)
        self.completed_files |= self._actual_writes(node)

        self._emit(
            "node.completed",
            NodeCompleted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node.id,
                node_type="SubgraphForkNode",
                files_written=sorted(node.writes),
                duration_ms=elapsed,
            ),
        )

        next_id = self._next_unconditional(node.id)
        if next_id:
            await self._execute_from(next_id)

    async def _execute_data_fork(self, node_id: str, node: DataNode) -> None:
        """Execute a DataNode via the data runtime (fork/join model)."""
        from factory.workflow.data_runtime import run_fork

        self.result.nodes_executed += 1
        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                node_type="DataNode",
            ),
        )

        start = time.monotonic()

        try:
            item_results = await run_fork(
                self.workflow,
                node,
                node_id,
                self.project_path,
                agent_pool=self.agent_pool,
                dry_run=self.dry_run,
                agent_fn=self._agent_fn,
                auto_write_outputs=self.auto_write_outputs,
                allowed_instance_ids=self._allowed_instance_ids,
                task=self._task,
                completed_files=self.completed_files,
                run_id=self.run_id,
                split=self._split,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self.result.halted = True
            self.result.halt_reason = f"DataNode '{node_id}' failed: {exc}"
            self._emit(
                "node.failed",
                NodeFailed(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type="DataNode",
                    error=str(exc),
                ),
            )
            return

        self.result.item_results = item_results

        # Check if all items errored (no ok or failed items)
        ok_items = [r for r in item_results if r.get("status") == "ok"]
        has_failed = any(r.get("status") == "failed" for r in item_results)
        if not ok_items and not has_failed:
            # All items errored — halt
            elapsed = (time.monotonic() - start) * 1000
            self.result.halted = True
            self.result.halt_reason = (
                f"DataNode '{node_id}': all {len(item_results)} items errored"
            )
            self.result.node_outputs[node_id] = json.dumps(item_results)
            self._emit(
                "node.failed",
                NodeFailed(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node_id,
                    node_type="DataNode",
                    error=self.result.halt_reason,
                ),
            )
            return

        elapsed = (time.monotonic() - start) * 1000
        self.result.node_outputs[node_id] = json.dumps(item_results)
        self.completed_files |= self._actual_writes(node)

        self._emit(
            "node.completed",
            NodeCompleted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node_id,
                node_type="DataNode",
                files_written=sorted(node.writes),
                duration_ms=elapsed,
            ),
        )

        # After DataNode completes, skip to the JoinNode (it's the barrier)
        # and then continue from there
        from factory.workflow.data_runtime import _find_branch_and_join
        try:
            _, join_id, _ = _find_branch_and_join(self.workflow, node_id)
            # Mark JoinNode as completed
            self.result.nodes_executed += 1
            self.result.node_outputs[join_id] = ""
            # Follow edges from JoinNode
            next_id = self._next_unconditional(join_id)
            if next_id:
                await self._execute_from(next_id)
        except ValueError:
            # No JoinNode found — just follow DataNode's unconditional edge
            next_id = self._next_unconditional(node_id)
            if next_id:
                await self._execute_from(next_id)

    async def _execute_selection(self, node: SelectionNode) -> None:
        """Compare parallel experiment results and select the best."""
        import subprocess as sp

        from factory.worktree import remove_worktree

        self.result.nodes_executed += 1

        self._emit(
            "node.started",
            NodeStarted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node.id,
                node_type="SelectionNode",
            ),
        )

        start = time.monotonic()

        # Find the SubgraphForkNode's output (branch results)
        fork_output = ""
        for nid, output in self.result.node_outputs.items():
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list) and parsed and "exp_id" in parsed[0]:
                    fork_output = output
                    break
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        if self.dry_run or not fork_output:
            selection_result: dict[str, Any] = {"strategy": node.strategy, "winner": None, "reason": "dry-run"}
            self.result.node_outputs[node.id] = json.dumps(selection_result)
            self.completed_files |= self._actual_writes(node)
            elapsed = (time.monotonic() - start) * 1000
            self._emit(
                "node.completed",
                NodeCompleted(
                    workflow_name=self.workflow.name,
                    run_id=self.run_id,
                    node_id=node.id,
                    node_type="SelectionNode",
                    files_written=sorted(node.writes),
                    duration_ms=elapsed,
                ),
            )
            next_id = self._next_unconditional(node.id)
            if next_id:
                await self._execute_from(next_id)
            return

        branches: list[dict[str, Any]] = json.loads(fork_output)
        successful = [b for b in branches if b.get("success")]

        if not successful:
            self.result.halted = True
            self.result.halt_reason = "all parallel experiment branches failed"
            return

        # best_score: read eval results from each worktree
        best: dict[str, Any] | None = None
        best_score = -1.0

        for branch in successful:
            wt_path = Path(branch["worktree_path"])
            eval_file = wt_path / ".factory" / "last_eval.json"
            score = 0.0
            if eval_file.exists():
                try:
                    data = json.loads(eval_file.read_text())
                    score = float(data.get("total", data.get("score", 0.0)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            branch["score"] = score
            if score > best_score:
                best_score = score
                best = branch

        if not best:
            best = successful[0]

        # Merge winner branch into baseline
        winner_branch = best["branch"]
        try:
            sp.run(
                ["git", "merge", winner_branch, "--no-edit", "-m",
                 f"Merge parallel experiment winner (exp {best['exp_id']})"],
                cwd=self.project_path,
                check=True,
                capture_output=True,
            )
        except sp.CalledProcessError as exc:
            log.error("selection_merge_failed", branch=winner_branch, error=str(exc))
            self.result.halted = True
            self.result.halt_reason = f"failed to merge winner branch {winner_branch}"
            return

        # Finalize losers as superseded, clean up all worktrees
        from factory.store import ExperimentStore

        store = ExperimentStore(self.project_path)
        for branch in branches:
            wt_path = Path(branch.get("worktree_path", ""))
            branch_name = branch.get("branch", "")
            exp_id = branch.get("exp_id")

            if branch is not best and exp_id is not None:
                from factory.models import ExperimentRecord
                record = ExperimentRecord(
                    id=exp_id,
                    timestamp=__import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc),
                    hypothesis=branch.get("hypothesis", ""),
                    change_summary="superseded by experiment " + str(best["exp_id"]),
                    issue_number=None,
                    pr_number=None,
                    score_before=None,
                    score_after=branch.get("score"),
                    delta=None,
                    verdict="superseded",
                    cost_usd=None,
                    notes="",
                )
                try:
                    await store.finalize(exp_id, record)
                except Exception as exc:
                    log.warning("finalize_superseded_failed", exp_id=exp_id, error=str(exc))

            if wt_path.exists() and branch_name:
                try:
                    remove_worktree(self.project_path, wt_path, branch_name)
                except Exception as exc:
                    log.warning("worktree_cleanup_failed", path=str(wt_path), error=str(exc))

        selection_result = {
            "strategy": node.strategy,
            "winner_exp_id": best["exp_id"],
            "winner_score": best.get("score", 0.0),
            "winner_hypothesis": best.get("hypothesis", ""),
            "total_branches": len(branches),
            "successful_branches": len(successful),
        }
        self.result.node_outputs[node.id] = json.dumps(selection_result)
        self.completed_files |= self._actual_writes(node)

        elapsed = (time.monotonic() - start) * 1000
        self._emit(
            "node.completed",
            NodeCompleted(
                workflow_name=self.workflow.name,
                run_id=self.run_id,
                node_id=node.id,
                node_type="SelectionNode",
                files_written=sorted(node.writes),
                duration_ms=elapsed,
            ),
        )

        next_id = self._next_unconditional(node.id)
        if next_id:
            await self._execute_from(next_id)

    async def _run_node(self, node: NodeType) -> str:
        """Execute a single node and return its output."""
        if self.dry_run:
            return f"[dry-run] {node.id} executed"

        if isinstance(node, Study):
            return await self._run_study(node)

        if isinstance(node, FnNode):
            return await self._run_fn(node)

        if isinstance(node, AgentNode):
            return await self._run_agent(node)

        if isinstance(node, LLMNode):
            return await self._run_llm(node)

        return f"[unknown node type] {type(node).__name__}"

    async def _run_study(self, node: Study) -> str:
        """Run factory study command."""
        cmd = f"factory study {shlex.quote(str(self.project_path))}"
        if node.focus:
            cmd += f' --focus "{node.focus}"'
        return await self._run_shell(cmd)

    async def _run_fn(self, node: FnNode) -> str:
        """Run a FnNode's callable or shell command.

        When ``callable_name`` is set (format ``module.path:fn_name``),
        the function is imported and called with ``project_dir=str(project_path)``
        as keyword argument.  Otherwise falls back to running the shell
        ``command``.
        """
        if node.callable_name:
            return await self._run_callable(node)
        if not node.command:
            return ""
        cmd = node.command.replace("{project_path}", shlex.quote(str(self.project_path)))
        return await self._run_shell(cmd)

    async def _run_callable(self, node: FnNode) -> str:
        """Import and call a FnNode's Python callable."""
        import importlib

        ref = node.callable_name
        if not ref or ":" not in ref:
            raise ValueError(
                f"FnNode '{node.id}' callable_name must be 'module.path:fn_name', "
                f"got {ref!r}"
            )
        module_path, fn_name = ref.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name, None)
        if fn is None:
            raise ValueError(
                f"Module {module_path!r} has no attribute {fn_name!r}"
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: fn(project_dir=str(self.project_path)),
        )
        return str(result) if result is not None else ""

    async def _run_agent(self, node: AgentNode) -> str:
        """Invoke an agent via factory/agents/runner.py."""
        task = node.prompt_template.replace(
            "{project_path}", str(self.project_path),
        )
        context = self.node_context.get(node.id, "")
        if context:
            task = f"{task}\n\n{context}"

        model = node.model
        if not model:
            pool_entry = self.agent_pool.get(node.role.value)
            if pool_entry:
                model = pool_entry.model

        timeout = node.timeout
        if timeout is None:
            pool_entry = self.agent_pool.get(node.role.value)
            if pool_entry:
                timeout = pool_entry.timeout

        stdout, code = await self._agent_fn(
            node.role.value,  # type: ignore[arg-type]
            task,
            self.project_path,
            model=model or None,
            timeout=float(timeout) if timeout is not None else 600.0,
            node_id=node.id,
        )

        if code != 0:
            log.warning(
                "agent_nonzero_exit",
                role=node.role.value,
                code=code,
                output_len=len(stdout),
            )

        # Persist output to node.writes paths (mirrors _run_llm pattern).
        # Skip when auto_write_outputs is False (e.g., tests using FakeAgent).
        if node.writes and self.auto_write_outputs:
            for wpath in node.writes:
                fpath = self.project_path / wpath
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(stdout)

        return stdout

    async def _run_llm(self, node: LLMNode) -> str:
        """Run an LLMNode via direct API tool-use loop."""
        from factory.workflow.llm_loop import run_llm_loop

        context_parts: list[str] = []
        for read_path in sorted(node.reads):
            full_path = self.project_path / read_path
            if full_path.exists():
                context_parts.append(full_path.read_text())
        gate_context = self.node_context.get(node.id, "")
        if gate_context:
            context_parts.append(gate_context)

        output = await asyncio.wait_for(
            run_llm_loop(
                node, self.project_path,
                instance_context="\n\n".join(context_parts),
            ),
            timeout=float(node.timeout),
        )

        output_path = self.project_path / ".factory" / "reviews" / "builder-latest.md"
        if node.writes:
            first_write = next(iter(node.writes))
            output_path = self.project_path / first_write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

        return output

    async def _evaluate_gate(self, node: GateNode) -> Verdict:
        """Evaluate a gate and return a verdict."""
        if self.dry_run:
            return Verdict.proceed()

        effective_max = node.max_iterations if node.max_iterations is not None else 3

        if node.evaluator_type == "user":
            if self.auto_approve:
                log.info("gate.auto_approved", gate_id=node.id, workflow=self.workflow.name)
                return Verdict.proceed()
            prompt_text = node.gate_prompt or (
                f"Gate '{node.id}' [{self.workflow.name}] — enter verdict: proceed / reloop / halt"
            )
            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, self._input_fn, f"\n{prompt_text}\n> ")
            except (EOFError, OSError):
                return Verdict.halt(reason=f"gate '{node.id}': stdin closed (non-interactive context)")
            r = response.strip().lower()
            log.info("gate.user_verdict", gate_id=node.id, raw=r)
            # halt first (most consequential), then reloop, then proceed.
            # Word-boundary matching avoids false-positives ("asphalt", "halting").
            if re.search(r"\bhalt\b", r):
                return Verdict.halt(reason=f"user halted at gate '{node.id}'")
            if re.search(r"\breloop\b", r):
                target = self._resolve_reloop_target(r, node.id)
                if target is None:
                    return Verdict.halt(reason=f"gate '{node.id}': reloop requested but no RELOOP edge configured")
                return Verdict.reloop(target=target, feedback=r, max_iterations=effective_max)
            if re.search(r"\bproceed\b", r):
                return Verdict.proceed()
            # Unrecognized input — fail closed, matching fn/agent gate convention.
            return Verdict.halt(
                reason=f"gate '{node.id}': unrecognized input '{response.strip()}' — type 'proceed', 'reloop', or 'halt'",
            )
        if node.evaluator_type == "fn":
            if node.evaluator_command:
                cmd = node.evaluator_command.replace(
                    "{project_path}",
                    shlex.quote(str(self.project_path)),
                )
                try:
                    output = await self._run_shell(cmd)
                    return self._parse_fn_verdict(output, node.id, max_iterations=effective_max)
                except RuntimeError:
                    return Verdict.halt(reason=f"gate command failed: {cmd}")
            return Verdict.halt(reason=f"gate '{node.id}' has no evaluator_command configured")

        prompt = self._build_gate_prompt(node)
        from factory.agents.runner import invoke_agent

        model = "opus"
        pool_entry = self.agent_pool.get("ceo")
        if pool_entry:
            model = pool_entry.model

        stdout, code = await invoke_agent(
            "ceo",
            prompt,
            self.project_path,
            model=model,
        )

        if code != 0:
            return Verdict.halt(reason=f"CEO gate agent exited with code {code}")

        return self._parse_agent_verdict(stdout, node.id, max_iterations=effective_max)

    def _resolve_reloop_target(self, response: str, node_id: str) -> str | None:
        """Extract the reloop target node ID from a user response.

        Tries in order:
        1. First non-filler word after "reloop" in the response.
        2. First RELOOP edge target declared in the workflow graph.
        Returns None when no target can be resolved.
        """
        parts = response.split()
        reloop_idx = next((i for i, p in enumerate(parts) if re.search(r"\breloop\b", p)), None)
        if reloop_idx is not None:
            for word in parts[reloop_idx + 1:]:
                if word not in _USER_GATE_RELOOP_FILLERS:
                    return word
        for edge in self._edge_index.get(node_id, []):
            if edge.condition == VerdictType.RELOOP:
                return edge.target
        return None

    def _build_gate_prompt(self, node: GateNode) -> str:
        """Build the lightweight CEO gate prompt."""
        if node.gate_prompt:
            return node.gate_prompt.replace(
                "{project_path}", str(self.project_path),
            )

        output_files = sorted(node.reads) if node.reads else ["(no specific file)"]
        context = self.node_context.get(node.id, "none")

        reloop_targets: list[str] = []
        for edge in self._edge_index.get(node.id, []):
            if edge.condition == VerdictType.RELOOP:
                reloop_targets.append(edge.target)

        return CEO_GATE_PROMPT.format(
            step_name=node.id,
            workflow_name=self.workflow.name,
            output_file=", ".join(output_files),
            previous_context=context,
            reloop_targets=", ".join(reloop_targets) if reloop_targets else "(use exact node IDs)",
        )

    def _parse_agent_verdict(self, output: str, gate_id: str, max_iterations: int = 3) -> Verdict:
        """Parse agent output into a Verdict by examining the last non-empty line."""

        lines = output.strip().splitlines()
        last_line = ""
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break

        text = last_line.upper()

        if text.startswith("HALT") or re.match(r"^HALT\b", text):
            reason_match = re.search(r'REASON="([^"]+)"', last_line, re.IGNORECASE)
            reason = reason_match.group(1) if reason_match else "gate halted"
            return Verdict.halt(reason=reason)

        if text.startswith("RELOOP") or re.match(r"^RELOOP\b", text):
            target_match = re.search(r'TARGET="([^"]+)"', last_line, re.IGNORECASE)
            feedback_match = re.search(r'FEEDBACK="([^"]+)"', last_line, re.IGNORECASE)
            target = target_match.group(1) if target_match else None

            if target and target not in self.workflow.nodes:
                matches = [nid for nid in self.workflow.nodes if target in nid]
                if len(matches) == 1:
                    target = matches[0]
                else:
                    target = self._next_conditional(gate_id, VerdictType.RELOOP)

            if not target:
                target = self._next_conditional(gate_id, VerdictType.RELOOP)
            if not target:
                return Verdict.halt(reason=f"RELOOP verdict from gate '{gate_id}' missing target and no RELOOP edge defined")
            feedback = feedback_match.group(1) if feedback_match else "needs improvement"
            return Verdict.reloop(target=target, feedback=feedback, max_iterations=max_iterations)

        if text.startswith("PROCEED") or re.match(r"^PROCEED\b", text):
            return Verdict.proceed()

        first_line = ""
        for line in lines:
            if line.strip():
                first_line = line.strip()
                break

        if first_line and first_line != last_line:
            ft = first_line.upper()

            if ft.startswith("HALT") or re.match(r"^HALT\b", ft):
                reason_match = re.search(r'REASON="([^"]+)"', first_line, re.IGNORECASE)
                reason = reason_match.group(1) if reason_match else "gate halted"
                return Verdict.halt(reason=reason)

            if ft.startswith("RELOOP") or re.match(r"^RELOOP\b", ft):
                target_match = re.search(r'TARGET="([^"]+)"', first_line, re.IGNORECASE)
                feedback_match = re.search(r'FEEDBACK="([^"]+)"', first_line, re.IGNORECASE)
                target = target_match.group(1) if target_match else None

                if target and target not in self.workflow.nodes:
                    matches = [nid for nid in self.workflow.nodes if target in nid]
                    if len(matches) == 1:
                        target = matches[0]
                    else:
                        target = self._next_conditional(gate_id, VerdictType.RELOOP)

                if not target:
                    target = self._next_conditional(gate_id, VerdictType.RELOOP)
                if not target:
                    return Verdict.halt(reason=f"RELOOP verdict from gate '{gate_id}' missing target and no RELOOP edge defined")
                feedback = feedback_match.group(1) if feedback_match else "needs improvement"
                return Verdict.reloop(target=target, feedback=feedback, max_iterations=max_iterations)

            if ft.startswith("PROCEED") or re.match(r"^PROCEED\b", ft):
                return Verdict.proceed()

        return Verdict.halt(
            reason=(
                f"gate '{gate_id}' returned unparseable verdict "
                f"(expected PROCEED | RELOOP target=... | HALT reason=...): "
                f"{output.strip()[:200]}"
            )
        )

    def _parse_fn_verdict(self, output: str, gate_id: str, max_iterations: int = 3) -> Verdict:
        """Parse function output into a Verdict."""
        text = output.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "passed" in data:
                if data["passed"]:
                    return Verdict.proceed()
                return Verdict.halt(
                    reason=f"precheck failed: {data.get('blocking_failures', [])!r}"[:200]
                )
        except (json.JSONDecodeError, TypeError):
            pass

        first_line = text.split("\n")[0].strip().lower()
        if first_line.startswith("pass") or first_line.startswith("proceed"):
            return Verdict.proceed()
        if first_line.startswith("fail") or first_line.startswith("revert"):
            return Verdict.halt(reason=f"precheck failed: {text[:200]}")
        if first_line.startswith("reloop"):
            target = self._next_conditional(gate_id, VerdictType.RELOOP)
            raw_line = text.split("\n")[0].strip()
            after_prefix = raw_line.split(":", 1)[1].strip() if ":" in raw_line else ""
            feedback = after_prefix if after_prefix else "fn gate requested reloop"
            if target:
                return Verdict.reloop(target=target, feedback=feedback, max_iterations=max_iterations)
            return Verdict.halt(reason="fn gate returned RELOOP but no RELOOP edge defined")
        return Verdict.halt(
            reason=(
                f"gate '{gate_id}' returned unparseable verdict "
                f"(expected pass | fail | revert | reloop | {{\"passed\": bool}}): "
                f"{text[:200]}"
            )
        )

    async def _run_shell_or_exec(self, cmd: str) -> str:
        """Run a command, using exec mode for python3 -c to avoid quote issues."""
        m = re.match(r"""^python3\s+-c\s+(['"])(.*)\1\s*$""", cmd, re.DOTALL)
        if m:
            code = m.group(2)
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
            )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode() if stdout_bytes else ""

        if proc.returncode != 0:
            stderr = stderr_bytes.decode() if stderr_bytes else ""
            raise RuntimeError(
                f"command failed (exit {proc.returncode}): {cmd}\n{stderr[:500]}"
            )

        return stdout

    async def _run_shell(self, cmd: str) -> str:
        """Run a shell command and return stdout."""
        return await self._run_shell_or_exec(cmd)

    async def _wait_for_reads(self, node: NodeType) -> None:
        """Wait until all files in node.reads are available in completed_files."""
        if not node.reads:
            return
        poll_interval = 0.1
        max_wait = 60.0
        waited = 0.0
        while True:
            missing = node.reads - self.completed_files
            if not missing:
                return
            if waited >= max_wait:
                # Diagnostic: show what files exist vs what's expected
                existing = sorted(self.completed_files)
                log.warning(
                    'wait_for_reads_timeout_diagnostic',
                    node_id=node.id,
                    missing_reads=sorted(missing),
                    completed_files_count=len(self.completed_files),
                    sample_completed=existing[:10],
                )
                self.result.halted = True
                self.result.halt_reason = (
                    f"node '{node.id}' timed out waiting for reads: {sorted(missing)}"
                )
                return
            log.debug(
                "node.waiting_for_reads",
                node=node.id,
                missing=sorted(missing),
                waited_s=round(waited, 1),
            )
            await asyncio.sleep(poll_interval)
            waited += poll_interval

    def _next_unconditional(self, node_id: str) -> str | None:
        """Find the next node via unconditional edge."""
        for edge in self._edge_index.get(node_id, []):
            if edge.condition is None:
                return edge.target
        return None

    def _next_conditional(self, node_id: str, verdict_type: VerdictType) -> str | None:
        """Find the next node via conditional edge matching the verdict."""
        for edge in self._edge_index.get(node_id, []):
            if edge.condition == verdict_type:
                return edge.target
        return None

    def _validate_post_checks(self, node: AgentNode) -> None:
        """Enforce SPEC §7.3: validate post_checks after an AgentNode completes.

        Raises RuntimeError if any ArtifactCheck fails.
        """

        for check in node.post_checks:
            fpath = self.project_path / check.path
            if check.must_exist and not fpath.exists():
                raise RuntimeError(
                    f"post_check failed for node '{node.id}': "
                    f"artifact '{check.path}' must exist but was not found"
                )
            if not fpath.exists():
                continue
            if check.min_size > 0 and fpath.stat().st_size < check.min_size:
                raise RuntimeError(
                    f"post_check failed for node '{node.id}': "
                    f"artifact '{check.path}' size {fpath.stat().st_size} < "
                    f"min_size {check.min_size}"
                )
            if check.must_contain:
                content = fpath.read_text()
                for substr in check.must_contain:
                    if substr not in content:
                        raise RuntimeError(
                            f"post_check failed for node '{node.id}': "
                            f"artifact '{check.path}' must contain '{substr}'"
                        )

    def _emit(self, event_type: str, event: Any) -> None:
        """Emit a workflow event."""
        self.result.events.append({"type": event_type, **event.model_dump(mode="python")})
        try:
            emit_workflow_event(self.project_path, event_type, event)
        except Exception:
            log.debug("event_emission_failed", event_type=event_type)


def _parse_hypotheses(strategy_file: Path) -> list[str]:
    """Extract individual hypotheses from the strategist's current.md output."""
    text = strategy_file.read_text()
    hypotheses: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Hypothesis") or stripped.startswith("### Hypothesis"):
            if current:
                hypotheses.append("\n".join(current).strip())
                current = []
            current.append(stripped)
        elif stripped.startswith("## ") and current:
            hypotheses.append("\n".join(current).strip())
            current = []
        elif current:
            current.append(line)

    if current:
        hypotheses.append("\n".join(current).strip())

    if not hypotheses:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- **") or stripped.startswith("1. **"):
                hypotheses.append(stripped.lstrip("- 0123456789.").strip())

    return hypotheses


def _collect_subgraph_nodes(
    workflow: Workflow,
    entry: str,
    exit_node: str,
) -> set[str]:
    """Collect all node IDs on paths from entry to exit_node (inclusive).

    Uses bidirectional BFS: intersect nodes reachable forward from entry
    with nodes reachable backward from exit_node.  This excludes stray
    branches that are reachable from entry but do not lead to exit_node.
    """
    edges_by_source: dict[str, list[str]] = {}
    edges_by_target: dict[str, list[str]] = {}
    for edge in workflow.edges:
        edges_by_source.setdefault(edge.source, []).append(edge.target)
        edges_by_target.setdefault(edge.target, []).append(edge.source)

    def _bfs(start: str, adjacency: dict[str, list[str]], stop_at: str) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            if nid == stop_at:
                continue
            for neighbour in adjacency.get(nid, []):
                if neighbour not in visited:
                    queue.append(neighbour)
        return visited

    forward = _bfs(entry, edges_by_source, stop_at=exit_node)
    backward = _bfs(exit_node, edges_by_target, stop_at=entry)
    return forward & backward
