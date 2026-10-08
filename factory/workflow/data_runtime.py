"""Data runtime — the single module that touches items and scores.

The executor delegates to ``run_fork()`` when it encounters a DataNode:
resolve items, create per-item workspaces, run setup → branch → verify,
record ``ItemResult``, and clean up workspaces in ``finally``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import structlog

from factory.models import ItemResult, ItemStatus
from factory.workflow.primitives import (
    DataItem,
    DataNode,
    JoinNode,
    Workflow,
)

log = structlog.get_logger()

_SYNC_EXCLUDE = {".git", ".factory-worktrees"}


def _sync_working_tree(src: Path, dst: Path) -> None:
    """Copy uncommitted files from *src* working tree to *dst* worktree."""
    for item in src.iterdir():
        if item.name in _SYNC_EXCLUDE:
            continue
        dest_item = dst / item.name
        if item.is_dir():
            if dest_item.exists():
                shutil.rmtree(dest_item)
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)


def _find_branch_and_join(
    workflow: Workflow,
    data_node_id: str,
) -> tuple[str, str, set[str]]:
    """Walk forward from the DataNode to a JoinNode, returning
    ``(entry_node_id, join_node_id, subgraph_node_ids)``.
    """
    edge_index: dict[str, list[str]] = {}
    for edge in workflow.edges:
        edge_index.setdefault(edge.source, []).append(edge.target)

    # Find entry: first edge target from the DataNode
    targets = edge_index.get(data_node_id, [])
    if not targets:
        raise ValueError(
            f"DataNode '{data_node_id}' has no outgoing edges — "
            f"cannot determine branch entry"
        )
    entry = targets[0]

    # Walk forward from entry to find the JoinNode, collecting subgraph nodes
    visited: set[str] = set()
    join_id: str | None = None
    queue = [entry]

    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        node = workflow.nodes.get(nid)
        if isinstance(node, JoinNode):
            join_id = nid
            continue
        visited.add(nid)
        for target in edge_index.get(nid, []):
            queue.append(target)

    if join_id is None:
        raise ValueError(
            f"DataNode '{data_node_id}' branch has no JoinNode — "
            f"cannot determine where branches converge"
        )

    return entry, join_id, visited


def _resolve_items(
    node: DataNode,
    data_node_id: str,
    project_path: Path,
    *,
    allowed_instance_ids: set[str] | None = None,
    task: Any | None = None,
    run_id: str = "",
) -> tuple[list[tuple[DataItem, Any | None]], Any | None]:
    """Resolve, filter, shuffle and limit data items from a DataNode.

    Returns ``(task_instances, resolved_task)`` — shared by
    ``run_fork`` and ``evaluate_fork``.
    """
    from factory.task import Task as _Task
    from factory.task import TaskInstance as _TaskInstance

    task_instances: list[tuple[DataItem, _TaskInstance | None]] = []
    resolved_task: _Task | None = None

    if node.inline_items:
        task_instances = [(item, None) for item in node.inline_items]
    elif node.task_ref:
        import sys
        tasks_dir = str(project_path / '.factory' / 'tasks')
        if tasks_dir not in sys.path:
            sys.path.insert(0, tasks_dir)

        from factory.task import TaskRef
        task_ref = TaskRef(ref=node.task_ref)
        resolved_task = task_ref.resolve()
        for _ti in resolved_task.instances():
            task_instances.append((
                DataItem(
                    id=_ti.id,
                    path=str(_ti.path) if _ti.path else None,
                    metadata=_ti.metadata,
                ),
                _ti,
            ))
    elif node.source_path:
        src = Path(node.source_path)
        if not src.is_absolute():
            src = project_path / src
        if not src.exists():
            raise FileNotFoundError(
                f"DataNode '{data_node_id}': source_path not found: {node.source_path}"
            )
        if node.source_format == "directory" and src.is_dir():
            for child in sorted(src.iterdir()):
                if child.is_dir():
                    task_instances.append((DataItem(id=child.name, path=str(child)), None))
        elif node.source_format == "jsonl" and src.is_file():
            for idx, line in enumerate(src.read_text().splitlines()):
                if line.strip():
                    try:
                        task_instances.append((DataItem(
                            id=str(idx),
                            metadata=json.loads(line),
                        ), None))
                    except json.JSONDecodeError:
                        log.warning("jsonl_parse_error", line_number=idx + 1)
        elif node.source_format == "csv" and src.is_file():
            import csv
            with src.open(newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    task_instances.append((DataItem(id=str(idx), metadata=dict(row)), None))

    # Fallback to InnerLoop's task
    if resolved_task is None and task is not None:
        resolved_task = task

    # Apply instance filter (train/val firewall)
    if allowed_instance_ids is not None and node.task_ref:
        task_instances = [
            (item, inst) for item, inst in task_instances
            if item.id in allowed_instance_ids
        ]

    # Apply shuffle/limit
    if node.shuffle:
        seed = (
            node.shuffle_seed
            if node.shuffle_seed is not None
            else int.from_bytes(
                hashlib.sha256(f"{data_node_id}:{run_id}".encode()).digest()[:8],
                "big",
            )
        )
        random.Random(seed).shuffle(task_instances)
    if node.limit is not None and node.limit > 0:
        task_instances = task_instances[:node.limit]

    if len(task_instances) == 0:
        raise ValueError(
            f"DataNode '{data_node_id}' resolved 0 items after filtering"
        )

    if len(task_instances) > node.max_items:
        raise ValueError(
            f"DataNode '{data_node_id}' resolved {len(task_instances)} items, "
            f"exceeding max_items={node.max_items}"
        )

    return task_instances, resolved_task


async def run_fork(
    workflow: Workflow,
    data_node: DataNode,
    data_node_id: str,
    project_path: Path,
    *,
    agent_pool: dict[str, Any] | None = None,
    dry_run: bool = False,
    agent_fn: Any | None = None,
    auto_write_outputs: bool = True,
    allowed_instance_ids: set[str] | None = None,
    task: Any | None = None,
    completed_files: set[str] | None = None,
    run_id: str = "",
) -> list[dict[str, Any]]:
    """Run the DataNode fork: resolve items, run branch per item, return results.

    Each item gets its own workspace snapshotted from the parent working
    tree (including uncommitted files).  Cleanup is in a finally block.
    """
    from factory.task import TaskInstance as _TaskInstance

    node = data_node

    # ── Resolve branch topology from edges ─────────────────────
    entry, join_id, subgraph_ids = _find_branch_and_join(
        workflow, data_node_id,
    )

    # Build branch sub-workflow
    sub_workflow = workflow.subgraph(
        subgraph_ids,
        name=f"{workflow.name}__data_item",
        start_node=entry,
    )

    # ── Resolve data items ─────────────────────────────────────
    task_instances, resolved_task = _resolve_items(
        node, data_node_id, project_path,
        allowed_instance_ids=allowed_instance_ids,
        task=task,
        run_id=run_id,
    )

    # ── Run items with parallelism ─────────────────────────────
    from factory.workflow.executor import WorkflowExecutor

    sem = asyncio.Semaphore(node.parallelism)
    worktrees_to_clean: list[tuple[Path, str]] = []
    item_results: list[dict[str, Any]] = []

    try:
        async def run_item(
            pair: tuple[DataItem, _TaskInstance | None],
            item_idx: int,
        ) -> dict[str, Any]:
            item, inst = pair
            item_project_path = project_path
            wt_branch: str | None = None
            t0 = time.monotonic()
            async with sem:
                try:
                    # Create per-item worktree snapshotted from parent working tree
                    wt_dir = (
                        project_path
                        / ".factory-worktrees"
                        / f"data-{run_id}-{item_idx}"
                    )
                    wt_branch = f"factory/data-{run_id}-{item_idx}"
                    wt_dir.parent.mkdir(parents=True, exist_ok=True)

                    # Get HEAD commit
                    rev_result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    base_commit = rev_result.stdout.strip()

                    # Create worktree from HEAD
                    subprocess.run(
                        [
                            "git", "worktree", "add",
                            str(wt_dir), "-b", wt_branch, base_commit,
                        ],
                        cwd=project_path,
                        check=True,
                        capture_output=True,
                    )
                    worktrees_to_clean.append((wt_dir, wt_branch))

                    # Copy uncommitted files from parent working tree
                    _sync_working_tree(project_path, wt_dir)

                    item_project_path = wt_dir

                    # Create TaskInstance from DataItem when task came from
                    # InnerLoop but items are inline/source_path
                    if resolved_task is not None and inst is None:
                        inst = _TaskInstance(
                            id=item.id,
                            path=Path(item.path) if item.path else None,
                            metadata=item.metadata or {},
                        )

                    if resolved_task is not None and inst is not None:
                        try:
                            resolved_task.setup(inst, item_project_path)
                        except Exception as setup_exc:
                            duration_s = time.monotonic() - t0
                            return ItemResult(
                                item_id=item.id,
                                status=ItemStatus.errored,
                                score=0.0,
                                error=f"setup_failed: {setup_exc}",
                                duration_s=duration_s,
                            ).model_dump()
                        item = DataItem(
                            id=inst.id,
                            path=str(inst.path) if inst.path else None,
                            metadata=inst.metadata,
                            prompt=resolved_task.prompt(inst),
                        )

                    # Write current_item.json
                    item_json_path = item_project_path / ".factory" / "current_item.json"
                    item_json_path.parent.mkdir(parents=True, exist_ok=True)
                    item_json_path.write_text(json.dumps(item.model_dump()))

                    try:
                        item_executor = WorkflowExecutor(
                            sub_workflow.model_copy(deep=True),
                            item_project_path,
                            agent_pool=agent_pool or {},
                            dry_run=dry_run,
                            agent_fn=agent_fn,
                            initial_context=item.prompt or None,
                            auto_write_outputs=auto_write_outputs,
                        )
                        # Include current_item.json so subgraph nodes don't block
                        base_files = (completed_files or set()) | {".factory/current_item.json"}
                        item_executor.completed_files = base_files
                        await item_executor.execute()
                    finally:
                        item_json_path.unlink(missing_ok=True)

                    # Verify
                    score = 0.0
                    passed = False
                    verify_details: dict[str, Any] = {}

                    if resolved_task is not None and inst is not None:
                        vr = resolved_task.verify(inst, item_project_path)
                        score = vr.score
                        passed = vr.passed
                        verify_details = vr.details or {}

                    duration_s = time.monotonic() - t0
                    if passed:
                        status = ItemStatus.ok
                    else:
                        status = ItemStatus.failed

                    return ItemResult(
                        item_id=item.id,
                        status=status,
                        score=score,
                        verify_details=verify_details,
                        duration_s=duration_s,
                    ).model_dump()

                except Exception as exc:
                    duration_s = time.monotonic() - t0
                    return ItemResult(
                        item_id=item.id,
                        status=ItemStatus.errored,
                        score=0.0,
                        error=str(exc),
                        duration_s=duration_s,
                    ).model_dump()

        tasks = [run_item(pair, idx) for idx, pair in enumerate(task_instances)]
        results = await asyncio.gather(*tasks)
        item_results = list(results)

    finally:
        # Clean up worktrees
        for wt_path, wt_branch_name in worktrees_to_clean:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", str(wt_path), "--force"],
                    cwd=project_path,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "branch", "-D", wt_branch_name],
                    cwd=project_path,
                    capture_output=True,
                )
            except Exception as wt_exc:
                log.warning(
                    "data_worktree_cleanup_failed",
                    path=str(wt_path),
                    error=str(wt_exc),
                )

    return item_results


async def evaluate_fork(
    workflow: Workflow,
    data_node: DataNode,
    data_node_id: str,
    project_path: Path,
    *,
    allowed_instance_ids: set[str] | None = None,
    task: Any | None = None,
    run_id: str = "",
) -> list[dict[str, Any]]:
    """Evaluate items via setup + verify only (no branch workflow execution).

    Used by the outer-loop evaluator for val / all_pass evaluations where
    only scores matter.  Avoids running the branch workflow (which would
    produce side effects like log entries and file mutations).
    """
    from factory.task import TaskInstance as _TaskInstance

    node = data_node

    # ── Resolve data items (shared with run_fork) ────────────
    task_instances, resolved_task = _resolve_items(
        node, data_node_id, project_path,
        allowed_instance_ids=allowed_instance_ids,
        task=task,
        run_id=run_id,
    )

    # ── Evaluate: setup + verify per item (no branch workflow) ──
    item_results: list[dict[str, Any]] = []

    for item, inst_or_none in task_instances:
        inst = inst_or_none
        t0 = time.monotonic()
        try:
            if resolved_task is not None and inst is None:
                inst = _TaskInstance(
                    id=item.id,
                    path=Path(item.path) if item.path else None,
                    metadata=item.metadata or {},
                )

            if resolved_task is not None and inst is not None:
                try:
                    resolved_task.setup(inst, project_path)
                except Exception as setup_exc:
                    duration_s = time.monotonic() - t0
                    item_results.append(ItemResult(
                        item_id=item.id,
                        status=ItemStatus.errored,
                        score=0.0,
                        error=f"setup_failed: {setup_exc}",
                        duration_s=duration_s,
                    ).model_dump())
                    continue

                vr = resolved_task.verify(inst, project_path)
                duration_s = time.monotonic() - t0
                status = ItemStatus.ok if vr.passed else ItemStatus.failed
                item_results.append(ItemResult(
                    item_id=item.id,
                    status=status,
                    score=vr.score,
                    verify_details=vr.details or {},
                    duration_s=duration_s,
                ).model_dump())
            else:
                duration_s = time.monotonic() - t0
                item_results.append(ItemResult(
                    item_id=item.id,
                    status=ItemStatus.ok,
                    score=0.0,
                    duration_s=duration_s,
                ).model_dump())

        except Exception as exc:
            duration_s = time.monotonic() - t0
            item_results.append(ItemResult(
                item_id=item.id,
                status=ItemStatus.errored,
                score=0.0,
                error=str(exc),
                duration_s=duration_s,
            ).model_dump())

    return item_results
