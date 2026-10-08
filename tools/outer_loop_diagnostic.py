#!/usr/bin/env python3
"""Outer loop diagnostic runner — live visibility into evolution.

Usage:
    python tools/outer_loop_diagnostic.py \\
        --task path/to/task.py:TaskClass \\
        --workflow path/to/workflow.py \\
        --budget 10 --population-size 4 \\
        --strategy executor

Shows per-generation:
    - Each individual: topology, node count, frozen subgraph intact?
    - Per-item scores from verify()
    - Designer variant validity (prompt_template populated, gate edges present)
    - Reflection summary
    - Score trajectory
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _load_task(task_ref: str):
    """Load a Task class from module:Class or file:Class ref."""
    if ":" in task_ref:
        parts = task_ref.rsplit(":", 1)
        if parts[0].endswith(".py"):
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("_task", parts[0])
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, parts[1])()
        else:
            from factory.task import TaskRef
            return TaskRef(ref=task_ref).resolve()
    raise ValueError(f"Invalid task ref: {task_ref}")


def _load_workflow(workflow_path: str):
    """Load a workflow from a .py file."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("_wf", workflow_path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.workflow() if hasattr(mod, "workflow") else mod.build_workflow()


def _check_individual(workflow, frozen_ids: set[str]) -> dict:
    """Diagnose an individual workflow for common issues."""
    from factory.workflow.primitives import AgentNode, DataNode, GateNode
    from factory.workflow.validation import validate_workflow

    issues = validate_workflow(workflow)
    diag = {
        "name": workflow.name,
        "nodes": len(workflow.nodes),
        "edges": len(workflow.edges),
        "valid": len(issues) == 0,
        "issues": issues,
    }

    for nid, node in workflow.nodes.items():
        if isinstance(node, DataNode):
            diag["datanode"] = nid
            diag["subgraph_entry"] = node.subgraph_entry
            diag["subgraph_exit"] = node.subgraph_exit
            entry_in_frozen = node.subgraph_entry in frozen_ids
            diag["frozen_subgraph_intact"] = entry_in_frozen

        if isinstance(node, AgentNode):
            if not node.prompt_template:
                diag.setdefault("empty_prompts", []).append(nid)

    gate_ids = {nid for nid, n in workflow.nodes.items() if isinstance(n, GateNode)}
    for gid in gate_ids:
        outgoing = [e for e in workflow.edges if e.source == gid]
        if outgoing:
            has_proceed = any(
                e.condition and hasattr(e.condition, 'value') and e.condition.value == "proceed"
                for e in outgoing
            )
            if not has_proceed:
                diag.setdefault("gates_no_proceed", []).append(gid)

    return diag


def _print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def _print_individual(idx: int, diag: dict):
    status = "✅" if diag["valid"] and not diag.get("empty_prompts") else "❌"
    frozen = "✅" if diag.get("frozen_subgraph_intact", True) else "❌ ORPHANED"
    print(f"\n  [{idx}] {diag['name']} — {diag['nodes']} nodes, {diag['edges']} edges {status}")
    if "datanode" in diag:
        print(f"      DataNode: entry={diag['subgraph_entry']}, exit={diag['subgraph_exit']} — frozen: {frozen}")
    if diag.get("empty_prompts"):
        print(f"      ❌ Empty prompts: {diag['empty_prompts']}")
    if diag.get("gates_no_proceed"):
        print(f"      ❌ Gates without PROCEED: {diag['gates_no_proceed']}")
    if diag["issues"]:
        for issue in diag["issues"][:3]:
            print(f"      ⚠ {issue}")


def main():
    parser = argparse.ArgumentParser(description="Outer loop diagnostic runner")
    parser.add_argument("--task", required=True, help="Task ref (file.py:Class or module:Class)")
    parser.add_argument("--workflow", required=True, help="Workflow .py file path")
    parser.add_argument("--project-dir", default=".", help="Project directory")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--strategy", default="executor", choices=["executor", "ceo-skill", "ceo-tool"])
    parser.add_argument("--frozen", nargs="*", help="Frozen node IDs (auto-detected if not set)")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    task_path = args.task
    if ":" in task_path and not task_path.split(":")[0].endswith(".py"):
        sys.path.insert(0, str(project_dir / ".factory" / "tasks"))

    _print_header("OUTER LOOP DIAGNOSTIC")
    print(f"  Task:       {args.task}")
    print(f"  Workflow:   {args.workflow}")
    print(f"  Strategy:   {args.strategy}")
    print(f"  Budget:     {args.budget}")
    print(f"  Population: {args.population_size}")

    task = _load_task(args.task)
    workflow = _load_workflow(args.workflow)

    instances = list(task.instances())
    print(f"  Instances:  {len(instances)} items")
    for inst in instances[:5]:
        print(f"    - {inst.id}")
    if len(instances) > 5:
        print(f"    ... and {len(instances) - 5} more")

    from factory.workflow.primitives import DataNode
    frozen_ids = set(args.frozen) if args.frozen else set()
    if not frozen_ids:
        for nid, node in workflow.nodes.items():
            if isinstance(node, DataNode):
                from factory.workflow.executor import _collect_subgraph_nodes
                sg = _collect_subgraph_nodes(workflow, node.subgraph_entry, node.subgraph_exit)
                frozen_ids = {nid} | sg
                break
    print(f"  Frozen:     {sorted(frozen_ids)}")

    _print_header("SEED WORKFLOW")
    seed_diag = _check_individual(workflow, frozen_ids)
    _print_individual(0, seed_diag)

    from factory.outer_loop import SwarmConfig, SwarmEngine
    from factory.outer_loop.evaluator import SwarmEvaluator
    from factory.outer_loop.designer import DesignerAgent

    task_module = args.task if ":" in args.task else args.task.replace(".py", "").replace("/", ".")
    config = SwarmConfig(
        benchmark="diagnostic",
        budget=args.budget,
        population_size=args.population_size,
        tournament_size=min(3, args.population_size),
        frozen_node_ids=list(frozen_ids),
        task_module=task_module,
        execution_strategy=args.strategy,
    )
    config.set_task(task)

    _print_header("DESIGNER VARIANTS")
    designer = DesignerAgent()
    for method_name in ["design_minimal", "design_thorough"]:
        method = getattr(designer, method_name)
        try:
            variant = method(
                "diagnostic",
                seed_workflow=workflow,
                frozen_node_ids=frozen_ids,
                execution_strategy=args.strategy,
            )
            diag = _check_individual(variant, frozen_ids)
            _print_individual(0, diag)
        except Exception as e:
            print(f"\n  ❌ {method_name} FAILED: {e}")

    _print_header("RUNNING OUTER LOOP")

    model = os.environ.get("ANTHROPIC_MODEL")
    if model:
        for nid, node in workflow.nodes.items():
            if hasattr(node, "model"):
                node.model = model

    evaluator = SwarmEvaluator(config, inner_loop_factory=True, project_dir=project_dir)
    engine = SwarmEngine(config, evaluator, project_dir=project_dir)

    start = time.time()
    result = engine.run(workflow, project_dir=str(project_dir))
    elapsed = time.time() - start

    _print_header("RESULTS")
    print(f"  Best score:    {result.best_score}")
    print(f"  Generations:   {result.generations_completed}")
    print(f"  Convergence:   {result.convergence_reason}")
    print(f"  Elapsed:       {elapsed:.1f}s")

    if hasattr(result, "trajectory") and result.trajectory:
        print("\n  Score trajectory:")
        for gen in result.trajectory:
            print(f"    gen {gen.get('generation', '?')}: best={gen.get('best_score', '?'):.4f} mean={gen.get('mean_score', '?'):.4f}")

    print()


if __name__ == "__main__":
    main()
