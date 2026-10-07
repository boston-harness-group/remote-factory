"""NetworkX-based graph validation for workflow definitions."""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from factory.workflow.primitives import Workflow

log = structlog.get_logger()


def _validate_start_node(workflow: Workflow, issues: list[str]) -> None:
    if workflow.start_node not in workflow.nodes:
        issues.append(f"start_node '{workflow.start_node}' not in nodes")


def _validate_edges(workflow: Workflow, issues: list[str]) -> None:
    for edge in workflow.edges:
        if edge.source not in workflow.nodes:
            issues.append(f"edge source '{edge.source}' not in nodes")
        if edge.target not in workflow.nodes:
            issues.append(f"edge target '{edge.target}' not in nodes")


def _validate_reachability(
    g: nx.DiGraph, workflow: Workflow, issues: list[str],  # type: ignore[type-arg]
) -> None:
    # Add implicit edges for fork/join semantics.
    # ForkNode.targets are reached implicitly (not via explicit edges).
    # JoinNode.sources flow into the join implicitly.
    # DataNode.subgraph_entry/exit are reached implicitly.
    nodes = workflow.nodes
    for nid, node in nodes.items():
        if type(node).__name__ == "ForkNode":
            for t in node.targets:  # type: ignore[union-attr]
                if t in nodes:
                    g.add_edge(nid, t)
        if type(node).__name__ == "JoinNode":
            for s in node.sources:  # type: ignore[union-attr]
                if s in nodes:
                    g.add_edge(s, nid)
        if type(node).__name__ == "DataNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            exit_node = node.subgraph_exit  # type: ignore[union-attr]
            if entry in nodes:
                g.add_edge(nid, entry)
            if exit_node in nodes:
                g.add_edge(nid, exit_node)

    reachable = nx.descendants(g, workflow.start_node) | {workflow.start_node}
    unreachable = set(workflow.nodes.keys()) - reachable
    for nid in sorted(unreachable):
        issues.append(f"node '{nid}' is unreachable from start_node")


def _validate_cycles(
    g: nx.DiGraph, workflow: Workflow, issues: list[str],  # type: ignore[type-arg]
) -> None:
    cycles = list(nx.simple_cycles(g))
    for cycle in cycles:
        cycle_edges = []
        for i in range(len(cycle)):
            src = cycle[i]
            tgt = cycle[(i + 1) % len(cycle)]
            cycle_edges.append((src, tgt))

        has_gate_with_limit = False
        for src, tgt in cycle_edges:
            if type(workflow.nodes.get(src)).__name__ == "GateNode":
                for edge in workflow.edges:
                    if edge.source == src and edge.target == tgt and edge.condition is not None:
                        has_gate_with_limit = True
                        break
            if has_gate_with_limit:
                break

        if not has_gate_with_limit:
            cycle_str = " -> ".join(cycle + [cycle[0]])
            issues.append(f"cycle without gate condition: {cycle_str}")


def _validate_data_dependencies(
    g: nx.DiGraph, workflow: Workflow, issues: list[str],  # type: ignore[type-arg]
) -> None:
    for nid, node in workflow.nodes.items():
        if node.reads:
            predecessors = nx.ancestors(g, nid)
            if not predecessors:
                continue
            available_writes: set[str] = set()
            for pred_id in predecessors:
                pred_node = workflow.nodes.get(pred_id)
                if pred_node:
                    available_writes |= pred_node.writes
                    # DataNode implicitly writes .factory/current_item.json
                    # before running its subgraph (executor.py L1012).
                    if type(pred_node).__name__ == 'DataNode':
                        available_writes.add('.factory/current_item.json')
            missing = node.reads - available_writes
            if missing:
                issues.append(
                    f"node '{nid}' reads {missing} but no predecessor writes them"
                )


def _validate_fork_join_nodes(workflow: Workflow, issues: list[str]) -> None:
    for nid, node in workflow.nodes.items():
        if type(node).__name__ == "ForkNode":
            for t in node.targets:  # type: ignore[union-attr]
                if t not in workflow.nodes:
                    issues.append(f"fork '{nid}' target '{t}' not in nodes")

        if type(node).__name__ == "JoinNode":
            for s in node.sources:  # type: ignore[union-attr]
                if s not in workflow.nodes:
                    issues.append(f"join '{nid}' source '{s}' not in nodes")

        if type(node).__name__ == "SubgraphForkNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            exit_node = node.subgraph_exit  # type: ignore[union-attr]
            if entry not in workflow.nodes:
                issues.append(f"subgraph_fork '{nid}' entry '{entry}' not in nodes")
            if exit_node not in workflow.nodes:
                issues.append(f"subgraph_fork '{nid}' exit '{exit_node}' not in nodes")

        if type(node).__name__ == "DataNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            exit_node = node.subgraph_exit  # type: ignore[union-attr]
            if entry not in workflow.nodes:
                issues.append(f"data_node '{nid}' entry '{entry}' not in nodes")
            if exit_node not in workflow.nodes:
                issues.append(f"data_node '{nid}' exit '{exit_node}' not in nodes")


def _validate_datanode_edges(workflow: Workflow, issues: list[str]) -> None:
    """Reject explicit edges from a DataNode to its own subgraph nodes.

    The executor handles subgraph execution internally — explicit edges
    would cause double-execution.
    """
    for nid, node in workflow.nodes.items():
        if type(node).__name__ != "DataNode":
            continue
        entry = node.subgraph_entry  # type: ignore[union-attr]
        exit_node = node.subgraph_exit  # type: ignore[union-attr]
        subgraph_ids = _collect_subgraph_nodes(workflow, entry, exit_node)
        for edge in workflow.edges:
            if edge.source == nid and edge.target in subgraph_ids:
                issues.append(
                    f"Edge from DataNode {nid} to its own subgraph node {edge.target} "
                    f"would cause double-execution. Remove explicit edges into DataNode "
                    f"subgraphs — the executor handles subgraph execution internally."
                )


def _validate_datanode_exit(workflow: Workflow, issues: list[str]) -> None:
    """Warn when a DataNode's subgraph_exit points to a Loop GateNode.

    When subgraph_exit is a GateNode that participates in a Loop (has
    outgoing RELOOP edges), _collect_subgraph_nodes stops BFS at the gate,
    excluding the PROCEED edge target (the real exit node).
    Workflow.subgraph() then drops the PROCEED edge, causing execution
    to silently halt after one loop iteration.

    Terminal GateNodes (no RELOOP edges) are fine as subgraph_exit — they
    don't have a PROCEED edge that would be dropped.
    """
    from factory.workflow.primitives import VerdictType

    for nid, node in workflow.nodes.items():
        if type(node).__name__ != "DataNode":
            continue
        exit_id = node.subgraph_exit  # type: ignore[union-attr]
        exit_node = workflow.nodes.get(exit_id)
        if exit_node is not None and type(exit_node).__name__ == "GateNode":
            has_reloop = any(
                e.source == exit_id and e.condition == VerdictType.RELOOP
                for e in workflow.edges
            )
            if not has_reloop:
                continue
            issues.append(
                f"DataNode '{nid}' has subgraph_exit pointing to GateNode '{exit_id}'. "
                f"This drops the PROCEED edge. Use the Loop's exit_node instead."
            )


def _collect_subgraph_nodes(
    workflow: Workflow,
    entry: str,
    exit_node: str,
) -> set[str]:
    from factory.workflow.executor import _collect_subgraph_nodes as _exec_collect
    return _exec_collect(workflow, entry, exit_node)


def _validate_agent_prompts(workflow: Workflow, issues: list[str]) -> None:
    """AgentNode.prompt_template must be non-empty and non-whitespace (ERROR)."""
    from factory.workflow.primitives import AgentNode

    for nid, node in workflow.nodes.items():
        if isinstance(node, AgentNode) and not node.prompt_template.strip():
            issues.append(
                f"AgentNode '{nid}' has empty prompt_template"
            )


def _validate_gate_edges(workflow: Workflow, issues: list[str]) -> None:
    """GateNode must have at least one PROCEED edge, unconditional edge, or be terminal (ERROR).

    Acceptable patterns:
    - Terminal gate (no outgoing edges) — workflow-terminating decision point.
    - Gate with PROCEED or unconditional edge — normal flow.
    - Gate with only RELOOP/HALT edges — halts after max iterations (valid loop-only gate).

    Only gates with outgoing edges that lack both PROCEED/unconditional edges
    AND lack RELOOP edges are flagged — they would stall on any verdict.
    """
    from factory.workflow.primitives import GateNode, VerdictType

    for nid, node in workflow.nodes.items():
        if not isinstance(node, GateNode):
            continue
        outgoing = [e for e in workflow.edges if e.source == nid]
        if not outgoing:
            # Terminal gate — no outgoing edges is fine
            continue
        has_proceed = any(e.condition == VerdictType.PROCEED for e in outgoing)
        has_unconditional = any(e.condition is None for e in outgoing)
        has_reloop = any(e.condition == VerdictType.RELOOP for e in outgoing)
        if not has_proceed and not has_unconditional and not has_reloop:
            issues.append(
                f"GateNode '{nid}' has outgoing edges but no PROCEED, "
                f"unconditional, or RELOOP edge — execution will stall"
            )


def _validate_datanode_entry_reads(workflow: Workflow) -> None:
    """Task-backed DataNode subgraph_entry should read current_item.json (WARNING).

    When a DataNode has a non-empty task_ref, the executor writes
    .factory/current_item.json before running the subgraph.  The entry
    node's reads set should include this path so that downstream nodes
    can rely on it being available.

    This is a WARNING, not an ERROR — some task-backed DataNodes may
    inject item data through a different mechanism.
    """
    from factory.workflow.primitives import DataNode

    for nid, node in workflow.nodes.items():
        if not isinstance(node, DataNode) or not node.task_ref:
            continue
        entry = workflow.nodes.get(node.subgraph_entry)
        if entry is None:
            continue
        if ".factory/current_item.json" not in entry.reads:
            log.warning(
                "datanode_entry_missing_current_item",
                data_node=nid,
                entry_node=node.subgraph_entry,
                entry_reads=sorted(entry.reads),
                hint="task-backed DataNode entry should read .factory/current_item.json",
            )


def validate_workflow(workflow: Workflow) -> list[str]:
    """Validate a workflow graph. Returns a list of issues (empty = valid)."""
    issues: list[str] = []

    _validate_start_node(workflow, issues)
    _validate_edges(workflow, issues)

    if issues:
        return issues

    g: nx.DiGraph[str] = nx.DiGraph()
    nodes = workflow.nodes
    for nid in nodes:
        g.add_node(nid)
    for edge in workflow.edges:
        g.add_edge(edge.source, edge.target, condition=edge.condition)

    # Add implicit edges for SubgraphForkNode and DataNode: node → subgraph_entry
    # so subgraph nodes are reachable in the graph
    for nid, node in nodes.items():
        if type(node).__name__ == "SubgraphForkNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            if entry in nodes:
                g.add_edge(nid, entry, condition=None)
        if type(node).__name__ == "DataNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            if entry in nodes:
                g.add_edge(nid, entry, condition=None)

    _validate_reachability(g, workflow, issues)
    _validate_cycles(g, workflow, issues)
    _validate_data_dependencies(g, workflow, issues)
    _validate_fork_join_nodes(workflow, issues)
    _validate_datanode_edges(workflow, issues)
    _validate_datanode_exit(workflow, issues)
    _validate_agent_prompts(workflow, issues)
    _validate_gate_edges(workflow, issues)
    _validate_datanode_entry_reads(workflow)  # WARNING only — does not add to issues

    for nid, node in nodes.items():
        if type(node).__name__ == "SubgraphForkNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            exit_node = node.subgraph_exit  # type: ignore[union-attr]
            if entry in nodes and exit_node in nodes:
                if not nx.has_path(g, entry, exit_node):
                    issues.append(
                        f"subgraph_fork '{nid}': no path from entry '{entry}' to exit '{exit_node}'"
                    )
        if type(node).__name__ == "DataNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            exit_node = node.subgraph_exit  # type: ignore[union-attr]
            if entry in nodes and exit_node in nodes:
                if not nx.has_path(g, entry, exit_node):
                    issues.append(
                        f"data_node '{nid}': no path from entry '{entry}' to exit '{exit_node}'"
                    )

    return issues
