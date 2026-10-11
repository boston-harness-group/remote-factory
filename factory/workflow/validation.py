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
        # DataNode: edges are explicit now (no subgraph_entry/exit)

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
    runtime_provided = set(workflow.runtime_inputs) if workflow.runtime_inputs else set()

    for nid, node in workflow.nodes.items():
        if node.reads:
            predecessors = nx.ancestors(g, nid)
            if not predecessors:
                if not runtime_provided:
                    continue  # ordinary workflow: first node reads are external
                available_writes: set[str] = set(runtime_provided)
            else:
                available_writes = set(runtime_provided)
            for pred_id in predecessors:
                pred_node = workflow.nodes.get(pred_id)
                if pred_node:
                    available_writes |= pred_node.writes
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

        # DataNode: validate it has outgoing edges and a downstream JoinNode
        if type(node).__name__ == "DataNode":
            has_outgoing = any(e.source == nid for e in workflow.edges)
            if not has_outgoing:
                issues.append(f"DataNode '{nid}' has no outgoing edges")


def _validate_datanode_fork(workflow: Workflow, issues: list[str]) -> None:
    """Validate DataNode fork/join structure via real edges."""
    from factory.workflow.primitives import DataNode, JoinNode

    for nid, node in workflow.nodes.items():
        if not isinstance(node, DataNode):
            continue
        # Check that there's a JoinNode downstream
        edge_targets = [e.target for e in workflow.edges if e.source == nid]
        if not edge_targets:
            continue  # Already reported by _validate_fork_join_nodes
        # BFS to find JoinNode
        visited: set[str] = set()
        queue = list(edge_targets)
        found_join = False
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            target_node = workflow.nodes.get(current)
            if isinstance(target_node, JoinNode):
                found_join = True
                break
            for e in workflow.edges:
                if e.source == current:
                    queue.append(e.target)
        if not found_join:
            issues.append(
                f"DataNode '{nid}' has no downstream JoinNode — "
                f"branches have no convergence point"
            )


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
    """Task-backed DataNode branch entry should read current_item.json (WARNING).

    When a DataNode has a non-empty task_ref, the runtime writes
    .factory/current_item.json before running the branch.  The entry
    node's reads set should include this path.

    This is a WARNING, not an ERROR.
    """
    from factory.workflow.primitives import DataNode

    for nid, node in workflow.nodes.items():
        if not isinstance(node, DataNode) or not node.task_ref:
            continue
        # Find entry via edges
        edge_targets = [e.target for e in workflow.edges if e.source == nid]
        if not edge_targets:
            continue
        entry_id = edge_targets[0]
        entry = workflow.nodes.get(entry_id)
        if entry is None:
            continue
        if ".factory/current_item.json" not in entry.reads:
            log.warning(
                "datanode_entry_missing_current_item",
                data_node=nid,
                entry_node=entry_id,
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

    # Add implicit edges for SubgraphForkNode: node → subgraph_entry
    # so subgraph nodes are reachable in the graph
    for nid, node in nodes.items():
        if type(node).__name__ == "SubgraphForkNode":
            entry = node.subgraph_entry  # type: ignore[union-attr]
            if entry in nodes:
                g.add_edge(nid, entry, condition=None)
        # DataNode: edges are explicit now (real edges in graph)

    _validate_reachability(g, workflow, issues)
    _validate_cycles(g, workflow, issues)
    _validate_data_dependencies(g, workflow, issues)
    _validate_fork_join_nodes(workflow, issues)
    _validate_datanode_fork(workflow, issues)
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

    return issues
