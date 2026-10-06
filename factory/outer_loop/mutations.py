"""Structured graph mutation operators and strategy protocol for workflow evolution."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import networkx as nx
import structlog

from factory.outer_loop.models import MutationRecord, MutationType
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode as _DataNodeType,
    Edge,
    ForkNode,
    JoinNode,
    NodeType,
    Workflow,
)

from factory.outer_loop.reflector import MutationSuggestion, ReflectionReport

log = structlog.get_logger()


def _is_in_data_subgraph(workflow: Workflow, node_id: str) -> bool:
    """Check if node_id is inside any DataNode's subgraph."""
    from factory.workflow.executor import _collect_subgraph_nodes
    from factory.workflow.primitives import DataNode

    for nid, node in workflow.nodes.items():
        if isinstance(node, DataNode):
            if (node.subgraph_entry in workflow.nodes
                    and node.subgraph_exit in workflow.nodes):
                subgraph = _collect_subgraph_nodes(
                    workflow, node.subgraph_entry, node.subgraph_exit
                )
                if node_id in subgraph:
                    return True
    return False


@runtime_checkable
class MutationStrategy(Protocol):
    """Protocol for pluggable mutation operator selection."""

    def select_operator(
        self, parent: Workflow, generation: int, archive_stats: dict[str, object]
    ) -> MutationType: ...

    def get_mutation_rate(self, generation: int) -> float: ...

    def get_designer_ratio(self, generation: int) -> float: ...


class WeightedRandomStrategy:
    """Default mutation strategy: select operators by configurable weights."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        mutation_rate: float = 0.3,
        designer_ratio: float = 0.3,
    ) -> None:
        self.weights = weights or {
            MutationType.NODE_INSERT.value: 0.15,
            MutationType.NODE_REMOVE.value: 0.10,
            MutationType.EDGE_REDIRECT.value: 0.15,
            MutationType.PARALLELIZE.value: 0.10,
            MutationType.SERIALIZE.value: 0.05,
            MutationType.PARAM_MUTATE.value: 0.10,
            MutationType.PROMPT_MUTATE.value: 0.10,
            MutationType.KNOB_MUTATE.value: 0.25,
        }
        self._mutation_rate = mutation_rate
        self._designer_ratio = designer_ratio

    def select_operator(
        self, parent: Workflow, generation: int, archive_stats: dict[str, object]
    ) -> MutationType:
        types = list(MutationType)
        w = [self.weights.get(t.value, 0.1) for t in types]
        return random.choices(types, weights=w, k=1)[0]

    def select_guided_operator(
        self,
        parent: Workflow,
        generation: int,
        reflection: ReflectionReport,
    ) -> MutationType:
        """Select an operator guided by reflection suggestions.

        Prefers typed_suggestions (direct enum matching) over string-based
        substring matching. Falls back to string path when typed_suggestions
        is empty (backward compat with persisted reflections from prior runs).
        """
        op_counts: dict[MutationType, int] = {}

        if reflection.typed_suggestions:
            _VALID_OPS = {t.value for t in MutationType}
            for ts in reflection.typed_suggestions:
                if ts.operator in _VALID_OPS:
                    mt = MutationType(ts.operator)
                    op_counts[mt] = op_counts.get(mt, 0) + 1
        else:
            for suggestion in reflection.mutation_suggestions + reflection.structural_recommendations:
                upper = suggestion.upper()
                if "NODE_INSERT" in upper:
                    op_counts[MutationType.NODE_INSERT] = op_counts.get(MutationType.NODE_INSERT, 0) + 1
                elif "NODE_REMOVE" in upper:
                    op_counts[MutationType.NODE_REMOVE] = op_counts.get(MutationType.NODE_REMOVE, 0) + 1
                elif "PARALLELIZE" in upper:
                    op_counts[MutationType.PARALLELIZE] = op_counts.get(MutationType.PARALLELIZE, 0) + 1
                elif "PARAM_MUTATE" in upper:
                    op_counts[MutationType.PARAM_MUTATE] = op_counts.get(MutationType.PARAM_MUTATE, 0) + 1
                elif "PROMPT_MUTATE" in upper:
                    op_counts[MutationType.PROMPT_MUTATE] = op_counts.get(MutationType.PROMPT_MUTATE, 0) + 1
                elif "KNOB" in upper:
                    op_counts[MutationType.KNOB_MUTATE] = op_counts.get(MutationType.KNOB_MUTATE, 0) + 1

        if not op_counts:
            return self.select_operator(parent, generation, {})

        types = list(op_counts.keys())
        weights = [float(op_counts[t]) for t in types]
        return random.choices(types, weights=weights, k=1)[0]

    def get_mutation_rate(self, generation: int) -> float:
        return self._mutation_rate

    def get_designer_ratio(self, generation: int) -> float:
        return self._designer_ratio

    def get_operator_weights(self) -> dict[str, float]:
        return dict(self.weights)

    def on_plateau(self) -> None:
        """Increase mutation rate when evolution stalls."""
        self._mutation_rate = min(self._mutation_rate + 0.2, 0.8)

    def on_improvement(self) -> None:
        """Reset mutation rate after improvement."""
        self._mutation_rate = 0.3


def validate_and_repair(workflow: Workflow) -> Workflow | None:
    """Validate a mutated workflow and attempt repair. Returns None if irreparable."""
    g: nx.DiGraph[str] = nx.DiGraph()
    for nid in workflow.nodes:
        g.add_node(nid)
    for edge in workflow.edges:
        if edge.source in workflow.nodes and edge.target in workflow.nodes:
            g.add_edge(edge.source, edge.target)
    # ForkNode.targets and JoinNode.sources declare implicit edges
    # that nx.descendants must follow for correct reachability.
    for nid, node in workflow.nodes.items():
        if isinstance(node, ForkNode):
            for target in node.targets:
                if target in workflow.nodes:
                    g.add_edge(nid, target)
        elif isinstance(node, JoinNode):
            for source in node.sources:
                if source in workflow.nodes:
                    g.add_edge(source, nid)

    if workflow.start_node not in workflow.nodes:
        return None

    # Prune unreachable nodes
    reachable = nx.descendants(g, workflow.start_node) | {workflow.start_node}
    unreachable = set(workflow.nodes.keys()) - reachable
    for nid in unreachable:
        del workflow.nodes[nid]
    workflow.edges = [
        e for e in workflow.edges
        if e.source in workflow.nodes and e.target in workflow.nodes
    ]

    # Rebuild graph and check for cycles without gate conditions
    g2: nx.DiGraph[str] = nx.DiGraph()
    for nid in workflow.nodes:
        g2.add_node(nid)
    for edge in workflow.edges:
        g2.add_edge(edge.source, edge.target)

    for cycle in nx.simple_cycles(g2):
        has_gated_edge = False
        for i in range(len(cycle)):
            src = cycle[i]
            tgt = cycle[(i + 1) % len(cycle)]
            if type(workflow.nodes.get(src)).__name__ == "GateNode":
                for e in workflow.edges:
                    if e.source == src and e.target == tgt and e.condition is not None:
                        has_gated_edge = True
                        break
            if has_gated_edge:
                break
        if not has_gated_edge:
            return None

    # Verify reads/writes chain
    # For nodes inside a DataNode subgraph, .factory/current_item.json is
    # a system-provided file (written by the executor before subgraph runs).
    _data_subgraph_nodes: set[str] = set()
    for _nid, _node in workflow.nodes.items():
        if isinstance(_node, _DataNodeType):
            if (_node.subgraph_entry in workflow.nodes
                    and _node.subgraph_exit in workflow.nodes):
                from factory.workflow.executor import _collect_subgraph_nodes as _csn
                _data_subgraph_nodes |= _csn(workflow, _node.subgraph_entry, _node.subgraph_exit)

    for nid, node in workflow.nodes.items():
        if node.reads:
            ancestors = nx.ancestors(g2, nid) if nid in g2 else set()
            available_writes: set[str] = set()
            for anc in ancestors:
                anc_node = workflow.nodes.get(anc)
                if anc_node:
                    available_writes |= anc_node.writes
            # Executor writes .factory/current_item.json for DataNode subgraphs
            if nid in _data_subgraph_nodes:
                available_writes.add(".factory/current_item.json")
            broken_reads = node.reads - available_writes
            if broken_reads:
                node_copy = node.model_copy(update={"reads": node.reads - broken_reads})
                workflow.nodes[nid] = node_copy  # type: ignore[assignment]

    return workflow


def _is_frozen(node_id: str, frozen_nodes: set[str]) -> bool:
    return node_id in frozen_nodes


def _deep_copy_workflow(workflow: Workflow) -> Workflow:
    """Deep copy a workflow for mutation."""
    nodes: dict[str, NodeType] = {}
    for nid, node in workflow.nodes.items():
        nodes[nid] = node.model_copy(deep=True)
    edges = [e.model_copy(deep=True) for e in workflow.edges]
    return Workflow(
        name=workflow.name,
        nodes=nodes,
        edges=edges,
        start_node=workflow.start_node,
        terminal=workflow.terminal,
        task=workflow.task,
        knob_values=dict(workflow.knob_values),
        knob_bounds={k: list(v) for k, v in workflow.knob_bounds.items()},
        knob_expandable=dict(workflow.knob_expandable),
        knob_specs={k: dict(v) for k, v in workflow.knob_specs.items()},
        declared_capabilities=frozenset(workflow.declared_capabilities),
    )


def insert_node(
    workflow: Workflow,
    new_node: NodeType,
    after_node_id: str,
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Insert a new node after an existing node, reconnecting edges."""
    frozen = frozen_nodes or set()
    if _is_frozen(after_node_id, frozen):
        return None

    wf = _deep_copy_workflow(workflow)
    if after_node_id not in wf.nodes:
        return None

    wf.nodes[new_node.id] = new_node

    outgoing = [e for e in wf.edges if e.source == after_node_id]
    if not outgoing:
        wf.edges.append(Edge(source=after_node_id, target=new_node.id))
    else:
        first_edge = outgoing[0]
        old_target = first_edge.target
        wf.edges = [e for e in wf.edges if not (e.source == after_node_id and e.target == old_target and e.condition is None)]
        wf.edges.append(Edge(source=after_node_id, target=new_node.id))
        wf.edges.append(Edge(source=new_node.id, target=old_target))

    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.NODE_INSERT,
        target_node=new_node.id,
        before={},
        after={"inserted_after": after_node_id},
        rationale=f"Inserted {new_node.id} after {after_node_id}",
    )
    return result, record


def remove_node(
    workflow: Workflow,
    node_id: str,
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Remove a node and short-circuit its edges."""
    frozen = frozen_nodes or set()
    if _is_frozen(node_id, frozen):
        return None

    wf = _deep_copy_workflow(workflow)
    if node_id not in wf.nodes or node_id == wf.start_node:
        return None

    incoming_sources = [e.source for e in wf.edges if e.target == node_id]
    outgoing_targets = [e.target for e in wf.edges if e.source == node_id]

    wf.edges = [e for e in wf.edges if e.source != node_id and e.target != node_id]

    for src in incoming_sources:
        for tgt in outgoing_targets:
            if not any(e.source == src and e.target == tgt for e in wf.edges):
                wf.edges.append(Edge(source=src, target=tgt))

    del wf.nodes[node_id]

    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.NODE_REMOVE,
        target_node=node_id,
        before={"node_existed": True},
        after={"short_circuited": True},
        rationale=f"Removed {node_id}, short-circuited edges",
    )
    return result, record


def redirect_edge(
    workflow: Workflow,
    source_id: str,
    old_target_id: str,
    new_target_id: str,
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Redirect an edge from old_target to new_target."""
    frozen = frozen_nodes or set()
    if _is_frozen(source_id, frozen):
        return None

    wf = _deep_copy_workflow(workflow)
    if new_target_id not in wf.nodes:
        return None

    found = False
    new_edges: list[Edge] = []
    for e in wf.edges:
        if e.source == source_id and e.target == old_target_id and not found:
            new_edges.append(Edge(source=source_id, target=new_target_id, condition=e.condition))
            found = True
        else:
            new_edges.append(e)

    if not found:
        return None

    wf.edges = new_edges
    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.EDGE_REDIRECT,
        target_node=source_id,
        before={"target": old_target_id},
        after={"target": new_target_id},
        rationale=f"Redirected edge from {source_id}: {old_target_id} → {new_target_id}",
    )
    return result, record


def parallelize(
    workflow: Workflow,
    node_ids: list[str],
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Convert sequential nodes to parallel execution via ForkNode + JoinNode."""
    frozen = frozen_nodes or set()
    if any(_is_frozen(nid, frozen) for nid in node_ids):
        return None
    if len(node_ids) < 2:
        return None

    wf = _deep_copy_workflow(workflow)
    for nid in node_ids:
        if nid not in wf.nodes:
            return None

    fork_id = f"fork_{'_'.join(node_ids[:2])}"
    join_id = f"join_{'_'.join(node_ids[:2])}"

    first_node = node_ids[0]
    last_node = node_ids[-1]

    predecessors = {e.source for e in wf.edges if e.target == first_node}
    successors = {e.target for e in wf.edges if e.source == last_node}

    for nid in node_ids:
        wf.edges = [e for e in wf.edges if e.source != nid and e.target != nid]

    wf.nodes[fork_id] = ForkNode(id=fork_id, targets=node_ids)
    wf.nodes[join_id] = JoinNode(id=join_id, sources=node_ids)

    for pred in predecessors:
        wf.edges.append(Edge(source=pred, target=fork_id))

    for nid in node_ids:
        wf.edges.append(Edge(source=fork_id, target=nid))
        wf.edges.append(Edge(source=nid, target=join_id))

    for succ in successors:
        wf.edges.append(Edge(source=join_id, target=succ))

    if wf.start_node == first_node:
        wf.start_node = fork_id

    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.PARALLELIZE,
        target_node=fork_id,
        before={"sequential": node_ids},
        after={"parallel": node_ids},
        rationale=f"Parallelized {node_ids}",
    )
    return result, record


def serialize(
    workflow: Workflow,
    fork_id: str,
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Collapse a fork/join pair back into sequential execution."""
    frozen = frozen_nodes or set()
    if _is_frozen(fork_id, frozen):
        return None

    wf = _deep_copy_workflow(workflow)
    fork_node = wf.nodes.get(fork_id)
    if fork_node is None or type(fork_node).__name__ != "ForkNode":
        return None

    targets = fork_node.targets  # type: ignore[union-attr]

    join_id: str | None = None
    for nid, node in wf.nodes.items():
        if type(node).__name__ == "JoinNode":
            sources = node.sources  # type: ignore[union-attr]
            if set(sources) == set(targets):
                join_id = nid
                break

    if join_id is None:
        return None

    predecessors = {e.source for e in wf.edges if e.target == fork_id}
    successors = {e.target for e in wf.edges if e.source == join_id}

    wf.edges = [
        e for e in wf.edges
        if e.source != fork_id and e.target != fork_id
        and e.source != join_id and e.target != join_id
        and not (e.source in targets and e.target == join_id)
    ]

    del wf.nodes[fork_id]
    del wf.nodes[join_id]

    chain = list(targets)
    for pred in predecessors:
        wf.edges.append(Edge(source=pred, target=chain[0]))

    for i in range(len(chain) - 1):
        wf.edges.append(Edge(source=chain[i], target=chain[i + 1]))

    for succ in successors:
        wf.edges.append(Edge(source=chain[-1], target=succ))

    if wf.start_node == fork_id:
        wf.start_node = chain[0]

    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.SERIALIZE,
        target_node=fork_id,
        before={"parallel": list(targets)},
        after={"sequential": chain},
        rationale=f"Serialized fork {fork_id}",
    )
    return result, record


def mutate_params(
    workflow: Workflow,
    node_id: str,
    changes: dict[str, object],
    *,
    frozen_nodes: set[str] | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Change parameters on a node (timeout, model, max_iterations)."""
    frozen = frozen_nodes or set()
    if _is_frozen(node_id, frozen):
        return None

    wf = _deep_copy_workflow(workflow)
    node = wf.nodes.get(node_id)
    if node is None:
        return None

    allowed_params = {"timeout", "max_iterations", "blocking"}
    filtered_changes = {k: v for k, v in changes.items() if k in allowed_params}
    if not filtered_changes:
        return None

    before: dict[str, object] = {}
    for k in filtered_changes:
        if hasattr(node, k):
            before[k] = getattr(node, k)

    try:
        updated_node = type(node)(**{**node.model_dump(), **filtered_changes})
        wf.nodes[node_id] = updated_node  # type: ignore[assignment]
    except Exception:
        return None

    result = validate_and_repair(wf)
    if result is None:
        return None

    record = MutationRecord(
        operator=MutationType.PARAM_MUTATE,
        target_node=node_id,
        before=before,
        after=dict(filtered_changes),
        rationale=f"Changed params on {node_id}: {filtered_changes}",
    )
    return result, record


_PROMPT_VARIANTS = [
    "Think step by step. Analyze the problem carefully before proposing changes.",
    "Focus on identifying the root cause of failures. Read error messages and trace back precisely.",
    "Prioritize minimal changes. Change only what is necessary to solve the problem.",
    "Start by reading all relevant context. Map dependencies before making changes.",
    "Verify each change before moving on. Confirm correctness at every step.",
    "Follow established patterns and conventions consistently.",
    "Check edge cases explicitly. Validate inputs and handle error paths.",
    "Consider performance implications. Avoid O(n^2) patterns when O(n) alternatives exist.",
]

MAX_NODES = 30

_COMPLEMENTARY_ROLES: dict[AgentRole, AgentRole] = {
    AgentRole.BUILDER: AgentRole.CODE_REVIEWER,
    AgentRole.RESEARCHER: AgentRole.STRATEGIST,
    AgentRole.STRATEGIST: AgentRole.BUILDER,
    AgentRole.HEALTH_CHECKER: AgentRole.BUILDER,
    AgentRole.CODE_REVIEWER: AgentRole.BUILDER,
    AgentRole.ADVERSARIAL_TESTER: AgentRole.BUILDER,
    AgentRole.FAILURE_ANALYST: AgentRole.STRATEGIST,
    AgentRole.ARCHIVIST: AgentRole.RESEARCHER,
    AgentRole.REFINER: AgentRole.BUILDER,
    AgentRole.CEO: AgentRole.STRATEGIST,
}

_ROLE_PROMPT_TEMPLATES: dict[AgentRole, str] = {
    AgentRole.BUILDER: "Implement the changes described in the input files.",
    AgentRole.CODE_REVIEWER: "Review the work done so far for correctness and quality issues.",
    AgentRole.RESEARCHER: "Investigate the problem space and gather relevant context.",
    AgentRole.STRATEGIST: "Analyze the available information and propose a clear plan of action.",
    AgentRole.HEALTH_CHECKER: "Run health checks and verify the system is functioning correctly.",
    AgentRole.ADVERSARIAL_TESTER: "Test the implementation against edge cases and failure modes.",
    AgentRole.FAILURE_ANALYST: "Analyze failures and identify root causes.",
    AgentRole.ARCHIVIST: "Record findings and decisions for future reference.",
    AgentRole.REFINER: "Refine the implementation based on feedback.",
    AgentRole.CEO: "Orchestrate the workflow and review agent outputs.",
}


PromptRewriter = Callable[[str, str, str | None], str | None]


def _cli_output_ok(returncode: int, stdout: str) -> bool:
    """Guard against accepting CLI error text as LLM output.

    `claude -p` prints auth/API errors to stdout (sometimes with exit 0),
    e.g. "Failed to authenticate. API Error: 403 ...". Without this check
    the error message can be captured as a rewritten prompt or knob value.
    """
    if returncode != 0:
        return False
    head = stdout.lstrip()[:200]
    return not head.startswith(("API Error", "Failed to authenticate"))


def default_prompt_rewriter(
    node_id: str,
    current_prompt: str,
    hint: str | None,
) -> str | None:
    """Default prompt rewriter: uses claude CLI to rewrite an agent prompt."""
    import subprocess

    context = f"Hint from reflection: {hint}" if hint else "No specific hint."
    prompt = (
        f"You are improving an AI agent's prompt. The agent's role is '{node_id}'.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"{context}\n\n"
        f"Write an improved version of this prompt. Keep the same role and format. "
        f"Make it more specific, fix any issues the hint identifies, and remove "
        f"any contradictory or redundant instructions. "
        f"Output ONLY the new prompt text, nothing else."
    )
    from factory.runners.claude import _claude_bin, _claude_model

    try:
        proc = subprocess.run(
            [_claude_bin(), "-p", prompt, "--model", _claude_model(),
             "--append-system-prompt", "Output only the prompt text.",
             "--max-turns", "1", "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
        )
        result = proc.stdout.strip()
        if not _cli_output_ok(proc.returncode, result):
            log.warning(
                "prompt_rewriter_cli_error", node=node_id,
                returncode=proc.returncode, head=result[:120],
            )
            return None
        if result:
            log.info("prompt_rewritten", node=node_id, len=len(result))
        else:
            log.warning("prompt_rewriter_empty", node=node_id)
        return result if result else None
    except subprocess.TimeoutExpired:
        log.warning("prompt_rewriter_timeout", node=node_id, timeout=120)
        return None
    except Exception as exc:
        log.warning("prompt_rewriter_error", node=node_id, error=str(exc))
        return None


async def async_prompt_rewriter(
    node_id: str,
    current_prompt: str,
    hint: str | None,
) -> str | None:
    """Async prompt rewriter: uses claude CLI without blocking the event loop."""
    import asyncio as _asyncio

    context = f"Hint from reflection: {hint}" if hint else "No specific hint."
    prompt = (
        f"You are improving an AI agent's prompt. The agent's role is '{node_id}'.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"{context}\n\n"
        f"Write an improved version of this prompt. Keep the same role and format. "
        f"Make it more specific, fix any issues the hint identifies, and remove "
        f"any contradictory or redundant instructions. "
        f"Output ONLY the new prompt text, nothing else."
    )
    from factory.runners.claude import _claude_bin, _claude_model

    try:
        proc = await _asyncio.create_subprocess_exec(
            _claude_bin(), "-p", prompt, "--model", _claude_model(),
            "--append-system-prompt", "Output only the prompt text.",
            "--max-turns", "1", "--output-format", "text",
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        stdout, _ = await _asyncio.wait_for(proc.communicate(), timeout=120.0)
        result = stdout.decode().strip() if stdout else ""
        returncode = proc.returncode if proc.returncode is not None else -1
        if not _cli_output_ok(returncode, result):
            log.warning(
                "prompt_rewriter_cli_error", node=node_id,
                returncode=returncode, head=result[:120],
            )
            return None
        if result:
            log.info("prompt_rewritten", node=node_id, len=len(result))
        else:
            log.warning("prompt_rewriter_empty", node=node_id)
        return result if result else None
    except _asyncio.TimeoutError:
        log.warning("prompt_rewriter_timeout", node=node_id, timeout=120)
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return None
    except Exception as exc:
        log.warning("prompt_rewriter_error", node=node_id, error=str(exc))
        return None


def mutate_prompt(
    workflow: Workflow,
    node_id: str,
    *,
    frozen_nodes: set[str] | None = None,
    prompt_hint: str | None = None,
    rewriter: PromptRewriter | None = default_prompt_rewriter,
) -> tuple[Workflow, MutationRecord] | None:
    """Mutate the prompt_template of an AgentNode.

    When a rewriter is provided, it REPLACES the prompt (informed by the
    current prompt + hint). When no rewriter is available, falls back to
    appending a hint or random variant.
    """
    frozen = frozen_nodes or set()
    if node_id in frozen:
        return None

    wf = _deep_copy_workflow(workflow)
    node = wf.nodes.get(node_id)
    if node is None or not isinstance(node, AgentNode):
        return None

    old_prompt = node.prompt_template or ""
    new_prompt: str | None = None

    if rewriter is not None:
        new_prompt = rewriter(node_id, old_prompt, prompt_hint)

    if not new_prompt:
        if prompt_hint:
            new_prompt = f"{old_prompt}\n\n{prompt_hint}" if old_prompt else prompt_hint
        else:
            variant = random.choice(_PROMPT_VARIANTS)
            new_prompt = f"{old_prompt}\n\n{variant}" if old_prompt else variant

    # If the node is inside a DataNode subgraph, ensure current_item.json
    # reference is preserved after prompt rewriting
    if (new_prompt
            and "current_item.json" not in new_prompt
            and "current_item.json" in old_prompt
            and _is_in_data_subgraph(wf, node_id)):
        new_prompt = (
            "Read .factory/current_item.json for the current item. "
            + new_prompt
        )

    try:
        updated = type(node)(**{**node.model_dump(), "prompt_template": new_prompt})
        wf.nodes[node_id] = updated  # type: ignore[assignment]
    except Exception as exc:
        log.warning("prompt_mutate_validation_failed", node=node_id, error=str(exc))
        return None

    # Persist in knob_values so the mutation survives Package.compile() round-trips
    prompt_knob = f"_prompt_{node_id}"
    wf.knob_values[prompt_knob] = new_prompt
    wf.knob_expandable[prompt_knob] = f"Prompt for {node_id}"

    record = MutationRecord(
        operator=MutationType.PROMPT_MUTATE,
        target_node=node_id,
        before={"prompt": old_prompt[:100]},
        after={"prompt": new_prompt[:100]},
        rationale=f"Mutated prompt on {node_id}",
    )
    return wf, record


KnobExpander = Callable[[str, str, str | float, list[str | float]], str | float | None]


def default_knob_expander(
    knob_name: str,
    hint: str,
    current: str | float,
    bounds: list[str | float],
) -> str | float | None:
    """Default expander: uses claude CLI to invent new knob values."""
    import subprocess

    prompt = (
        f"Invent a new value for the parameter '{knob_name}'.\n"
        f"Context: {hint}\n"
        f"Current value: {current}\n"
        f"Existing options: {bounds}\n\n"
        f"Write ONLY the new value (a short name if it's a prompt knob, "
        f"or a number if it's a threshold). Nothing else."
    )
    from factory.runners.claude import _claude_bin, _claude_model

    try:
        proc = subprocess.run(
            [_claude_bin(), "-p", prompt, "--model", _claude_model(),
             "--append-system-prompt", "Output only the value, no explanation.",
             "--max-turns", "1", "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
        )
        result = proc.stdout.strip()
        if not _cli_output_ok(proc.returncode, result):
            log.warning(
                "knob_expander_cli_error", knob=knob_name,
                returncode=proc.returncode, head=result[:120],
            )
            return None
        if not result:
            log.warning("knob_expander_empty", knob=knob_name)
            return None
        log.info("knob_expanded_via_cli", knob=knob_name, value=result[:40])
        try:
            return float(result)
        except ValueError:
            return result[:80]
    except subprocess.TimeoutExpired:
        log.warning("knob_expander_timeout", knob=knob_name, timeout=120)
        return None
    except Exception as exc:
        log.warning("knob_expander_error", knob=knob_name, error=str(exc))
        return None


def _parse_knob_suggestion(
    suggestion: str | MutationSuggestion,
) -> tuple[str, str] | None:
    """Extract (knob_name, best_value) from a KNOB_MUTATE suggestion.

    Accepts either a typed MutationSuggestion or a legacy string.
    """
    if isinstance(suggestion, MutationSuggestion):
        if suggestion.operator == "knob_mutate" and suggestion.value is not None and suggestion.value.strip():
            return (suggestion.target, suggestion.value)
        return None
    if not suggestion.startswith("KNOB_MUTATE:"):
        return None
    rest = suggestion[len("KNOB_MUTATE:"):].strip()
    if "=" not in rest:
        return None
    name, _, after = rest.partition("=")
    value = after.split()[0].rstrip("()") if after else ""
    return (name.strip(), value) if name and value else None


def mutate_knob(
    workflow: Workflow,
    *,
    expander: KnobExpander | None = default_knob_expander,
    reflection_report: ReflectionReport | None = None,
) -> tuple[Workflow, MutationRecord] | None:
    """Mutate a single knob value within its declared bounds.

    When a reflection_report is provided with KNOB_MUTATE suggestions,
    70% of the time picks the suggested knob and value (exploitation).
    30% of the time picks randomly (exploration).

    When all bounds are exhausted and the knob is expandable, calls
    ``expander(knob_name, expansion_hint, current_value, bounds)`` to
    generate a new value.
    """
    if not workflow.knob_values:
        return None

    wf = _deep_copy_workflow(workflow)
    knob_names = list(wf.knob_values.keys())

    # Try guided mutation from reflection suggestions (70% of the time)
    guided_knob: str | None = None
    guided_val: str | float | None = None
    if reflection_report and random.random() < 0.7:
        # Prefer typed suggestions over string parsing
        typed_knobs = [
            ts for ts in reflection_report.typed_suggestions
            if ts.operator == "knob_mutate" and ts.value is not None
        ]
        if typed_knobs:
            suggestions = [_parse_knob_suggestion(ts) for ts in typed_knobs]
        else:
            suggestions = [
                _parse_knob_suggestion(s) for s in reflection_report.mutation_suggestions
            ]
        valid = [(k, v) for parsed in suggestions if parsed
                 for k, v in [parsed] if k in wf.knob_values]
        if valid:
            guided_knob, guided_val = random.choice(valid)

    if guided_knob and guided_val is not None:
        knob_name = guided_knob
        old_val = wf.knob_values[knob_name]
        new_val: str | float | None = guided_val
        # Coerce type to match existing value
        if isinstance(old_val, float) and isinstance(new_val, str):
            try:
                new_val = float(new_val)
            except ValueError:
                pass
        # Skip no-op: guided value same as current
        if str(new_val) == str(old_val):
            new_val = None
            guided_knob = None
    if not guided_knob:
        # Exclude synthetic _prompt_* knobs (handled by PROMPT_MUTATE)
        real_knobs = [k for k in knob_names if not k.startswith("_prompt_")]
        if not real_knobs:
            return None
        knob_name = random.choice(real_knobs)
        old_val = wf.knob_values[knob_name]
        bounds = wf.knob_bounds.get(knob_name, [])
        new_val = None

        if bounds:
            alternatives = [v for v in bounds if v != old_val]
            if alternatives:
                new_val = random.choice(alternatives)
            elif knob_name in wf.knob_expandable and expander is not None:
                hint = wf.knob_expandable[knob_name]
                new_val = expander(knob_name, hint, old_val, bounds)
                if new_val is not None:
                    wf.knob_bounds.setdefault(knob_name, []).append(new_val)
                    log.info("knob_expanded", knob=knob_name, new_value=new_val)

    if new_val is None:
        if isinstance(old_val, bool):
            new_val = not old_val
        elif isinstance(old_val, (int, float)):
            delta = random.choice([-1, 1]) * max(1, abs(old_val) * 0.2)
            new_val = type(old_val)(old_val + delta)
        else:
            return None

    wf.knob_values[knob_name] = new_val

    record = MutationRecord(
        operator=MutationType.KNOB_MUTATE,
        target_node=knob_name,
        before={"value": str(old_val)},
        after={"value": str(new_val)},
        rationale=f"Mutated knob {knob_name}: {old_val} -> {new_val}",
    )
    return wf, record


def apply_random_mutation(
    workflow: Workflow,
    strategy: MutationStrategy,
    generation: int,
    *,
    frozen_nodes: set[str] | None = None,
    archive_stats: dict[str, object] | None = None,
    reflection_report: ReflectionReport | None = None,
    knob_expander: KnobExpander | None = default_knob_expander,
    max_attempts: int = 10,
) -> tuple[Workflow, MutationRecord] | None:
    """Apply a mutation using the given strategy. Retries on failure.

    When reflection_report is provided, guided mutations are attempted first
    (70% of the time), falling back to random mutations.

    When knob_expander is provided, KNOB_MUTATE can generate new values
    beyond the declared bounds for expandable knobs.
    """
    frozen = frozen_nodes or set()
    stats = archive_stats or {}
    use_guided = (
        reflection_report is not None
        and hasattr(strategy, "select_guided_operator")
        and (reflection_report.typed_suggestions
             or reflection_report.mutation_suggestions
             or reflection_report.structural_recommendations)
    )

    for attempt in range(max_attempts):
        if use_guided and random.random() < 0.7:
            op = strategy.select_guided_operator(  # type: ignore[attr-defined]
                workflow, generation, reflection_report,
            )
        else:
            op = strategy.select_operator(workflow, generation, stats)

        if op == MutationType.NODE_INSERT and len(workflow.nodes) >= MAX_NODES:
            op = MutationType.PARAM_MUTATE

        prompt_hint = _extract_prompt_hint(reflection_report) if reflection_report else None
        result = _try_mutation(workflow, op, frozen, prompt_hint=prompt_hint,
                               expander=knob_expander,
                               reflection_report=reflection_report)
        if result is not None:
            wf, rec = result
            if len(wf.nodes) > MAX_NODES:
                continue
            return result

    return None


def _extract_prompt_hint(report: ReflectionReport) -> str | None:
    """Extract a prompt improvement hint from a ReflectionReport."""
    if report.prompt_improvements:
        return random.choice(report.prompt_improvements)
    if report.success_patterns:
        return random.choice(report.success_patterns)
    return None


def _find_nearest_upstream_agent(
    workflow: Workflow, node_id: str,
) -> AgentNode | None:
    """Walk upstream edges to find the nearest AgentNode."""
    visited: set[str] = set()
    queue = [node_id]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        node = workflow.nodes.get(current)
        if node is not None and isinstance(node, AgentNode) and current != node_id:
            return node
        for edge in workflow.edges:
            if edge.target == current and edge.source not in visited:
                queue.append(edge.source)
    return None


def _generate_unique_agent_id(existing_keys: set[str], role: AgentRole) -> str:
    """Generate an agent ID that doesn't collide with existing node keys."""
    for _ in range(100):
        new_id = f"{role.value}_{random.randint(100, 999)}"
        if new_id not in existing_keys:
            return new_id
    return f"{role.value}_{random.randint(10000, 99999)}"


def _try_mutation(
    workflow: Workflow,
    op: MutationType,
    frozen: set[str],
    *,
    prompt_hint: str | None = None,
    **kwargs: object,
) -> tuple[Workflow, MutationRecord] | None:
    """Attempt a single mutation of the given type."""
    # Fix 1: two separate node lists for structural vs content mutations
    structurally_mutable = [
        nid for nid in workflow.nodes if nid not in frozen and nid != workflow.start_node
    ]
    content_mutable = [
        nid for nid in workflow.nodes if nid not in frozen
    ]

    structural_ops = {
        MutationType.NODE_REMOVE, MutationType.EDGE_REDIRECT,
        MutationType.PARALLELIZE, MutationType.SERIALIZE,
    }
    content_ops = {MutationType.PARAM_MUTATE, MutationType.PROMPT_MUTATE}
    exempt_ops = {MutationType.NODE_INSERT, MutationType.KNOB_MUTATE}

    if op in structural_ops and not structurally_mutable:
        return None
    if op in content_ops and not content_mutable:
        return None
    if op not in structural_ops and op not in content_ops and op not in exempt_ops:
        return None

    if op == MutationType.NODE_INSERT:
        # Fix 2: contextual complementary node with inherited file wiring
        target = random.choice(list(workflow.nodes.keys()))
        target_node = workflow.nodes[target]

        all_agent_nodes = [
            n for n in workflow.nodes.values() if isinstance(n, AgentNode)
        ]

        if all_agent_nodes:
            # Find the nearest upstream AgentNode from the insertion point
            upstream_agent: AgentNode | None = None
            if isinstance(target_node, AgentNode):
                upstream_agent = target_node
            else:
                upstream_agent = _find_nearest_upstream_agent(workflow, target)

            if upstream_agent is not None:
                base_role = upstream_agent.role
            else:
                base_role = random.choice(all_agent_nodes).role

            complementary_role = _COMPLEMENTARY_ROLES.get(base_role, random.choice(list(AgentRole)))

            # Inherit file wiring from context
            new_reads = upstream_agent.writes.copy() if (
                upstream_agent and upstream_agent.writes
            ) else (upstream_agent.reads.copy() if upstream_agent else set())
            # Find the downstream node to set writes
            new_writes: set[str] = set()
            for edge in workflow.edges:
                if edge.source == target and edge.condition is None:
                    downstream = workflow.nodes.get(edge.target)
                    if downstream and downstream.reads:
                        new_writes = downstream.reads.copy()
                        break

            # Fix 5: unique ID generation
            new_id = _generate_unique_agent_id(set(workflow.nodes.keys()), complementary_role)
            prompt_tmpl = _ROLE_PROMPT_TEMPLATES.get(complementary_role, "")

            new_node = AgentNode(
                id=new_id,
                role=complementary_role,
                reads=new_reads,
                writes=new_writes,
                prompt_template=prompt_tmpl,
            )

            # If inserting into a DataNode subgraph, make the new node data-aware
            if _is_in_data_subgraph(workflow, target):
                data_reads = set(new_node.reads or set()) | {".factory/current_item.json"}
                data_prompt = (
                    "As a " + new_node.role.value.replace("_", " ") + ", "
                    + "read .factory/current_item.json for the current item. "
                    + (new_node.prompt_template or "")
                )
                new_node = AgentNode(
                    id=new_node.id,
                    role=new_node.role,
                    reads=data_reads,
                    writes=new_node.writes,
                    prompt_template=data_prompt,
                )
        else:
            # Fallback: no AgentNodes exist at all
            role = random.choice(list(AgentRole))
            new_id = _generate_unique_agent_id(set(workflow.nodes.keys()), role)
            new_node = AgentNode(id=new_id, role=role)

        return insert_node(workflow, new_node, target, frozen_nodes=frozen)

    elif op == MutationType.NODE_REMOVE:
        target = random.choice(structurally_mutable)
        # Fix 3: protect the last AgentNode
        if isinstance(workflow.nodes[target], AgentNode):
            agent_count = sum(
                1 for n in workflow.nodes.values() if isinstance(n, AgentNode)
            )
            if agent_count <= 1:
                return None
        return remove_node(workflow, target, frozen_nodes=frozen)

    elif op == MutationType.EDGE_REDIRECT:
        edges_from_mutable = [
            e for e in workflow.edges if e.source not in frozen
        ]
        if not edges_from_mutable:
            return None
        edge = random.choice(edges_from_mutable)

        # Fix 4: pre-filter targets to exclude ungated cycle sources
        unconditional_graph: nx.DiGraph[str] = nx.DiGraph()
        for nid in workflow.nodes:
            unconditional_graph.add_node(nid)
        for e in workflow.edges:
            if e.condition is None:
                unconditional_graph.add_edge(e.source, e.target)

        ancestors_of_source = (
            nx.ancestors(unconditional_graph, edge.source)
            if edge.source in unconditional_graph
            else set()
        )
        possible_targets = [
            nid for nid in workflow.nodes
            if nid != edge.target
            and nid != edge.source
            and nid not in ancestors_of_source
        ]
        if not possible_targets:
            return None
        new_target = random.choice(possible_targets)
        return redirect_edge(workflow, edge.source, edge.target, new_target, frozen_nodes=frozen)

    elif op == MutationType.PARALLELIZE:
        if len(structurally_mutable) < 2:
            return None
        pair = random.sample(structurally_mutable, 2)
        return parallelize(workflow, pair, frozen_nodes=frozen)

    elif op == MutationType.SERIALIZE:
        fork_ids = [
            nid for nid, n in workflow.nodes.items()
            if type(n).__name__ == "ForkNode" and nid not in frozen
        ]
        if not fork_ids:
            return None
        return serialize(workflow, random.choice(fork_ids), frozen_nodes=frozen)

    elif op == MutationType.PARAM_MUTATE:
        agent_nodes = [
            nid for nid in content_mutable
            if isinstance(workflow.nodes[nid], AgentNode)
        ]
        if not agent_nodes:
            return None
        target = random.choice(agent_nodes)
        changes: dict[str, object] = {"timeout": random.choice([300, 600, 900, 1200, 1800])}
        return mutate_params(workflow, target, changes, frozen_nodes=frozen)

    elif op == MutationType.PROMPT_MUTATE:
        agent_nodes = [
            nid for nid in content_mutable
            if isinstance(workflow.nodes[nid], AgentNode)
        ]
        if not agent_nodes:
            return None
        target = random.choice(agent_nodes)
        return mutate_prompt(workflow, target, frozen_nodes=frozen, prompt_hint=prompt_hint)

    elif op == MutationType.KNOB_MUTATE:
        exp = kwargs.get("expander")
        ref = kwargs.get("reflection_report")
        return mutate_knob(
            workflow,
            expander=exp if callable(exp) else None,
            reflection_report=ref if isinstance(ref, ReflectionReport) else None,
        )

    return None
