"""Dual-engine comparison harness.

Defines the SAME workflow in both representations and compares execution:
1. Current engine simulation: dict[str, SimNode] + list[SimEdge] → mock walker
2. pydantic-graph: BaseNode subclasses → Graph → Graph.run()

The comparison workflow covers all patterns from Phases 2-4:
  - Sequential chain: builder executes, feeds into gate
  - Gate with RELOOP: qa_gate routes PROCEED/RELOOP/HALT
  - Parallel fork/join: 3 QA agents via asyncio.gather
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from pg_factory.deps import FactoryDeps
from pg_factory.nodes.gates import GateBaseNode
from pg_factory.nodes.parallel import ChildAgent, ForkJoinNode
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult, VerdictType


# ═══════════════════════════════════════════════════════════════════
# Part 1: Current Engine Simulation
# ═══════════════════════════════════════════════════════════════════


class NodeKind(str, Enum):
    ACTION = "action"
    GATE = "gate"
    FORK = "fork"
    JOIN = "join"


@dataclass(frozen=True)
class SimEdge:
    source: str
    target: str
    condition: str | None = None


@dataclass
class SimNode:
    id: str
    kind: NodeKind
    fork_targets: list[str] = field(default_factory=list)


@dataclass
class SimWorkflow:
    name: str
    nodes: dict[str, SimNode]
    edges: list[SimEdge]
    start_node: str


@dataclass
class TraceEvent:
    node: str
    action: str
    detail: dict[str, Any] = field(default_factory=dict)


def build_current_engine_workflow() -> SimWorkflow:
    """Define the comparison workflow as dict[str, Node] + list[Edge].

    Mirrors the definitions.py pattern: explicit node dict, explicit edge list,
    ForkNode.targets for parallel dispatch, string-based wiring.
    """
    nodes = {
        "builder": SimNode(id="builder", kind=NodeKind.ACTION),
        "qa_gate": SimNode(id="qa_gate", kind=NodeKind.GATE),
        "fork_qa": SimNode(
            id="fork_qa",
            kind=NodeKind.FORK,
            fork_targets=["health_checker", "code_reviewer", "adversarial_tester"],
        ),
        "health_checker": SimNode(id="health_checker", kind=NodeKind.ACTION),
        "code_reviewer": SimNode(id="code_reviewer", kind=NodeKind.ACTION),
        "adversarial_tester": SimNode(id="adversarial_tester", kind=NodeKind.ACTION),
        "join_qa": SimNode(id="join_qa", kind=NodeKind.JOIN),
    }
    edges = [
        SimEdge(source="builder", target="qa_gate"),
        SimEdge(source="qa_gate", target="fork_qa", condition="proceed"),
        SimEdge(source="qa_gate", target="builder", condition="reloop"),
        SimEdge(source="fork_qa", target="health_checker"),
        SimEdge(source="fork_qa", target="code_reviewer"),
        SimEdge(source="fork_qa", target="adversarial_tester"),
        SimEdge(source="fork_qa", target="join_qa"),
    ]
    return SimWorkflow(
        name="comparison-workflow",
        nodes=nodes,
        edges=edges,
        start_node="builder",
    )


def simulate_current_engine(
    workflow: SimWorkflow,
    verdict_fn: Callable[[int], VerdictType],
    max_iterations: int = 3,
) -> list[TraceEvent]:
    """Walk the graph, recording what WorkflowExecutor would do.

    Simplified mock of executor.py's _execute_from:
    - ACTION nodes: record execution, follow unconditional edge
    - GATE nodes: call verdict_fn(call_count), follow conditional edge
    - FORK nodes: record each target execution, follow to join
    - JOIN nodes: record barrier, follow unconditional edge

    verdict_fn receives the gate evaluation count (0-indexed).
    """
    events: list[TraceEvent] = []
    iteration_counts: dict[tuple[str, str], int] = {}
    gate_call_count = 0

    def _find_edge(source: str, condition: str | None = None) -> str | None:
        for edge in workflow.edges:
            if edge.source != source or edge.condition != condition:
                continue
            if condition is None:
                node = workflow.nodes.get(source)
                if node and node.kind == NodeKind.FORK and edge.target in node.fork_targets:
                    continue
            return edge.target
        return None

    def _walk(node_id: str) -> None:
        nonlocal gate_call_count

        node = workflow.nodes.get(node_id)
        if node is None:
            return

        if node.kind == NodeKind.ACTION:
            events.append(TraceEvent(node=node_id, action="execute"))
            nxt = _find_edge(node_id)
            if nxt:
                _walk(nxt)

        elif node.kind == NodeKind.GATE:
            verdict = verdict_fn(gate_call_count)
            gate_call_count += 1

            if verdict == VerdictType.HALT:
                events.append(TraceEvent(
                    node=node_id, action="gate_verdict",
                    detail={"verdict": "halt"},
                ))
                return

            if verdict == VerdictType.RELOOP:
                target = _find_edge(node_id, "reloop")
                if not target:
                    return
                key = (node_id, target)
                count = iteration_counts.get(key, 0) + 1
                iteration_counts[key] = count
                if count > max_iterations:
                    events.append(TraceEvent(
                        node=node_id, action="gate_verdict",
                        detail={"verdict": "halt", "reason": "max_iterations"},
                    ))
                    return
                events.append(TraceEvent(
                    node=node_id, action="gate_verdict",
                    detail={"verdict": "reloop", "target": target, "iteration": count},
                ))
                _walk(target)
                return

            events.append(TraceEvent(
                node=node_id, action="gate_verdict",
                detail={"verdict": "proceed"},
            ))
            target = _find_edge(node_id, "proceed")
            if target:
                _walk(target)

        elif node.kind == NodeKind.FORK:
            events.append(TraceEvent(
                node=node_id, action="fork",
                detail={"targets": list(node.fork_targets)},
            ))
            for t in node.fork_targets:
                events.append(TraceEvent(node=t, action="execute"))
            events.append(TraceEvent(
                node=node_id, action="fork_join_complete",
                detail={"children": list(node.fork_targets)},
            ))
            nxt = _find_edge(node_id)
            if nxt:
                _walk(nxt)

        elif node.kind == NodeKind.JOIN:
            events.append(TraceEvent(node=node_id, action="join"))
            nxt = _find_edge(node_id)
            if nxt:
                _walk(nxt)

    _walk(workflow.start_node)
    return events


# ═══════════════════════════════════════════════════════════════════
# Part 2: pydantic-graph Implementation
# ═══════════════════════════════════════════════════════════════════


class CompareQAForkJoinNode(ForkJoinNode):
    """Fork/join running 3 QA agents concurrently, then ends."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> End[HaltResult]:
        await self.execute_children(ctx)
        return End(HaltResult(reason="qa_complete"))


class CompareBuilderNode(BaseNode[FactoryState, FactoryDeps, HaltResult]):
    """Builder that feeds into the QA gate."""

    def __init__(
        self,
        verdict_fn: Callable[[FactoryState], VerdictType] | None = None,
        max_iterations: int = 3,
    ) -> None:
        self.verdict_fn = verdict_fn
        self.max_iterations = max_iterations

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> "CompareQAGateNode":
        ctx.state.node_outputs["builder"] = "build_output"
        ctx.state.events.append({"node": "builder", "action": "execute"})
        return CompareQAGateNode(
            gate_id="qa_gate",
            verdict_fn=self.verdict_fn,
            max_iterations=self.max_iterations,
        )


class CompareQAGateNode(GateBaseNode):
    """Gate routing: builder (RELOOP) | fork/join (PROCEED) | End (HALT)."""

    async def run(
        self, ctx: GraphRunContext[FactoryState, FactoryDeps]
    ) -> CompareBuilderNode | CompareQAForkJoinNode | End[HaltResult]:
        verdict = self.evaluate_verdict(ctx.state)

        if verdict == VerdictType.HALT:
            self.record_verdict_event(ctx, verdict)
            return End(HaltResult(reason="gate_halted"))

        if verdict == VerdictType.RELOOP:
            target_id = "CompareBuilderNode"
            halt = self.check_and_increment_iteration(ctx, target_id)
            if halt is not None:
                self.record_verdict_event(ctx, VerdictType.HALT, target_id)
                return halt
            iteration = ctx.state.iteration_counts[(self.gate_id, target_id)]
            self.inject_feedback(ctx, target_id, f"iteration {iteration}")
            self.record_verdict_event(ctx, verdict, target_id)
            return CompareBuilderNode(
                verdict_fn=self.verdict_fn,
                max_iterations=self.max_iterations,
            )

        self.record_verdict_event(ctx, verdict)
        return CompareQAForkJoinNode(children=_qa_children(), node_id="fork_qa")


def _qa_children() -> list[ChildAgent]:
    async def _health_check(ctx: GraphRunContext[FactoryState, FactoryDeps]) -> str:
        return "health_check: pass"

    async def _code_review(ctx: GraphRunContext[FactoryState, FactoryDeps]) -> str:
        return "code_review: pass"

    async def _adversarial(ctx: GraphRunContext[FactoryState, FactoryDeps]) -> str:
        return "adversarial: pass"

    return [
        ChildAgent(name="health_checker", fn=_health_check),
        ChildAgent(name="code_reviewer", fn=_code_review),
        ChildAgent(name="adversarial_tester", fn=_adversarial),
    ]


def build_pydantic_graph(
    verdict_fn: Callable[[FactoryState], VerdictType] | None = None,
    max_iterations: int = 3,
) -> tuple[
    GraphBuilder[FactoryState, FactoryDeps, CompareBuilderNode, HaltResult],
    CompareBuilderNode,
]:
    """Build the comparison workflow as pydantic-graph BaseNode subclasses."""
    builder: GraphBuilder[FactoryState, FactoryDeps, CompareBuilderNode, HaltResult] = (
        GraphBuilder(
            name="comparison-workflow",
            state_type=FactoryState,
            deps_type=FactoryDeps,
            input_type=CompareBuilderNode,
            output_type=HaltResult,
        )
    )
    builder.add_edge(builder.start_node, CompareBuilderNode)
    builder.add(builder.node(CompareBuilderNode))
    builder.add(builder.node(CompareQAGateNode))
    builder.add(builder.node(CompareQAForkJoinNode))

    start = CompareBuilderNode(verdict_fn=verdict_fn, max_iterations=max_iterations)
    return builder, start


# ═══════════════════════════════════════════════════════════════════
# Part 3: Comparison Utilities
# ═══════════════════════════════════════════════════════════════════


def normalize_pg_events(state: FactoryState) -> list[TraceEvent]:
    """Convert FactoryState.events to normalized TraceEvents."""
    return [
        TraceEvent(
            node=ev.get("node", ""),
            action=ev.get("action", ""),
            detail={k: v for k, v in ev.items() if k not in ("node", "action")},
        )
        for ev in state.events
    ]


def extract_active_nodes(events: list[TraceEvent]) -> list[str]:
    """Extract ordered list of nodes that did work.

    Includes action "execute" events and fork children ("child_completed").
    Deduplicates consecutive repeats from the same node.
    """
    nodes: list[str] = []
    for e in events:
        if e.action in ("execute", "child_completed"):
            node_name = e.detail.get("child", e.node) if e.action == "child_completed" else e.node
            nodes.append(node_name)
    return nodes


def extract_gate_verdicts(events: list[TraceEvent]) -> list[str]:
    """Extract the sequence of gate verdict values."""
    return [
        e.detail.get("verdict", "")
        for e in events
        if e.action == "gate_verdict"
    ]


def extract_fork_children(events: list[TraceEvent]) -> set[str]:
    """Extract the set of children that ran in a fork/join."""
    children: set[str] = set()
    for e in events:
        if e.action == "fork_join_complete":
            children.update(e.detail.get("children", []))
        elif e.action == "child_completed":
            children.add(e.detail.get("child", ""))
    return children


async def run_comparison(
    verdict_sequence: list[VerdictType],
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Run both engines with the same verdict sequence and return comparison data.

    verdict_sequence controls the gate: element i is the verdict for the i-th
    gate evaluation (both engines use the same sequence).
    """
    call_count_sim = 0

    def sim_verdict(call_idx: int) -> VerdictType:
        if call_idx < len(verdict_sequence):
            return verdict_sequence[call_idx]
        return VerdictType.PROCEED

    pg_call_count = 0

    def pg_verdict(_state: FactoryState) -> VerdictType:
        nonlocal pg_call_count
        idx = pg_call_count
        pg_call_count += 1
        if idx < len(verdict_sequence):
            return verdict_sequence[idx]
        return VerdictType.PROCEED

    sim_workflow = build_current_engine_workflow()
    sim_events = simulate_current_engine(sim_workflow, sim_verdict, max_iterations)

    state = FactoryState()
    deps = FactoryDeps(dry_run=True)
    graph_builder, start = build_pydantic_graph(pg_verdict, max_iterations)
    graph = graph_builder.build()
    result = await graph.run(state=state, deps=deps, inputs=start)
    pg_events = normalize_pg_events(state)

    sim_active = extract_active_nodes(sim_events)
    pg_active = extract_active_nodes(pg_events)
    sim_verdicts = extract_gate_verdicts(sim_events)
    pg_verdicts = extract_gate_verdicts(pg_events)
    sim_children = extract_fork_children(sim_events)
    pg_children = extract_fork_children(pg_events)

    return {
        "current_engine": {
            "events": sim_events,
            "active_nodes": sim_active,
            "gate_verdicts": sim_verdicts,
            "fork_children": sim_children,
        },
        "pydantic_graph": {
            "events": pg_events,
            "active_nodes": pg_active,
            "gate_verdicts": pg_verdicts,
            "fork_children": pg_children,
            "result": result,
        },
        "match": {
            "active_nodes": set(sim_active) == set(pg_active),
            "gate_verdicts": sim_verdicts == pg_verdicts,
            "fork_children": sim_children == pg_children,
        },
        "mermaid": graph.render(),
    }
