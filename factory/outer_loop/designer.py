"""Designer Agent — dual-mode workflow designer and informed mutation proposer.

Design mode: creates from-scratch workflow designs (minimal, thorough, custom).
Mutation mode: proposes targeted mutations based on failure telemetry.

v1 uses deterministic templates. LLM integration comes when the outer loop
runs against real benchmarks.
"""

from __future__ import annotations

import structlog

from factory.outer_loop.models import EvalResult, MutationRecord, MutationType
from factory.workflow.executor import _collect_subgraph_nodes
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    GateNode,
    NodeType,
    VerdictType,
    Workflow,
)

log = structlog.get_logger()


def _detect_frozen_data_node(
    seed_workflow: Workflow | None,
    frozen_node_ids: set[str] | None,
) -> DataNode | None:
    """Return the frozen DataNode if one exists, else None."""
    if not seed_workflow or not frozen_node_ids:
        return None
    for node_id in frozen_node_ids:
        node = seed_workflow.nodes.get(node_id)
        if isinstance(node, DataNode):
            return node
    return None


class DesignerAgent:
    """LLM-guided workflow designer with design and mutation modes.

    Design mode produces from-scratch workflows for seed diversity.
    Mutation mode proposes targeted mutations from failure telemetry.
    """

    def design_minimal(
        self,
        benchmark_spec: str,
        seed_workflow: Workflow | None = None,
        frozen_node_ids: set[str] | None = None,
        execution_strategy: str = "executor",
    ) -> Workflow:
        """Create a 3-4 node workflow optimized for speed.

        Structure: researcher → builder → gate
        """
        nodes: dict[str, NodeType] = {
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                writes={".factory/strategy/research.md"},
                timeout=300,
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                reads={".factory/strategy/research.md"},
                writes={".factory/reviews/builder-latest.md"},
                timeout=600,
            ),
            "gate_qa": GateNode(
                id="gate_qa",
                evaluator_type="agent",
                evaluator_role=AgentRole.HEALTH_CHECKER,
                reads={".factory/reviews/builder-latest.md"},
            ),
        }

        edges = [
            Edge(source="researcher", target="builder"),
            Edge(source="builder", target="gate_qa"),
        ]

        # Detect if we're creating a subgraph for a DataNode
        is_data_subgraph = _detect_frozen_data_node(seed_workflow, frozen_node_ids) is not None

        _propagate_prompts_from_seed(nodes, seed_workflow)
        _inject_frozen_nodes(nodes, edges, seed_workflow, frozen_node_ids)

        if execution_strategy == "executor":
            _populate_executor_fields(nodes, edges, is_data_subgraph=is_data_subgraph)

        start_node = "researcher"
        new_start = _rewire_data_nodes(
            nodes, edges, start_node, seed_workflow, frozen_node_ids
        )
        if new_start is not None:
            start_node = new_start

        wf = Workflow(
            name=f"minimal_{_slug(benchmark_spec)}",
            nodes=nodes,  # type: ignore[arg-type]
            edges=edges,
            start_node=start_node,
        )
        wf = _validate_and_fix(wf, seed_workflow)
        log.info("designed_minimal", nodes=len(wf.nodes), benchmark=benchmark_spec[:40])
        return wf

    def design_thorough(
        self,
        benchmark_spec: str,
        seed_workflow: Workflow | None = None,
        frozen_node_ids: set[str] | None = None,
        execution_strategy: str = "executor",
    ) -> Workflow:
        """Create an 8-10 node workflow optimized for thoroughness.

        Structure: study → researcher → strategist → fork(builder_a, builder_b)
                   → join → code_reviewer → adversarial_tester → gate
        """
        from factory.workflow.primitives import ForkNode, JoinNode

        # Detect if we're creating a subgraph for a DataNode
        is_data_subgraph = _detect_frozen_data_node(seed_workflow, frozen_node_ids) is not None

        if is_data_subgraph:
            # Data-native thorough template: omit study, strategist,
            # code_reviewer, adversarial_tester (not relevant per-item)
            nodes: dict[str, NodeType] = {
                "researcher": AgentNode(
                    id="researcher",
                    role=AgentRole.RESEARCHER,
                    writes={".factory/strategy/research.md"},
                    timeout=600,
                ),
                "fork_builders": ForkNode(
                    id="fork_builders",
                    targets=["builder_a", "builder_b"],
                    reads={".factory/strategy/research.md"},
                ),
                "builder_a": AgentNode(
                    id="builder_a",
                    role=AgentRole.BUILDER,
                    reads={".factory/strategy/research.md"},
                    writes={".factory/reviews/builder-a.md"},
                    timeout=1200,
                ),
                "builder_b": AgentNode(
                    id="builder_b",
                    role=AgentRole.BUILDER,
                    reads={".factory/strategy/research.md"},
                    writes={".factory/reviews/builder-b.md"},
                    timeout=1200,
                ),
                "join_builders": JoinNode(
                    id="join_builders",
                    sources=["builder_a", "builder_b"],
                ),
                "gate_qa": GateNode(
                    id="gate_qa",
                    evaluator_type="agent",
                    evaluator_role=AgentRole.HEALTH_CHECKER,
                    reads={".factory/reviews/builder-a.md", ".factory/reviews/builder-b.md"},
                ),
            }

            edges = [
                Edge(source="researcher", target="fork_builders"),
                Edge(source="fork_builders", target="builder_a"),
                Edge(source="fork_builders", target="builder_b"),
                Edge(source="builder_a", target="join_builders"),
                Edge(source="builder_b", target="join_builders"),
                Edge(source="join_builders", target="gate_qa"),
            ]

            start_node = "researcher"
        else:
            nodes = {
                "study": FnNode(
                    id="study",
                    command="factory study {project_path}",
                    writes={".factory/strategy/observations.md"},
                ),
                "researcher": AgentNode(
                    id="researcher",
                    role=AgentRole.RESEARCHER,
                    reads={".factory/strategy/observations.md"},
                    writes={".factory/strategy/research.md"},
                    timeout=600,
                ),
                "strategist": AgentNode(
                    id="strategist",
                    role=AgentRole.STRATEGIST,
                    reads={".factory/strategy/research.md"},
                    writes={".factory/strategy/current.md"},
                    timeout=600,
                ),
                "fork_builders": ForkNode(
                    id="fork_builders",
                    targets=["builder_a", "builder_b"],
                    reads={".factory/strategy/current.md"},
                ),
                "builder_a": AgentNode(
                    id="builder_a",
                    role=AgentRole.BUILDER,
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-a.md"},
                    timeout=1200,
                ),
                "builder_b": AgentNode(
                    id="builder_b",
                    role=AgentRole.BUILDER,
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-b.md"},
                    timeout=1200,
                ),
                "join_builders": JoinNode(
                    id="join_builders",
                    sources=["builder_a", "builder_b"],
                ),
                "code_reviewer": AgentNode(
                    id="code_reviewer",
                    role=AgentRole.CODE_REVIEWER,
                    reads={".factory/reviews/builder-a.md", ".factory/reviews/builder-b.md"},
                    writes={".factory/reviews/code-review.md"},
                    timeout=900,
                ),
                "adversarial_tester": AgentNode(
                    id="adversarial_tester",
                    role=AgentRole.ADVERSARIAL_TESTER,
                    reads={".factory/reviews/code-review.md"},
                    writes={".factory/reviews/adversarial-qa.md"},
                    timeout=1800,
                ),
                "gate_qa": GateNode(
                    id="gate_qa",
                    evaluator_type="agent",
                    evaluator_role=AgentRole.CEO,
                    reads={".factory/reviews/adversarial-qa.md"},
                ),
            }

            edges = [
                Edge(source="study", target="researcher"),
                Edge(source="researcher", target="strategist"),
                Edge(source="strategist", target="fork_builders"),
                Edge(source="fork_builders", target="builder_a"),
                Edge(source="fork_builders", target="builder_b"),
                Edge(source="builder_a", target="join_builders"),
                Edge(source="builder_b", target="join_builders"),
                Edge(source="join_builders", target="code_reviewer"),
                Edge(source="code_reviewer", target="adversarial_tester"),
                Edge(source="adversarial_tester", target="gate_qa"),
            ]

            start_node = "study"

        _propagate_prompts_from_seed(nodes, seed_workflow)
        _inject_frozen_nodes(nodes, edges, seed_workflow, frozen_node_ids)

        if execution_strategy == "executor":
            _populate_executor_fields(nodes, edges, is_data_subgraph=is_data_subgraph)

        new_start = _rewire_data_nodes(
            nodes, edges, start_node, seed_workflow, frozen_node_ids
        )
        if new_start is not None:
            start_node = new_start

        wf = Workflow(
            name=f"thorough_{_slug(benchmark_spec)}",
            nodes=nodes,  # type: ignore[arg-type]
            edges=edges,
            start_node=start_node,
        )
        wf = _validate_and_fix(wf, seed_workflow)
        log.info("designed_thorough", nodes=len(wf.nodes), benchmark=benchmark_spec[:40])
        return wf

    def design_custom(
        self,
        benchmark_spec: str,
        constraints: dict[str, object],
        seed_workflow: Workflow | None = None,
        frozen_node_ids: set[str] | None = None,
        execution_strategy: str = "executor",
    ) -> Workflow:
        """Create a custom from-scratch workflow with optional constraints.

        Constraints can specify:
        - max_nodes: int — cap on node count
        - require_roles: list[str] — roles that must be present
        - parallel: bool — whether to include fork/join parallelism
        """
        raw_max = constraints.get("max_nodes", 6)
        max_nodes = int(raw_max) if isinstance(raw_max, (int, float, str)) else 6
        raw_roles = constraints.get("require_roles", [])
        require_roles: list[object] = list(raw_roles) if isinstance(raw_roles, list) else []

        # Detect if we're creating a subgraph for a DataNode
        is_data_subgraph = _detect_frozen_data_node(seed_workflow, frozen_node_ids) is not None

        nodes: dict[str, NodeType] = {}
        edges: list[Edge] = []
        prev_id: str | None = None

        if is_data_subgraph:
            # Data subgraphs: exclude STRATEGIST and CODE_REVIEWER from defaults
            core_roles: list[tuple[str, AgentRole]] = [
                ("researcher", AgentRole.RESEARCHER),
                ("builder", AgentRole.BUILDER),
            ]
        else:
            core_roles = [
                ("researcher", AgentRole.RESEARCHER),
                ("strategist", AgentRole.STRATEGIST),
                ("builder", AgentRole.BUILDER),
            ]

        for role_str in require_roles:
            if isinstance(role_str, str) and not any(r[0] == role_str for r in core_roles):
                try:
                    role_enum = AgentRole(role_str)
                    core_roles.append((role_str, role_enum))
                except ValueError:
                    pass

        node_budget = max_nodes - 1
        for node_id, role in core_roles:
            if len(nodes) >= node_budget:
                break
            nodes[node_id] = AgentNode(
                id=node_id,
                role=role,
                timeout=600,
            )
            if prev_id is not None:
                edges.append(Edge(source=prev_id, target=node_id))
            prev_id = node_id

        if prev_id is not None:
            gate_id = "gate_qa"
            nodes[gate_id] = GateNode(  # type: ignore[assignment]
                id=gate_id,
                evaluator_type="agent",
                evaluator_role=AgentRole.HEALTH_CHECKER,
            )
            edges.append(Edge(source=prev_id, target=gate_id))

        _propagate_prompts_from_seed(nodes, seed_workflow)
        _inject_frozen_nodes(nodes, edges, seed_workflow, frozen_node_ids)

        if execution_strategy == "executor":
            _populate_executor_fields(nodes, edges, is_data_subgraph=is_data_subgraph)

        start = core_roles[0][0] if core_roles else "gate_qa"
        new_start = _rewire_data_nodes(
            nodes, edges, start, seed_workflow, frozen_node_ids
        )
        if new_start is not None:
            start = new_start

        wf = Workflow(
            name=f"custom_{_slug(benchmark_spec)}",
            nodes=nodes,  # type: ignore[arg-type]
            edges=edges,
            start_node=start,
        )
        wf = _validate_and_fix(wf, seed_workflow)
        log.info("designed_custom", nodes=len(wf.nodes), benchmark=benchmark_spec[:40])
        return wf

    def propose(
        self,
        parent_workflow: Workflow,
        telemetry: dict[str, object],
        archive_stats: dict[str, object],
        benchmark_spec: str,
    ) -> list[MutationRecord]:
        """Propose 1-3 targeted mutations based on failure telemetry.

        Heuristics:
        - High failure rate on a node → propose removing or replacing it
        - Dominant failure is timeout → propose reducing parallelism or increasing timeout
        - Low diversity → propose inserting a new agent role not yet present
        """
        proposals: list[MutationRecord] = []

        node_stats = telemetry.get("node_stats", {})
        if isinstance(node_stats, dict):
            for node_id, stats in node_stats.items():
                if not isinstance(stats, dict):
                    continue
                failure_rate = stats.get("failure_rate", 0.0)
                if isinstance(failure_rate, (int, float)) and failure_rate > 0.5:
                    proposals.append(MutationRecord(
                        operator=MutationType.NODE_REMOVE,
                        target_node=node_id,
                        before={"failure_rate": failure_rate},
                        after={"action": "remove_failing_node"},
                        rationale=f"Node {node_id} has {failure_rate:.0%} failure rate",
                    ))

        dominant_failure = telemetry.get("dominant_failure", "")
        if dominant_failure == "timeout":
            agent_nodes = [
                nid for nid, node in parent_workflow.nodes.items()
                if type(node).__name__ == "AgentNode"
            ]
            if agent_nodes:
                target = agent_nodes[0]
                current_timeout = getattr(parent_workflow.nodes[target], "timeout", 600)
                new_timeout = min((current_timeout or 600) * 2, 3600)
                proposals.append(MutationRecord(
                    operator=MutationType.PARAM_MUTATE,
                    target_node=target,
                    before={"timeout": current_timeout},
                    after={"timeout": new_timeout},
                    rationale="Dominant failure is timeout — increase timeout",
                ))

        diversity = archive_stats.get("diversity", 1.0)
        if isinstance(diversity, (int, float)) and diversity < 0.3:
            present_roles = {
                node.role.value  # type: ignore[union-attr]
                for node in parent_workflow.nodes.values()
                if hasattr(node, "role")
            }
            missing = set(AgentRole) - {AgentRole(r) for r in present_roles if r in [ar.value for ar in AgentRole]}
            if missing:
                new_role = next(iter(missing))
                proposals.append(MutationRecord(
                    operator=MutationType.NODE_INSERT,
                    target_node=None,
                    before={"present_roles": sorted(present_roles)},
                    after={"new_role": new_role.value},
                    rationale=f"Low diversity ({diversity:.2f}) — insert {new_role.value}",
                ))

        if not proposals:
            proposals.append(MutationRecord(
                operator=MutationType.PARAM_MUTATE,
                target_node=None,
                before={},
                after={"action": "explore"},
                rationale="No specific failure signal — propose parameter exploration",
            ))

        return proposals[:3]


def _generate_prompt_template(node: AgentNode, is_data_subgraph: bool = False) -> str:
    """Derive a functional prompt_template from an AgentNode's metadata.

    For data subgraphs this is the FALLBACK — seed prompt propagation
    (via ``_inject_frozen_nodes``) should have already set a better prompt.
    """
    role_name = node.role.value.replace("_", " ")
    if is_data_subgraph:
        # Fallback: role framing + data contract. No domain-specific language.
        # Seed prompt propagation provides domain knowledge when available.
        parts = [f"As a {role_name},"]
        parts.append("read .factory/current_item.json for the current item.")
        if node.reads:
            parts.append("Read: " + ", ".join(sorted(node.reads)) + ".")
        if node.writes:
            parts.append("Write your output to: " + ", ".join(sorted(node.writes)) + ".")
        return " ".join(parts)
    else:
        parts = [f"Act as a {role_name} for the project at {{project_path}}."]
        if node.reads:
            parts.append("Read: " + ", ".join(sorted(node.reads)) + ".")
        if node.writes:
            parts.append("Write your output to: " + ", ".join(sorted(node.writes)) + ".")
        return " ".join(parts)


def _populate_executor_fields(
    nodes: dict[str, NodeType],
    edges: list[Edge],
    *,
    is_data_subgraph: bool = False,
) -> None:
    """Populate prompt_template on AgentNodes and PROCEED edges on non-terminal GateNodes.

    Required when execution_strategy='executor' so WorkflowExecutor can
    construct agent prompts and follow success paths through gates.

    When *is_data_subgraph* is True, prompts use data-item language and
    the first AgentNode gets '.factory/current_item.json' added to reads.
    """
    # Find terminal nodes (no outgoing edges)
    sources_with_targets = {e.source for e in edges}
    first_agent_seen = False

    for node_id, node in nodes.items():
        if isinstance(node, AgentNode):
            update: dict[str, object] = {}

            if not node.prompt_template:
                # No propagated prompt — use generated fallback
                update["prompt_template"] = _generate_prompt_template(node, is_data_subgraph)
            elif is_data_subgraph and "current_item.json" not in node.prompt_template:
                # Has propagated prompt but missing data awareness — add prefix
                update["prompt_template"] = (
                    "Read .factory/current_item.json for the current item. "
                    + node.prompt_template
                )

            # Add current_item.json to reads for data subgraph entry node
            if is_data_subgraph and not first_agent_seen:
                update["reads"] = set(node.reads or set()) | {".factory/current_item.json"}
                first_agent_seen = True

            if update:
                nodes[node_id] = node.model_copy(update=update)
        elif isinstance(node, GateNode) and node_id in sources_with_targets:
            # Non-terminal gate: check if it already has a PROCEED edge
            has_proceed = any(
                e.source == node_id
                and e.condition is not None
                and (
                    e.condition == VerdictType.PROCEED
                    or (isinstance(e.condition, str) and e.condition.lower() == "proceed")
                )
                for e in edges
            )
            if not has_proceed:
                # Find the unconditional target and add a PROCEED edge
                for e in edges:
                    if e.source == node_id and e.condition is None:
                        edges.append(Edge(
                            source=node_id,
                            target=e.target,
                            condition=VerdictType.PROCEED,
                        ))
                        break


def _inject_frozen_nodes(
    nodes: dict[str, NodeType],
    edges: list[Edge],
    seed_workflow: Workflow | None,
    frozen_node_ids: set[str] | None,
) -> None:
    """Inject frozen nodes from a seed workflow into a template nodes dict.

    Frozen nodes take precedence over template nodes on ID collision.

    When a frozen node is a DataNode, its subgraph nodes are NOT injected —
    the template already provides new subgraph nodes (the evolution surface).
    Instead, prompt_template values are propagated from seed subgraph nodes
    to template nodes that share the same AgentRole, preserving learned prompts.

    Limitation: nested DataNodes within a subgraph are NOT recursively expanded.
    """
    if not seed_workflow or not frozen_node_ids:
        return

    for frozen_id in frozen_node_ids:
        if frozen_id in seed_workflow.nodes:
            if frozen_id in nodes:
                log.warning(
                    "frozen_node_collision",
                    node_id=frozen_id,
                    action="preferring_frozen_over_template",
                )
            nodes[frozen_id] = seed_workflow.nodes[frozen_id]

            node = seed_workflow.nodes[frozen_id]
            if isinstance(node, DataNode):
                # DO NOT inject subgraph nodes — template nodes replace them.
                # Instead, propagate prompt_template from seed subgraph to
                # template by matching AgentRole.
                if (node.subgraph_entry in seed_workflow.nodes
                        and node.subgraph_exit in seed_workflow.nodes):
                    subgraph_ids = _collect_subgraph_nodes(
                        seed_workflow, node.subgraph_entry, node.subgraph_exit
                    )
                    # Build role→prompt_template map from seed subgraph
                    seed_prompts: dict[str, str] = {}
                    for sg_id in subgraph_ids:
                        sg_node = seed_workflow.nodes.get(sg_id)
                        if isinstance(sg_node, AgentNode) and sg_node.prompt_template:
                            role_key = sg_node.role.value if sg_node.role else sg_id
                            seed_prompts[role_key] = sg_node.prompt_template

                    # Propagate to template nodes with matching roles
                    for tid, tnode in nodes.items():
                        if isinstance(tnode, AgentNode) and not tnode.prompt_template:
                            role_key = tnode.role.value if tnode.role else tid
                            if role_key in seed_prompts:
                                nodes[tid] = tnode.model_copy(
                                    update={"prompt_template": seed_prompts[role_key]}
                                )

                    log.debug(
                        "propagating_seed_prompts",
                        data_node_id=frozen_id,
                        roles_propagated=list(seed_prompts.keys()),
                    )
        else:
            log.warning("frozen_node_missing_in_seed", node_id=frozen_id)


def _rewire_data_nodes(
    nodes: dict[str, NodeType],
    edges: list[Edge],
    original_start: str,
    seed_workflow: Workflow | None,
    frozen_node_ids: set[str] | None,
) -> str | None:
    """Rewire injected frozen DataNodes so they integrate into the template.

    For each frozen DataNode, unconditionally update:
    - subgraph_entry → template's original start_node (the first template node)
    - subgraph_exit  → template's terminal node (no outgoing edges)

    The template nodes ARE the new subgraph — the DataNode's old subgraph
    references are replaced because the outer loop evolves the subgraph
    topology, not the DataNode infrastructure.

    No explicit edge is added from the DataNode to subgraph_entry — the
    executor reads subgraph_entry directly from the DataNode object.

    Returns the DataNode ID (new start_node) or None if no DataNode was injected.
    """
    if not seed_workflow or not frozen_node_ids:
        return None

    # Find terminal node: the node with no outgoing edges (among template edges)
    sources = {e.source for e in edges}
    all_node_ids = set(nodes.keys())
    terminal_candidates = all_node_ids - sources
    # Exclude the frozen DataNodes themselves from terminal candidates
    frozen_data_ids: set[str] = set()

    for fid in frozen_node_ids:
        node = nodes.get(fid)
        if isinstance(node, DataNode):
            frozen_data_ids.add(fid)

    if not frozen_data_ids:
        return None

    terminal_candidates -= frozen_data_ids
    terminal_node = next(iter(terminal_candidates)) if terminal_candidates else original_start

    new_start: str | None = None
    for data_id in frozen_data_ids:
        data_node = nodes[data_id]
        assert isinstance(data_node, DataNode)

        # Determine subgraph entry: if the DataNode ID collides with the
        # template's original_start, follow edges to find the actual first
        # template node (otherwise subgraph_entry would point to itself).
        # Also remove the now-stale edges from original_start — they would
        # become invalid edges from the DataNode to its own subgraph.
        entry = original_start
        if data_id == original_start:
            for edge in edges:
                if edge.source == original_start:
                    entry = edge.target
                    break
            edges[:] = [e for e in edges if e.source != original_start]

        updated = data_node.model_copy(
            update={"subgraph_entry": entry, "subgraph_exit": terminal_node}
        )
        nodes[data_id] = updated

        # Part D: If the terminal node is a GateNode that now serves as
        # subgraph_exit, it needs a PROCEED edge to advance past the
        # subgraph boundary.  The original template had it as a terminal
        # node (zero outgoing edges — valid for SKILL.md), but inside a
        # DataNode subgraph the executor expects a PROCEED edge.
        exit_node = nodes.get(terminal_node)
        if exit_node is not None and type(exit_node).__name__ == "GateNode":
            has_proceed = any(
                e.source == terminal_node and e.condition == VerdictType.PROCEED
                for e in edges
            )
            if not has_proceed:
                # Find the next node after the DataNode in the edge list
                after_data_targets = [
                    e.target for e in edges
                    if e.source == data_id and e.target != terminal_node
                ]
                proceed_target = after_data_targets[0] if after_data_targets else data_id
                edges.append(
                    Edge(
                        source=terminal_node,
                        target=proceed_target,
                        condition=VerdictType.PROCEED,
                    )
                )
        new_start = data_id

    return new_start


def extract_telemetry(eval_result: EvalResult) -> dict[str, object]:
    """Extract structured diagnostics from an EvalResult.

    Returns a dict with:
    - node_stats: per-node success/failure data (from details if available)
    - dominant_failure: most common failure category
    - benchmark_score: the raw benchmark score
    - cost_usd: evaluation cost
    - complexity: workflow complexity metric
    """
    details = eval_result.details or {}

    node_stats: dict[str, object] = {}
    raw_stats = details.get("node_stats", {})
    if isinstance(raw_stats, dict):
        node_stats = dict(raw_stats)

    dominant_failure = ""
    raw_failure = details.get("dominant_failure", "")
    if isinstance(raw_failure, str):
        dominant_failure = raw_failure

    return {
        "node_stats": node_stats,
        "dominant_failure": dominant_failure,
        "benchmark_score": eval_result.benchmark_score,
        "hygiene_score": eval_result.hygiene_score,
        "cost_usd": eval_result.cost_usd,
        "complexity": eval_result.complexity,
        "score": eval_result.score,
    }


def _propagate_prompts_from_seed(
    nodes: dict[str, NodeType],
    seed_workflow: Workflow | None,
) -> None:
    """Copy prompt_template from seed AgentNodes to template AgentNodes by role.

    Builds a {AgentRole → prompt_template} lookup from the seed workflow and
    fills in any template AgentNode whose prompt_template is empty.  Uses
    model_copy(update={...}) because AgentNode is strict=True, extra=forbid.

    No-op when seed_workflow is None.
    """
    if seed_workflow is None:
        return

    # Build role → prompt lookup from seed
    role_prompts: dict[AgentRole, str] = {}
    for node in seed_workflow.nodes.values():
        if type(node).__name__ == "AgentNode" and isinstance(node, AgentNode):
            if node.prompt_template:
                role_prompts[node.role] = node.prompt_template

    if not role_prompts:
        return

    # Fill empty prompts in template nodes
    for nid, node in list(nodes.items()):
        if type(node).__name__ == "AgentNode" and isinstance(node, AgentNode):
            if not node.prompt_template and node.role in role_prompts:
                nodes[nid] = node.model_copy(
                    update={"prompt_template": role_prompts[node.role]}
                )


def _validate_and_fix(
    wf: Workflow,
    seed_workflow: Workflow | None,
) -> Workflow:
    """Safety-net: validate and attempt to fix remaining issues on a constructed Workflow.

    Fixes applied:
    - Empty prompt_template on AgentNodes → generic default
    - Non-terminal GateNodes missing a PROCEED edge → adds one

    Re-validates after fixes and logs warnings for any unfixable issues.
    """
    issues = wf.validate_graph()
    if not issues:
        return wf

    # Attempt fixes
    updated_nodes = dict(wf.nodes)
    updated_edges = list(wf.edges)

    # Fix 1: fill empty prompt_templates with a generic default
    for nid, node in updated_nodes.items():
        if type(node).__name__ == "AgentNode" and isinstance(node, AgentNode):
            if not node.prompt_template:
                default_prompt = (
                    f"You are the {node.role.value} agent. "
                    f"Analyze the project at {{project_path}} and complete your assigned task."
                )
                updated_nodes[nid] = node.model_copy(
                    update={"prompt_template": default_prompt}
                )

    # Fix 2: add PROCEED edges to non-terminal GateNodes missing them
    gate_ids = {
        nid for nid, node in updated_nodes.items()
        if type(node).__name__ == "GateNode"
    }
    for gate_id in gate_ids:
        outgoing = [e for e in updated_edges if e.source == gate_id]
        if not outgoing:
            # Terminal gate — skip
            continue
        has_proceed = any(
            e.condition == VerdictType.PROCEED or e.condition is None
            for e in outgoing
        )
        if not has_proceed:
            # Find a reasonable target — use the first unconditional edge target
            # from any edge after the gate, or fall back to the gate itself
            targets_from_gate = {e.target for e in outgoing}
            # Use the first RELOOP target as a fallback PROCEED target
            fallback_target = next(iter(targets_from_gate))
            updated_edges.append(
                Edge(source=gate_id, target=fallback_target, condition=VerdictType.PROCEED)
            )

    # Rebuild workflow with fixes
    wf = Workflow(
        name=wf.name,
        nodes=updated_nodes,  # type: ignore[arg-type]
        edges=updated_edges,
        start_node=wf.start_node,
    )

    # Re-validate and warn about unfixable issues
    remaining = wf.validate_graph()
    if remaining:
        log.warning(
            "validate_and_fix_unfixable",
            workflow=wf.name,
            remaining_issues=remaining,
        )

    return wf


def _slug(text: str) -> str:
    """Convert text to a short slug for workflow naming."""
    clean = text.lower().replace(" ", "_")[:20]
    return "".join(c for c in clean if c.isalnum() or c == "_").strip("_") or "default"
