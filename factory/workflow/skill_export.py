"""Convert WorkflowSkill (Pydantic graph) → Claude Code SKILL.md files.

The converter parses the graph structure — nodes, edges, gates, fork/join
topology — and generates standardized prose instructions. Two execution
formats from one source: flexible prose (SKILL.md) for interactive use,
rigid graph (WorkflowExecutor) for headless automation.

The templatize path emits {{slot_name::default_value}} markers and
<!-- --> annotation comments for the verified skill generation pipeline.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

import structlog

from factory.workflow.primitives import (
    AgentNode,
    DEFAULT_AGENT_POOL,
    DataNode,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    LLMNode,
    SelectionNode,
    Study,
    SubgraphForkNode,
    VerdictType,
    Workflow,
)
from factory.workflow.templates import emit

log = structlog.get_logger()


# ── metadata per workflow (enriched descriptions, phases, triggers) ──


WORKFLOW_META: dict[str, dict[str, str | list[str]]] = {
    "design": {
        "description": (
            "Interactive design mode — build with a user approval gate at strategy, "
            "plus conditional study for existing projects. Use when the user says "
            "'design X', 'plan X', 'let's discuss what to build', or wants to review "
            "the strategy before building. Works for both new and existing projects. "
            "Supports --from-plan to load an existing plan and skip research. "
            "With --just-plan, runs plan-only (research + strategy + GitHub publish, NO implementation)."
        ),
        "argument_hint": "<project_path> [idea or spec] [--from-plan <path_or_url>] [--just-plan]",
    },
    "create": {
        "description": (
            "Create mode — meta-mode for creating new factory modes or updating existing ones. "
            "For new modes: takes a description and produces a fully working workflow definition, "
            'SKILL.md, CLI wiring, and tests. For updates: use --focus "mode_name: change description" '
            'to modify an existing registered mode (e.g. --focus "improve: add plateau detection"). '
            "Use when the user says 'create a mode for X', 'update the improve mode', "
            "'add a new workflow', or wants to extend/modify factory pipelines."
        ),
        "argument_hint": '"mode description" or "existing_mode: change description"',
    },
    "swebench": {
        "description": (
            "SWE-bench benchmark mode — minimal 4-node pipeline for solving "
            "GitHub issues in containerized evaluation. Reads the task instruction, "
            "fixes the bug, runs tests, and merges to main. No eval infrastructure, "
            "no deep-QA, no research phases. Use when invoked with --mode swebench "
            "inside a Harbor benchmark container."
        ),
        "argument_hint": "<project_path> --prompt /tmp/task-instruction.md",
    },
    "outer-loop": {
        "description": (
            "Outer loop evolutionary search — evolve workflow DAGs against benchmarks. "
            "Runs seed → evaluate → reflect → evolve → convergence gate with RELOOP. "
            "Terminal mode — does not chain to other modes. "
            "Use when the user says 'outer-loop', 'evolve workflows', or wants "
            "evolutionary search for optimal workflow topologies."
        ),
        "argument_hint": "<project_path>",
    },
    "create-v2": {
        "description": (
            "Create mode with inference-time scaling — dynamic research, "
            "multi-strategy with intent fidelity, workflow-specific QA (mandatory "
            "workflow-validate and cli-integration testers), and Overwatch "
            "verification. Use when the user says 'create a mode' and wants "
            "the v2 pipeline with directors and intent tracking."
        ),
        "argument_hint": '"mode description" or "existing_mode: change description"',
    },
}


# ── topological sort ────────────────────────────────────────────


def _topological_sort(workflow: Workflow) -> list[str]:
    """Topological sort of node IDs respecting edge directions.

    Handles RELOOP edges by ignoring back-edges (edges whose target
    is already visited) and treating conditional edges as optional paths.
    Returns nodes in execution order.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in workflow.nodes}

    back_edges: set[tuple[str, str]] = set()
    for edge in workflow.edges:
        if edge.condition == VerdictType.RELOOP:
            back_edges.add((edge.source, edge.target))
            continue
        adj[edge.source].append(edge.target)
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    # Add implicit edges for fork/join semantics so fork targets sort
    # after the fork node and join sources sort before the join node.
    for nid, node in workflow.nodes.items():
        if type(node).__name__ == "ForkNode":
            for t in node.targets:  # type: ignore
                if t in workflow.nodes:
                    adj[nid].append(t)
                    in_degree[t] = in_degree.get(t, 0) + 1
        if type(node).__name__ == "JoinNode":
            for s in node.sources:  # type: ignore
                if s in workflow.nodes:
                    adj[s].append(nid)
                    in_degree[nid] = in_degree.get(nid, 0) + 1

    queue: deque[str] = deque()
    for nid in workflow.nodes:
        if in_degree.get(nid, 0) == 0:
            queue.append(nid)

    if not queue:
        queue.append(workflow.start_node)

    ordered: list[str] = []
    visited: set[str] = set()

    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        ordered.append(nid)

        for target in adj.get(nid, []):
            in_degree[target] -= 1
            if in_degree[target] <= 0 and target not in visited:
                queue.append(target)

    for nid in workflow.nodes:
        if nid not in visited:
            ordered.append(nid)

    return ordered


# ── edge helpers ──────────────────────────────────────────────────


def _outgoing_edges(workflow: Workflow, node_id: str) -> list[Edge]:
    """Return all edges originating from node_id."""
    return [e for e in workflow.edges if e.source == node_id]


def _format_edges(edges: list[Edge]) -> str:
    """Format outgoing edges for annotation comments."""
    if not edges:
        return "none"
    parts = []
    for e in edges:
        cond = e.condition_label or "unconditional"
        parts.append(f"{cond} → {e.target}")
    return ", ".join(parts)


# ── node → instruction converters ──────────────────────────────


def _agent_to_instruction(
    node: AgentNode,
    workflow: Workflow,
    *,
    is_parallel: bool = False,
) -> str:
    """Convert an AgentNode to a CLI invocation instruction with template slots."""
    role = node.role.value
    pool_entry = DEFAULT_AGENT_POOL.get(role)
    default_timeout = node.timeout or (pool_entry.timeout if pool_entry else 600)
    model_flag = " --model haiku" if role == "archivist" else ""

    prompt = (node.prompt_template or f"Execute {role} task for the project.").replace(
        "{project_path}", "$PROJECT_PATH",
    )

    if node.reads:
        reads_str = ", ".join(sorted(node.reads))
        prompt += f"\nRead: {reads_str}"
    if node.writes:
        writes_str = ", ".join(sorted(node.writes))
        prompt += f"\nWrite output to: {writes_str}"

    bg_suffix = " &" if is_parallel or not node.blocking else ""
    tag_flag = ""
    if is_parallel and role == "researcher":
        tag = node.id.replace("researcher_", "")
        tag_flag = f" --review-tag {tag}"

    timeout_slot = emit(f"timeout_{node.id}", str(default_timeout))
    task_slot = emit(f"task_prompt_{node.id}", prompt)

    cmd = (
        f'factory agent {role}{tag_flag} --task "{task_slot}"'
        f' --project "$PROJECT_PATH" --timeout {timeout_slot}{model_flag}{bg_suffix}'
    )

    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)
    reads_ann = ", ".join(sorted(node.reads)) if node.reads else "none"
    writes_ann = ", ".join(sorted(node.writes)) if node.writes else "none"

    annotations = [
        f"<!-- node: AgentNode id={node.id} role={role} blocking={str(node.blocking).lower()} -->",
        f"<!-- reads: {reads_ann} -->",
        f"<!-- writes: {writes_ann} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    lines = [*annotations, "", f"```bash\n{cmd}\n```"]

    if not node.blocking:
        lines.append("*(fire-and-forget — CEO continues immediately)*")
    elif not is_parallel and (node.writes or node.post_checks):
        from factory.workflow.verification import compile_agent_verification

        verify_script = compile_agent_verification(node)
        if verify_script:
            lines.append("")
            lines.append(f"```bash\n{verify_script}\n```")
            lines.append("*(harness verification — DO NOT SKIP)*")

    return "\n".join(lines)


def _llm_to_instruction(node: LLMNode, workflow: Workflow) -> str:
    """Convert an LLMNode to a direct API call instruction with template slots."""
    nid = node.id
    out_edges = _outgoing_edges(workflow, node.id)
    tools_str = ", ".join(t.name for t in node.tools) or "none"

    system = emit(f"system_prompt_{nid}", node.system_prompt)
    instance = emit(f"instance_prompt_{nid}", node.instance_prompt)

    lines = [
        f"<!-- node: LLMNode id={nid} model={node.model} provider={node.provider}"
        f" tools=[{tools_str}] max_turns={node.max_turns} timeout={node.timeout} -->",
        f"<!-- edges: {_format_edges(out_edges)} -->",
        "",
        f"**Model:** {node.model} | **Provider:** {node.provider}"
        f" | **Tools:** {tools_str}"
        f" | **Max turns:** {emit(f'max_turns_{nid}', str(node.max_turns))}"
        f" | **Timeout:** {emit(f'timeout_{nid}', str(node.timeout))}s",
        "",
        "**System prompt:**",
        system,
        "",
        "**Instance prompt:**",
        instance,
    ]

    if node.reads:
        lines.append("")
        lines.append(f"**Reads:** {', '.join(sorted(node.reads))}")
    if node.writes:
        lines.append(f"**Writes:** {', '.join(sorted(node.writes))}")

    return "\n".join(lines)


def _fn_to_instruction(node: FnNode, workflow: Workflow) -> str:
    """Convert an FnNode to a CLI command instruction with template slots."""
    cmd = node.command.replace("{project_path}", "$PROJECT_PATH")

    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)
    reads_ann = ", ".join(sorted(node.reads)) if node.reads else "none"
    writes_ann = ", ".join(sorted(node.writes)) if node.writes else "none"

    annotations = [
        f"<!-- node: FnNode id={node.id} -->",
        f"<!-- command: {node.command} -->",
        f"<!-- reads: {reads_ann} -->",
        f"<!-- writes: {writes_ann} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    prose = f"{node.notes}\n\n" if node.notes else ""

    if _has_template_placeholders(cmd):
        finalize_slot = emit(f"finalize_command_{node.id}", cmd)
        annotations.append(
            "<!-- NOTE: command contains template values requiring CEO substitution -->"
        )
        lines = [*annotations, "", f"{prose}```bash\n{finalize_slot}\n```"]
    else:
        lines = [*annotations, "", f"{prose}```bash\n{cmd}\n```"]

    return "\n".join(lines)


def _has_template_placeholders(text: str) -> bool:
    """Check if a command has $VARIABLE placeholders that need CEO substitution."""
    placeholders = {
        "$EXP_ID",
        "$VERDICT",
        "$HYPOTHESIS",
        "$REQUEST",
        "$PR_NUMBER",
        "$SCORE_BEFORE",
        "$SCORE_AFTER",
    }
    return any(p in text for p in placeholders)


def _study_to_instruction(node: Study, workflow: Workflow) -> str:
    """Convert a Study node to a factory study instruction."""
    cmd = node.command.replace("{project_path}", "$PROJECT_PATH")
    focus = ""
    if node.focus:
        focus = f' --focus "{node.focus}"'

    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)
    writes_ann = ", ".join(sorted(node.writes)) if node.writes else "none"

    annotations = [
        f"<!-- node: Study id={node.id} -->",
        f"<!-- command: {node.command} -->",
        f"<!-- writes: {writes_ann} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    focus_hint = ""
    if not node.focus:
        focus_hint = (
            "\n\nIf your task includes a focus directive or focus topic, "
            "pass it to the study command:\n"
            '`factory study $PROJECT_PATH --focus "<your focus topic>"`'
        )

    return (
        "\n".join(annotations) + "\n\n"
        f"Run local study to gather observations:\n\n"
        f"```bash\n{cmd}{focus}\n```\n\n"
        f"Writes observations to `.factory/strategy/observations.md`."
        f"{focus_hint}"
    )


def _gate_to_checkpoint(
    node: GateNode,
    reloop_edges: list[Edge],
    workflow: Workflow,
) -> str:
    """Convert a GateNode to a steering checkpoint with template slots."""
    gate_name = node.id.replace("gate_", "").replace("_", " ").title()

    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)
    reads_ann = ", ".join(sorted(node.reads)) if node.reads else "none"

    halt_edges = [e for e in out_edges if e.condition == VerdictType.HALT]
    proceed_edges = [e for e in out_edges if e.condition == VerdictType.PROCEED]

    lines: list[str] = []

    if node.evaluator_type == "user":
        ann = [
            f"<!-- gate: GateNode id={node.id} evaluator_type=user -->",
            f"<!-- reads: {reads_ann} -->",
            f"<!-- edges: {edges_str} -->",
        ]
        lines.extend(ann)
        lines.append("")
        lines.append(f"### Steering Point — {gate_name} (User Approval)")
        lines.append("")
        lines.append(
            "**This is a USER approval gate, NOT a CEO review gate. Do NOT self-approve.**"
        )
        lines.append("")
        lines.append(
            "Present the strategy/findings to the user by summarizing key points in your output."
        )
        lines.append(
            'Then explicitly ask the user: "Do you approve this plan, or do you have feedback?"'
        )
        lines.append("")
        lines.append("**You MUST wait for the user's response before proceeding.**")
        lines.append(
            '- The user says "approve", "yes", "looks good", or similar → proceed to next step'
        )
        lines.append(
            "- The user provides feedback or corrections → re-run the previous step incorporating their feedback"
        )
        lines.append(
            "- Do NOT write a verdict file and auto-proceed — this gate requires human input"
        )
    elif node.evaluator_type == "fn":
        evaluator_cmd = ""
        if node.evaluator_command:
            evaluator_cmd = node.evaluator_command
        ann = [
            f"<!-- gate: GateNode id={node.id} evaluator_type=fn -->",
            f"<!-- evaluator_command: {evaluator_cmd} -->",
            f"<!-- reads: {reads_ann} -->",
            f"<!-- edges: {edges_str} -->",
        ]
        lines.extend(ann)
        lines.append("")
        lines.append(f"### Gate — {gate_name} (Automated)")
        lines.append("")
        lines.append(
            "**MANDATORY:** Wait for the preceding agent to finish, then run this "
            "check BEFORE spawning the next agent. Do NOT run agents in parallel "
            "across this gate."
        )
        lines.append("")
        if node.evaluator_command:
            cmd = node.evaluator_command.replace("{project_path}", "$PROJECT_PATH")
            lines.append(f"```bash\n{cmd}\n```")

        if proceed_edges:
            proceed_target = proceed_edges[0].target
            lines.append(
                f"\n- **PROCEED** (exit 0 / no FAIL in output) → continue to `{proceed_target}`"
            )
            if halt_edges:
                halt_target = halt_edges[0].target
                lines.append(
                    f"- **HALT** (exit non-zero / FAIL in output) → "
                    f"continue to `{halt_target}` instead."
                )
            elif reloop_edges:
                reloop_target = reloop_edges[0].target
                lines.append(
                    f"- **RELOOP** (exit non-zero / FAIL in output) → "
                    f"return to `{reloop_target}` for the next iteration."
                )
            else:
                lines.append(
                    f"- **HALT** (exit non-zero / FAIL in output) → do NOT spawn `{proceed_target}`. "
                    "Skip to the next CEO review gate or finalize as error."
                )
        elif halt_edges:
            halt_target = halt_edges[0].target
            lines.append(
                f"\n- **HALT** (exit non-zero / FAIL in output) → "
                f"route to `{halt_target}` for error handling."
            )
    else:
        gate_prompt_slot = emit(f"gate_prompt_{node.id}", node.gate_prompt)
        ann = [
            f"<!-- gate: GateNode id={node.id} evaluator_type=agent evaluator_role={node.evaluator_role.value if node.evaluator_role else 'CEO'} -->",
            f"<!-- reads: {reads_ann} -->",
            f"<!-- edges: {edges_str} -->",
        ]
        lines.extend(ann)
        lines.append("")
        lines.append(f"### CEO Review — {gate_name}")
        lines.append("")
        lines.append("Apply the CEO Review Gate protocol:")
        lines.append("1. Read the agent output for the preceding step")
        if node.reads:
            reads = ", ".join(f"`{r}`" for r in sorted(node.reads))
            lines.append(f"2. Read artifacts: {reads}")
        lines.append(f"3. Assess: {gate_prompt_slot}")
        lines.append(
            f"4. Write verdict to `.factory/reviews/ceo-verdict-{gate_name.lower().replace(' ', '-')}.md`"
        )
        lines.append("5. **PROCEED** → continue to next step")
        lines.append("6. **REDIRECT** → re-invoke the preceding agent with corrections (max 2)")
        lines.append("7. **ABORT** → log failure and skip to archival")

    for edge in reloop_edges:
        max_iter = _resolve_max_iterations(edge, workflow)
        max_iter_slot = emit(f"max_iterations_{node.id}", str(max_iter))
        lines.append(f"\n*On RELOOP: return to `{edge.target}` (max {max_iter_slot} iterations)*")

    return "\n".join(lines)


def _resolve_max_iterations(edge: Edge, workflow: Workflow) -> int:
    """Resolve max_iterations from the RELOOP edge target's AgentNode."""
    target_node = workflow.nodes.get(edge.target)
    if isinstance(target_node, AgentNode) and target_node.max_iterations != 1:
        return target_node.max_iterations
    return 3


def _fork_to_instruction(node: ForkNode, workflow: Workflow) -> str:
    """Convert a ForkNode to parallel agent spawning instructions."""
    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)

    annotations = [
        f"<!-- node: ForkNode id={node.id} targets={','.join(node.targets)} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    lines = [*annotations, "", f"Spawn {len(node.targets)} agents in parallel:\n"]

    for target_id in node.targets:
        target_node = workflow.nodes.get(target_id)
        if isinstance(target_node, AgentNode):
            lines.append(_agent_to_instruction(target_node, workflow, is_parallel=True))
            lines.append("")
        elif isinstance(target_node, FnNode):
            lines.append(_fn_to_instruction(target_node, workflow))
            lines.append("")

    lines.append("```bash\nwait\n```")

    agent_nodes: list[AgentNode] = [
        workflow.nodes[tid]  # type: ignore[misc]
        for tid in node.targets
        if isinstance(workflow.nodes.get(tid), AgentNode)
    ]
    if agent_nodes:
        # Calculate the maximum timeout among all parallel agents
        max_timeout = max((node.timeout or 600 for node in agent_nodes), default=600)

        # Add timeout guidance if max_timeout exceeds Bash tool's default (120s)
        if max_timeout > 120:
            lines.append("")
            lines.append(
                f"\n**Important:** Run ALL commands above in a **single** Bash tool call "
                f"with timeout set to at least {max_timeout} seconds.\n"
            )

        from factory.workflow.verification import compile_fork_verification

        verify_script = compile_fork_verification(agent_nodes)
        if verify_script:
            lines.append("")
            lines.append(f"```bash\n{verify_script}\n```")
            lines.append("*(post-barrier harness verification — DO NOT SKIP)*")

    return "\n".join(lines)


def _join_to_instruction(node: JoinNode, workflow: Workflow) -> str:
    """Convert a JoinNode to a wait-for-all instruction."""
    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)
    reads_ann = ", ".join(sorted(node.reads)) if node.reads else "none"
    writes_ann = ", ".join(sorted(node.writes)) if node.writes else "none"

    annotations = [
        f"<!-- node: JoinNode id={node.id} sources={','.join(node.sources)} -->",
        f"<!-- reads: {reads_ann} -->",
        f"<!-- writes: {writes_ann} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    sources = ", ".join(f"`{s}`" for s in node.sources)
    lines = [*annotations, "", f"Wait for all parallel agents to complete: {sources}"]
    if node.reads:
        reads = ", ".join(f"`{r}`" for r in sorted(node.reads))
        lines.append(f"\nRead combined outputs: {reads}")
    if node.writes:
        writes = ", ".join(f"`{w}`" for w in sorted(node.writes))
        lines.append(f"\nWrite combined result to: {writes}")
    return "\n".join(lines)


def _subgraph_fork_to_instruction(node: SubgraphForkNode, workflow: Workflow) -> str:
    """Convert a SubgraphForkNode to parallel worktree experiment instructions."""
    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)

    annotations = [
        f"<!-- node: SubgraphForkNode id={node.id} entry={node.subgraph_entry} exit={node.subgraph_exit} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    lines = [
        *annotations,
        "",
        f"Fork up to {node.parallelism} parallel experiment branches, each in an isolated worktree:",
        "",
        "For each hypothesis from the strategy:",
        "1. Create an experiment worktree branching from the current commit",
        f"2. Run the experiment subgraph (`{node.subgraph_entry}` → `{node.subgraph_exit}`)",
        "3. Each branch runs independently: begin → builder → QA → eval",
        "",
        "All branches run concurrently. Results are collected at the barrier.",
    ]
    return "\n".join(lines)


def _selection_to_instruction(node: SelectionNode, workflow: Workflow) -> str:
    """Convert a SelectionNode to selection protocol instructions."""
    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)

    annotations = [
        f"<!-- node: SelectionNode id={node.id} strategy={node.strategy} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    lines = [
        *annotations,
        "",
        f"**Selection strategy: `{node.strategy}`**",
        "",
        "Compare all completed experiment branches:",
        "1. Read eval results from each branch's worktree",
        "2. Select the branch with the highest composite score",
        "3. Merge the winner's branch into the baseline",
        "4. Mark losing experiments as `superseded`",
        "5. Clean up all experiment worktrees",
    ]
    return "\n".join(lines)


def _data_to_instruction(node: DataNode, workflow: Workflow) -> str:
    """Convert a DataNode to concrete data-iteration instructions for the CEO.

    Raises ValueError because ceo-skill DataNode support is deferred to PR B.
    """
    raise ValueError(
        "DataNode workflows are not supported with ceo-skill / ceo-tool "
        "until PR B.  Use execution_strategy='executor'."
    )
    out_edges = _outgoing_edges(workflow, node.id)
    edges_str = _format_edges(out_edges)

    # Source description for annotations
    source_desc = "inline items"
    if node.task_ref:
        source_desc = f"task_ref `{node.task_ref}`"
    elif node.source_path:
        source_desc = f"source_path `{node.source_path}` (format: {node.source_format})"

    annotations = [
        f"<!-- node: DataNode id={node.id} entry={node.subgraph_entry} exit={node.subgraph_exit} -->",
        f"<!-- source: {source_desc} -->",
        f"<!-- edges: {edges_str} -->",
    ]

    lines: list[str] = [*annotations, ""]

    # Step 1: Get items
    if node.task_ref:
        lines.extend([
            "**Step 1 — Get items:**",
            "",
            "```bash",
            f"factory task instances --task-ref {node.task_ref}",
            "```",
            "",
            "Each line is a JSON object with fields: `id`, `path`, `metadata`.",
            "Parse each line and collect the item IDs.",
            "",
        ])
    elif node.source_path:
        lines.extend([
            "**Step 1 — Get items:**",
            "",
            f"Read items from `{node.source_path}` (format: {node.source_format}).",
            "",
        ])
        if node.source_format == "jsonl":
            lines.append("Each line is a JSON object. Parse each line as one item.")
        elif node.source_format == "directory":
            lines.append("Each subdirectory is one item. The directory name is the item ID.")
        elif node.source_format == "csv":
            lines.append("Each row is one item. The first column is the item ID.")
        lines.append("")
    else:
        # inline_items
        lines.extend([
            "**Step 1 — Items are inline:**",
            "",
        ])
        for item in node.inline_items:
            prompt_preview = item.prompt[:80] if item.prompt else "(no prompt)"
            lines.append(f"- Item `{item.id}`: {prompt_preview}")
        lines.append("")

    # Step 2: Per-item processing
    if node.task_ref:
        lines.extend([
            f"**Step 2 — For each item, run the subgraph "
            f"(`{node.subgraph_entry}` → `{node.subgraph_exit}`):**",
            "",
            "For each item ID from Step 1:",
            "",
            f"1. **Setup:** `factory task setup --task-ref {node.task_ref}"
            f" --instance-id <id> --workspace $PROJECT_PATH`",
            "2. **Write context:** Write the item JSON to"
            " `.factory/current_item.json` so subgraph agents can read it.",
            f"3. **Run subgraph:** Execute agents from"
            f" `{node.subgraph_entry}` through `{node.subgraph_exit}`.",
            f"4. **Verify:** `factory task verify --task-ref {node.task_ref}"
            f" --instance-id <id> --workspace $PROJECT_PATH`",
            '   The verify command prints JSON: `{"score": 0.85, "passed": true, "details": ...}`',
            "",
        ])
    else:
        lines.extend([
            f"**Step 2 — For each item, run the subgraph "
            f"(`{node.subgraph_entry}` → `{node.subgraph_exit}`):**",
            "",
            "For each item:",
            "",
            "1. **Write context:** Write the item data to `.factory/current_item.json`.",
            f"2. **Run subgraph:** Execute agents from"
            f" `{node.subgraph_entry}` through `{node.subgraph_exit}`.",
            "3. **Check result:** Score 1.0 if subgraph succeeded, 0.0 if it failed.",
            "",
        ])

    # Step 3: Aggregate
    lines.extend([
        "**Step 3 — Aggregate scores:**",
        "",
        "Average all per-item scores to get the final score for this DataNode.",
        "",
    ])

    # Configuration
    lines.extend([
        "**Configuration:**",
        "",
        f"- Parallelism: {node.parallelism} (process this many items concurrently)",
    ])
    if node.shuffle:
        lines.append("- Shuffle: yes (randomize item order)")
    if node.limit is not None:
        lines.append(f"- Limit: {node.limit} items")
    lines.append(f"- Max items (safety ceiling): {node.max_items}")
    lines.append("")
    lines.append(
        "**Fault isolation:** A failing item scores 0.0"
        " but does not halt the iteration. Continue with remaining items."
    )

    return "\n".join(lines)


# ── frontmatter builder ────────────────────────────────────────


def _build_frontmatter(
    name: str,
    description: str,
    argument_hint: str | None = None,
) -> str:
    """Build SKILL.md YAML frontmatter."""
    lines = [
        "---",
        f"name: workflow-{name}",
        f'description: "{description}"',
        "disable-model-invocation: true",
    ]
    if argument_hint:
        lines.append(f'argument-hint: "{argument_hint}"')
    lines.append("---")
    return "\n".join(lines)


# ── main converter ──────────────────────────────────────────────


def workflow_to_skill_md(workflow: Workflow) -> str:
    """Convert a Workflow into a Claude Code SKILL.md string.

    Parses the workflow graph structure (nodes, edges, gates, fork/join)
    and generates standardized prose instructions that the CEO follows
    flexibly. Gates become steering points for user interaction.

    Emits {{slot_name::default_value}} template markers and <!-- -->
    annotation comments for the verified skill generation pipeline.
    """
    name = workflow.name
    meta = WORKFLOW_META.get(name, {})
    description = str(meta.get("description", f"Run the {name} workflow."))
    argument_hint = str(meta.get("argument_hint", "<project_path>"))

    frontmatter = _build_frontmatter(name, description, argument_hint)

    title = name.replace("_", " ").replace("-", " ").title()
    header = f"# {title} Workflow\n\nThe user wants: **$ARGUMENTS**"

    preamble = meta.get("preamble")
    if preamble:
        header += f"\n\n{preamble}"

    reloop_map: dict[str, list[Edge]] = defaultdict(list)
    for edge in workflow.edges:
        if edge.condition == VerdictType.RELOOP:
            reloop_map[edge.source].append(edge)

    sorted_nodes = _topological_sort(workflow)
    fork_targets: set[str] = set()
    subgraph_nodes: set[str] = set()
    for nid in sorted_nodes:
        node = workflow.nodes[nid]
        if isinstance(node, ForkNode):
            fork_targets.update(node.targets)
        elif isinstance(node, SubgraphForkNode):
            from factory.workflow.executor import _collect_subgraph_nodes

            subgraph_nodes |= _collect_subgraph_nodes(
                workflow, node.subgraph_entry, node.subgraph_exit
            )
        elif isinstance(node, DataNode):
            # DataNode uses real edges; collect branch nodes via BFS
            from factory.workflow.data_runtime import _find_branch_and_join

            try:
                _, _, branch_ids = _find_branch_and_join(workflow, nid)
                subgraph_nodes |= branch_ids
            except ValueError:
                pass  # No JoinNode found — will error in _data_to_instruction

    sections: list[str] = []
    phase_num = 1

    for nid in sorted_nodes:
        if nid in fork_targets or nid in subgraph_nodes:
            continue

        node = workflow.nodes[nid]

        if isinstance(node, SubgraphForkNode):
            node_title = nid.replace("fork_", "").replace("_", " ").title()
            sections.append(f"## Phase {phase_num}: {node_title} (Parallel Experiments)\n")
            sections.append(_subgraph_fork_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, SelectionNode):
            sections.append(f"## Phase {phase_num}: Select Best Experiment\n")
            sections.append(_selection_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, ForkNode):
            node_title = nid.replace("fork_", "").replace("_", " ").title()
            sections.append(f"## Phase {phase_num}: {node_title} (Parallel)\n")
            sections.append(_fork_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, JoinNode):
            node_title = nid.replace("join_", "").replace("_", " ").title()
            sections.append(f"## Barrier: {node_title}\n")
            sections.append(_join_to_instruction(node, workflow))

        elif isinstance(node, GateNode):
            sections.append(_gate_to_checkpoint(node, reloop_map.get(nid, []), workflow))

        elif isinstance(node, Study):
            node_title = "Observe"
            sections.append(f"## Phase {phase_num}: {node_title}\n")
            sections.append(_study_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, AgentNode):
            role_title = node.role.value.replace("_", " ").title()
            node_title = nid.replace("_", " ").title()
            if role_title.lower() in node_title.lower():
                section_title = node_title
            else:
                section_title = f"{role_title} — {node_title}"
            sections.append(f"## Phase {phase_num}: {section_title}\n")
            sections.append(_agent_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, LLMNode):
            node_title = nid.replace("_", " ").title()
            sections.append(f"## Phase {phase_num}: {node_title} (LLM API)\n")
            sections.append(_llm_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, DataNode):
            node_title = nid.replace("_", " ").title()
            sections.append(f"## Phase {phase_num}: {node_title} (Data Iteration)\n")
            sections.append(_data_to_instruction(node, workflow))
            phase_num += 1

        elif isinstance(node, FnNode):
            node_title = nid.replace("_", " ").title()
            sections.append(f"## Step: {node_title}\n")
            sections.append(_fn_to_instruction(node, workflow))

        else:
            log.warning(
                "skill_export.unmatched_node_type",
                node_id=nid,
                node_type=type(node).__name__,
            )

    body = "\n\n".join(sections)

    result = f"{frontmatter}\n\n{header}\n\n{body}\n"

    line_count = result.count("\n") + 1
    if line_count > 1200:
        log.warning(
            "skill_export.oversized",
            workflow=name,
            lines=line_count,
            limit=1200,
        )

    return result


# ── bulk export ─────────────────────────────────────────────────


def export_all_skills(
    output_dir: Path,
    workflows: dict[str, Workflow] | None = None,
) -> list[Path]:
    """Export all registered workflows as SKILL.md files.

    Generates templatized content, then resolves it to clean prose for
    SKILL.md and writes structured annotations to SKILL.annotations.yaml.
    Returns paths to generated SKILL.md files.
    """
    from factory.workflow.splitter import annotations_to_yaml, split_skill

    if workflows is None:
        from factory.workflow.definitions import register_all

        workflows = register_all()

    generated: list[Path] = []

    for name, wf in workflows.items():
        templatized = workflow_to_skill_md(wf)
        clean_md, annotations = split_skill(templatized)

        skill_dir = output_dir / f"workflow-{name}"
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(clean_md)

        if annotations:
            ann_path = skill_dir / "SKILL.annotations.yaml"
            ann_path.write_text(annotations_to_yaml(annotations))

        generated.append(skill_path)
        log.info("skill_export.wrote", path=str(skill_path), lines=clean_md.count("\n") + 1)

    return generated


# ── validation ──────────────────────────────────────────────────


def validate_skill(content: str) -> list[str]:
    """Validate a generated SKILL.md string. Returns list of issues."""
    issues: list[str] = []

    if not content.startswith("---"):
        issues.append("Missing frontmatter (must start with ---)")
        return issues

    parts = content.split("---", 2)
    if len(parts) < 3:
        issues.append("Malformed frontmatter (missing closing ---)")
        return issues

    fm = parts[1]

    name_match = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
    if not name_match:
        issues.append("Missing 'name' in frontmatter")
    else:
        name_val = name_match.group(1).strip()
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,63}$", name_val):
            issues.append(f"Name '{name_val}' is not valid kebab-case (1-64 chars, a-z0-9-)")

    desc_match = re.search(r'^description:\s*"(.+)"$', fm, re.MULTILINE)
    if not desc_match:
        issues.append("Missing 'description' in frontmatter")
    else:
        desc_val = desc_match.group(1)
        if len(desc_val) > 1024:
            issues.append(f"Description exceeds 1024 chars ({len(desc_val)})")

    line_count = content.count("\n") + 1
    if line_count > 1200:
        issues.append(f"Body exceeds 1200 lines ({line_count})")

    return issues
