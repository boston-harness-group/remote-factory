"""Agent-scoped guardrails: ``Guard``, ``with_guards`` and compile-time expansion.

``Package.compile()`` splices a checker stage after each guarded agent
(``A -> reset -> fork -> checkers -> join -> gate``); the gate PROCEEDs to A's
successors and RELOOPs to A on violation. Expansion is pure and idempotent and
rejects unsupported placements. Guards on a graph that never calls ``compile()``
are inert.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    Guard,
    JoinNode,
    NodeType,
    SubgraphForkNode,
    VerdictType,
    Workflow,
)

__all__ = [
    "DEFAULT_AUDIT_HINT",
    "GUARD_ARTIFACT_ROOT",
    "GUARDRAILS_HEADER",
    "GUARD_SLUG_RE",
    "MAX_RULES_PAYLOAD_BYTES",
    "Guard",
    "expand_agent_guards",
    "guard_dir",
    "guard_rules_payload",
    "guard_slug",
    "with_guards",
]

GUARDRAILS_HEADER: str = "## Guardrails (binding)"

GUARD_ARTIFACT_ROOT: str = ".factory/guards"

MAX_RULES_PAYLOAD_BYTES: int = 65536

DEFAULT_AUDIT_HINT: str = (
    "Inspect the project's actual state (git diff, git status, the produced artifacts). "
    "Do not rely on any agent's self-report. Cite the exact command output or file:line that "
    "proves or disproves the rule. If you cannot verify, treat it as violated."
)

GUARD_SLUG_RE: re.Pattern[str] = re.compile(r"^[a-z0-9_]+$")

_GUARD_SLUG_MAX_LEN: int = 40
_CHECKER_ROLE: AgentRole = AgentRole.CODE_REVIEWER
_RESERVED_STAGE_SUFFIXES: tuple[str, ...] = ("reset", "fork", "join", "gate")
_GUARD_CLI: str = "factory workflow guard"


def guard_slug(rule: str) -> str:
    tokens = [token for token in re.split(r"[^a-z0-9]+", rule.lower()) if token]
    parts: list[str] = []
    for index, token in enumerate(tokens):
        if index == 0 and len(token) > _GUARD_SLUG_MAX_LEN:
            parts.append(token[:_GUARD_SLUG_MAX_LEN])
            break
        candidate = "_".join([*parts, token])
        if len(candidate) > _GUARD_SLUG_MAX_LEN:
            break
        parts.append(token)
    slug = "_".join(parts).strip("_")
    return slug or "guard"


def _unique_slugs(guards: Sequence[Guard]) -> list[str]:
    used: set[str] = set(_RESERVED_STAGE_SUFFIXES)
    slugs: list[str] = []
    for guard in guards:
        base = guard_slug(guard.rule)
        slug = base
        suffix = 2
        while slug in used:
            slug = f"{base}_{suffix}"
            suffix += 1
        used.add(slug)
        slugs.append(slug)
    return slugs


def _collision_error(node_id: str, agent_id: str, owners: dict[str, str]) -> ValueError:
    owner = owners.get(node_id)
    holder = (
        f"the guard stage generated for agent {owner!r}"
        if owner is not None
        else "an existing node"
    )
    return ValueError(
        f"generated guard node id {node_id!r} for agent {agent_id!r} collides with "
        f"{holder}; generated guard ids must be unique across guarded agents"
    )


def _agent_segment(agent_id: str) -> str:
    segment = re.sub(r"[^a-z0-9]+", "_", agent_id.lower()).strip("_")
    if not segment:
        raise ValueError(f"guard segment for agent id {agent_id!r} is empty")
    return segment


def guard_dir(agent_id_or_segment: str) -> Path:
    return Path(GUARD_ARTIFACT_ROOT) / _agent_segment(agent_id_or_segment)


def guard_rules_payload(slugs: Sequence[str], guards: Sequence[Guard]) -> str:
    """Canonical JSON ``{slug: rule}`` payload (unquoted; callers shlex-quote)."""
    return json.dumps(
        dict(zip(slugs, (guard.rule for guard in guards))),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _guard_block(node: AgentNode) -> str:
    if GUARDRAILS_HEADER in (node.prompt_template or ""):
        return ""
    rules = "".join(f"- {guard.rule.strip()}\n" for guard in node.guards)
    return f"\n\n{GUARDRAILS_HEADER}\n{rules}"


def _checker_prompt(agent_id: str, slug: str, guard: Guard) -> str:
    stage_dir = guard_dir(agent_id)
    verdict_path = (stage_dir / f"{slug}.json").as_posix()
    token_path = (stage_dir / "token").as_posix()
    hint = guard.audit_hint or DEFAULT_AUDIT_HINT
    lines = [
        f"You are the guard checker for agent '{agent_id}' in this project. "
        "Judge ONLY the single rule below. Do not review, summarise, or change anything else.",
        "",
        f"Rule: {guard.rule}",
        "",
        f"How to establish it: {hint}",
        "",
        f"Write exactly one JSON object to {verdict_path} with these keys:",
        f'- "guard": the string "{slug}"',
        f'- "token": the contents of {token_path} with the surrounding whitespace trimmed '
        "(the file ends with a newline; do not put a raw newline inside the JSON string)",
        '- "violated": a JSON boolean (true or false)',
        '- "evidence": a string citing the exact command output or file:line that proves or '
        "disproves the rule",
        '- "fix": a string; empty when not violated',
        'Example shape: {"guard": "<slug>", "token": "<token>", "violated": false, '
        '"evidence": "<evidence>", "fix": ""}',
        "",
        "Do not modify any other file in the project. "
        "Do not run git push, create remotes, or open PRs.",
        "",
        'If you cannot verify the rule, set "violated": true and explain why in "evidence".',
        "",
    ]
    return "\n".join(lines)


def _reset_command(segment: str, node: AgentNode) -> str:
    payload = guard_rules_payload(_unique_slugs(node.guards), node.guards)
    return f"{_GUARD_CLI} reset {{project_path}} {shlex.quote(segment)} {shlex.quote(payload)}"


def _check_command(segment: str, slugs: Sequence[str]) -> str:
    slug_args = " ".join(shlex.quote(slug) for slug in slugs)
    return f"{_GUARD_CLI} check {{project_path}} {shlex.quote(segment)} {slug_args}"


def _reset_node(segment: str, node: AgentNode) -> FnNode:
    stage_dir = guard_dir(segment)
    return FnNode(
        id=f"guard_{segment}_reset",
        command=_reset_command(segment, node),
        writes={
            (stage_dir / "token").as_posix(),
            (stage_dir / "rules.json").as_posix(),
        },
        reads=set(),
        notes=(
            f"Reset guard stage for {node.id}: clears stale verdicts, persists the rule "
            "registry, and writes a fresh iteration token"
        ),
    )


def _checker_node(segment: str, agent_id: str, slug: str, guard: Guard) -> AgentNode:
    return AgentNode(
        id=f"guard_{segment}_{slug}",
        role=_CHECKER_ROLE,
        model="",
        prompt_template=_checker_prompt(agent_id, slug, guard),
        writes={(guard_dir(segment) / f"{slug}.json").as_posix()},
        reads=set(),
        timeout=None,
        max_iterations=1,
        post_checks=[],
        blocking=True,
    )


def _gate_node(segment: str, slugs: Sequence[str], node: AgentNode) -> GateNode:
    return GateNode(
        id=f"guard_{segment}_gate",
        evaluator_type="fn",
        evaluator_command=_check_command(segment, slugs),
        evaluator_role=None,
        gate_prompt="",
        max_iterations=node.guard_max_iterations,
        reads=set(),
    )


def with_guards(
    node: AgentNode, guards: Sequence[Guard], *, max_iterations: int = 3
) -> AgentNode:
    if not isinstance(node, AgentNode):
        raise TypeError(f"with_guards expects an AgentNode, got {type(node).__name__}")
    if max_iterations < 1:
        raise ValueError(f"with_guards max_iterations must be >= 1, got {max_iterations}")
    return node.model_copy(
        deep=True,
        update={
            "guards": [guard.model_copy() for guard in guards],
            "guard_max_iterations": max_iterations,
        },
    )


def _fork_targets(workflow: Workflow) -> set[str]:
    targets: set[str] = set()
    for node in workflow.nodes.values():
        if isinstance(node, ForkNode):
            targets.update(node.targets)
    return targets


def _edge_index(workflow: Workflow, *, reverse: bool = False) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in workflow.edges:
        source, target = (edge.target, edge.source) if reverse else (edge.source, edge.target)
        adjacency.setdefault(source, []).append(target)
    return adjacency


def _reachable(workflow: Workflow, start: str, *, reverse: bool = False) -> set[str]:
    adjacency = _edge_index(workflow, reverse=reverse)
    seen: set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in workflow.nodes:
            continue
        seen.add(node_id)
        stack.extend(adjacency.get(node_id, ()))
    return seen


def _datanode_branch_region(workflow: Workflow, data_node_id: str) -> set[str]:
    adjacency = _edge_index(workflow)
    region: set[str] = set()
    stack = list(adjacency.get(data_node_id, ()))
    while stack:
        node_id = stack.pop()
        if node_id in region:
            continue
        node = workflow.nodes.get(node_id)
        if node is None or isinstance(node, JoinNode):
            continue
        region.add(node_id)
        stack.extend(adjacency.get(node_id, ()))
    return region


def _per_branch_contexts(workflow: Workflow) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for node_id, node in workflow.nodes.items():
        if isinstance(node, DataNode):
            for member in _datanode_branch_region(workflow, node_id):
                contexts.setdefault(member, f"DataNode '{node_id}' branch")
        elif isinstance(node, SubgraphForkNode):
            entry, exit_id = node.subgraph_entry, node.subgraph_exit
            if entry not in workflow.nodes or exit_id not in workflow.nodes:
                continue
            region = _reachable(workflow, entry) & _reachable(workflow, exit_id, reverse=True)
            for member in region:
                contexts.setdefault(member, f"SubgraphForkNode '{node_id}' subgraph")
    return contexts


def _stage_mismatches(
    workflow: Workflow,
    agent_id: str,
    segment: str,
    node: AgentNode,
    slugs: list[str],
    checker_ids: list[str],
) -> list[str]:
    """Name every part of an existing stage that no longer matches the agent's guards."""
    reset = workflow.nodes[f"guard_{segment}_reset"]
    fork = workflow.nodes[f"guard_{segment}_fork"]
    join = workflow.nodes[f"guard_{segment}_join"]
    gate = workflow.nodes[f"guard_{segment}_gate"]
    expected_reset = _reset_command(segment, node)
    expected_check = _check_command(segment, slugs)
    mismatches: list[str] = []
    if not isinstance(reset, FnNode):
        mismatches.append("reset node type")
    elif reset.command != expected_reset:
        mismatches.append("reset command")
    if not isinstance(fork, ForkNode):
        mismatches.append("fork node type")
    elif list(fork.targets) != checker_ids:
        mismatches.append("fork target list")
    if not isinstance(join, JoinNode):
        mismatches.append("join node type")
    elif list(join.sources) != checker_ids:
        mismatches.append("join source list")
    if not isinstance(gate, GateNode):
        mismatches.append("gate node type")
    else:
        if gate.max_iterations != node.guard_max_iterations:
            mismatches.append("gate iteration budget")
        if gate.evaluator_command != expected_check:
            mismatches.append("gate command")
    for checker_id, slug, guard in zip(checker_ids, slugs, node.guards, strict=True):
        checker = workflow.nodes[checker_id]
        if not isinstance(checker, AgentNode):
            mismatches.append(f"checker {checker_id!r} node type")
        elif checker.prompt_template != _checker_prompt(agent_id, slug, guard):
            mismatches.append(f"checker {checker_id!r} prompt")
    return mismatches


def _verify_stage(
    workflow: Workflow,
    agent_id: str,
    segment: str,
    node: AgentNode,
    slugs: list[str],
    checker_ids: list[str],
) -> None:
    mismatches = _stage_mismatches(workflow, agent_id, segment, node, slugs, checker_ids)
    if mismatches:
        raise ValueError(
            f"guard stage for agent {agent_id!r} exists but does not match its current "
            f"guards; differs in: {', '.join(mismatches)}; rebuild from the source package"
        )


def _add_stage_nodes(
    workflow: Workflow,
    stage: dict[str, NodeType],
    agent_id: str,
    owners: dict[str, str],
) -> None:
    for node_id in stage:
        if node_id in workflow.nodes:
            raise _collision_error(node_id, agent_id, owners)
    workflow.nodes.update(stage)
    owners.update({node_id: agent_id for node_id in stage})


def expand_agent_guards(workflow: Workflow) -> Workflow:
    """Expand every guarded agent's stage; pure, idempotent, raises when unsupported."""
    out = workflow.model_copy(deep=True)
    fork_targets = _fork_targets(out)
    per_branch = _per_branch_contexts(out)
    seen_segments: dict[str, str] = {}
    owners: dict[str, str] = {}
    for agent_id in sorted(out.nodes):
        node = out.nodes[agent_id]
        if not isinstance(node, AgentNode) or not node.guards:
            continue
        if not node.blocking:
            raise ValueError(
                f"guarded agent {agent_id!r} is blocking=False; "
                "guarded agents must be blocking=True"
            )
        segment = _agent_segment(agent_id)
        if segment in seen_segments:
            raise ValueError(
                f"guarded agents {seen_segments[segment]!r} and {agent_id!r} "
                f"share guard segment {segment!r}"
            )
        seen_segments[segment] = agent_id
        if agent_id in fork_targets:
            raise ValueError(
                f"guarded agent {agent_id!r} is a ForkNode target; a guard stage cannot "
                "run inside a fork branch"
            )
        if agent_id in per_branch:
            raise ValueError(
                f"guarded agent {agent_id!r} is inside a {per_branch[agent_id]}; a guard "
                "stage runs once per item there and would share its artifact dir across "
                "items — attach the guard to an agent on the single-run path instead"
            )
        slugs = _unique_slugs(node.guards)
        checker_ids = [f"guard_{segment}_{slug}" for slug in slugs]
        payload = guard_rules_payload(slugs, node.guards)
        payload_bytes = len(payload.encode("ascii"))
        if payload_bytes > MAX_RULES_PAYLOAD_BYTES:
            raise ValueError(
                f"guard payload for agent {agent_id!r} is {payload_bytes} bytes; "
                f"max {MAX_RULES_PAYLOAD_BYTES}"
            )

        block = _guard_block(node)
        if block:
            out.nodes[agent_id] = node.model_copy(
                update={"prompt_template": (node.prompt_template or "") + block}
            )

        reset_id = f"guard_{segment}_reset"
        fork_id = f"guard_{segment}_fork"
        join_id = f"guard_{segment}_join"
        gate_id = f"guard_{segment}_gate"
        structural = {reset_id, fork_id, join_id, gate_id}
        present = structural & out.nodes.keys()
        occupied = present | (set(checker_ids) & out.nodes.keys())
        if occupied:
            foreign = sorted(
                node_id for node_id in occupied if owners.get(node_id, agent_id) != agent_id
            )
            if foreign:
                raise _collision_error(foreign[0], agent_id, owners)
            missing = sorted(structural - present) + [
                node_id for node_id in checker_ids if node_id not in out.nodes
            ]
            if missing:
                raise ValueError(
                    f"guard stage for agent {agent_id!r} is partially present "
                    f"(missing stage ids: {', '.join(missing)}); rebuild from the "
                    "source package"
                )
            _verify_stage(out, agent_id, segment, node, slugs, checker_ids)
            continue

        stage: dict[str, NodeType] = {
            reset_id: _reset_node(segment, node),
            fork_id: ForkNode(id=fork_id, targets=checker_ids),
        }
        for checker_id, slug, guard in zip(checker_ids, slugs, node.guards, strict=True):
            stage[checker_id] = _checker_node(segment, agent_id, slug, guard)
        stage[join_id] = JoinNode(id=join_id, sources=checker_ids)
        stage[gate_id] = _gate_node(segment, slugs, node)

        succ_edges = [edge for edge in out.edges if edge.source == agent_id]
        for edge in succ_edges:
            label = edge.condition_label
            if label is not None and label.lower() != VerdictType.PROCEED.value:
                raise ValueError(
                    f"guarded agent {agent_id!r} has an out-edge labelled {label!r}; "
                    "only unconditional or PROCEED edges are supported"
                )
        out.edges = [edge for edge in out.edges if edge.source != agent_id]
        out.edges += [
            Edge(source=agent_id, target=reset_id),
            Edge(source=reset_id, target=fork_id),
            Edge(source=fork_id, target=join_id),
            Edge(source=join_id, target=gate_id),
        ]
        out.edges += [
            Edge(source=gate_id, target=edge.target, condition=VerdictType.PROCEED)
            for edge in succ_edges
        ]
        out.edges.append(Edge(source=gate_id, target=agent_id, condition=VerdictType.RELOOP))

        _add_stage_nodes(out, stage, agent_id, owners)
    return out
