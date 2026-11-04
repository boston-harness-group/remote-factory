"""Agent-scoped guardrails — attach, expand, inject (INT-002, PR 1).

A guard is data: a rule plus an optional audit hint attached to one
:class:`~factory.workflow.primitives.AgentNode`. Expansion splices a generated
parallel checker stage immediately after that agent, and reloops to that agent on
violation::

    A -> guard_<seg>_reset(Fn) -> guard_<seg>_fork(Fork)
      -> {guard_<seg>_<slug>(Agent) ...} -> guard_<seg>_join(Join)
      -> guard_<seg>_gate(Gate, fn)
      -> PROCEED -> A's original successor(s)
      -> RELOOP  -> A

What does **not** get expanded (§3.4, ASM-004): expansion runs only where a flat
graph is produced through :meth:`factory.workflow.package.Package.compile`. A
hand-built ``Workflow`` that never calls ``compile()`` keeps its ``guards`` as
inert data — they serialize, they do not enforce. Guards on an agent that is a
``ForkNode`` target, on a ``blocking=False`` agent, or in a per-item context —
reachable through a ``DataNode`` branch or inside a ``SubgraphForkNode``
subgraph — are rejected at expansion (ASM-005): guard artifacts are keyed by the
agent segment alone, so a stage that runs once per item would race on the shared
``token``/``rules.json``/verdict files.

Expansion is pure (the input ``Workflow`` is untouched) and idempotent **by
ensuring, not by skipping**: the prompt block is re-appended whenever
``GUARDRAILS_HEADER`` is absent, and an already-present stage is verified against
a fresh build (``fork.targets``, ``join.sources``, ``gate.max_iterations``, the
generated reset and check commands, and every generated checker prompt) and
rejected with ``ValueError`` on any mismatch. The caller rebuilds from the source
package.

Generated ids must never collide with a node that is already in the graph: the
four structural suffixes are pre-seeded in :func:`_unique_slugs`, so a rule that
reduces to ``fork`` yields the checker id ``guard_<seg>_fork_2`` rather than the
``ForkNode`` id, and any remaining generated-id collision (e.g. two distinct
agent segments deriving the same checker id) raises ``ValueError`` naming the id
and the agents instead of overwriting a node.

Dependency direction: this module imports :mod:`factory.workflow.primitives` and
nothing else from the engine; the runtime half is the standalone
:mod:`factory.workflow.guardrails_aggregate` CLI. It is unrelated to
:mod:`factory.workflow.guard`, the structural diff checker used by verified
skill generation.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
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
"""Prompt block header appended to a guarded agent's ``prompt_template`` (§5.2)."""

GUARD_ARTIFACT_ROOT: str = ".factory/guards"
"""Project-relative root of the per-agent guard artifact dirs (§4.6)."""

MAX_RULES_PAYLOAD_BYTES: int = 65536
"""Per-agent argv budget for the ``{slug: rule}`` reset payload.

The payload is one ``execve`` argument and Linux's ``MAX_ARG_STRLEN`` is 131072
bytes; half of it leaves headroom for the interpreter path and other argv.
"""

DEFAULT_AUDIT_HINT: str = (
    "Inspect the project's actual state (git diff, git status, the produced artifacts). "
    "Do not rely on any agent's self-report. Cite the exact command output or file:line that "
    "proves or disproves the rule. If you cannot verify, treat it as violated."
)

GUARD_SLUG_RE: re.Pattern[str] = re.compile(r"^[a-z0-9_]+$")
"""One source of truth for the generated-id / artifact-filename charset (§6.3)."""

_GUARD_SLUG_MAX_LEN: int = 40
_CHECKER_ROLE: AgentRole = AgentRole.CODE_REVIEWER
# The four fixed stage-node suffixes (§4.4). A checker slug equal to one of them
# would produce the id of a structural node; the stage would then overwrite it,
# silently dropping the reset/fork/join/gate node (M1-001).
_RESERVED_STAGE_SUFFIXES: tuple[str, ...] = ("reset", "fork", "join", "gate")


# ── id derivation (§6.3, SD-004) ─────────────────────────────────


def guard_slug(rule: str) -> str:
    """Deterministic, path- and node-id-safe slug for a rule.

    Lowercases, splits on every maximal run outside ``[a-z0-9]``, joins whole
    tokens with ``_`` while the result fits in 40 chars (stopping at the first
    token that would overflow — never a mid-token cut, except that a single
    over-long *first* token is hard-truncated to 40 chars), strips trailing
    ``_`` and falls back to ``"guard"`` when nothing is left.
    """
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
    """Fold :func:`guard_slug` over attachment order, disambiguating collisions.

    The four structural stage suffixes are pre-seeded as taken, so a rule that
    reduces to exactly ``reset``/``fork``/``join``/``gate`` is disambiguated like
    any other repeated slug (``fork`` ⇒ ``fork_2``) instead of reproducing a
    ``guard_<seg>_<suffix>`` stage id. This is the SD-004 reading recorded for
    M1-001: reserved ids stay reserved, the stage keeps its ``4 + N`` shape, and
    nothing is renamed for every other rule.
    """
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
    """Error naming a generated id that is already taken (M1-001, NIT-2).

    ``owners`` maps ids generated earlier in this expansion to their agent, so the
    message can name both colliding agents instead of claiming a "partially
    present" stage. An id with no recorded owner is a node that was in the graph
    before this call.
    """
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
    """Node-id/file segment for an agent id: lowercase ``[a-z0-9_]+``, never empty."""
    segment = re.sub(r"[^a-z0-9]+", "_", agent_id.lower()).strip("_")
    if not segment:
        raise ValueError(f"guard segment for agent id {agent_id!r} is empty")
    return segment


def guard_dir(agent_id_or_segment: str) -> Path:
    """Project-relative artifact dir for an agent: ``GUARD_ARTIFACT_ROOT / segment``.

    Accepts either the raw agent id or the already-normalised segment
    (:func:`_agent_segment` is idempotent on ``[a-z0-9_]+``); raises
    ``ValueError`` when the segment is empty.
    """
    return Path(GUARD_ARTIFACT_ROOT) / _agent_segment(agent_id_or_segment)


# ── generated text (§4.5, §5.2, §6.4) ────────────────────────────


def guard_rules_payload(slugs: Sequence[str], guards: Sequence[Guard]) -> str:
    """Compact JSON object slug → rule (sorted keys, no trailing newline), UNQUOTED.

    Returns exactly ``json.dumps(dict(zip(slugs, rules)), ensure_ascii=True,
    sort_keys=True, separators=(",", ":"))``. The command builder applies
    ``shlex.quote``; the ``reset`` verb writes these canonical bytes plus a
    trailing ``"\\n"``.
    """
    return json.dumps(
        dict(zip(slugs, (guard.rule for guard in guards))),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _guard_block(node: AgentNode) -> str:
    """Block to append, or ``""`` when ``GUARDRAILS_HEADER`` is already present."""
    if GUARDRAILS_HEADER in (node.prompt_template or ""):
        return ""
    rules = "".join(f"- {guard.rule.strip()}\n" for guard in node.guards)
    return f"\n\n{GUARDRAILS_HEADER}\n{rules}"


def _checker_prompt(agent_id: str, slug: str, guard: Guard) -> str:
    """Deterministic checker prompt per §4.5 (embeds the rule + audit hint; no markers)."""
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
    """Deterministic reset FnNode command per §6.4 (embeds the quoted rules payload)."""
    payload = guard_rules_payload(_unique_slugs(node.guards), node.guards)
    return (
        f"{shlex.quote(sys.executable)} -m factory.workflow.guardrails_aggregate reset "
        f"{{project_path}} {shlex.quote(segment)} {shlex.quote(payload)}"
    )


def _check_command(segment: str, slugs: Sequence[str]) -> str:
    """Deterministic gate evaluator_command per §6.4 (segment + slugs, no rule text)."""
    slug_args = " ".join(shlex.quote(slug) for slug in slugs)
    return (
        f"{shlex.quote(sys.executable)} -m factory.workflow.guardrails_aggregate check "
        f"{{project_path}} {shlex.quote(segment)} {slug_args}"
    )


# ── generated nodes (§4.4) ───────────────────────────────────────


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


# ── attach (§6.2, R-004/SD-013) ──────────────────────────────────


def with_guards(
    node: AgentNode, guards: Sequence[Guard], *, max_iterations: int = 3
) -> AgentNode:
    """Return a copy of ``node`` with its guard list **replaced** and the budget set."""
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


# ── expansion (§5.1) ─────────────────────────────────────────────


def _fork_targets(workflow: Workflow) -> set[str]:
    targets: set[str] = set()
    for node in workflow.nodes.values():
        if isinstance(node, ForkNode):
            targets.update(node.targets)
    return targets


def _edge_index(workflow: Workflow, *, reverse: bool = False) -> dict[str, list[str]]:
    """Adjacency index: source → targets, or target → sources when ``reverse``."""
    adjacency: dict[str, list[str]] = {}
    for edge in workflow.edges:
        source, target = (edge.target, edge.source) if reverse else (edge.source, edge.target)
        adjacency.setdefault(source, []).append(target)
    return adjacency


def _reachable(workflow: Workflow, start: str, *, reverse: bool = False) -> set[str]:
    """Node ids reachable from ``start`` (inclusive), following or reversing edges."""
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
    """Node ids the executor runs once per data item: the DataNode's branch, minus joins."""
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
    """Map node id → the per-item fork context it runs in (ASM-005, rejection at expansion)."""
    contexts: dict[str, str] = {}
    for node_id, node in workflow.nodes.items():
        if isinstance(node, DataNode):
            for member in _datanode_branch_region(workflow, node_id):
                contexts.setdefault(member, f"DataNode '{node_id}' branch")
        elif isinstance(node, SubgraphForkNode):
            entry, exit_id = node.subgraph_entry, node.subgraph_exit
            if entry not in workflow.nodes or exit_id not in workflow.nodes:
                continue
            # Mirror the executor: only nodes on paths entry → exit run in the branch.
            region = _reachable(workflow, entry) & _reachable(workflow, exit_id, reverse=True)
            for member in region:
                contexts.setdefault(member, f"SubgraphForkNode '{node_id}' subgraph")
    return contexts


def _verify_stage(
    workflow: Workflow,
    agent_id: str,
    segment: str,
    node: AgentNode,
    slugs: list[str],
    checker_ids: list[str],
) -> None:
    """Raise when a present stage disagrees with the node's *current* guards (R-006)."""
    reset = workflow.nodes[f"guard_{segment}_reset"]
    fork = workflow.nodes[f"guard_{segment}_fork"]
    join = workflow.nodes[f"guard_{segment}_join"]
    gate = workflow.nodes[f"guard_{segment}_gate"]
    stale = (
        not isinstance(reset, FnNode)
        or not isinstance(fork, ForkNode)
        or not isinstance(join, JoinNode)
        or not isinstance(gate, GateNode)
        or list(fork.targets) != checker_ids
        or list(join.sources) != checker_ids
        or gate.max_iterations != node.guard_max_iterations
        or reset.command != _reset_command(segment, node)
        or gate.evaluator_command != _check_command(segment, slugs)
    )
    for checker_id, slug, guard in zip(checker_ids, slugs, node.guards, strict=True):
        checker = workflow.nodes[checker_id]
        if not isinstance(checker, AgentNode):
            stale = True
            break
        if checker.prompt_template != _checker_prompt(agent_id, slug, guard):
            stale = True
            break
    if stale:
        raise ValueError(
            f"guard stage for agent {agent_id!r} exists but does not match its current "
            "guards; rebuild from the source package"
        )


def _add_stage_nodes(
    workflow: Workflow,
    stage: dict[str, NodeType],
    agent_id: str,
    owners: dict[str, str],
) -> None:
    """Insert a generated stage, refusing to overwrite any existing node (M1-001).

    The presence gate in :func:`expand_agent_guards` already routes a complete
    same-stage id set to :func:`_verify_stage`, so every generated id is new when
    this is called today. The check stays because a silent ``dict`` overwrite is
    the exact M1-001 failure mode: a checker id that equals ``guard_<seg>_fork``
    (or two segments deriving the same checker id) must fail loudly even if that
    gate is ever edited.
    """
    for node_id in stage:
        if node_id in workflow.nodes:
            raise _collision_error(node_id, agent_id, owners)
    workflow.nodes.update(stage)
    owners.update({node_id: agent_id for node_id in stage})


def expand_agent_guards(workflow: Workflow) -> Workflow:
    """Splice one guard-checker stage after every guarded ``AgentNode`` (R-005).

    Pure: the input workflow is untouched, and the caller receives a deep copy.
    The stage is spliced only when absent; when present and consistent with the
    node's current guards it is left untouched, but the prompt block is still
    ensured (R-006). Any partial or stale stage raises ``ValueError``, as does a
    guard on an unsupported placement (``ForkNode`` target, ``blocking=False``, or
    inside a ``DataNode`` branch / ``SubgraphForkNode`` subgraph).

    The generated reset/check commands embed ``sys.executable``, so a persisted
    expanded graph is not portable across interpreter paths; rebuild from the
    source package instead of reusing a serialized expanded graph elsewhere.
    """
    out = workflow.model_copy(deep=True)
    fork_targets = _fork_targets(out)
    per_branch = _per_branch_contexts(out)
    seen_segments: dict[str, str] = {}
    # Generated id → agent that generated it, so a collision can name both sides.
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

        # 1. Prompt ENSURE (not skip): repair a block wiped by a PROMPT_MUTATE.
        block = _guard_block(node)
        if block:
            out.nodes[agent_id] = node.model_copy(
                update={"prompt_template": (node.prompt_template or "") + block}
            )

        # 2. Splice the stage when absent; verify (never silently reuse) when present.
        reset_id = f"guard_{segment}_reset"
        fork_id = f"guard_{segment}_fork"
        join_id = f"guard_{segment}_join"
        gate_id = f"guard_{segment}_gate"
        structural = {reset_id, fork_id, join_id, gate_id}
        present = structural & out.nodes.keys()
        occupied = present | (set(checker_ids) & out.nodes.keys())
        if occupied:
            # An occupied id that another agent's stage generated in this same call
            # is a cross-agent collision, not a partial stage: name it (M1-001).
            foreign = sorted(
                node_id for node_id in occupied if owners.get(node_id, agent_id) != agent_id
            )
            if foreign:
                raise _collision_error(foreign[0], agent_id, owners)
            if present != structural or not set(checker_ids) <= out.nodes.keys():
                raise ValueError(
                    f"guard stage for agent {agent_id!r} is partially present "
                    f"(structural ids present: {sorted(present)}); rebuild from the "
                    "source package"
                )
            _verify_stage(out, agent_id, segment, node, slugs, checker_ids)
            continue

        # 3. Build the whole stage before touching the graph. The presence gate above
        # already routes a complete same-stage id set to _verify_stage, so every
        # generated id is new today; `_add_stage_nodes` below is the guard that makes
        # the silent-overwrite failure mode (a checker id equal to guard_<seg>_fork,
        # or two segments deriving the same checker id) impossible even if that gate
        # is edited (M1-001).
        stage: dict[str, NodeType] = {
            reset_id: _reset_node(segment, node),
            fork_id: ForkNode(id=fork_id, targets=checker_ids),
        }
        for checker_id, slug, guard in zip(checker_ids, slugs, node.guards, strict=True):
            stage[checker_id] = _checker_node(segment, agent_id, slug, guard)
        stage[join_id] = JoinNode(id=join_id, sources=checker_ids)
        stage[gate_id] = _gate_node(segment, slugs, node)

        # 4. Resplice outgoing edges — rewrite, never append a second out-edge.
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

        # 5. Add the stage nodes (the helper re-checks that no id is taken).
        _add_stage_nodes(out, stage, agent_id, owners)
    return out
