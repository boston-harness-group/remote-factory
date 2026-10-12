"""Canonical workflow serialization + structural diff for parity testing.

The "Dataset of harness engineering" needs a trustworthy parity oracle when
migrating monolithic workflow definitions (``definitions.py``) to Package
compositions (``packages.py``).  Node-set ``issubset`` checks pass over real
behavioural regressions — lost reloop edges, dropped gate conditions, a missing
``terminal`` flag — because they only inspect *which* nodes exist, never *how*
they're wired or configured.

This module produces a deterministic, sorted, normalised representation of a
compiled ``Workflow`` and a structured diff between two of them, so that drift
between a monolithic mode and its composed equivalent surfaces in code review
instead of hiding behind magic-number assertions.

Only behavioural attributes are compared.  Volatile / non-serialisable fields
(``knob_values``, ``knob_bounds``, ``knob_expandable``, ``trigger``) are
excluded — they belong to the optimisation surface, not to structural parity.
"""

from __future__ import annotations

from typing import Any

from factory.workflow.primitives import Workflow


# Fields that carry behavioural meaning and should survive canonicalisation.
# Everything else (Pydantic defaults, internal book-keeping) is dropped so the
# diff stays focused on real drift.
_NODE_FIELDS: tuple[str, ...] = (
    "id",
    "_type",
    "role",
    "reads",
    "writes",
    "blocking",
    # AgentNode / Study
    "model",
    "prompt_template",
    "tools",
    "timeout",
    "max_iterations",
    "post_checks",
    # FnNode / Study
    "command",
    "callable_name",
    "notes",
    # GateNode
    "evaluator_type",
    "evaluator_role",
    "evaluator_command",
    "gate_prompt",
    # ForkNode / JoinNode
    "targets",
    "sources",
    # SubgraphForkNode (subgraph_entry/exit also used by SubgraphForkNode)
    "subgraph_entry",
    "subgraph_exit",
    "parallelism",
    "worktree_isolated",
    # SelectionNode
    "strategy",
)

# Pydantic-default values that add noise when present; strip them so two nodes
# that differ only in "explicit default vs. unset" compare equal.
_DEFAULTS = {
    "reads": set(),
    "writes": set(),
    "blocking": True,
    "model": "",
    "prompt_template": "",
    "tools": [],
    "timeout": None,
    "max_iterations": 1,
    "post_checks": [],
    "command": "",
    "callable_name": None,
    "notes": "",
    "evaluator_type": "agent",
    "evaluator_role": None,
    "evaluator_command": None,
    "gate_prompt": "",
    "targets": [],
    "sources": [],
    "subgraph_entry": "",
    "subgraph_exit": "",
    "parallelism": 3,
    "worktree_isolated": True,
    "strategy": "best_score",
}


def _canonical_node(node: Any) -> dict[str, Any]:
    """Normalise a single node to its behavioural attributes, sorted."""
    d = node.model_dump(mode="json")
    d["_type"] = type(node).__name__
    out: dict[str, Any] = {}
    for field in _NODE_FIELDS:
        if field not in d:
            continue
        val = d[field]
        if val == _DEFAULTS.get(field):
            continue
        # Sort set/list fields for determinism.
        if isinstance(val, list):
            val = sorted(val)
        out[field] = val
    return out


def _redundant_edges(wf: Workflow) -> set[tuple[str, str]]:
    """Edges the executor ignores because they duplicate node attributes.

    The executor fans out from a ``ForkNode`` via ``node.targets`` and fans in
    to a ``JoinNode`` via ``node.sources`` — the corresponding explicit edges
    are dead.  Dropping them lets a workflow that expresses fan-out as
    ``targets``/``sources`` compare equal to one that also emits explicit
    fork→child / child→join edges.
    """
    from factory.workflow.primitives import ForkNode, JoinNode

    redundant: set[tuple[str, str]] = set()
    for nid, node in wf.nodes.items():
        if isinstance(node, ForkNode):
            for t in node.targets:
                redundant.add((nid, t))
        elif isinstance(node, JoinNode):
            for s in node.sources:
                redundant.add((s, nid))
    return redundant


def canonicalize(wf: Workflow) -> dict[str, Any]:
    """Return a deterministic, sorted, JSON-safe representation of ``wf``.

    Two workflows that behave identically produce identical canonical dicts;
    any behavioural drift (missing edges, changed gate conditions, dropped
    reads/writes, altered prompts) produces a non-empty ``diff_workflows``
    result.

    Fork/Join fan-out edges are normalised away (see ``_redundant_edges``):
    the executor reads ``ForkNode.targets`` / ``JoinNode.sources``, so
    explicit edges that duplicate them are ignored for parity.
    """
    nodes = {
        nid: _canonical_node(node)
        for nid, node in sorted(wf.nodes.items())
    }
    redundant = _redundant_edges(wf)
    edges = sorted(
        (
            {
                "source": e.source,
                "target": e.target,
                "condition": e.condition_label,
            }
            for e in wf.edges
            if (e.source, e.target) not in redundant
        ),
        key=lambda e: (e["source"], e["target"], str(e["condition"])),
    )
    return {
        "name": wf.name,
        "start_node": wf.start_node,
        "terminal": wf.terminal,
        "nodes": nodes,
        "edges": edges,
    }


def diff_workflows(a: Workflow, b: Workflow) -> dict[str, Any]:
    """Structural diff of ``a`` vs ``b`` (monolithic vs composed, typically).

    Returns a dict with::

        {
          "attrs": {<field>: {"a": ..., "b": ...}, ...},   # name/start/terminal differ
          "nodes": {
            "only_a": [...], "only_b": [...],
            "changed": {<id>: {<field>: {"a": ..., "b": ...}, ...}},
          },
          "edges": {
            "only_a": [(s, t, cond), ...],
            "only_b": [(s, t, cond), ...],
          },
        }

    An empty diff (all keys empty) means the two workflows are structurally
    equivalent.  ``only_a``/``only_b`` for edges use ``(source, target,
    condition)`` tuples so a missing-vs-present *condition* on the same
    source→target pair shows up as one edge in each set.
    """
    ca = canonicalize(a)
    cb = canonicalize(b)

    # ── scalar attrs ─────────────────────────────────────────────
    attrs: dict[str, dict[str, Any]] = {}
    for field in ("name", "start_node", "terminal"):
        if ca[field] != cb[field]:
            attrs[field] = {"a": ca[field], "b": cb[field]}

    # ── nodes ────────────────────────────────────────────────────
    a_nodes = ca["nodes"]
    b_nodes = cb["nodes"]
    only_a = sorted(set(a_nodes) - set(b_nodes))
    only_b = sorted(set(b_nodes) - set(a_nodes))
    changed: dict[str, dict[str, dict[str, Any]]] = {}
    for nid in sorted(set(a_nodes) & set(b_nodes)):
        na, nb = a_nodes[nid], b_nodes[nid]
        if na == nb:
            continue
        field_diffs: dict[str, dict[str, Any]] = {}
        for field in sorted(set(na) | set(nb)):
            va, vb = na.get(field), nb.get(field)
            if va != vb:
                field_diffs[field] = {"a": va, "b": vb}
        if field_diffs:
            changed[nid] = field_diffs

    # ── edges ────────────────────────────────────────────────────
    def _edge_tuple(e: dict[str, Any]) -> tuple[str, str, Any]:
        return (e["source"], e["target"], e["condition"])

    a_edges = {_edge_tuple(e) for e in ca["edges"]}
    b_edges = {_edge_tuple(e) for e in cb["edges"]}

    return {
        "attrs": attrs,
        "nodes": {"only_a": only_a, "only_b": only_b, "changed": changed},
        "edges": {
            "only_a": sorted(a_edges - b_edges),
            "only_b": sorted(b_edges - a_edges),
        },
    }


def diff_is_empty(diff: dict[str, Any]) -> bool:
    """True if a ``diff_workflows`` result represents structural parity."""
    if diff["attrs"]:
        return False
    if diff["nodes"]["only_a"] or diff["nodes"]["only_b"] or diff["nodes"]["changed"]:
        return False
    if diff["edges"]["only_a"] or diff["edges"]["only_b"]:
        return False
    return True


def format_diff(diff: dict[str, Any]) -> str:
    """Human-readable rendering of a ``diff_workflows`` result."""
    if diff_is_empty(diff):
        return "(structural parity — no diff)"
    lines: list[str] = []
    if diff["attrs"]:
        lines.append("attrs:")
        for field, vals in diff["attrs"].items():
            lines.append(f"  {field}: a={vals['a']!r}  b={vals['b']!r}")
    nd = diff["nodes"]
    if nd["only_a"] or nd["only_b"] or nd["changed"]:
        lines.append("nodes:")
        if nd["only_a"]:
            lines.append(f"  only in a: {nd['only_a']}")
        if nd["only_b"]:
            lines.append(f"  only in b: {nd['only_b']}")
        for nid, fields in nd["changed"].items():
            lines.append(f"  changed {nid}:")
            for field, vals in fields.items():
                lines.append(f"    {field}: a={vals['a']!r}  b={vals['b']!r}")
    ed = diff["edges"]
    if ed["only_a"] or ed["only_b"]:
        lines.append("edges:")
        if ed["only_a"]:
            lines.append(f"  only in a: {ed['only_a']}")
        if ed["only_b"]:
            lines.append(f"  only in b: {ed['only_b']}")
    return "\n".join(lines)
