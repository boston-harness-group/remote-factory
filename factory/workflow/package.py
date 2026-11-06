"""Package ecosystem — composable workflow subgraphs with typed interfaces.

A Package wraps a workflow subgraph behind a typed interface of Ports
(data plane) and a StateContract (control plane). Packages compose via
Sequential, Parallel, Conditional, and Loop operators.

Three representations:
- Author-time: Python compositions (readable, version-controlled)
- Optimize-time: compiled to mutable Workflow IR via compile()
- Distribution-time: serialized via save()/load() as package.toml + graph.json
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from factory.workflow.guardrails import expand_agent_guards
from factory.workflow.primitives import (
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    NodeType,
    VerdictType,
    Workflow,
)


# ── interface types ────────────────────────────────────────────────


class Port(BaseModel):
    """A named artifact slot — the data plane of a package interface."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    artifact_path: str
    media_type: str = "text/markdown"


class StateContract(BaseModel):
    """Preconditions and postconditions on project state — the control plane."""

    model_config = ConfigDict(strict=True, extra="forbid")

    requires: frozenset[str] = frozenset()
    produces: frozenset[str] = frozenset()
    capabilities: list[str] = Field(default_factory=list)


class OptKnob(BaseModel):
    """A parameter the outer loop is allowed to mutate.

    When ``expandable=True``, the outer loop may propose values beyond
    the initial ``bounds``.  For prompt knobs this means authoring new
    prompt text; for threshold knobs it means extrapolating the range.
    ``expansion_hint`` tells the optimizer how to generate new values.
    ``description`` is prose for humans and carries no optimizer meaning.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    kind: Literal["prompt", "model", "threshold", "topology"]
    node_id: str
    default: str | float
    bounds: list[str | float] = Field(default_factory=list)
    expandable: bool = False
    expansion_hint: str = ""
    description: str = ""


class MemoryDeclaration(BaseModel):
    """Declares what a package persists and how it should be stored."""

    model_config = ConfigDict(strict=True, extra="forbid")

    namespace: str
    kind: Literal["kv", "vector", "graph", "log"]
    schema_def: dict[str, str] = Field(default_factory=dict)
    retention: Literal["ephemeral", "run", "persistent"] = "persistent"


# ── the package primitive ──────────────────────────────────────────


class Package(BaseModel):
    """A composable workflow subgraph with a typed interface.

    The nn.Module of factory workflows. Declares typed inputs/outputs,
    hides internal graph structure, exposes tunable knobs, and composes
    with other packages via Sequential/Parallel/Conditional/Loop.
    """

    model_config = ConfigDict(strict=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    version: str = "0.0.0"
    description: str = ""

    inputs: list[Port] = Field(default_factory=list)
    outputs: list[Port] = Field(default_factory=list)
    contract: StateContract = Field(default_factory=StateContract)

    graph: Workflow
    entry_node: str
    exit_node: str

    knobs: list[OptKnob] = Field(default_factory=list)
    frozen: bool = False
    memory: list[MemoryDeclaration] = Field(default_factory=list)
    terminal: bool = False

    @property
    def input_paths(self) -> set[str]:
        return {p.artifact_path for p in self.inputs}

    @property
    def output_paths(self) -> set[str]:
        return {p.artifact_path for p in self.outputs}

    def configure(self, **knob_values: str | float) -> Package:
        """Return a copy with knob defaults overridden."""
        new_knobs = []
        for k in self.knobs:
            if k.name in knob_values:
                new_knobs.append(k.model_copy(update={"default": knob_values[k.name]}))
            else:
                new_knobs.append(k)
        return self.model_copy(update={"knobs": new_knobs})

    def compile(self) -> Workflow:
        """Lower this package to a flat, mutable Workflow IR.

        Prompt knobs (``_prompt_<node_id>`` in knob_values) are applied
        back to node prompt_templates so that PROMPT_MUTATE mutations
        survive compile() round-trips.
        """
        wf = self.graph.model_copy(deep=True)
        # Preserve _prompt_* entries from previous mutations before overwriting
        saved_prompts = {
            k: v for k, v in wf.knob_values.items()
            if k.startswith("_prompt_")
        }
        saved_expandable = {
            k: v for k, v in wf.knob_expandable.items()
            if k.startswith("_prompt_")
        }
        if self.knobs:
            wf.knob_values = {k.name: k.default for k in self.knobs}
            wf.knob_bounds = {k.name: list(k.bounds) for k in self.knobs if k.bounds}
            wf.knob_expandable = {
                k.name: k.expansion_hint for k in self.knobs if k.expandable
            }
            wf.knob_specs = {
                k.name: {
                    "kind": k.kind,
                    "node_id": k.node_id,
                    "default": k.default,
                    "bounds": list(k.bounds),
                    "expandable": k.expandable,
                    "expansion_hint": k.expansion_hint,
                    "description": k.description,
                }
                for k in self.knobs
            }
        wf.knob_values.update(saved_prompts)
        wf.knob_expandable.update(saved_expandable)
        for key, val in list(wf.knob_values.items()):
            if key.startswith("_prompt_") and isinstance(val, str):
                node_id = key[len("_prompt_"):]
                node = wf.nodes.get(node_id)
                if node and hasattr(node, "prompt_template"):
                    try:
                        wf.nodes[node_id] = node.model_copy(
                            update={"prompt_template": val}
                        )
                    except Exception:
                        pass
        wf.terminal = self.terminal
        if self.contract.capabilities:
            wf.declared_capabilities = frozenset(self.contract.capabilities)
        return expand_agent_guards(wf)


# ── composition operators ──────────────────────────────────────────


def _merge_graphs(
    packages: list[Package],
    extra_edges: list[Edge] | None = None,
    *,
    name: str,
    start_node: str,
) -> Workflow:
    """Merge multiple package graphs into a single flat Workflow."""
    all_nodes: dict[str, NodeType] = {}
    all_edges: list[Edge] = []

    for pkg in packages:
        for nid, node in pkg.graph.nodes.items():
            if nid in all_nodes:
                raise ValueError(
                    f"Node ID collision: '{nid}' exists in multiple packages. "
                    "Use unique node IDs or namespace them by package."
                )
            all_nodes[nid] = node
        all_edges.extend(pkg.graph.edges)

    if extra_edges:
        all_edges.extend(extra_edges)

    return Workflow(
        name=name,
        nodes=all_nodes,
        edges=all_edges,
        start_node=start_node,
    )


def _merge_contracts(packages: list[Package]) -> StateContract:
    """Merge state contracts: union of requires/produces/capabilities."""
    requires: set[str] = set()
    produces: set[str] = set()
    capabilities: list[str] = []
    seen_caps: set[str] = set()

    for pkg in packages:
        requires |= pkg.contract.requires
        produces |= pkg.contract.produces
        for cap in pkg.contract.capabilities:
            if cap not in seen_caps:
                capabilities.append(cap)
                seen_caps.add(cap)

    requires -= produces

    return StateContract(
        requires=frozenset(requires),
        produces=frozenset(produces),
        capabilities=capabilities,
    )


def Sequential(
    *packages: Package,
    name: str = "",
    extra_edges: list[Edge] | None = None,
    terminal: bool | None = None,
) -> Package:
    """Compose packages in sequence: exit of A wires to entry of B.

    ``extra_edges`` lets a composition declare cross-package back-edges that
    plain sequential wiring can't express — e.g. a QA gate in a later package
    relooping to a builder node in an earlier one.  These are forwarded to the
    graph merge alongside the forward bridge edges.

    Bridge edges (exit → next entry) carry ``PROCEED`` when the exit node is a
    ``GateNode``: a gate's forward path is its PROCEED branch, and its RELOOP
    branch is internal to the package.  Non-gate exit nodes keep an
    unconditional bridge, matching prior behaviour.
    """
    pkg_list = list(packages)
    if not pkg_list:
        raise ValueError("Sequential requires at least one package")
    if len(pkg_list) == 1:
        return pkg_list[0]

    bridge_edges = []
    for i in range(len(pkg_list) - 1):
        src_pkg = pkg_list[i]
        exit_node = src_pkg.graph.nodes.get(src_pkg.exit_node)
        condition = VerdictType.PROCEED if isinstance(exit_node, GateNode) else None
        bridge_edges.append(
            Edge(
                source=src_pkg.exit_node,
                target=pkg_list[i + 1].entry_node,
                condition=condition,
            )
        )
    if extra_edges:
        bridge_edges.extend(extra_edges)

    composed_name = name or "seq_" + "_".join(p.name for p in pkg_list)
    graph = _merge_graphs(
        pkg_list,
        bridge_edges,
        name=composed_name,
        start_node=pkg_list[0].entry_node,
    )

    all_inputs = pkg_list[0].inputs
    all_outputs = pkg_list[-1].outputs
    all_knobs = [k for p in pkg_list for k in p.knobs]
    all_memory = [m for p in pkg_list for m in p.memory]
    is_terminal = terminal if terminal is not None else any(p.terminal for p in pkg_list)

    return Package(
        name=composed_name,
        inputs=all_inputs,
        outputs=all_outputs,
        contract=_merge_contracts(pkg_list),
        graph=graph,
        entry_node=pkg_list[0].entry_node,
        exit_node=pkg_list[-1].exit_node,
        knobs=all_knobs,
        memory=all_memory,
        terminal=is_terminal,
    )


class JoinStrategy(str):
    CONCATENATE = "concatenate"
    BEST_SCORE = "best_score"
    ALL_MUST_PASS = "all_must_pass"


def Parallel(
    *packages: Package,
    join: str = JoinStrategy.CONCATENATE,
    name: str = "",
) -> Package:
    """Fork into N packages in parallel, join results."""
    pkg_list = list(packages)
    if not pkg_list:
        raise ValueError("Parallel requires at least one package")

    composed_name = name or "par_" + "_".join(p.name for p in pkg_list)
    fork_id = f"fork_{composed_name}"
    join_id = f"join_{composed_name}"

    fork_node = ForkNode(
        id=fork_id,
        targets=[p.entry_node for p in pkg_list],
    )

    join_node = JoinNode(
        id=join_id,
        sources=[p.exit_node for p in pkg_list],
        reads={path for p in pkg_list for path in p.output_paths},
    )

    all_nodes: dict[str, NodeType] = {fork_id: fork_node, join_id: join_node}
    all_edges: list[Edge] = [Edge(source=fork_id, target=join_id)]

    for pkg in pkg_list:
        for nid, node in pkg.graph.nodes.items():
            if nid in all_nodes and nid not in (fork_id, join_id):
                raise ValueError(
                    f"Node ID collision: '{nid}' exists in multiple packages. "
                    "Use unique node IDs or namespace them by package."
                )
            all_nodes[nid] = node
        all_edges.extend(pkg.graph.edges)

    graph = Workflow(
        name=composed_name,
        nodes=all_nodes,
        edges=all_edges,
        start_node=fork_id,
    )

    all_inputs = [inp for p in pkg_list for inp in p.inputs]
    all_outputs = [out for p in pkg_list for out in p.outputs]
    all_knobs = [k for p in pkg_list for k in p.knobs]
    all_memory = [m for p in pkg_list for m in p.memory]

    return Package(
        name=composed_name,
        inputs=all_inputs,
        outputs=all_outputs,
        contract=_merge_contracts(pkg_list),
        graph=graph,
        entry_node=fork_id,
        exit_node=join_id,
        knobs=all_knobs,
        memory=all_memory,
    )


def Conditional(
    gate: GateNode,
    branches: dict[str, Package],
    *,
    name: str = "",
) -> Package:
    """Route to one of several packages based on a gate verdict.

    A branch label is normally a ``VerdictType`` name (``PROCEED`` / ``HALT`` /
    ``RELOOP``), but any label is accepted: the gate's evaluator may name its own
    outcomes (``"DRAFT"``, ``"IMPROVE"``, ``"DEBUG"``), which become the edge's
    ``condition`` in lowercase. Consumers match conditions case-insensitively.
    """
    composed_name = name or "cond_" + "_".join(branches.keys())
    pkg_list = list(branches.values())

    exit_id = f"join_{composed_name}"
    exit_node = FnNode(id=exit_id, command="true", notes="Conditional join point")

    all_nodes: dict[str, NodeType] = {gate.id: gate, exit_id: exit_node}
    all_edges: list[Edge] = []

    condition_map: dict[str, str] = {
        "PROCEED": VerdictType.PROCEED,
        "HALT": VerdictType.HALT,
        "RELOOP": VerdictType.RELOOP,
    }

    for label, pkg in branches.items():
        for nid, node in pkg.graph.nodes.items():
            all_nodes[nid] = node
        all_edges.extend(pkg.graph.edges)

        condition = condition_map.get(label, label.lower())
        all_edges.append(Edge(source=gate.id, target=pkg.entry_node, condition=condition))
        all_edges.append(Edge(source=pkg.exit_node, target=exit_id))

    graph = Workflow(
        name=composed_name,
        nodes=all_nodes,
        edges=all_edges,
        start_node=gate.id,
    )

    all_inputs = [inp for p in pkg_list for inp in p.inputs]
    all_outputs = [out for p in pkg_list for out in p.outputs]
    all_knobs = [k for p in pkg_list for k in p.knobs]
    all_memory = [m for p in pkg_list for m in p.memory]

    return Package(
        name=composed_name,
        inputs=all_inputs,
        outputs=all_outputs,
        contract=_merge_contracts(pkg_list),
        graph=graph,
        entry_node=gate.id,
        exit_node=exit_id,
        knobs=all_knobs,
        memory=all_memory,
    )


def Loop(
    body: Package,
    gate: GateNode,
    max_iterations: int = 5,
    *,
    name: str = "",
) -> Package:
    """Wrap a package with a gate that re-enters on RELOOP."""
    composed_name = name or f"loop_{body.name}"
    exit_id = f"exit_{composed_name}"
    exit_node = FnNode(id=exit_id, command="true", notes="Loop exit point")

    all_nodes: dict[str, NodeType] = {gate.id: gate, exit_id: exit_node}
    all_edges: list[Edge] = []

    for nid, node in body.graph.nodes.items():
        all_nodes[nid] = node
    all_edges.extend(body.graph.edges)

    all_edges.append(Edge(source=body.exit_node, target=gate.id))
    all_edges.append(Edge(source=gate.id, target=body.entry_node, condition=VerdictType.RELOOP))
    all_edges.append(Edge(source=gate.id, target=exit_id, condition=VerdictType.PROCEED))

    graph = Workflow(
        name=composed_name,
        nodes=all_nodes,
        edges=all_edges,
        start_node=body.entry_node,
    )

    return Package(
        name=composed_name,
        inputs=body.inputs,
        outputs=body.outputs,
        contract=body.contract,
        graph=graph,
        entry_node=body.entry_node,
        exit_node=exit_id,
        knobs=body.knobs,
        memory=body.memory,
    )
