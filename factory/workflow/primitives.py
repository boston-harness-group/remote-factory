"""Workflow graph primitives — composable types for factory orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── project state (inlined so the graph language has zero runtime deps) ──


class ProjectState(str, Enum):
    """The five possible states of a target project."""

    NO_REPO = "no_repo"
    REPO_INCOMPLETE = "incomplete"
    NO_FACTORY = "no_factory"
    EVALS_PENDING_REVIEW = "evals_pending_review"
    HAS_FACTORY = "has_factory"


# ── agent pool ───────────────────────────────────────────────────


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    STRATEGIST = "strategist"
    BUILDER = "builder"
    HEALTH_CHECKER = "health_checker"
    CODE_REVIEWER = "code_reviewer"
    ADVERSARIAL_TESTER = "adversarial_tester"
    FAILURE_ANALYST = "failure_analyst"
    CEO = "ceo"
    ARCHIVIST = "archivist"
    REFINER = "refiner"


class AgentConfig(BaseModel):
    """Configuration for an agent in the pool."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: AgentRole
    model: str
    timeout: int = 600


DEFAULT_AGENT_POOL: dict[str, AgentConfig] = {
    "researcher": AgentConfig(role=AgentRole.RESEARCHER, model="sonnet", timeout=600),
    "strategist": AgentConfig(role=AgentRole.STRATEGIST, model="opus", timeout=600),
    "builder": AgentConfig(role=AgentRole.BUILDER, model="opus", timeout=1200),
    "health_checker": AgentConfig(role=AgentRole.HEALTH_CHECKER, model="opus", timeout=600),
    "code_reviewer": AgentConfig(role=AgentRole.CODE_REVIEWER, model="opus", timeout=900),
    "adversarial_tester": AgentConfig(
        role=AgentRole.ADVERSARIAL_TESTER, model="opus", timeout=1800
    ),
    "failure_analyst": AgentConfig(role=AgentRole.FAILURE_ANALYST, model="opus", timeout=600),
    "ceo": AgentConfig(role=AgentRole.CEO, model="opus", timeout=3600),
    "archivist": AgentConfig(role=AgentRole.ARCHIVIST, model="haiku", timeout=300),
    "refiner": AgentConfig(role=AgentRole.REFINER, model="opus", timeout=600),
}


# ── verdicts ─────────────────────────────────────────────────────


class VerdictType(str, Enum):
    PROCEED = "proceed"
    RELOOP = "reloop"
    HALT = "halt"


class Verdict(BaseModel):
    """Algebraic verdict type: Proceed | Reloop(target, feedback, max_iterations) | Halt(reason)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: VerdictType
    target: str | None = None
    feedback: str | None = None
    max_iterations: int = 3
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_variant(self) -> Verdict:
        if self.type == VerdictType.RELOOP:
            if not self.target:
                raise ValueError("Reloop verdict requires a target node")
        if self.type == VerdictType.HALT:
            if not self.reason:
                raise ValueError("Halt verdict requires a reason")
        return self

    @staticmethod
    def proceed() -> Verdict:
        return Verdict(type=VerdictType.PROCEED)

    @staticmethod
    def reloop(target: str, feedback: str, max_iterations: int = 3) -> Verdict:
        return Verdict(
            type=VerdictType.RELOOP,
            target=target,
            feedback=feedback,
            max_iterations=max_iterations,
        )

    @staticmethod
    def halt(reason: str) -> Verdict:
        return Verdict(type=VerdictType.HALT, reason=reason)


# ── nodes ────────────────────────────────────────────────────────


class Node(BaseModel):
    """Base node in the workflow graph."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    reads: set[str] = Field(default_factory=set)
    writes: set[str] = Field(default_factory=set)
    blocking: bool = True


class ArtifactCheck(BaseModel):
    """Validation rule for an agent-produced artifact."""

    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    must_exist: bool = True
    min_size: int = 0
    must_contain: list[str] = Field(default_factory=list)


class AgentNode(Node):
    """Node that invokes a Claude Code agent."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: AgentRole
    model: str = ""
    prompt_template: str = ""
    tools: list[str] = Field(default_factory=list)
    timeout: int | None = None
    max_iterations: int = 1
    post_checks: list[ArtifactCheck] = Field(default_factory=list)


class FnNode(Node):
    """Node that runs a deterministic shell command or Python callable."""

    model_config = ConfigDict(strict=True, extra="forbid")

    command: str = ""
    callable_name: str | None = None
    notes: str = ""


class GateNode(Node):
    """Decision node that produces a Verdict."""

    model_config = ConfigDict(strict=True, extra="forbid")

    evaluator_type: Literal["agent", "fn", "user"] = "agent"
    evaluator_role: AgentRole | None = None
    evaluator_command: str | None = None
    gate_prompt: str = ""
    max_iterations: int | None = None
    """Cap for this gate's reloop edge. ``None`` leaves the cap to the runtime."""


class ForkNode(Node):
    """Parallel execution node — launches all targets concurrently."""

    model_config = ConfigDict(strict=True, extra="forbid")

    targets: list[str]


class JoinNode(Node):
    """Barrier node — waits for all sources to complete."""

    model_config = ConfigDict(strict=True, extra="forbid")

    sources: list[str]


class SubgraphForkNode(Node):
    """Fan-out to N copies of a subgraph, each in an isolated worktree.

    The executor creates independent WorkflowExecutor instances per branch,
    each with its own worktree branching from the same base commit.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    subgraph_entry: str
    subgraph_exit: str
    parallelism: int = 3
    worktree_isolated: bool = True


class DataItem(BaseModel):
    """Standardized data payload for per-item workflow iteration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""


class DataNode(Node):
    """Node that loads data items and drives per-item execution of a subgraph."""

    model_config = ConfigDict(strict=True, extra="forbid")

    task_ref: str | None = None
    source_path: str | None = None
    source_format: Literal["directory", "jsonl", "csv"] | None = None
    inline_items: list[DataItem] = Field(default_factory=list)
    subgraph_entry: str
    subgraph_exit: str
    parallelism: int = Field(default=1, ge=1)
    split: Literal["train", "val", "test", "all"] = "all"
    shuffle: bool = False
    shuffle_seed: int | None = None
    limit: int | None = None
    max_items: int = 500

    @model_validator(mode="after")
    def _validate_source(self) -> DataNode:
        sources = [
            self.task_ref is not None,
            self.source_path is not None,
            bool(self.inline_items),
        ]
        if sum(sources) > 1:
            raise ValueError(
                "At most one of task_ref, source_path, or inline_items may be set"
            )
        if self.source_path is not None and self.source_format is None:
            raise ValueError("source_path requires source_format to be set")
        return self


class SelectionNode(Node):
    """Compare N completed experiment branches and select the best."""

    model_config = ConfigDict(strict=True, extra="forbid")

    strategy: Literal["best_score"] = "best_score"


class Study(FnNode):
    """Distinguished FnNode wrapping `factory study`."""

    model_config = ConfigDict(strict=True, extra="forbid")

    focus: str | None = None


class ToolDef(BaseModel):
    """A tool available to the LLM during a tool-use loop."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    executor: Literal["bash", "file_read", "file_write", "file_edit"] = "bash"


class LLMNode(Node):
    """Node that makes direct LLM API calls with a configurable tool-use loop.

    Unlike AgentNode (full CLI subprocess), this runs the API loop in-process
    with a minimal, configurable tool set. ``model`` and ``provider`` default to
    neutral values (``""`` / ``"auto"``): a graph that does not pin them leaves
    the choice to the runtime, so the same graph runs on any provider.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    system_prompt: str = ""
    instance_prompt: str = ""
    model: str = ""
    provider: Literal["auto", "anthropic", "vertex", "litellm"] = "auto"
    max_tokens: int = 8192
    max_turns: int = 50
    temperature: float = 0.0
    stop_sequences: list[str] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    tool_choice: Literal["auto", "any", "none"] = "auto"
    timeout: int = 600


# ── edges ────────────────────────────────────────────────────────


class Edge(BaseModel):
    """Directed edge in the workflow graph with an optional labelled condition.

    ``condition`` is normally one of the three ``VerdictType`` labels, but a gate
    may name its own forward outcomes (a switch rather than a binary decision).
    Arbitrary labels are stored verbatim and lowercased by consumers before
    matching, so ``"DRAFT"`` from an evaluator matches ``condition="draft"``.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    source: str
    target: str
    condition: VerdictType | str | None = None

    @property
    def condition_label(self) -> str | None:
        """Return the condition as a plain string, or ``None``.

        ``VerdictType.PROCEED`` → ``"proceed"`` (not ``"VerdictType.PROCEED"``).
        Plain ``str`` conditions are returned as-is.
        """
        if self.condition is None:
            return None
        return self.condition.value if isinstance(self.condition, VerdictType) else self.condition


# ── workflow ─────────────────────────────────────────────────────


NodeType = (
    AgentNode | FnNode | GateNode | ForkNode | JoinNode | SubgraphForkNode
    | SelectionNode | Study | LLMNode | DataNode
)


TriggerFn = Callable[[ProjectState, dict[str, Any]], bool]


class Workflow(BaseModel):
    """A directed graph of typed nodes with labeled edges and a state-based trigger."""

    model_config = ConfigDict(strict=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    nodes: dict[str, NodeType]
    edges: list[Edge]
    start_node: str
    terminal: bool = False
    task: str | None = Field(default=None)
    trigger: TriggerFn | None = Field(default=None, exclude=True)
    knob_values: dict[str, str | float] = Field(default_factory=dict)
    knob_bounds: dict[str, list[str | float]] = Field(default_factory=dict)
    knob_expandable: dict[str, str] = Field(default_factory=dict)
    # Full OptKnob metadata (kind, node_id, description, …) so the IR is the
    # complete contract for the outer loop: it must know what each knob tunes
    # and what kind of mutation is legal, not just the value and bounds.
    knob_specs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    declared_capabilities: frozenset[str] = frozenset()

    def validate_graph(self) -> list[str]:
        """Validate workflow graph structure using NetworkX. Returns list of issues."""
        from factory.workflow.validation import validate_workflow

        return validate_workflow(self)

    def subgraph(
        self,
        node_ids: set[str],
        *,
        name: str,
        start_node: str,
    ) -> Workflow:
        """Extract a subgraph containing only the specified nodes.

        Deep-copies requested nodes and filters edges to only those
        where both source and target are in node_ids.
        """
        nodes: dict[str, NodeType] = {}
        for nid in node_ids:
            if nid not in self.nodes:
                raise ValueError(f"node '{nid}' not found in workflow '{self.name}'")
            nodes[nid] = self.nodes[nid].model_copy(deep=True)
        edges = [
            e.model_copy(deep=True)
            for e in self.edges
            if e.source in node_ids and e.target in node_ids
        ]
        return Workflow(name=name, nodes=nodes, edges=edges, start_node=start_node)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow to a JSON-safe dict.

        ``reads`` and ``writes`` are sets, whose iteration order is not stable
        across processes; they are sorted here so that a given graph always
        serializes to identical bytes.
        """
        nodes_out: dict[str, Any] = {}
        for nid, node in self.nodes.items():
            d = node.model_dump(mode="json")
            for field_name in ("reads", "writes"):
                value = d.get(field_name)
                if isinstance(value, list):
                    d[field_name] = sorted(value)
            d["_type"] = type(node).__name__
            nodes_out[nid] = d

        edges_out = [e.model_dump(mode="json") for e in self.edges]

        result: dict[str, Any] = {
            "name": self.name,
            "nodes": nodes_out,
            "edges": edges_out,
            "start_node": self.start_node,
            "terminal": self.terminal,
        }
        if self.task is not None:
            result["task"] = self.task
        if self.knob_values:
            result["knob_values"] = dict(self.knob_values)
        if self.knob_bounds:
            result["knob_bounds"] = {k: list(v) for k, v in self.knob_bounds.items()}
        if self.knob_expandable:
            result["knob_expandable"] = dict(self.knob_expandable)
        if self.knob_specs:
            result["knob_specs"] = {k: dict(v) for k, v in self.knob_specs.items()}
        if self.declared_capabilities:
            result["declared_capabilities"] = sorted(self.declared_capabilities)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        """Reconstruct a Workflow from a dict produced by ``to_dict``."""
        _NODE_TYPE_MAP: dict[str, type[Node]] = {
            "AgentNode": AgentNode,
            "FnNode": FnNode,
            "GateNode": GateNode,
            "ForkNode": ForkNode,
            "JoinNode": JoinNode,
            "SubgraphForkNode": SubgraphForkNode,
            "SelectionNode": SelectionNode,
            "Study": Study,
            "LLMNode": LLMNode,
            "DataNode": DataNode,
        }
        _SET_FIELDS = {"reads", "writes"}

        nodes: dict[str, NodeType] = {}
        for nid, node_data in data["nodes"].items():
            node_data = dict(node_data)
            type_name = node_data.pop("_type", "FnNode")
            node_cls = _NODE_TYPE_MAP.get(type_name)
            if node_cls is None:
                raise ValueError(f"Unknown node type: {type_name}")
            for fld in _SET_FIELDS:
                if fld in node_data and isinstance(node_data[fld], list):
                    node_data[fld] = set(node_data[fld])
            nodes[nid] = node_cls.model_validate(node_data, strict=False)  # type: ignore[assignment]

        edges = [Edge.model_validate(e, strict=False) for e in data["edges"]]

        return cls(
            name=data["name"],
            nodes=nodes,
            edges=edges,
            start_node=data["start_node"],
            terminal=data.get("terminal", False),
            task=data.get("task"),
            knob_values=data.get("knob_values", {}),
            knob_bounds={k: list(v) for k, v in data.get("knob_bounds", {}).items()},
            knob_expandable=data.get("knob_expandable", {}),
            knob_specs=data.get("knob_specs", {}),
            declared_capabilities=frozenset(data.get("declared_capabilities", [])),
        )


# ── plugin role registration ─────────────────────────────────────


# Roles registered through register_agent_role, keyed by value. Lets
# re-registration be idempotent while builtin collisions still raise.
_registered_plugin_roles: dict[str, AgentRole] = {}


def _rebuild_role_schemas() -> None:
    """Rebuild the Pydantic core schemas of models defined in this module.

    Pydantic freezes an enum's valid values into the model's core schema at
    class-definition time, so a newly added ``AgentRole`` member is invisible
    to validation until every model referencing it is rebuilt. Iterating in
    ``vars()`` order visits children before the models that embed them
    (``Workflow``, ``Factory``), which is the order the schema graph needs.
    """
    import sys

    module = sys.modules[__name__]
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            obj.model_rebuild(force=True)


def register_agent_role(role: str, name: str | None = None) -> AgentRole:
    """Register a plugin-defined agent role, making it usable in workflow graphs.

    Extends ``AgentRole`` with a new member so the role can appear in
    ``AgentNode.role`` and ``GateNode.evaluator_role`` — plain enum mutation
    (the workaround plugins used before this API) leaves Pydantic's frozen
    schema stale, and workflows containing the role then fail JSON
    validation. This function performs the insertion and rebuilds the
    affected schemas, so serialization roundtrips work.

    ``role`` is the role value used in graphs, prompts, and the CLI (e.g.
    ``"paper-reader"``). The enum member name is derived from it (``PAPER_READER``)
    unless ``name`` is given explicitly.

    Idempotent: re-registering an existing name with the same value returns
    the existing member. Raises ``ValueError`` on name or value collisions
    with existing members, or when the derived member name is not a valid
    Python identifier.
    """
    if not role or not role.strip():
        raise ValueError("agent role value must be a non-empty string")
    role = role.strip()
    if name is None:
        name = role.upper().replace("-", "_")
    if not name.isidentifier():
        raise ValueError(
            f"agent role {role!r} does not map to a valid enum member name ({name!r})"
        )

    previously = _registered_plugin_roles.get(role)
    if previously is not None:
        if previously.name != name:
            raise ValueError(
                f"agent role {role!r} is already registered with member name "
                f"{previously.name!r}, not {name!r}"
            )
        return previously

    by_name = AgentRole._member_map_.get(name)
    if by_name is not None:
        raise ValueError(
            f"agent role name {name!r} already exists with value {by_name.value!r}"
        )
    if role in AgentRole._value2member_map_:
        raise ValueError(
            f"agent role value {role!r} is already taken by member "
            f"{AgentRole._value2member_map_[role].name!r}"
        )

    member = str.__new__(AgentRole, role)
    member._name_ = name
    member._value_ = role
    AgentRole._member_map_[name] = member
    AgentRole._value2member_map_[role] = member
    type.__setattr__(AgentRole, name, member)
    if name not in AgentRole._member_names_:
        AgentRole._member_names_.append(name)

    _registered_plugin_roles[role] = member
    _rebuild_role_schemas()
    return member


# ── factory ──────────────────────────────────────────────────────


class Factory(BaseModel):
    """Top-level container: agent pool + workflows + config."""

    model_config = ConfigDict(strict=True, extra="forbid", arbitrary_types_allowed=True)

    agent_pool: dict[str, AgentConfig]
    workflows: dict[str, Workflow]
    config: dict[str, Any] | None = None
