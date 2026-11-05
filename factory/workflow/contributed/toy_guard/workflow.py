"""Toy guard mode — deterministic end-to-end fixture for agent guardrails (INT-002).

A two-node mode (``draft`` → ``publish``) composed as a
:class:`~factory.workflow.package.Package` and returned through
``Package.compile()``, with two guards attached to ``draft`` via
:func:`factory.workflow.guardrails.with_guards`. Compiling splices the generated
checker stage, so the returned graph is::

    draft (AgentNode, 2 guards, budget 2)
      -> guard_draft_reset (FnNode)  -> guard_draft_fork (ForkNode)
      -> {guard_draft_<slug> ...}    -> guard_draft_join (JoinNode)
      -> guard_draft_gate (GateNode, fn)
         -> PROCEED -> publish (AgentNode)
         -> RELOOP  -> draft

The graph returned by ``workflow()`` has exactly 8 nodes / 6 edges: 2 authored
mode nodes + 4 fixed stage nodes + 2 checkers (the authored graph before
expansion is 2 nodes / 1 edge), and the gate reloops at most
``GUARD_MAX_ITERATIONS = 2`` times.

``WorkflowRegistry`` returns whatever ``workflow()`` returns, so a
package-composed mode must return ``…compile()`` for ``show`` / ``validate`` /
``export-skills`` to see the stage — this mode does exactly that. What does
**not** get expanded (§3.4, ASM-004): ``expand_agent_guards`` runs **only**
where a flat graph is produced through
:meth:`factory.workflow.package.Package.compile`. A hand-built ``Workflow`` that
never calls ``compile()`` keeps its ``guards`` list as inert data — it
serializes, it does not enforce — including builtin modes in ``definitions.py``,
contributed workflow files that return a hand-built graph, and YAML overrides
via ``factory.skillopt.yaml_surface.yaml_to_workflow``. Guards on an
``AgentNode`` that is a ``ForkNode`` target, on a ``blocking=False`` agent, or in
a per-item context (a ``DataNode`` branch or a ``SubgraphForkNode`` subgraph)
are rejected by expansion; they are not silently ignored (ASM-005).

This mode is a deterministic test fixture: the e2e tests drive it through
``WorkflowExecutor(agent_fn=...)`` with stub agents, so no model, CLI, network
or paid service is involved.
"""

from __future__ import annotations

from factory.workflow.guardrails import Guard, with_guards
from factory.workflow.package import Package, Port
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    NodeType,
    Workflow,
)

meta = {
    "name": "toy-guard",
    "description": (
        "Toy guardrail mode — a 2-node draft -> publish pipeline whose draft "
        "agent is guarded by two generated checker rules, composed as a Package "
        "and compiled so the guard stage expands. Deterministic e2e fixture."
    ),
}

TOY_AGENT_ID: str = "draft"

GUARD_MAX_ITERATIONS: int = 2

GUARDS: tuple[Guard, ...] = (
    Guard(rule="draft.md must state Status: ready"),
    Guard(rule="draft.md must not contain TBD"),
)

_DRAFT_PROMPT = (
    "You are drafting a short note for the toy-guard mode.\n\n"
    "Write the draft to draft.md. It must end with a line that reads exactly "
    "'Status: ready' and it must not contain the placeholder 'TBD'.\n\n"
    "Keep the draft to one short paragraph."
)

_PUBLISH_PROMPT = (
    "Publish the approved draft: copy the contents of draft.md to published.md "
    "and reply with the path you wrote."
)


def workflow() -> Workflow:
    """Build the toy-guard package and return its compiled (guard-expanded) graph."""
    draft = with_guards(
        AgentNode(
            id=TOY_AGENT_ID,
            role=AgentRole.BUILDER,
            prompt_template=_DRAFT_PROMPT,
            writes={"draft.md"},
        ),
        GUARDS,
        max_iterations=GUARD_MAX_ITERATIONS,
    )
    publish = AgentNode(
        id="publish",
        role=AgentRole.ARCHIVIST,
        prompt_template=_PUBLISH_PROMPT,
        reads={"draft.md"},
        writes={"published.md"},
    )

    nodes: dict[str, NodeType] = {draft.id: draft, publish.id: publish}
    graph = Workflow(
        name="toy-guard",
        nodes=nodes,
        edges=[Edge(source=draft.id, target=publish.id)],
        start_node=draft.id,
    )
    package = Package(
        name="toy-guard",
        version="1.0.0",
        description=meta["description"],
        outputs=[
            Port(name="draft", artifact_path="draft.md"),
            Port(name="published", artifact_path="published.md"),
        ],
        graph=graph,
        entry_node=draft.id,
        exit_node=publish.id,
        terminal=True,
    )
    return package.compile()
