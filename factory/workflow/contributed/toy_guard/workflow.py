"""Toy guard mode — a 2-node ``draft`` -> ``publish`` package whose draft agent
carries two guards and is compiled, so the checker stage expands (8 nodes / 6
edges). Deterministic fixture: stub agents only, no model/CLI/network.
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
