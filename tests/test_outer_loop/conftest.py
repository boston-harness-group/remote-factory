"""Shared fixtures for outer loop tests."""

from __future__ import annotations

import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow,
)


@pytest.fixture()
def simple_workflow() -> Workflow:
    """A simple 5-node workflow for mutation testing.

    study → researcher → strategist → builder → gate_qa
    """
    nodes = {
        "study": FnNode(
            id="study",
            command="factory study {project_path}",
            writes={".factory/strategy/observations.md"},
        ),
        "researcher": AgentNode(
            id="researcher",
            role=AgentRole.RESEARCHER,
            prompt_template="Research the project at {project_path}.",
            reads={".factory/strategy/observations.md"},
            writes={".factory/strategy/research.md"},
        ),
        "strategist": AgentNode(
            id="strategist",
            role=AgentRole.STRATEGIST,
            prompt_template="Strategize improvements for the project at {project_path}.",
            reads={".factory/strategy/research.md"},
            writes={".factory/strategy/current.md"},
        ),
        "builder": AgentNode(
            id="builder",
            role=AgentRole.BUILDER,
            prompt_template="Build the solution for the project at {project_path}.",
            reads={".factory/strategy/current.md"},
            writes={".factory/reviews/builder-latest.md"},
        ),
        "gate_qa": GateNode(
            id="gate_qa",
            evaluator_type="agent",
            evaluator_role=AgentRole.CEO,
            reads={".factory/reviews/builder-latest.md"},
        ),
    }
    edges = [
        Edge(source="study", target="researcher"),
        Edge(source="researcher", target="strategist"),
        Edge(source="strategist", target="builder"),
        Edge(source="builder", target="gate_qa"),
        Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
        Edge(source="gate_qa", target="study", condition=VerdictType.PROCEED),
    ]
    return Workflow(
        name="test_simple",
        nodes=nodes,
        edges=edges,
        start_node="study",
    )
