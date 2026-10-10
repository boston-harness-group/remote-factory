"""WorkflowMutator — GEPA-style prompt evolution for workflow AgentNode prompt_templates."""

from __future__ import annotations

import structlog

from factory.cycle_analyzer import CycleRecord
from factory.workflow.primitives import AgentNode, Workflow

log = structlog.get_logger()


class WorkflowMutator:
    """Mutates workflow AgentNode prompt_templates based on technique performance history.

    Respects frozen_nodes: raises ValueError if asked to mutate a frozen node.
    Uses model_copy(deep=True) for safe workflow cloning.
    """

    def __init__(
        self,
        frozen_nodes: frozenset[str] = frozenset(),
        mutation_model: str = "sonnet",
    ) -> None:
        self.frozen_nodes = frozenset(frozen_nodes)
        self.mutation_model = mutation_model

    def classify_techniques(self, history: list[CycleRecord]) -> dict:
        """Categorize techniques as successful, failed, or plateau based on score deltas."""
        successful: list[str] = []
        failed: list[str] = []
        plateau: list[str] = []

        for record in history:
            for exp in record.experiments:
                technique = exp.hypothesis or "unknown"
                if exp.score_delta is not None:
                    if exp.score_delta > 0.01:
                        successful.append(technique)
                    elif exp.score_delta < -0.01:
                        failed.append(technique)
                    else:
                        plateau.append(technique)
                elif exp.verdict == "keep":
                    successful.append(technique)
                elif exp.verdict == "revert":
                    failed.append(technique)
                else:
                    plateau.append(technique)

        return {
            "successful": successful,
            "failed": failed,
            "plateau": plateau,
        }

    def build_prompt_amendments(self, technique_perf: dict) -> dict[str, str]:
        """Generate prompt amendments from technique performance classification.

        Failed techniques become constraints (avoid), successful become priorities.
        """
        amendments: dict[str, str] = {}

        failed = technique_perf.get("failed", [])
        if failed:
            constraint_lines = [f"- Avoid: {t}" for t in failed]
            amendments["constraints"] = (
                "Based on prior experiments, avoid these approaches:\n"
                + "\n".join(constraint_lines)
            )

        successful = technique_perf.get("successful", [])
        if successful:
            priority_lines = [f"- Prioritize: {t}" for t in successful]
            amendments["priorities"] = (
                "These approaches have shown improvement:\n"
                + "\n".join(priority_lines)
            )

        return amendments

    def mutate(
        self,
        workflow: Workflow,
        history: list[CycleRecord],
        focus_nodes: list[str] | None = None,
    ) -> Workflow:
        """Deep-copy workflow and apply prompt amendments to mutable AgentNodes.

        Raises ValueError if a focus_node is frozen.
        """
        target_nodes = focus_nodes or [
            nid for nid, node in workflow.nodes.items()
            if isinstance(node, AgentNode) and nid not in self.frozen_nodes
        ]

        for nid in target_nodes:
            if nid in self.frozen_nodes:
                raise ValueError(
                    f"Cannot mutate frozen node {nid!r}. "
                    f"Frozen nodes: {sorted(self.frozen_nodes)}"
                )

        technique_perf = self.classify_techniques(history)
        amendments = self.build_prompt_amendments(technique_perf)

        mutated = workflow.model_copy(deep=True)

        for nid in target_nodes:
            node = mutated.nodes.get(nid)
            if not isinstance(node, AgentNode):
                continue

            amendment_text = ""
            if "constraints" in amendments:
                amendment_text += f"\n\n{amendments['constraints']}"
            if "priorities" in amendments:
                amendment_text += f"\n\n{amendments['priorities']}"

            if amendment_text:
                node.prompt_template += amendment_text
                log.info("mutated_node", node_id=nid, amendment_length=len(amendment_text))

        return mutated
