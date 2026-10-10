"""Tests for the task-setup workflow — v2 director topology for Task scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    GateNode,
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry, _load_workflow_file


# ── Fixtures ────────────────────────────────────────────────────

WORKFLOW_FILE = Path(__file__).resolve().parent.parent / ".factory" / "workflows" / "task-setup.py"


@pytest.fixture(scope="module")
def task_setup_meta():
    meta, _ = _load_workflow_file(WORKFLOW_FILE)
    return meta


@pytest.fixture(scope="module")
def task_setup_wf():
    _, wf_fn = _load_workflow_file(WORKFLOW_FILE)
    return wf_fn()


# ── Module-level metadata ──────────────────────────────────────


class TestMeta:
    def test_meta_has_name(self, task_setup_meta: dict) -> None:
        assert "name" in task_setup_meta
        assert task_setup_meta["name"] == "task-setup"

    def test_meta_has_description(self, task_setup_meta: dict) -> None:
        assert "description" in task_setup_meta
        assert len(task_setup_meta["description"]) > 0

    def test_workflow_fn_returns_workflow(self) -> None:
        _, wf_fn = _load_workflow_file(WORKFLOW_FILE)
        wf = wf_fn()
        assert isinstance(wf, Workflow)
        assert wf.name == "task-setup"


# ── Graph structure ─────────────────────────────────────────────


class TestGraphStructure:
    def test_node_count(self, task_setup_wf: Workflow) -> None:
        assert len(task_setup_wf.nodes) == 11

    def test_edge_count(self, task_setup_wf: Workflow) -> None:
        assert len(task_setup_wf.edges) == 15

    def test_start_node(self, task_setup_wf: Workflow) -> None:
        assert task_setup_wf.start_node == "research_director"
        assert "research_director" in task_setup_wf.nodes

    def test_terminal(self, task_setup_wf: Workflow) -> None:
        assert task_setup_wf.terminal is True

    def test_validate_graph_passes(self, task_setup_wf: Workflow) -> None:
        issues = task_setup_wf.validate_graph()
        assert issues == [], f"Graph validation issues: {issues}"

    def test_all_expected_nodes_present(self, task_setup_wf: Workflow) -> None:
        expected = {
            "research_director",
            "gate_research",
            "strategy_director",
            "gate_strategy",
            "builder",
            "qa_director",
            "gate_qa",
            "validate_task",
            "scoring_validator",
            "gate_scoring",
            "archivist",
        }
        assert set(task_setup_wf.nodes.keys()) == expected


# ── Node types ──────────────────────────────────────────────────


class TestNodeTypes:
    def test_research_director_is_agent_ceo(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["research_director"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.CEO
        assert node.timeout == 600

    def test_gate_research_is_agent_gate(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["gate_research"]
        assert isinstance(node, GateNode)
        assert node.evaluator_type == "agent"
        assert node.evaluator_role == AgentRole.CEO
        assert node.max_iterations == 2

    def test_strategy_director_is_agent_ceo(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["strategy_director"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.CEO
        assert node.timeout == 600

    def test_gate_strategy_is_user_gate(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["gate_strategy"]
        assert isinstance(node, GateNode)
        assert node.evaluator_type == "user"
        assert node.evaluator_role is None

    def test_builder_is_agent_builder(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["builder"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.BUILDER
        assert node.timeout == 600

    def test_qa_director_is_agent_ceo(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["qa_director"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.CEO
        assert node.timeout == 600

    def test_gate_qa_is_agent_gate(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["gate_qa"]
        assert isinstance(node, GateNode)
        assert node.evaluator_type == "agent"
        assert node.evaluator_role == AgentRole.CEO
        assert node.max_iterations == 2

    def test_validate_task_is_gate_node(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["validate_task"]
        assert isinstance(node, GateNode)
        assert node.evaluator_type == "fn"
        assert "factory task validate" in node.evaluator_command
        assert "{project_path}" in node.evaluator_command

    def test_scoring_validator_is_agent_ceo(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["scoring_validator"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.CEO
        assert node.timeout == 900

    def test_gate_scoring_is_agent_gate(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["gate_scoring"]
        assert isinstance(node, GateNode)
        assert node.evaluator_type == "agent"
        assert node.evaluator_role == AgentRole.CEO

    def test_archivist_is_non_blocking(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["archivist"]
        assert isinstance(node, AgentNode)
        assert node.role == AgentRole.ARCHIVIST
        assert node.blocking is False


# ── Edge topology ───────────────────────────────────────────────


class TestTopology:
    """Verify exact 15-edge wiring matches the specification."""

    EXPECTED_EDGES = [
        ("research_director", "gate_research", None),
        ("gate_research", "strategy_director", VerdictType.PROCEED),
        ("gate_research", "research_director", VerdictType.RELOOP),
        ("strategy_director", "gate_strategy", None),
        ("gate_strategy", "builder", VerdictType.PROCEED),
        ("gate_strategy", "strategy_director", VerdictType.RELOOP),
        ("builder", "qa_director", None),
        ("qa_director", "gate_qa", None),
        ("gate_qa", "validate_task", VerdictType.PROCEED),
        ("gate_qa", "builder", VerdictType.RELOOP),
        ("validate_task", "scoring_validator", VerdictType.PROCEED),
        ("validate_task", "builder", VerdictType.RELOOP),
        ("scoring_validator", "gate_scoring", None),
        ("gate_scoring", "archivist", VerdictType.PROCEED),
        ("gate_scoring", "builder", VerdictType.RELOOP),
    ]

    def test_all_expected_edges_present(self, task_setup_wf: Workflow) -> None:
        actual = {(e.source, e.target): e.condition for e in task_setup_wf.edges}
        for src, tgt, cond in self.EXPECTED_EDGES:
            key = (src, tgt)
            assert key in actual, f"Missing edge {src} -> {tgt}"
            assert actual[key] == cond, (
                f"Edge {src}->{tgt}: condition={actual[key]}, expected {cond}"
            )

    def test_no_extra_edges(self, task_setup_wf: Workflow) -> None:
        expected = {(src, tgt) for src, tgt, _ in self.EXPECTED_EDGES}
        actual = {(e.source, e.target) for e in task_setup_wf.edges}
        extra = actual - expected
        assert not extra, f"Extra edges found: {extra}"

    def test_back_edges_exist(self, task_setup_wf: Workflow) -> None:
        """Verify five reloop back-edges."""
        edges = {(e.source, e.target, e.condition) for e in task_setup_wf.edges}
        assert ("gate_research", "research_director", VerdictType.RELOOP) in edges
        assert ("gate_strategy", "strategy_director", VerdictType.RELOOP) in edges
        assert ("gate_qa", "builder", VerdictType.RELOOP) in edges
        assert ("validate_task", "builder", VerdictType.RELOOP) in edges
        assert ("gate_scoring", "builder", VerdictType.RELOOP) in edges

    def test_reloop_to_builder_edges_count(self, task_setup_wf: Workflow) -> None:
        """Verify three gates can reloop back to builder."""
        reloop_to_builder = [
            e for e in task_setup_wf.edges
            if e.target == "builder" and e.condition == VerdictType.RELOOP
        ]
        assert len(reloop_to_builder) == 3
        sources = {e.source for e in reloop_to_builder}
        assert sources == {"gate_qa", "validate_task", "gate_scoring"}


# ── Prompt quality ──────────────────────────────────────────────


class TestPromptQuality:
    def test_research_director_spawns_researchers(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["research_director"]
        assert isinstance(node, AgentNode)
        assert "factory agent researcher" in node.prompt_template

    def test_research_director_has_outer_loop(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["research_director"]
        assert isinstance(node, AgentNode)
        assert "outer loop" in node.prompt_template.lower()

    def test_strategy_director_spawns_strategists(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["strategy_director"]
        assert isinstance(node, AgentNode)
        assert "factory agent strategist" in node.prompt_template

    def test_qa_director_spawns_testers(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["qa_director"]
        assert isinstance(node, AgentNode)
        assert "factory agent adversarial_tester" in node.prompt_template

    def test_qa_director_checks_evolution(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["qa_director"]
        assert isinstance(node, AgentNode)
        assert "evolution" in node.prompt_template.lower()

    def test_builder_references_learning_signal(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["builder"]
        assert isinstance(node, AgentNode)
        assert "learning signal" in node.prompt_template.lower()

    def test_builder_has_chess_evolve_example(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["builder"]
        assert isinstance(node, AgentNode)
        assert "blunder_count" in node.prompt_template

    def test_builder_has_anti_example(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["builder"]
        assert isinstance(node, AgentNode)
        assert "Anti-example" in node.prompt_template

    def test_prompts_reference_project_path(self, task_setup_wf: Workflow) -> None:
        for nid in ("research_director", "strategy_director", "qa_director"):
            node = task_setup_wf.nodes[nid]
            assert isinstance(node, AgentNode)
            assert "$PROJECT_PATH" in node.prompt_template, (
                f"{nid} prompt missing $PROJECT_PATH"
            )

    def test_scoring_validator_prompt_references_verify(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["scoring_validator"]
        assert isinstance(node, AgentNode)
        assert "verify()" in node.prompt_template
        assert "score" in node.prompt_template

    def test_no_hardcoded_reflector_references(self, task_setup_wf: Workflow) -> None:
        """Verify prompts don't hardcode references to current reflector algorithm."""
        for nid, node in task_setup_wf.nodes.items():
            text = ""
            if isinstance(node, AgentNode):
                text = node.prompt_template
            elif isinstance(node, GateNode):
                text = node.gate_prompt
            assert "contrastive analysis" not in text, (
                f"Node {nid} hardcodes 'contrastive analysis'"
            )
            assert "top-K vs bottom-K" not in text, (
                f"Node {nid} hardcodes 'top-K vs bottom-K'"
            )


# ── Post-checks and I/O ────────────────────────────────────────


class TestArtifactChecks:
    def test_research_director_post_checks(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["research_director"]
        assert isinstance(node, AgentNode)
        assert len(node.post_checks) == 3
        paths = {c.path for c in node.post_checks}
        assert ".factory/strategy/research-domain.md" in paths
        assert ".factory/strategy/research-verification.md" in paths
        assert ".factory/strategy/research-outer-loop.md" in paths

    def test_strategy_director_post_checks(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["strategy_director"]
        assert isinstance(node, AgentNode)
        assert len(node.post_checks) == 1
        check = node.post_checks[0]
        assert check.path == ".factory/strategy/current.md"
        assert check.must_exist is True
        assert check.min_size == 500
        assert "## Task Specification" in check.must_contain
        assert "### 1. Data" in check.must_contain
        assert "### 2. Environment" in check.must_contain
        assert "### 3. Task Objective" in check.must_contain
        assert "### 4. Verification" in check.must_contain
        assert "### 5. Evaluation Details" in check.must_contain

    def test_builder_post_checks(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["builder"]
        assert isinstance(node, AgentNode)
        assert len(node.post_checks) == 1
        assert node.post_checks[0].path == ".factory/generated-task-name.txt"

    def test_qa_director_post_checks(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["qa_director"]
        assert isinstance(node, AgentNode)
        assert len(node.post_checks) == 1
        assert node.post_checks[0].path == ".factory/strategy/qa-report.md"

    def test_validate_task_reads(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["validate_task"]
        assert ".factory/generated-task-name.txt" in node.reads

    def test_scoring_validator_reads(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["scoring_validator"]
        assert isinstance(node, AgentNode)
        assert ".factory/generated-task-name.txt" in node.reads
        assert ".factory/strategy/current.md" in node.reads

    def test_scoring_validator_writes(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["scoring_validator"]
        assert isinstance(node, AgentNode)
        assert ".factory/strategy/scoring-report.md" in node.writes

    def test_scoring_validator_post_checks(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["scoring_validator"]
        assert isinstance(node, AgentNode)
        assert len(node.post_checks) == 1
        assert node.post_checks[0].path == ".factory/strategy/scoring-report.md"
        assert node.post_checks[0].must_exist is True
        assert node.post_checks[0].min_size == 200

    def test_gate_scoring_reads(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["gate_scoring"]
        assert isinstance(node, GateNode)
        assert ".factory/strategy/scoring-report.md" in node.reads

    def test_archivist_reads_scoring_report(self, task_setup_wf: Workflow) -> None:
        node = task_setup_wf.nodes["archivist"]
        assert isinstance(node, AgentNode)
        assert ".factory/strategy/scoring-report.md" in node.reads


# ── Registry discovery ──────────────────────────────────────────


class TestRegistryDiscovery:
    def test_discoverable_from_project_path(self, tmp_path: Path) -> None:
        """Verify the workflow is discoverable when placed in a project."""
        import shutil

        wf_dir = tmp_path / ".factory" / "workflows"
        wf_dir.mkdir(parents=True)
        shutil.copy(WORKFLOW_FILE, wf_dir / "task-setup.py")

        WorkflowRegistry.reset()
        entries = WorkflowRegistry.discover(tmp_path)
        assert "task-setup" in entries
        assert entries["task-setup"].source == "project"

    def test_get_workflow_returns_instance(self, tmp_path: Path) -> None:
        """Verify get_workflow returns a valid Workflow from a project dir."""
        import shutil

        wf_dir = tmp_path / ".factory" / "workflows"
        wf_dir.mkdir(parents=True)
        shutil.copy(WORKFLOW_FILE, wf_dir / "task-setup.py")

        WorkflowRegistry.reset()
        wf = WorkflowRegistry.get_workflow("task-setup", tmp_path)
        assert wf is not None
        assert isinstance(wf, Workflow)
        assert wf.name == "task-setup"
        assert len(wf.nodes) == 11
        assert len(wf.edges) == 15
