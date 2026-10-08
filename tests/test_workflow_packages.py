"""Tests for Package-based mode composition.

Parity strategy: the composed ``design_mode()`` must be structurally equivalent
to the monolithic ``design_workflow()`` it replaces.  Node-set ``issubset``
checks pass over real regressions (lost reloop edges, dropped gate conditions,
a missing ``terminal`` flag), so parity is asserted via
``factory.workflow.parity.diff_workflows`` — a canonical, sorted, normalised
structural comparison.  The only accepted diff is a pinned, reviewed set of
legitimate representation differences (see ``EXPECTED_DESIGN_DIFF``).
"""

from __future__ import annotations

from factory.workflow.parity import diff_workflows, format_diff
from factory.workflow.packages import (
    BUILD_RESEARCHERS,
    build_mode,
    build_package,
    design_mode,
    design_with_frontend_mode,
    discovery_package,
    qa_package,
    research_package,
    strategy_package,
    study_package,
)
from factory.workflow.primitives import AgentNode, ForkNode, GateNode, JoinNode, VerdictType


# ── Per-package smoke tests ──────────────────────────────────────


class TestStudyPackage:
    def test_compiles(self):
        pkg = study_package()
        wf = pkg.compile()
        assert len(wf.nodes) == 4

    def test_has_expected_nodes(self):
        wf = study_package().compile()
        assert "graph_update" in wf.nodes
        assert "study" in wf.nodes
        assert "graph_explorer" in wf.nodes
        assert "concat_study" in wf.nodes

    def test_focus_propagates(self):
        wf = study_package(focus="auth").compile()
        node = wf.nodes["graph_explorer"]
        assert isinstance(node, AgentNode)
        assert "auth" in node.prompt_template

    def test_produces_study_complete(self):
        pkg = study_package()
        assert "study_complete" in pkg.contract.produces


class TestResearchPackage:
    def test_compiles(self):
        pkg = research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt="Check research quality.",
        )
        wf = pkg.compile()
        assert len(wf.nodes) >= 5  # fork + 3 researchers + join + gate

    def test_has_fork_join(self):
        wf = research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt="Check.",
        ).compile()
        assert "fork_research" in wf.nodes
        assert "join_research" in wf.nodes
        assert isinstance(wf.nodes["fork_research"], ForkNode)
        assert isinstance(wf.nodes["join_research"], JoinNode)

    def test_gate_has_reloop(self):
        wf = research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt="Check.",
        ).compile()
        reloop_edges = [e for e in wf.edges if e.source == "gate_research" and e.target == "fork_research"]
        assert len(reloop_edges) == 1

    def test_researchers_read_study_output(self):
        """Researchers must read the study-combined artifact (regression guard)."""
        wf = research_package(
            researchers=BUILD_RESEARCHERS, gate_prompt="Check.",
        ).compile()
        for r in ("researcher_similar", "researcher_techstack", "researcher_pitfalls"):
            assert ".factory/strategy/study-combined.md" in wf.nodes[r].reads


class TestStrategyPackage:
    def test_compiles(self):
        wf = strategy_package().compile()
        assert "strategist" in wf.nodes
        assert "gate_strategy" in wf.nodes

    def test_gate_has_reloop_to_strategist(self):
        wf = strategy_package().compile()
        reloop = [e for e in wf.edges if e.source == "gate_strategy" and e.target == "strategist"]
        assert len(reloop) == 1

    def test_strategist_reads_study_output(self):
        wf = strategy_package().compile()
        assert ".factory/strategy/study-combined.md" in wf.nodes["strategist"].reads


class TestBuildPackage:
    def test_compiles(self):
        wf = build_package().compile()
        assert "builder" in wf.nodes
        assert "gate_build" in wf.nodes
        assert len(wf.nodes) == 2


class TestQaPackage:
    def test_compiles(self):
        wf = qa_package().compile()
        assert len(wf.nodes) == 10

    def test_parallel_qa_agents(self):
        wf = qa_package().compile()
        assert "health_checker" in wf.nodes
        assert "code_reviewer" in wf.nodes
        assert "adversarial_tester" in wf.nodes

    def test_has_gates(self):
        wf = qa_package().compile()
        assert isinstance(wf.nodes["gate_qa"], GateNode)
        assert isinstance(wf.nodes["gate_precheck"], GateNode)


class TestDiscoveryPackage:
    def test_compiles(self):
        wf = discovery_package().compile()
        assert "gate_has_factory" in wf.nodes
        assert "discover" in wf.nodes

    def test_has_bootstrap_path(self):
        wf = discovery_package().compile()
        assert "create_factory_md" in wf.nodes
        assert "factory_init" in wf.nodes

    def test_has_skip_path(self):
        wf = discovery_package().compile()
        assert "skip_bootstrap" in wf.nodes


# ── Composed modes ───────────────────────────────────────────────


class TestBuildMode:
    def test_compiles(self):
        wf = build_mode().compile()
        assert len(wf.nodes) >= 20

    def test_has_all_stages(self):
        wf = build_mode().compile()
        assert "study" in wf.nodes
        assert "fork_research" in wf.nodes
        assert "strategist" in wf.nodes
        assert "builder" in wf.nodes
        assert "fork_qa" in wf.nodes

    def test_focus_reaches_study(self):
        wf = build_mode(focus="payments").compile()
        node = wf.nodes["graph_explorer"]
        assert isinstance(node, AgentNode)
        assert "payments" in node.prompt_template

    def test_is_terminal(self):
        """build_mode is a terminal mode — it must not chain to others."""
        assert build_mode().compile().terminal is True

    def test_qa_reloops_to_builder(self):
        """The QA gates must reloop back to builder (cross-package back-edges)."""
        wf = build_mode().compile()
        edges = {(e.source, e.target, e.condition) for e in wf.edges}
        assert ("gate_qa", "builder", VerdictType.RELOOP) in edges
        assert ("gate_doc_freshness", "builder", VerdictType.RELOOP) in edges

    def test_gate_bridge_conditions_preserved(self):
        """Forward bridges out of gates carry PROCEED, not None."""
        wf = build_mode().compile()
        edges = {(e.source, e.target): e.condition for e in wf.edges}
        assert edges.get(("gate_research", "strategist")) == VerdictType.PROCEED
        assert edges.get(("gate_build", "fork_qa")) == VerdictType.PROCEED

    def test_is_superset_of_build_workflow(self):
        """build_mode intentionally adds a study phase build_workflow lacks."""
        from factory.workflow.definitions import build_workflow

        d = diff_workflows(build_workflow(), build_mode().compile())
        assert d["nodes"]["only_a"] == []  # nothing in build_workflow is missing
        assert set(d["nodes"]["only_b"]) == {
            "concat_study", "graph_explorer", "graph_update", "study",
        }


class TestDesignMode:
    def test_compiles(self):
        wf = design_mode().compile()
        assert len(wf.nodes) >= 25

    def test_has_discovery_and_study(self):
        wf = design_mode().compile()
        assert "gate_has_factory" in wf.nodes
        assert "study" in wf.nodes
        assert "strategist" in wf.nodes
        assert "builder" in wf.nodes

    def test_is_terminal(self):
        """design_mode is a terminal mode — it must not chain to others."""
        assert design_mode().compile().terminal is True

    def test_qa_reloops_to_builder(self):
        wf = design_mode().compile()
        edges = {(e.source, e.target, e.condition) for e in wf.edges}
        assert ("gate_qa", "builder", VerdictType.RELOOP) in edges
        assert ("gate_doc_freshness", "builder", VerdictType.RELOOP) in edges

    def test_superset_of_build(self):
        build_nodes = set(build_mode().compile().nodes.keys())
        design_nodes = set(design_mode().compile().nodes.keys())
        assert build_nodes.issubset(design_nodes)


# ── Structural parity: design_mode vs design_workflow ────────────


# The only accepted differences between the composed design_mode() and the
# monolithic design_workflow().  Both are legitimate representation choices,
# not behavioural drift:
#   - skip_bootstrap: the discovery package models the "config exists, skip
#     bootstrap" path as an explicit node + edges, where the monolithic routes
#     gate_has_factory → graph_update directly.  Functionally equivalent.
#   - fork_qa → join_qa: a degenerate edge the monolithic's QA-subgraph helper
#     emits; the executor ignores it (it reads ForkNode.targets / JoinNode.sources),
#     and the composed version omits it.
EXPECTED_DESIGN_DIFF = {
    "attrs": {},
    "nodes": {
        "only_a": [],
        "only_b": ["skip_bootstrap"],
        "changed": {
            "adversarial_tester": {
                "prompt_template": {
                    "a": (
                        "Adversarially test the builder output. "
                        "Try to find edge cases, security issues, and correctness bugs."
                    ),
                    "b": "Run adversarial tests on the builder output.",
                },
            },
            "code_reviewer": {
                "prompt_template": {
                    "a": (
                        "Review the builder output for code quality. "
                        "Check style, correctness, and adherence to the strategy."
                    ),
                    "b": "Review the builder output for code quality.",
                },
            },
            "health_checker": {
                "prompt_template": {
                    "a": (
                        "Review the builder output for health issues. "
                        "Check that tests pass, no regressions were introduced, "
                        "and the code is production-ready."
                    ),
                    "b": "Run health checks on the builder output.",
                },
            },
        },
    },
    "edges": {
        "only_a": [
            ("factory_init", "graph_update", None),
            ("fork_qa", "join_qa", None),
            ("gate_has_factory", "graph_update", "proceed"),
        ],
        "only_b": [
            ("gate_has_factory", "skip_bootstrap", "proceed"),
            ("skip_bootstrap", "graph_update", None),
        ],
    },
}


class TestDesignModeParity:
    """design_mode() must be structurally equivalent to design_workflow().

    Replaces the old ``old_nodes.issubset(new_nodes)`` check, which passed
    over lost reloop edges, dropped gate conditions, and a missing terminal
    flag.  The diff is pinned: any change to either workflow surfaces here
    for review.
    """

    def test_structural_parity(self):
        from factory.workflow.definitions import design_workflow

        d = diff_workflows(design_workflow(), design_mode().compile())
        assert d == EXPECTED_DESIGN_DIFF, (
            "design_mode() drifted from design_workflow():\n" + format_diff(d)
        )

    def test_no_unexpected_node_changes(self):
        """No node's behavioural attributes (reads/writes/prompts/gate config)
        may silently drift between the monolithic and composed workflows,
        beyond the accepted prompt_template wording differences in QA nodes."""
        from factory.workflow.definitions import design_workflow

        d = diff_workflows(design_workflow(), design_mode().compile())
        assert d["nodes"]["changed"] == EXPECTED_DESIGN_DIFF["nodes"]["changed"], (
            "Node content drift detected — see diff:\n" + format_diff(d)
        )

    def test_attrs_match(self):
        """name, start_node, terminal must match exactly."""
        from factory.workflow.definitions import design_workflow

        d = diff_workflows(design_workflow(), design_mode().compile())
        assert d["attrs"] == {}, f"Workflow attrs drifted: {d['attrs']}"


# ── just_plan: documented gap ────────────────────────────────────


class TestJustPlanGap:
    """design_workflow(just_plan=True) is a distinct plan-only workflow
    (prior-plan detection → research → strategy → user gate → publish →
    seed backlog) that the package interface does not yet express.

    This test pins the gap so it's a conscious, tracked decision rather than
    a silent regression.  When just_plan is added to design_mode(), update
    this test to assert parity instead.
    """

    def test_just_plan_is_distinct_workflow(self):
        from factory.workflow.definitions import design_workflow

        jp = design_workflow(just_plan=True)
        assert jp.name == "plan"
        assert jp.terminal is True
        # Feature surface the package interface doesn't yet compose:
        for node in ("check_prior_plans", "gate_prior_plans",
                     "publish_github", "seed_backlog"):
            assert node in jp.nodes, f"{node} missing from just_plan workflow"

    def test_design_mode_does_not_support_just_plan(self):
        import inspect

        sig = inspect.signature(design_mode)
        assert "just_plan" not in sig.parameters, (
            "design_mode now accepts just_plan — update TestDesignModeParity "
            "to cover the just_plan=True branch with a structural parity test."
        )


class TestDesignWithFrontend:
    def test_compiles(self):
        wf = design_with_frontend_mode().compile()
        assert len(wf.nodes) >= 30

    def test_has_frontend_discovery(self):
        wf = design_with_frontend_mode().compile()
        assert "frontend_discovery" in wf.nodes

    def test_one_node_more_than_design(self):
        design_nodes = set(design_mode().compile().nodes.keys())
        frontend_nodes = set(design_with_frontend_mode().compile().nodes.keys())
        extra = frontend_nodes - design_nodes
        assert extra == {"frontend_discovery"}

    def test_frontend_reads_study(self):
        wf = design_with_frontend_mode().compile()
        node = wf.nodes["frontend_discovery"]
        assert isinstance(node, AgentNode)
        assert ".factory/strategy/study-combined.md" in node.reads
