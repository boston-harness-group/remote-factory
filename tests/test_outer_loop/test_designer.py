"""Tests for DesignerAgent — design mode and mutation mode."""

from __future__ import annotations

from factory.outer_loop.designer import DesignerAgent, _seed_agent_params
from factory.outer_loop.models import MutationType
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    Edge,
    FnNode,
    GateNode,
    Workflow)


class TestDesignMinimal:
    def test_produces_3_to_4_nodes(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        assert 3 <= len(wf.nodes) <= 4

    def test_valid_workflow(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        issues = wf.validate_graph()
        assert issues == [], f"Validation issues: {issues}"

    def test_has_builder(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        roles = {
            node.role.value
            for node in wf.nodes.values()
            if hasattr(node, "role")
        }
        assert "builder" in roles

    def test_has_gate(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        gate_nodes = [
            n for n in wf.nodes.values()
            if type(n).__name__ == "GateNode"
        ]
        assert len(gate_nodes) >= 1

    def test_name_includes_benchmark(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_minimal("feature_bench")
        assert "minimal" in wf.name
        assert "feature_bench" in wf.name

    def test_serialization_roundtrip(self) -> None:
        from factory.workflow.primitives import Workflow

        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        data = wf.to_dict()
        restored = Workflow.from_dict(data)
        assert len(restored.nodes) == len(wf.nodes)
        assert restored.start_node == wf.start_node


class TestDesignThorough:
    def test_produces_8_to_10_nodes(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        assert 8 <= len(wf.nodes) <= 10

    def test_valid_workflow(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        issues = wf.validate_graph()
        assert issues == [], f"Validation issues: {issues}"

    def test_has_parallel_builders(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        fork_nodes = [
            n for n in wf.nodes.values()
            if type(n).__name__ == "ForkNode"
        ]
        assert len(fork_nodes) >= 1

    def test_has_code_reviewer(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        roles = {
            node.role.value
            for node in wf.nodes.values()
            if hasattr(node, "role")
        }
        assert "code_reviewer" in roles

    def test_has_adversarial_tester(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        roles = {
            node.role.value
            for node in wf.nodes.values()
            if hasattr(node, "role")
        }
        assert "adversarial_tester" in roles

    def test_has_study_node(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        assert "study" in wf.nodes

    def test_serialization_roundtrip(self) -> None:
        from factory.workflow.primitives import Workflow

        designer = DesignerAgent()
        wf = designer.design_thorough("test benchmark")
        data = wf.to_dict()
        restored = Workflow.from_dict(data)
        assert len(restored.nodes) == len(wf.nodes)
        assert restored.start_node == wf.start_node


class TestDesignCustom:
    def test_respects_max_nodes(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_custom("bench", {"max_nodes": 5})
        assert len(wf.nodes) <= 5

    def test_valid_workflow(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_custom("bench", {"max_nodes": 6})
        issues = wf.validate_graph()
        assert issues == [], f"Validation issues: {issues}"

    def test_includes_required_roles(self) -> None:
        designer = DesignerAgent()
        wf = designer.design_custom(
            "bench", {"max_nodes": 8, "require_roles": ["health_checker"]}
        )
        roles = {
            node.role.value
            for node in wf.nodes.values()
            if hasattr(node, "role")
        }
        assert "health_checker" in roles


class TestPropose:
    def test_returns_mutation_records(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={"node_stats": {}, "dominant_failure": ""},
            archive_stats={"diversity": 0.5},
            benchmark_spec="test")
        assert len(proposals) >= 1
        assert len(proposals) <= 3

    def test_high_failure_rate_proposes_removal(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={
                "node_stats": {"researcher": {"failure_rate": 0.8}},
                "dominant_failure": "",
            },
            archive_stats={"diversity": 0.5},
            benchmark_spec="test")
        remove_proposals = [
            p for p in proposals if p.operator == MutationType.NODE_REMOVE
        ]
        assert len(remove_proposals) >= 1
        assert remove_proposals[0].target_node == "researcher"

    def test_timeout_failure_proposes_param_mutate(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={
                "node_stats": {},
                "dominant_failure": "timeout",
            },
            archive_stats={"diversity": 0.5},
            benchmark_spec="test")
        timeout_proposals = [
            p for p in proposals if p.operator == MutationType.PARAM_MUTATE
        ]
        assert len(timeout_proposals) >= 1

    def test_low_diversity_proposes_insertion(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={"node_stats": {}, "dominant_failure": ""},
            archive_stats={"diversity": 0.1},
            benchmark_spec="test")
        insert_proposals = [
            p for p in proposals if p.operator == MutationType.NODE_INSERT
        ]
        assert len(insert_proposals) >= 1

    def test_no_signal_still_returns_proposal(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={},
            archive_stats={},
            benchmark_spec="test")
        assert len(proposals) >= 1

    def test_max_3_proposals(self, simple_workflow) -> None:  # type: ignore[no-untyped-def]
        designer = DesignerAgent()
        proposals = designer.propose(
            simple_workflow,
            telemetry={
                "node_stats": {
                    "researcher": {"failure_rate": 0.9},
                    "strategist": {"failure_rate": 0.9},
                    "builder": {"failure_rate": 0.9},
                    "gate_qa": {"failure_rate": 0.9},
                },
                "dominant_failure": "timeout",
            },
            archive_stats={"diversity": 0.1},
            benchmark_spec="test")
        assert len(proposals) <= 3


class TestFrozenNodePreservation:
    """Tests for frozen node injection in designer methods."""

    @staticmethod
    def _seed_with_positions() -> Workflow:
        """Create a seed workflow containing a FnNode with id='positions'."""
        return Workflow(
            name="seed",
            nodes={
                "positions": FnNode(
                    id="positions",
                    command="load_positions",
                    writes={".factory/positions.json"}),
                "researcher": AgentNode(
                    id="researcher",
                    role=AgentRole.RESEARCHER),
            },
            edges=[Edge(source="positions", target="researcher")],
            start_node="positions")

    def test_design_minimal_preserves_frozen_nodes(self) -> None:
        designer = DesignerAgent()
        seed = self._seed_with_positions()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        assert "positions" in result.nodes
        assert result.nodes["positions"].command == "load_positions"  # type: ignore[union-attr]

    def test_design_thorough_preserves_frozen_nodes(self) -> None:
        designer = DesignerAgent()
        seed = self._seed_with_positions()
        result = designer.design_thorough(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        assert "positions" in result.nodes
        assert result.nodes["positions"].command == "load_positions"  # type: ignore[union-attr]

    def test_design_custom_preserves_frozen_nodes(self) -> None:
        designer = DesignerAgent()
        seed = self._seed_with_positions()
        result = designer.design_custom(
            "bench",
            {"max_nodes": 6},
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        assert "positions" in result.nodes
        assert result.nodes["positions"].command == "load_positions"  # type: ignore[union-attr]

    def test_frozen_node_overwrites_template_on_collision(self) -> None:
        """When a frozen node ID collides with a template node, frozen wins."""
        seed = Workflow(
            name="seed",
            nodes={
                "researcher": AgentNode(
                    id="researcher",
                    role=AgentRole.RESEARCHER,
                    timeout=999),
            },
            edges=[],
            start_node="researcher")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"researcher"})
        assert result.nodes["researcher"].timeout == 999  # type: ignore[union-attr]

    def test_design_without_frozen_nodes_unchanged(self) -> None:
        """Calling design_minimal() without seed/frozen params works as before."""
        designer = DesignerAgent()
        wf = designer.design_minimal("test benchmark")
        assert 3 <= len(wf.nodes) <= 4
        issues = wf.validate_graph()
        assert issues == [], f"Validation issues: {issues}"

    def test_design_minimal_preserves_frozen_data_node(self) -> None:
        """DataNode auto-frozen via _auto_frozen_nodes should be preserved."""
        from factory.workflow.primitives import DataItem, DataNode

        designer = DesignerAgent()
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        assert "positions" in result.nodes
        assert type(result.nodes["positions"]).__name__ == "DataNode"

    def test_engine_designer_includes_auto_frozen_data_nodes(self) -> None:
        """_add_designer_variants should include auto-frozen DataNodes."""
        from factory.outer_loop.engine import _auto_frozen_nodes
        from factory.workflow.primitives import DataItem, DataNode

        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")

        # Verify _auto_frozen_nodes detects the DataNode but NOT subgraph nodes
        auto_frozen = _auto_frozen_nodes(seed)
        assert "positions" in auto_frozen
        assert "solver" not in auto_frozen


class TestDataNodeRewiring:
    """Tests that frozen DataNodes are properly wired into designer templates."""

    @staticmethod
    def _seed_with_data_node() -> Workflow:
        """Seed workflow containing a DataNode with subgraph refs to 'solver'."""
        return Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")

    def test_minimal_start_node_is_data_node(self) -> None:
        """Designer variant with DataNode has start_node == DataNode ID."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        assert wf.start_node == "positions"

    def test_minimal_datanode_has_outgoing_edge(self) -> None:
        """DataNode has an outgoing edge to the branch entry."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        data_node = wf.nodes["positions"]
        assert isinstance(data_node, DataNode)
        outgoing = [e for e in wf.edges if e.source == "positions"]
        assert len(outgoing) >= 1

    def test_minimal_has_join_node(self) -> None:
        """Designer-produced DataNode workflow has a JoinNode."""
        from factory.workflow.primitives import JoinNode
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        join_nodes = [n for n in wf.nodes.values() if isinstance(n, JoinNode)]
        assert len(join_nodes) >= 1

    def test_minimal_datanode_edge_to_branch_entry(self) -> None:
        """DataNode has an edge to the branch entry node."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        outgoing = [e.target for e in wf.edges if e.source == "positions"]
        assert len(outgoing) >= 1

    def test_minimal_without_data_node_unchanged(self) -> None:
        """Designer without frozen DataNode retains original start_node."""
        designer = DesignerAgent()
        wf = designer.design_minimal("bench")
        assert wf.start_node == "researcher"
        edge_sources = {e.source for e in wf.edges}
        assert "positions" not in edge_sources

    def test_rewired_workflow_validates_graph(self) -> None:
        """Rewired workflow with DataNode passes validate_graph() without structural issues."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        issues = wf.validate_graph()
        # No unreachable-node warnings since subgraph nodes are not injected.
        # Only data dependency warnings may remain.
        structural = [
            i for i in issues
            if "no predecessor writes" not in i
        ]
        assert structural == [], f"Validation issues: {structural}"


class TestInjectFrozenDataNodeSubgraph:
    """Tests for DataNode injection in _inject_frozen_nodes.

    With the correct evolution model, subgraph nodes are NOT injected —
    template nodes replace them. Only the DataNode itself is injected.
    """

    @staticmethod
    def _seed_with_multi_node_subgraph() -> Workflow:
        """Seed with DataNode whose subgraph spans generator → processor → validator."""
        return Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "generator": AgentNode(
                    id="generator",
                    role=AgentRole.BUILDER,
                    timeout=300),
                "processor": AgentNode(
                    id="processor",
                    role=AgentRole.RESEARCHER,
                    timeout=300),
                "validator": AgentNode(
                    id="validator",
                    role=AgentRole.CODE_REVIEWER,
                    timeout=300),
            },
            edges=[
                Edge(source="positions", target="generator"),
                Edge(source="generator", target="processor"),
                Edge(source="processor", target="validator"),
            ],
            start_node="positions")

    def test_frozen_data_node_subgraph_not_injected(self) -> None:
        """Freezing a DataNode does NOT inject its seed subgraph nodes.
        Template nodes replace the subgraph."""
        designer = DesignerAgent()
        seed = self._seed_with_multi_node_subgraph()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # Seed subgraph nodes should NOT be present
        assert "generator" not in result.nodes
        assert "processor" not in result.nodes
        assert "validator" not in result.nodes
        # Template nodes should be present instead
        assert "researcher" in result.nodes
        assert "builder" in result.nodes
        assert "gate_qa" in result.nodes

    def test_inject_frozen_data_node_validates(self) -> None:
        """Variant with frozen DataNode passes validation cleanly.
        No orphaned subgraph nodes since they are not injected."""
        designer = DesignerAgent()
        seed = self._seed_with_multi_node_subgraph()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        issues = result.validate_graph()
        # No unreachable-node warnings should occur since subgraph nodes
        # are not injected. Only data dependency warnings may remain.
        structural = [
            i for i in issues
            if "no predecessor writes" not in i
        ]
        assert structural == [], f"Structural issues: {structural}"


class TestDataSubgraphDesignerAwareness:
    """Tests for data-aware Designer prompts and templates."""

    @staticmethod
    def _seed_with_data_node() -> Workflow:
        return Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")

    def test_data_subgraph_prompt_language(self) -> None:
        """design_minimal with frozen DataNode + executor strategy → entry node
        prompt contains role framing + 'current_item.json' but NOT generic 'data item' filler."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"},
            execution_strategy="executor")
        # Find branch entry via edges from the DataNode
        entry_targets = [e.target for e in wf.edges if e.source == "positions"]
        assert len(entry_targets) >= 1
        entry_node = wf.nodes[entry_targets[0]]
        assert isinstance(entry_node, AgentNode)
        prompt = entry_node.prompt_template or ""
        assert "current_item.json" in prompt
        assert "data item" not in prompt
        # Role framing: prompt starts with 'As a <role>,'
        role_name = entry_node.role.value.replace("_", " ")
        assert f"As a {role_name}," in prompt

    def test_data_subgraph_generic_prompt_unchanged(self) -> None:
        """design_minimal WITHOUT frozen DataNode → entry node prompt contains
        'project at {project_path}' and NOT data-specific language."""
        designer = DesignerAgent()
        wf = designer.design_minimal("bench", execution_strategy="executor")
        researcher = wf.nodes["researcher"]
        assert isinstance(researcher, AgentNode)
        assert "project at {project_path}" in (researcher.prompt_template or "")
        assert "current_item.json" not in (researcher.prompt_template or "")

    def test_data_thorough_fewer_nodes(self) -> None:
        """design_thorough with frozen DataNode has fewer nodes
        (no study, strategist, code_reviewer, adversarial_tester)."""
        designer = DesignerAgent()
        seed = self._seed_with_data_node()
        wf = designer.design_thorough(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # DataNode + 6 template nodes = 7 total
        # Nodes that should NOT be present in data subgraph template
        assert "study" not in wf.nodes
        assert "strategist" not in wf.nodes
        assert "code_reviewer" not in wf.nodes
        assert "adversarial_tester" not in wf.nodes
        # Nodes that SHOULD be present
        assert "researcher" in wf.nodes
        assert "fork_builders" in wf.nodes
        assert "builder_a" in wf.nodes
        assert "builder_b" in wf.nodes
        assert "join_builders" in wf.nodes
        assert "gate_qa" in wf.nodes
        assert "positions" in wf.nodes

    def test_rewire_no_longer_patches_prompt(self) -> None:
        """_rewire_data_nodes only does topology — no reads/prompt changes.
        The entry node's prompt comes from _populate_executor_fields, not from
        _rewire_data_nodes patching."""
        from factory.outer_loop.designer import _rewire_data_nodes

        nodes: dict = {
            "positions": DataNode(
                id="positions",
                inline_items=[DataItem(id="pos1", prompt="test")]),
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                writes={".factory/strategy/research.md"},
                prompt_template="Original prompt."),
            "gate_qa": AgentNode(
                id="gate_qa",
                role=AgentRole.BUILDER),
        }
        edges = [Edge(source="researcher", target="gate_qa")]
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(id="solver", role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        _rewire_data_nodes(nodes, edges, "researcher", seed, {"positions"})
        # _rewire_data_nodes should NOT have changed researcher's prompt or reads
        researcher = nodes["researcher"]
        assert isinstance(researcher, AgentNode)
        assert researcher.prompt_template == "Original prompt."
        assert ".factory/current_item.json" not in (researcher.reads or set())


class TestDataNodeEvolutionModel:
    """Tests for the correct DataNode evolution model.

    DataNode = infrastructure (frozen). Its subgraph = evolution surface (mutable).
    Template nodes replace seed subgraph nodes. Prompt templates propagate by role.
    """

    def test_subgraph_nodes_not_injected(self) -> None:
        """When DataNode is frozen, its seed subgraph nodes should NOT appear
        in the variant — template nodes replace them."""
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "generator": AgentNode(
                    id="generator",
                    role=AgentRole.BUILDER,
                    timeout=300),
                "processor": AgentNode(
                    id="processor",
                    role=AgentRole.RESEARCHER,
                    timeout=300),
                "validator": AgentNode(
                    id="validator",
                    role=AgentRole.CODE_REVIEWER,
                    timeout=300),
            },
            edges=[
                Edge(source="positions", target="generator"),
                Edge(source="generator", target="processor"),
                Edge(source="processor", target="validator"),
            ],
            start_node="positions")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # Seed subgraph nodes (generator, processor, validator) should NOT be present
        assert "generator" not in result.nodes
        assert "processor" not in result.nodes
        assert "validator" not in result.nodes
        # Template nodes (researcher, builder, gate_qa) should be present
        assert "researcher" in result.nodes
        assert "builder" in result.nodes
        assert "gate_qa" in result.nodes

    def test_datanode_edge_points_to_template(self) -> None:
        """After design_minimal with frozen DataNode, the DataNode edge should
        point to template's researcher (not seed's generator)."""
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "generator": AgentNode(
                    id="generator",
                    role=AgentRole.BUILDER,
                    timeout=300),
            },
            edges=[Edge(source="positions", target="generator")],
            start_node="positions")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # DataNode should have an edge to the template's entry node
        entry_targets = [e.target for e in result.edges if e.source == "positions"]
        assert len(entry_targets) >= 1
        assert "researcher" in entry_targets

    def test_prompt_propagation_by_role(self) -> None:
        """When seed subgraph has a BUILDER node with prompt_template,
        the template's BUILDER node should receive that prompt."""
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER,
                    prompt_template="Build chess engine"),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"},
            execution_strategy="none",  # Skip _populate_executor_fields
        )
        builder_node = result.nodes["builder"]
        assert isinstance(builder_node, AgentNode)
        assert builder_node.prompt_template == "Build chess engine"

    def test_branch_entry_reads_current_item(self) -> None:
        """After design_minimal with frozen DataNode, the branch entry node
        must have '.factory/current_item.json' in its reads set."""
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # Find branch entry via edges
        entry_targets = [e.target for e in result.edges if e.source == "positions"]
        assert len(entry_targets) >= 1
        entry_node = result.nodes[entry_targets[0]]
        assert isinstance(entry_node, AgentNode)
        assert ".factory/current_item.json" in entry_node.reads

    def test_branch_entry_prompt_mentions_current_item(self) -> None:
        """After design_minimal with frozen DataNode, the entry node's
        prompt_template must contain 'current_item.json'."""
        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        designer = DesignerAgent()
        result = designer.design_minimal(
            "bench",
            seed_workflow=seed,
            frozen_node_ids={"positions"})
        # Find branch entry via edges
        entry_targets = [e.target for e in result.edges if e.source == "positions"]
        assert len(entry_targets) >= 1
        entry_node = result.nodes[entry_targets[0]]
        assert isinstance(entry_node, AgentNode)
        assert "current_item.json" in (entry_node.prompt_template or "")

    def test_auto_frozen_only_datanode(self) -> None:
        """_auto_frozen_nodes() returns only DataNode IDs, not subgraph nodes."""
        from factory.outer_loop.engine import _auto_frozen_nodes

        seed = Workflow(
            name="seed",
            nodes={
                "positions": DataNode(
                    id="positions",
                    inline_items=[DataItem(id="pos1", prompt="test")]),
                "solver": AgentNode(
                    id="solver",
                    role=AgentRole.BUILDER),
            },
            edges=[Edge(source="positions", target="solver")],
            start_node="positions")
        frozen = _auto_frozen_nodes(seed)
        assert frozen == {"positions"}
        assert "solver" not in frozen


# ── Fix 8: designer inherits model/timeout from seed ──────────────


class TestFix8DesignerInheritsModelTimeout:
    """AgentNodes in designed workflows inherit model and timeout from seed."""

    def _make_seed_with_model(self, model: str = "claude-haiku-4-5-20251001") -> Workflow:
        return Workflow(
            name="seed-with-model",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    model=model,
                    timeout=999,
                ),
                "gate_qa": GateNode(
                    id="gate_qa",
                    evaluator_type="agent",
                    evaluator_role=AgentRole.HEALTH_CHECKER,
                ),
            },
            edges=[Edge(source="builder", target="gate_qa")],
            start_node="builder",
        )

    def test_seed_agent_params_extracts_model(self) -> None:
        seed = self._make_seed_with_model()
        params = _seed_agent_params(seed)
        assert params["model"] == "claude-haiku-4-5-20251001"
        assert params["timeout"] == 999

    def test_seed_agent_params_none_workflow(self) -> None:
        assert _seed_agent_params(None) == {}

    def test_minimal_inherits_model(self) -> None:
        seed = self._make_seed_with_model()
        designer = DesignerAgent()
        wf = designer.design_minimal("bench", seed_workflow=seed)
        for node in wf.nodes.values():
            if isinstance(node, AgentNode):
                assert node.model == "claude-haiku-4-5-20251001", (
                    f"Node {node.id} has model={node.model!r}, expected seed model"
                )

    def test_thorough_inherits_model(self) -> None:
        seed = self._make_seed_with_model()
        designer = DesignerAgent()
        wf = designer.design_thorough("bench", seed_workflow=seed)
        for node in wf.nodes.values():
            if isinstance(node, AgentNode):
                assert node.model == "claude-haiku-4-5-20251001", (
                    f"Node {node.id} has model={node.model!r}, expected seed model"
                )

    def test_custom_inherits_model(self) -> None:
        seed = self._make_seed_with_model()
        designer = DesignerAgent()
        wf = designer.design_custom("bench", {}, seed_workflow=seed)
        for node in wf.nodes.values():
            if isinstance(node, AgentNode):
                assert node.model == "claude-haiku-4-5-20251001", (
                    f"Node {node.id} has model={node.model!r}, expected seed model"
                )

    def test_no_seed_model_keeps_defaults(self) -> None:
        """Without a seed model, designed AgentNodes keep their defaults."""
        designer = DesignerAgent()
        wf = designer.design_minimal("bench")
        for node in wf.nodes.values():
            if isinstance(node, AgentNode):
                assert node.model == "", (
                    f"Node {node.id} should have empty model without seed"
                )
