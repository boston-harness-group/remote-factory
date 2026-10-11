"""Tests for factory/workflow/wrapping.py — mechanical DataNode wrapping."""
import pytest
from factory.workflow.primitives import (
    DataNode, Edge, FnNode, JoinNode, Workflow)
from factory.workflow.wrapping import wrap_with_data_node


class TestWrapWithDataNode:
    def test_simple_linear_chain(self) -> None:
        """Wrap a 3-node linear chain: a → b → c."""
        wf = Workflow(
            name="test",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
                "c": FnNode(id="c", command="echo c"),
            },
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            start_node="a")
        wrapped = wrap_with_data_node(wf, task_ref="my.mod:MyTask")
        assert wrapped.start_node == "data"
        assert "data" in wrapped.nodes
        dn = wrapped.nodes["data"]
        assert isinstance(dn, DataNode)
        assert dn.task_ref == "my.mod:MyTask"
        assert wrapped.task == "my.mod:MyTask"
        # All original nodes preserved
        assert "a" in wrapped.nodes
        assert "b" in wrapped.nodes
        assert "c" in wrapped.nodes
        # JoinNode added
        join_ids = [nid for nid, n in wrapped.nodes.items() if isinstance(n, JoinNode)]
        assert len(join_ids) == 1
        # Edges: data→a, a→b, b→c, c→join
        assert len(wrapped.edges) == 4

    def test_late_bound_no_task_ref(self) -> None:
        """Wrap without task_ref — DataNode has no source (late-bound)."""
        wf = Workflow(
            name="test",
            nodes={
                "start": FnNode(id="start", command="echo go"),
                "end": FnNode(id="end", command="echo done"),
            },
            edges=[Edge(source="start", target="end")],
            start_node="start")
        wrapped = wrap_with_data_node(wf)
        dn = wrapped.nodes["data"]
        assert isinstance(dn, DataNode)
        assert dn.task_ref is None
        assert dn.source_path is None
        assert dn.inline_items == []

    def test_single_node_workflow(self) -> None:
        """A single-node workflow: entry == terminal."""
        wf = Workflow(
            name="solo",
            nodes={"only": FnNode(id="only", command="echo solo")},
            edges=[],
            start_node="only")
        wrapped = wrap_with_data_node(wf, task_ref="x:Y")
        dn = wrapped.nodes["data"]
        assert isinstance(dn, DataNode)
        # JoinNode should be present
        join_ids = [nid for nid, n in wrapped.nodes.items() if isinstance(n, JoinNode)]
        assert len(join_ids) == 1
        # Edges: data→only, only→join
        edge_srcs = {(e.source, e.target) for e in wrapped.edges}
        assert ("data", "only") in edge_srcs

    def test_id_collision_resolved(self) -> None:
        """If 'data' already exists, fallback to '_data'."""
        wf = Workflow(
            name="test",
            nodes={
                "data": FnNode(id="data", command="echo orig"),
                "end": FnNode(id="end", command="echo end"),
            },
            edges=[Edge(source="data", target="end")],
            start_node="data")
        wrapped = wrap_with_data_node(wf, task_ref="x:Y")
        assert "_data" in wrapped.nodes
        assert wrapped.start_node == "_data"

    def test_empty_workflow_raises(self) -> None:
        """Empty workflow raises ValueError."""
        wf = Workflow(name="empty", nodes={}, edges=[], start_node="x")
        with pytest.raises(ValueError, match="empty workflow"):
            wrap_with_data_node(wf)

    def test_validates_after_wrapping(self) -> None:
        """Wrapped workflow passes validate_graph()."""
        wf = Workflow(
            name="test",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[Edge(source="a", target="b")],
            start_node="a")
        wrapped = wrap_with_data_node(wf, task_ref="x:Y")
        issues = wrapped.validate_graph()
        assert len(issues) == 0

    def test_parallelism_setting(self) -> None:
        """Custom parallelism is passed through."""
        wf = Workflow(
            name="test",
            nodes={
                "a": FnNode(id="a", command="echo a"),
            },
            edges=[],
            start_node="a")
        wrapped = wrap_with_data_node(wf, parallelism=4)
        dn = wrapped.nodes["data"]
        assert isinstance(dn, DataNode)
        assert dn.parallelism == 4

    def test_double_collision_raises(self) -> None:
        """If both 'data' and '_data' exist, raise ValueError."""
        wf = Workflow(
            name="test",
            nodes={
                "data": FnNode(id="data", command="echo 1"),
                "_data": FnNode(id="_data", command="echo 2"),
                "end": FnNode(id="end", command="echo 3"),
            },
            edges=[
                Edge(source="data", target="_data"),
                Edge(source="_data", target="end"),
            ],
            start_node="data")
        with pytest.raises(ValueError, match="collision"):
            wrap_with_data_node(wf)

    def test_invalid_start_node_raises(self) -> None:
        """start_node not found in workflow nodes raises ValueError."""
        wf = Workflow(
            name="t",
            nodes={"a": FnNode(id="a", command="echo a")},
            edges=[],
            start_node="missing")
        with pytest.raises(ValueError, match="start_node"):
            wrap_with_data_node(wf)

    def test_no_terminal_node_raises(self) -> None:
        """Cycle with no terminal node (every node has outgoing edges) raises ValueError."""
        wf = Workflow(
            name="t",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[Edge(source="a", target="b"), Edge(source="b", target="a")],
            start_node="a")
        with pytest.raises(ValueError, match="No terminal"):
            wrap_with_data_node(wf)

    def test_multiple_terminal_nodes_raises(self) -> None:
        """Multiple terminal nodes (>1 node with no outgoing edges) raises ValueError."""
        wf = Workflow(
            name="t",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
                "c": FnNode(id="c", command="echo c"),
            },
            edges=[Edge(source="a", target="b")],
            start_node="a")
        with pytest.raises(ValueError, match="Multiple terminal"):
            wrap_with_data_node(wf)
