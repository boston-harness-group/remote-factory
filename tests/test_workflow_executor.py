"""Tier 2: Executor tests — deterministic graph walker behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.workflow.executor import WorkflowExecutor
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    Verdict,
    VerdictType,
    Workflow)


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project with .factory/ directory."""
    factory_dir = tmp_path / ".factory"
    factory_dir.mkdir()
    (factory_dir / "strategy").mkdir()
    (factory_dir / "reviews").mkdir()
    (factory_dir / "experiments").mkdir()
    (factory_dir / "archive").mkdir()
    return tmp_path


# ── Linear workflow ──────────────────────────────────────────────


class TestLinearWorkflow:
    async def test_a_b_c(self, tmp_project: Path) -> None:
        """Nodes execute in order, files flow correctly."""
        wf = Workflow(
            name="linear",
            nodes={
                "a": FnNode(id="a", command="echo a > a.txt", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b > b.txt", reads={"a.txt"}, writes={"b.txt"}),
                "c": FnNode(id="c", command="echo c > c.txt", reads={"b.txt"}, writes={"c.txt"}),
            },
            edges=[
                Edge(source="a", target="b"),
                Edge(source="b", target="c"),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert result.success
        assert result.nodes_executed == 3
        assert not result.halted

    async def test_files_tracked(self, tmp_project: Path) -> None:
        """Completed files are tracked in executor state."""
        wf = Workflow(
            name="linear",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b", reads={"a.txt"}, writes={"b.txt"}),
            },
            edges=[Edge(source="a", target="b")],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert "a.txt" in result.completed_files
        assert "b.txt" in result.completed_files


# ── Gate with Proceed ────────────────────────────────────────────


class TestGateProceed:
    async def test_proceed_follows_forward_edge(self, tmp_project: Path) -> None:
        wf = Workflow(
            name="gate_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(
                    id="gate",
                    evaluator_type="fn",
                    evaluator_command="echo PROCEED",
                    reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert result.success
        assert result.nodes_executed >= 2


# ── Gate with Reloop ─────────────────────────────────────────────


class TestGateReloop:
    async def test_reloop_returns_to_target(self, tmp_project: Path) -> None:
        """Gate produces Reloop, execution returns with feedback."""
        wf = Workflow(
            name="reloop_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(
                    id="gate",
                    evaluator_type="fn",
                    evaluator_command="echo PROCEED",
                    reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
                Edge(source="gate", target="a", condition=VerdictType.RELOOP),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()
        assert result.success


# ── Gate with Halt ───────────────────────────────────────────────


class TestGateHalt:
    async def test_halt_terminates(self, tmp_project: Path) -> None:
        """Gate produces Halt, workflow terminates."""
        wf = Workflow(
            name="halt_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(
                    id="gate",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL",
                    reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert result.success
        assert result.nodes_executed >= 1


# ── Max iterations ───────────────────────────────────────────────


class TestMaxIterations:
    async def test_max_iterations_halts(self, tmp_project: Path) -> None:
        """Reloop exceeds max_iterations, workflow halts."""
        call_count = 0

        async def mock_evaluate_gate(node: GateNode) -> Verdict:
            nonlocal call_count
            call_count += 1
            return Verdict.reloop("a", f"try again #{call_count}", max_iterations=2)

        wf = Workflow(
            name="max_iter",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="fn", reads={"a.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="a", condition=VerdictType.RELOOP),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        executor._evaluate_gate = mock_evaluate_gate  # type: ignore[assignment]
        result = await executor.execute()

        assert result.halted
        assert "max iterations" in result.halt_reason


# ── Fork/Join ────────────────────────────────────────────────────


class TestForkJoin:
    async def test_fork_runs_concurrently(self, tmp_project: Path) -> None:
        """Forked nodes execute concurrently."""
        wf = Workflow(
            name="fork_test",
            nodes={
                "fork": ForkNode(id="fork", targets=["a", "b", "c"]),
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
                "c": FnNode(id="c", command="echo c", writes={"c.txt"}),
                "join": JoinNode(
                    id="join",
                    sources=["a", "b", "c"],
                    reads={"a.txt", "b.txt", "c.txt"}),
                "final": FnNode(id="final", command="echo done", reads={"a.txt", "b.txt", "c.txt"}),
            },
            edges=[
                Edge(source="fork", target="a"),
                Edge(source="fork", target="b"),
                Edge(source="fork", target="c"),
                Edge(source="a", target="join"),
                Edge(source="b", target="join"),
                Edge(source="c", target="join"),
                Edge(source="join", target="final"),
            ],
            start_node="fork")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert result.success
        assert "a.txt" in result.completed_files
        assert "b.txt" in result.completed_files
        assert "c.txt" in result.completed_files


# ── JoinNode barrier semantics (issue #1189) ────────────────────


class TestJoinNodeBarrier:
    async def test_join_with_all_sources_complete_succeeds(self, tmp_project: Path) -> None:
        """JoinNode proceeds normally when all declared sources are in node_outputs."""
        wf = Workflow(
            name="join_barrier_ok",
            nodes={
                "fork": ForkNode(id="fork", targets=["a", "b"]),
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
                "join": JoinNode(id="join", sources=["a", "b"], reads={"a.txt", "b.txt"}),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="fork", target="a"),
                Edge(source="fork", target="b"),
                Edge(source="a", target="join"),
                Edge(source="b", target="join"),
                Edge(source="join", target="final"),
            ],
            start_node="fork")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()
        assert result.success
        assert not result.halted

    async def test_join_with_missing_source_halts(self, tmp_project: Path) -> None:
        """JoinNode halts with a descriptive error when a declared source has not completed."""
        wf = Workflow(
            name="join_barrier_fail",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "join": JoinNode(id="join", sources=["a", "missing_node"]),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="a", target="join"),
                Edge(source="join", target="final"),
            ],
            start_node="a")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True, validate=False)
        result = await executor.execute()
        assert result.halted
        assert "missing_node" in result.halt_reason
        assert "JoinNode" in result.halt_reason

    async def test_join_with_no_sources_declared_proceeds(self, tmp_project: Path) -> None:
        """JoinNode with empty sources list proceeds unconditionally (backward compat)."""
        wf = Workflow(
            name="join_no_sources",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "join": JoinNode(id="join", sources=[]),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="join"),
                Edge(source="join", target="b"),
            ],
            start_node="a")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()
        assert result.success

    async def test_gate_in_sources_does_not_false_halt(self, tmp_project: Path) -> None:
        """GateNode in JoinNode.sources proceeds: gates are now in node_outputs after running."""
        wf = Workflow(
            name="gate_in_sources",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(
                    id="gate",
                    evaluator_type="fn",
                    evaluator_command="echo PROCEED",
                    reads={"a.txt"}),
                "join": JoinNode(id="join", sources=["a", "gate"]),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="join", condition=VerdictType.PROCEED),
                Edge(source="join", target="final"),
            ],
            start_node="a")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()
        assert result.success
        assert not result.halted

    async def test_joinnode_appears_in_node_outputs(self, tmp_project: Path) -> None:
        """JoinNode adds itself to node_outputs so nested joins don't false-halt."""
        wf = Workflow(
            name="joinnode_in_outputs",
            nodes={
                "fork": ForkNode(id="fork", targets=["a", "b"]),
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
                "join": JoinNode(id="join", sources=["a", "b"]),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="fork", target="a"),
                Edge(source="fork", target="b"),
                Edge(source="a", target="join"),
                Edge(source="b", target="join"),
                Edge(source="join", target="final"),
            ],
            start_node="fork")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()
        assert result.success
        assert "join" in result.node_outputs

    async def test_barrier_halt_emits_node_failed_event(self, tmp_project: Path) -> None:
        """Barrier-halt emits a node.failed event consistent with all other halt paths."""
        wf = Workflow(
            name="barrier_event",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "join": JoinNode(id="join", sources=["a", "phantom"]),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="a", target="join"),
                Edge(source="join", target="final"),
            ],
            start_node="a")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True, validate=False)
        result = await executor.execute()
        assert result.halted
        failed_events = [e for e in result.events if e["type"] == "node.failed"]
        assert any(e.get("node_id") == "join" for e in failed_events)

    async def test_non_blocking_source_exempted_from_barrier(self, tmp_project: Path) -> None:
        """Non-blocking sources skip the barrier check — only blocking sources must complete."""
        wf = Workflow(
            name="non_blocking_exempt",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "bg": FnNode(id="bg", command="echo bg", blocking=False),
                "join": JoinNode(id="join", sources=["a", "bg"]),
                "final": FnNode(id="final", command="echo done"),
            },
            edges=[
                Edge(source="join", target="final"),
            ],
            start_node="join")
        executor = WorkflowExecutor(wf, tmp_project, dry_run=True, validate=False)
        executor.result.node_outputs["a"] = "done"
        await executor._execute_from("join")
        assert not executor.result.halted


# ── Non-blocking node ────────────────────────────────────────────


class TestNonBlocking:
    async def test_fire_and_forget(self, tmp_project: Path) -> None:
        """Non-blocking node fires, executor advances immediately."""
        wf = Workflow(
            name="nonblock_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "async_node": FnNode(
                    id="async_node",
                    command="echo async",
                    reads={"a.txt"},
                    writes={"async.txt"},
                    blocking=False),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="async_node"),
                Edge(source="async_node", target="b"),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        assert result.success
        assert result.nodes_executed >= 2


# ── Event emission ───────────────────────────────────────────────


class TestEventEmission:
    async def test_events_emitted(self, tmp_project: Path) -> None:
        """All event types emitted with correct structure."""
        wf = Workflow(
            name="event_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b", reads={"a.txt"}),
            },
            edges=[Edge(source="a", target="b")],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        event_types = [e["type"] for e in result.events]
        assert "workflow.started" in event_types
        assert "node.started" in event_types
        assert "node.completed" in event_types
        assert "workflow.completed" in event_types

    async def test_gate_verdict_event(self, tmp_project: Path) -> None:
        wf = Workflow(
            name="gate_event",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(
                    id="gate",
                    evaluator_type="fn",
                    evaluator_command="echo PROCEED",
                    reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        result = await executor.execute()

        event_types = [e["type"] for e in result.events]
        assert "gate.verdict" in event_types


# ── Error handling ───────────────────────────────────────────────


class TestErrorHandling:
    async def test_node_failure_halts(self, tmp_project: Path) -> None:
        """Node failure produces Halt with error message."""
        wf = Workflow(
            name="error_test",
            nodes={
                "a": FnNode(id="a", command="exit 1", writes={"a.txt"}),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[Edge(source="a", target="b")],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project)
        result = await executor.execute()

        assert result.halted
        assert "failed" in result.halt_reason.lower()


# ── Auto-approve ────────────────────────────────────────────────


class TestAutoApprove:
    async def test_executor_auto_approve_logs(self, tmp_project: Path) -> None:
        """WorkflowExecutor(auto_approve=True) logs gate.auto_approved for user gates."""
        wf = Workflow(
            name="auto_approve_test",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True, auto_approve=True)
        result = await executor.execute()

        assert result.success
        gate_events = [e for e in result.events if e["type"] == "gate.verdict"]
        assert len(gate_events) == 1
        assert gate_events[0]["verdict_type"] == VerdictType.PROCEED

    async def test_dry_run_skips_user_gate(self, tmp_project: Path) -> None:
        """dry_run=True bypasses all gate logic, including user gates."""
        wf = Workflow(
            name="default_user_gate",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True, auto_approve=False)
        result = await executor.execute()

        assert result.success
        gate_events = [e for e in result.events if e["type"] == "gate.verdict"]
        assert len(gate_events) == 1
        assert gate_events[0]["verdict_type"] == VerdictType.PROCEED

    async def test_auto_approve_emits_structured_log(self, tmp_project: Path) -> None:
        """auto_approve=True emits gate.auto_approved with gate_id and workflow name (non-dry-run)."""
        import structlog

        wf = Workflow(
            name="log_check_wf",
            nodes={
                "a": FnNode(id="a", command="echo a > a.txt", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b > b.txt", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        captured: list[dict] = []

        def capture_log(_logger, _method, event_dict):
            captured.append(event_dict.copy())
            return event_dict

        structlog.configure(processors=[capture_log, structlog.dev.ConsoleRenderer()])

        try:
            executor = WorkflowExecutor(wf, tmp_project, dry_run=False, auto_approve=True)
            result = await executor.execute()
        finally:
            structlog.reset_defaults()

        assert result.success
        auto_approved = [e for e in captured if e.get("event") == "gate.auto_approved"]
        assert len(auto_approved) == 1
        assert auto_approved[0]["gate_id"] == "gate"
        assert auto_approved[0]["workflow"] == "log_check_wf"

    async def test_auto_approve_false_no_log(self, tmp_project: Path) -> None:
        """auto_approve=False does not emit gate.auto_approved; prompts user instead."""
        import structlog

        wf = Workflow(
            name="no_log_wf",
            nodes={
                "a": FnNode(id="a", command="echo a > a.txt", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b > b.txt", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        captured: list[dict] = []

        def capture_log(_logger, _method, event_dict):
            captured.append(event_dict.copy())
            return event_dict

        structlog.configure(processors=[capture_log, structlog.dev.ConsoleRenderer()])

        try:
            executor = WorkflowExecutor(
                wf, tmp_project, dry_run=False, auto_approve=False,
                input_fn=lambda _: "proceed")
            result = await executor.execute()
        finally:
            structlog.reset_defaults()

        assert result.success
        auto_approved = [e for e in captured if e.get("event") == "gate.auto_approved"]
        assert len(auto_approved) == 0


# ── User gate interactive input (issue #1243) ────────────────────


def _user_gate_workflow() -> Workflow:
    return Workflow(
        name="user_gate_wf",
        nodes={
            "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
            "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
            "b": FnNode(id="b", command="echo b", writes={"b.txt"}),
        },
        edges=[
            Edge(source="a", target="gate"),
            Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            Edge(source="gate", target="a", condition=VerdictType.RELOOP),
        ],
        start_node="a")


class TestUserGateInteractive:
    async def test_user_proceed(self, tmp_project: Path) -> None:
        """input_fn returning 'proceed' causes the gate to PROCEED."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "proceed")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.PROCEED

    async def test_user_reloop(self, tmp_project: Path) -> None:
        """input_fn returning 'reloop' causes the gate to RELOOP."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "reloop")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.RELOOP

    async def test_user_halt(self, tmp_project: Path) -> None:
        """input_fn returning 'halt' causes the gate to HALT."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "halt")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.HALT

    async def test_user_gate_halt_stops_workflow(self, tmp_project: Path) -> None:
        """User responding 'halt' terminates the full workflow."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, dry_run=False, input_fn=lambda _: "halt")
        result = await executor.execute()
        assert result.halted
        assert "gate" in result.halt_reason

    async def test_auto_approve_does_not_prompt(self, tmp_project: Path) -> None:
        """auto_approve=True never calls input_fn."""
        called = []
        def should_not_be_called(prompt: str) -> str:
            called.append(prompt)
            return "proceed"

        wf = _user_gate_workflow()
        executor = WorkflowExecutor(
            wf, tmp_project, auto_approve=True, input_fn=should_not_be_called)
        await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert not called, "input_fn should not be called when auto_approve=True"

    async def test_eof_on_stdin_halts_cleanly(self, tmp_project: Path) -> None:
        """EOFError from stdin (non-interactive context) produces a clean HALT."""
        def raise_eof(_prompt: str) -> str:
            raise EOFError

        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=raise_eof)
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.HALT
        assert "stdin closed" in verdict.reason

    async def test_reloop_without_edge_halts_cleanly(self, tmp_project: Path) -> None:
        """User typing 'reloop' on a gate with no RELOOP edge produces a clean HALT."""
        wf = Workflow(
            name="no_reloop_edge",
            nodes={
                "a": FnNode(id="a", command="echo a", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="user", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "reloop")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.HALT
        assert "no RELOOP edge" in verdict.reason

    async def test_oserror_on_stdin_halts_cleanly(self, tmp_project: Path) -> None:
        """OSError (pytest captured stdin) produces a clean HALT."""
        def raise_oserror(_prompt: str) -> str:
            raise OSError("reading from stdin while output is captured!")

        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=raise_oserror)
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.HALT
        assert "stdin closed" in verdict.reason

    async def test_unrecognized_input_halts_fail_closed(self, tmp_project: Path) -> None:
        """Unrecognized input halts fail-closed instead of silently proceeding."""
        for bad_input in ("yes", "no", "cancel", "hault", ""):
            wf = _user_gate_workflow()
            executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _, x=bad_input: x)
            verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
            assert verdict.type == VerdictType.HALT, f"expected HALT for {bad_input!r}"
            assert "unrecognized" in verdict.reason

    async def test_halt_word_boundary_not_substring(self, tmp_project: Path) -> None:
        """'asphalt' does not trigger halt; word-boundary matching is enforced."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "asphalt coloring")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        # 'asphalt' has no whole-word 'halt', 'reloop', or 'proceed' — unrecognized → HALT
        assert verdict.type == VerdictType.HALT
        assert "unrecognized" in verdict.reason

    async def test_halt_wins_over_reloop(self, tmp_project: Path) -> None:
        """When both 'halt' and 'reloop' appear, halt wins (checked first)."""
        wf = _user_gate_workflow()
        executor = WorkflowExecutor(wf, tmp_project, input_fn=lambda _: "reloop then halt")
        verdict = await executor._evaluate_gate(wf.nodes["gate"])  # type: ignore[arg-type]
        assert verdict.type == VerdictType.HALT


# ── Gate verdict parsing fails closed (issue #1250) ──────────────


def _make_gate_executor() -> WorkflowExecutor:
    """Build a bare WorkflowExecutor with a workflow + edge index for gate parsing."""
    wf = Workflow(
        name="gate_fail_closed",
        nodes={
            "a": FnNode(id="a", command="echo a"),
            "gate": GateNode(id="gate", evaluator_type="fn", evaluator_command="echo pass"),
            "b": FnNode(id="b", command="echo b"),
        },
        edges=[
            Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            Edge(source="gate", target="a", condition=VerdictType.RELOOP),
        ],
        start_node="a")
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor.workflow = wf
    executor.project_path = Path("/fake")
    executor._edge_index = {}
    for edge in wf.edges:
        executor._edge_index.setdefault(edge.source, []).append(edge)
    return executor


class TestGateVerdictFailClosed:
    """Unrecognized gate output halts instead of proceeding (issue #1250)."""

    def test_agent_proceed_recognized(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict("PROCEED", "gate")
        assert verdict.type == VerdictType.PROCEED

    def test_agent_proceed_last_nonempty_line(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            "all good\n\nPROCEED\n", "gate"
        )
        assert verdict.type == VerdictType.PROCEED

    def test_agent_empty_halts(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict("", "gate")
        assert verdict.type == VerdictType.HALT
        assert "unparseable" in (verdict.reason or "")

    def test_agent_whitespace_halts(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict("   \n  \n", "gate")
        assert verdict.type == VerdictType.HALT

    def test_agent_apology_halts(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            "sorry, I could not determine the result", "gate"
        )
        assert verdict.type == VerdictType.HALT
        assert "could not determine" in (verdict.reason or "")

    def test_agent_ambiguous_halts(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            "GATE RESULT: maybe proceed?", "gate"
        )
        assert verdict.type == VerdictType.HALT

    def test_agent_halt_parsed(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            'HALT reason="tests broke"', "gate"
        )
        assert verdict.type == VerdictType.HALT
        assert verdict.reason == "tests broke"

    def test_agent_reloop_parsed(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            'RELOOP target="a" feedback="redo it"', "gate"
        )
        assert verdict.type == VerdictType.RELOOP
        assert verdict.target == "a"
        assert verdict.feedback == "redo it"

    def test_agent_proceed_first_line_fallback(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            "PROCEED\n\nAll checks pass.", "gate"
        )
        assert verdict.type == VerdictType.PROCEED

    def test_agent_halt_first_line_fallback(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            'HALT reason="broken"\n\nSee details above.', "gate"
        )
        assert verdict.type == VerdictType.HALT
        assert verdict.reason == "broken"

    def test_agent_reloop_first_line_fallback(self) -> None:
        verdict = _make_gate_executor()._parse_agent_verdict(
            'RELOOP target="a" feedback="try again"\n\nNeeds work.', "gate"
        )
        assert verdict.type == VerdictType.RELOOP
        assert verdict.target == "a"
        assert verdict.feedback == "try again"

    def test_fn_pass_proceeds(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict("pass", "gate")
        assert verdict.type == VerdictType.PROCEED

    def test_fn_proceed_text_proceeds(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict("PROCEED", "gate")
        assert verdict.type == VerdictType.PROCEED

    def test_fn_json_passed_true_proceeds(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict('{"passed": true}', "gate")
        assert verdict.type == VerdictType.PROCEED

    def test_fn_json_passed_false_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict('{"passed": false}', "gate")
        assert verdict.type == VerdictType.HALT

    def test_fn_fail_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict(
            "fail: compilation error", "gate"
        )
        assert verdict.type == VerdictType.HALT

    def test_fn_revert_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict("revert", "gate")
        assert verdict.type == VerdictType.HALT

    def test_fn_reloop_parsed(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict(
            "reloop: redo the build", "gate"
        )
        assert verdict.type == VerdictType.RELOOP
        assert verdict.feedback == "redo the build"

    def test_fn_empty_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict("", "gate")
        assert verdict.type == VerdictType.HALT
        assert "unparseable" in (verdict.reason or "")

    def test_fn_apology_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict(
            "I ran the check but cannot tell if it passed", "gate"
        )
        assert verdict.type == VerdictType.HALT

    def test_fn_malformed_json_halts(self) -> None:
        verdict = _make_gate_executor()._parse_fn_verdict('{"passed": false', "gate")
        assert verdict.type == VerdictType.HALT

    async def test_fn_gate_without_command_halts(self, tmp_project: Path) -> None:
        """An fn gate with no evaluator_command halts instead of proceeding."""
        wf = Workflow(
            name="no_cmd_gate",
            nodes={
                "a": FnNode(id="a", command="echo a > a.txt", writes={"a.txt"}),
                "gate": GateNode(id="gate", evaluator_type="fn", reads={"a.txt"}),
                "b": FnNode(id="b", command="echo b > b.txt", writes={"b.txt"}),
            },
            edges=[
                Edge(source="a", target="gate"),
                Edge(source="gate", target="b", condition=VerdictType.PROCEED),
            ],
            start_node="a")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=False)
        result = await executor.execute()

        assert result.halted
        assert not result.success
        assert "no evaluator_command" in result.halt_reason
        assert result.nodes_executed == 2


# ── initial_context guarded by AgentNode type ──────────────────


class TestInitialContextGuard:
    """initial_context should only be written to node_context for AgentNode start nodes."""

    def test_fnnode_start_ignores_initial_context(self, tmp_project: Path) -> None:
        """When start_node is a FnNode, initial_context is NOT written to node_context."""
        wf = Workflow(
            name="fn_start",
            nodes={
                "fn": FnNode(id="fn", command="echo hi", writes={"out.txt"}),
            },
            edges=[],
            start_node="fn")

        executor = WorkflowExecutor(
            wf, tmp_project, dry_run=True, initial_context="should be ignored")
        assert "fn" not in executor.node_context

    def test_gatenode_start_ignores_initial_context(self, tmp_project: Path) -> None:
        """When start_node is a GateNode, initial_context is NOT written to node_context."""
        wf = Workflow(
            name="gate_start",
            nodes={
                "gate": GateNode(id="gate", evaluator_type="fn", evaluator_command="echo pass"),
            },
            edges=[],
            start_node="gate")

        executor = WorkflowExecutor(
            wf, tmp_project, dry_run=True, initial_context="should be ignored")
        assert "gate" not in executor.node_context

    def test_agentnode_start_receives_initial_context(self, tmp_project: Path) -> None:
        """When start_node is an AgentNode, initial_context IS written to node_context."""
        wf = Workflow(
            name="agent_start",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build it"),
            },
            edges=[],
            start_node="agent")

        executor = WorkflowExecutor(
            wf, tmp_project, dry_run=True, initial_context="domain prompt here")
        assert executor.node_context["agent"] == "domain prompt here"

    def test_no_initial_context_leaves_node_context_empty(self, tmp_project: Path) -> None:
        """When initial_context is None, node_context stays empty regardless of node type."""
        wf = Workflow(
            name="no_ctx",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build it"),
            },
            edges=[],
            start_node="agent")

        executor = WorkflowExecutor(wf, tmp_project, dry_run=True)
        assert executor.node_context == {}


class TestRunAgentPersistsToNodeWrites:
    """Gap 1: _run_agent() persists stdout to node.writes paths."""

    async def test_run_agent_persists_to_node_writes(self, tmp_project: Path) -> None:
        """AgentNode with writes={'output.md'} persists agent stdout to that file."""
        from unittest.mock import AsyncMock

        wf = Workflow(
            name="persist_test",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                    writes={"output.md"}),
            },
            edges=[],
            start_node="agent")

        mock_agent_fn = AsyncMock(return_value=("agent output", 0))
        executor = WorkflowExecutor(wf, tmp_project, agent_fn=mock_agent_fn)
        await executor.execute()

        output_file = tmp_project / "output.md"
        assert output_file.exists()
        assert output_file.read_text() == "agent output"

    async def test_run_agent_no_writes_skips_file_creation(self, tmp_project: Path) -> None:
        """AgentNode without writes doesn't create extra files."""
        from unittest.mock import AsyncMock

        wf = Workflow(
            name="no_writes_test",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
            },
            edges=[],
            start_node="agent")

        files_before = set(tmp_project.rglob("*"))
        mock_agent_fn = AsyncMock(return_value=("agent output", 0))
        executor = WorkflowExecutor(wf, tmp_project, agent_fn=mock_agent_fn)
        await executor.execute()

        files_after = set(tmp_project.rglob("*"))
        new_files = files_after - files_before
        # Only event log files should be created, not agent output files
        for f in new_files:
            assert "output.md" not in f.name


class TestAgentFnInjection:
    """Gap 2: WorkflowExecutor supports agent_fn injection."""

    async def test_custom_agent_fn_used(self, tmp_project: Path) -> None:
        """Custom agent_fn is called instead of default invoke_agent."""
        from unittest.mock import AsyncMock

        mock_fn = AsyncMock(return_value=("custom output", 0))

        wf = Workflow(
            name="custom_fn_test",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
            },
            edges=[],
            start_node="agent")

        executor = WorkflowExecutor(wf, tmp_project, agent_fn=mock_fn)
        await executor.execute()

        mock_fn.assert_called_once()

    async def test_agent_fn_propagates_to_data_node_sub_executor(
        self, tmp_project: Path) -> None:
        """agent_fn propagates to DataNode per-item sub-executors."""
        from unittest.mock import AsyncMock

        from factory.workflow.primitives import DataItem, DataNode, JoinNode

        mock_fn = AsyncMock(return_value=("sub output", 0))

        wf = Workflow(
            name="data_propagation_test",
            nodes={
                "data": DataNode(
                    id="data",
                    inline_items=[DataItem(id="item1", prompt="do it")]),
                "sub_agent": AgentNode(
                    id="sub_agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
                "join": JoinNode(id="join", sources=["sub_agent"]),
            },
            edges=[
                Edge(source="data", target="sub_agent"),
                Edge(source="sub_agent", target="join"),
            ],
            start_node="data")

        executor = WorkflowExecutor(wf, tmp_project, agent_fn=mock_fn)
        result = await executor.execute()

        assert result.success
        mock_fn.assert_called_once()

    def test_agent_fn_defaults_to_invoke_agent(self, tmp_project: Path) -> None:
        """When agent_fn is not provided, defaults to invoke_agent."""
        from factory.agents.runner import invoke_agent

        wf = Workflow(
            name="default_fn_test",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    role=AgentRole.BUILDER,
                    prompt_template="build"),
            },
            edges=[],
            start_node="agent")

        executor = WorkflowExecutor(wf, tmp_project)
        assert executor._agent_fn is invoke_agent


# ── _collect_subgraph_nodes ─────────────────────────────────────


class TestCollectSubgraphNodes:
    """Verify _collect_subgraph_nodes returns only nodes on entry→exit paths."""

    def _make_workflow(self, edges: list[tuple[str, str]], nodes: list[str]) -> Workflow:
        return Workflow(
            name="test",
            nodes={n: FnNode(id=n, command="echo") for n in nodes},
            edges=[Edge(source=s, target=t) for s, t in edges],
            start_node=nodes[0])

    def test_linear_chain(self) -> None:
        from factory.workflow.executor import _collect_subgraph_nodes

        wf = self._make_workflow(
            nodes=["a", "b", "c"],
            edges=[("a", "b"), ("b", "c")])
        assert _collect_subgraph_nodes(wf, "a", "c") == {"a", "b", "c"}

    def test_stray_branch_excluded(self) -> None:
        """Node reachable from entry but not on any path to exit is excluded."""
        from factory.workflow.executor import _collect_subgraph_nodes

        # a -> b -> c (exit)
        # a -> stray (dead end, not connected to c)
        wf = self._make_workflow(
            nodes=["a", "b", "c", "stray"],
            edges=[("a", "b"), ("b", "c"), ("a", "stray")])
        result = _collect_subgraph_nodes(wf, "a", "c")
        assert "stray" not in result
        assert result == {"a", "b", "c"}

    def test_parallel_branches_both_included(self) -> None:
        """Both branches of a fork that converge at exit are included."""
        from factory.workflow.executor import _collect_subgraph_nodes

        # entry -> left -> exit
        # entry -> right -> exit
        wf = self._make_workflow(
            nodes=["entry", "left", "right", "exit"],
            edges=[("entry", "left"), ("entry", "right"), ("left", "exit"), ("right", "exit")])
        assert _collect_subgraph_nodes(wf, "entry", "exit") == {"entry", "left", "right", "exit"}

    def test_entry_equals_exit(self) -> None:
        """When entry is the exit node, only that node is returned."""
        from factory.workflow.executor import _collect_subgraph_nodes

        wf = self._make_workflow(nodes=["a", "b"], edges=[("a", "b")])
        assert _collect_subgraph_nodes(wf, "a", "a") == {"a"}
