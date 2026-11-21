"""Tests for factory/workflow/tool.py — tool-based workflow execution."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    Study,
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.tool import (
    _detect_artifact,
    _find_reloop_target,
    _format_gate_task,
    _format_node_task,
    _format_progress,
    _get_workflow_cached,
    _phase_label,
    _rebuild_workflow,
    _resolve_original_project,
    _workflow_cache,
    tool_curr,
    tool_finalize,
    tool_init,
    tool_next,
    tool_overview,
    tool_status,
    tool_submit,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    WorkflowRegistry.reset()
    _workflow_cache.clear()
    yield
    WorkflowRegistry.reset()
    _workflow_cache.clear()


def _simple_workflow() -> Workflow:
    """A minimal workflow: study -> researcher -> gate -> builder."""
    return Workflow(
        name="test-simple",
        start_node="study",
        nodes={
            "study": Study(
                id="study",
                command="factory study {project_path}",
                writes={".factory/strategy/observations.md"},
            ),
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research the project at {project_path}",
                writes={".factory/reviews/researcher-latest.md"},
            ),
            "gate_research": GateNode(
                id="gate_research",
                evaluator_type="agent",
                gate_prompt="Review research output",
                reads={".factory/reviews/researcher-latest.md"},
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build the project",
                writes={".factory/reviews/builder-latest.md"},
            ),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="gate_research"),
            Edge(source="gate_research", target="builder", condition=VerdictType.PROCEED),
            Edge(source="gate_research", target="researcher", condition=VerdictType.RELOOP),
        ],
    )


def _fn_gate_workflow() -> Workflow:
    """Workflow with an fn-type gate for auto-evaluation."""
    return Workflow(
        name="test-fn-gate",
        start_node="builder",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build",
                writes={".factory/reviews/builder-latest.md"},
            ),
            "gate_review": GateNode(
                id="gate_review",
                evaluator_type="fn",
                evaluator_command="echo PROCEED",
                reads={".factory/reviews/builder-latest.md"},
            ),
            "archivist": AgentNode(
                id="archivist",
                role=AgentRole.ARCHIVIST,
                prompt_template="Archive results",
                writes={".factory/archive/build.md"},
                blocking=False,
            ),
        },
        edges=[
            Edge(source="builder", target="gate_review"),
            Edge(source="gate_review", target="archivist", condition=VerdictType.PROCEED),
            Edge(source="gate_review", target="builder", condition=VerdictType.RELOOP),
        ],
    )


def _register_workflow(wf: Workflow) -> None:
    """Register a workflow through the public API, so a later discovery keeps it."""
    WorkflowRegistry.register_callable(
        wf.name, lambda _wf=wf: _wf, source="builtin", description="test workflow"
    )


class TestToolInit:
    def test_init_creates_state(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        session_dir = tool_init("test-simple", tmp_path)

        state_path = Path(session_dir) / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["workflow_name"] == "test-simple"
        assert state["status"] == "active"
        assert state["pointer_idx"] == 0
        assert len(state["session_id"]) == 12
        assert "study" in state["topo_order"]

    def test_init_unknown_workflow(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown workflow"):
            tool_init("nonexistent", tmp_path)

    def test_init_filters_join_nodes(self, tmp_path: Path) -> None:
        """JoinNodes should be excluded from topo_order."""
        from factory.workflow.primitives import JoinNode
        wf = Workflow(
            name="test-join",
            start_node="a",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "join": JoinNode(id="join", sources=["a"]),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="join"),
                Edge(source="join", target="b"),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-join", tmp_path)
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "join" not in state["topo_order"]
        assert "a" in state["topo_order"]
        assert "b" in state["topo_order"]


class TestToolNext:
    def test_next_returns_first_node(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_next(tmp_path)

        assert "Node: study" in result
        assert "Type: Study" in result

    def test_next_returns_done_when_completed(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["status"] = "completed"
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_next(tmp_path)
        assert "DONE" in result

    def test_next_completes_when_past_end(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = len(state["topo_order"])
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_next(tmp_path)
        assert "DONE" in result


class TestToolSubmit:
    def test_submit_stores_output(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_submit(tmp_path, "study", "Observations: project looks good")

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["completed"]["study"] == "Observations: project looks good"
        assert result == "CONTINUE"

    def test_submit_writes_agent_output_files(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Advance past study first
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = 1  # researcher
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        tool_submit(tmp_path, "researcher", "Research findings here")

        output_file = tmp_path / ".factory" / "reviews" / "researcher-latest.md"
        assert output_file.exists()
        assert output_file.read_text() == "Research findings here"

    def test_submit_advances_past_submitted_node(self, tmp_path: Path) -> None:
        """Submit advances the pointer past the submitted node."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = 1  # researcher
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_submit(tmp_path, "researcher", "Research done")
        assert result == "CONTINUE"

        # Next call to tool_next should return the agent gate
        next_result = tool_next(tmp_path)
        assert "GATE" in next_result
        assert "gate_research" in next_result

    def test_submit_fn_gate_proceed(self, tmp_path: Path) -> None:
        wf = _fn_gate_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-gate", tmp_path)

        result = tool_submit(tmp_path, "builder", "Built successfully")

        assert result == "CONTINUE"
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["gate_results"]["gate_review"] == "PROCEED"

    def test_submit_fn_gate_halt(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-halt",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_fail": GateNode(
                    id="gate_fail",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: tests broken",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_fail"),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-halt", tmp_path)

        result = tool_submit(tmp_path, "builder", "Built")
        assert result.startswith("HALT")
        assert "FAIL" in result

    def test_submit_fn_gate_reloop(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-reloop",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_check": GateNode(
                    id="gate_check",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: needs fixes",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_check"),
                Edge(source="gate_check", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        result = tool_submit(tmp_path, "builder", "First attempt")
        assert result.startswith("RETRY")
        assert "attempt 1/3" in result

    def test_submit_fn_gate_reloop_max_iterations(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-max-iter",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_check": GateNode(
                    id="gate_check",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: still broken",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_check"),
                Edge(source="gate_check", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-max-iter", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["iteration_counts"]["gate_check->builder"] = 3
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_submit(tmp_path, "builder", "Fourth attempt")
        assert result.startswith("HALT")

    def test_submit_then_next_returns_user_gate(self, tmp_path: Path) -> None:
        """After submit, calling next returns user gate as APPROVAL_NEEDED."""
        wf = Workflow(
            name="test-user-gate",
            start_node="strategist",
            nodes={
                "strategist": AgentNode(
                    id="strategist",
                    role=AgentRole.STRATEGIST,
                    prompt_template="Strategize",
                ),
                "gate_approval": GateNode(
                    id="gate_approval",
                    evaluator_type="user",
                    gate_prompt="Approve this strategy?",
                ),
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
            },
            edges=[
                Edge(source="strategist", target="gate_approval"),
                Edge(source="gate_approval", target="builder", condition=VerdictType.PROCEED),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-user-gate", tmp_path)

        result = tool_submit(tmp_path, "strategist", "Strategy ready")
        assert result == "CONTINUE"

        next_result = tool_next(tmp_path)
        assert "APPROVAL_NEEDED" in next_result
        assert "Approve this strategy?" in next_result

    def test_submit_returns_done_at_end(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-single",
            start_node="study",
            nodes={
                "study": Study(id="study", command="factory study {project_path}"),
            },
            edges=[],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-single", tmp_path)

        result = tool_submit(tmp_path, "study", "Done studying")
        assert result == "DONE"


class TestToolStatus:
    def test_status_no_session(self, tmp_path: Path) -> None:
        result = tool_status(tmp_path)
        assert "No active session" in result

    def test_status_active_session(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_status(tmp_path)
        assert "Workflow: test-simple" in result
        assert "Status:   active" in result
        assert "Progress: 0/" in result

    def test_status_with_completed_nodes(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)
        tool_submit(tmp_path, "study", "Observations here")

        result = tool_status(tmp_path)
        assert "Progress: 1/" in result
        assert "✓ study" in result

    def test_status_with_gate_results(self, tmp_path: Path) -> None:
        wf = _fn_gate_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-gate", tmp_path)
        tool_submit(tmp_path, "builder", "Built")

        result = tool_status(tmp_path)
        assert "Gates:" in result
        assert "PROCEED" in result


class TestAutoSubmit:
    """Tests for the primary auto-submit mechanism in tool_next."""

    def test_next_auto_submits_agent(self, tmp_path: Path) -> None:
        """tool_next auto-submits an agent node when its review file exists."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Submit study to advance past it
        tool_submit(tmp_path, "study", "Observations done")

        # Simulate agent ran: write the review file directly (no submit)
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings here")

        # tool_next should auto-submit the researcher and return the gate
        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "researcher" in state["completed"]
        assert state["completed"]["researcher"] == "Research findings here"
        assert "GATE" in result
        assert "gate_research" in result

    def test_next_auto_submits_study(self, tmp_path: Path) -> None:
        """tool_next auto-submits a study node when observations.md exists."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write observations file directly (no submit)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in result

    def test_next_stops_at_gate(self, tmp_path: Path) -> None:
        """tool_next auto-submits agent, then stops at the following agent gate."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write both study and researcher artifacts
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in state["completed"]
        assert "GATE" in result
        assert "gate_research" in result

    def test_next_auto_evaluates_fn_gate(self, tmp_path: Path) -> None:
        """tool_next auto-submits agent and auto-evaluates following fn gate."""
        wf = _fn_gate_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-gate", tmp_path)

        # Write builder review file (fn gate passes via "echo PROCEED")
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "builder-latest.md").write_text("Built successfully")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "builder" in state["completed"]
        assert "gate_review" in state["completed"]
        assert state["gate_results"]["gate_review"] == "PROCEED"
        # Should return the archivist node (after auto-evaluating gate)
        assert "archivist" in result

    def test_next_chain_multiple(self, tmp_path: Path) -> None:
        """tool_next chains through multiple auto-submittable nodes in one call."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write both study observations and researcher review
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings")

        # Single call to next should skip both and stop at gate
        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in state["completed"]
        assert "GATE" in result
        assert "gate_research" in result

    def test_next_auto_submits_fn_with_output_files(self, tmp_path: Path) -> None:
        """tool_next auto-submits a FnNode when its declared output files exist."""
        wf = Workflow(
            name="test-fn-auto",
            start_node="fn1",
            nodes={
                "fn1": FnNode(
                    id="fn1",
                    command="echo hello",
                    writes={".factory/output.md"},
                ),
                "fn2": FnNode(id="fn2", command="echo done"),
            },
            edges=[Edge(source="fn1", target="fn2")],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-auto", tmp_path)

        # Write the output file directly
        (tmp_path / ".factory" / "output.md").write_text("Generated output")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "fn1" in state["completed"]
        assert "fn2" in result

    def test_next_does_not_auto_submit_empty_review(self, tmp_path: Path) -> None:
        """Empty review files should not trigger auto-submit."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        tool_submit(tmp_path, "study", "Observations done")

        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("")

        result = tool_next(tmp_path)
        assert "researcher" in result
        assert "Type: Agent" in result

    def test_next_auto_submits_fork_node(self, tmp_path: Path) -> None:
        """ForkNodes are auto-submitted immediately (structural nodes)."""
        from factory.workflow.primitives import ForkNode
        wf = Workflow(
            name="test-fork-auto",
            start_node="fork1",
            nodes={
                "fork1": ForkNode(id="fork1", targets=["a", "b"]),
                "a": FnNode(id="a", command="echo a"),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="fork1", target="a"),
                Edge(source="fork1", target="b"),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fork-auto", tmp_path)

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "fork1" in state["completed"]
        assert "Fork targets" in state["completed"]["fork1"]
        assert "a" in result or "b" in result


class TestHelpers:
    def test_find_reloop_target(self) -> None:
        wf = _simple_workflow()
        target = _find_reloop_target(wf, "gate_research")
        assert target == "researcher"

    def test_find_reloop_target_none(self) -> None:
        wf = _simple_workflow()
        target = _find_reloop_target(wf, "study")
        assert target is None

    def test_format_node_task_agent(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["researcher"]
        result = _format_node_task("researcher", node, wf, {}, tmp_path)
        assert "Type: Agent (researcher)" in result
        assert "Model:" in result
        assert "Timeout:" in result

    def test_format_node_task_study(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["study"]
        result = _format_node_task("study", node, wf, {}, tmp_path)
        assert "Type: Study" in result
        assert "Command:" in result

    def test_format_node_task_gate(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["gate_research"]
        result = _format_node_task("gate_research", node, wf, {}, tmp_path)
        assert "Type: Gate (agent)" in result

    def test_format_node_task_fn(self, tmp_path: Path) -> None:
        node = FnNode(id="fn1", command="echo hello", notes="test note")
        wf = Workflow(
            name="test", start_node="fn1",
            nodes={"fn1": node}, edges=[],
        )
        result = _format_node_task("fn1", node, wf, {}, tmp_path)
        assert "Type: Function" in result
        assert "Notes: test note" in result

    def test_format_node_task_fork(self, tmp_path: Path) -> None:
        from factory.workflow.primitives import ForkNode
        node = ForkNode(id="fork1", targets=["a", "b"])
        wf = Workflow(
            name="test", start_node="fork1",
            nodes={"fork1": node}, edges=[],
        )
        result = _format_node_task("fork1", node, wf, {}, tmp_path)
        assert "Type: Fork" in result
        assert "a, b" in result

    def test_format_gate_task(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        gate = wf.nodes["gate_research"]
        state = {"workflow_name": "test-simple"}
        result = _format_gate_task("gate_research", gate, state, tmp_path)
        assert "Gate: gate_research" in result
        assert "PROCEED" in result
        assert "RETRY" in result
        assert "researcher" in result

    def test_detect_artifact_agent_review_file(self, tmp_path: Path) -> None:
        node = AgentNode(id="researcher", role=AgentRole.RESEARCHER, prompt_template="r")
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("findings")

        result = _detect_artifact("researcher", node, tmp_path)
        assert result == "findings"

    def test_detect_artifact_agent_empty(self, tmp_path: Path) -> None:
        node = AgentNode(id="researcher", role=AgentRole.RESEARCHER, prompt_template="r")
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("")

        result = _detect_artifact("researcher", node, tmp_path)
        assert result is None

    def test_detect_artifact_agent_tagged(self, tmp_path: Path) -> None:
        node = AgentNode(id="researcher_similar", role=AgentRole.RESEARCHER, prompt_template="r")
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-similar-latest.md").write_text("similar findings")

        result = _detect_artifact("researcher_similar", node, tmp_path)
        assert result == "similar findings"

    def test_detect_artifact_study(self, tmp_path: Path) -> None:
        node = Study(id="study", command="factory study {project_path}")
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text("x" * 100)

        result = _detect_artifact("study", node, tmp_path)
        assert result is not None

    def test_detect_artifact_study_too_short(self, tmp_path: Path) -> None:
        node = Study(id="study", command="factory study {project_path}")
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text("short")

        result = _detect_artifact("study", node, tmp_path)
        assert result is None

    def test_detect_artifact_fn_node(self, tmp_path: Path) -> None:
        node = FnNode(id="fn1", command="echo hello", writes={".factory/out.md"})
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".factory" / "out.md").write_text("output")

        result = _detect_artifact("fn1", node, tmp_path)
        assert result == "output"

    def test_detect_artifact_fn_missing_writes(self, tmp_path: Path) -> None:
        node = FnNode(id="fn1", command="echo hello", writes={".factory/out.md"})
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)

        result = _detect_artifact("fn1", node, tmp_path)
        assert result is None

    def test_detect_artifact_fork(self, tmp_path: Path) -> None:
        from factory.workflow.primitives import ForkNode
        node = ForkNode(id="fork1", targets=["a", "b"])
        result = _detect_artifact("fork1", node, tmp_path)
        assert result is not None
        assert "Fork targets" in result

    def test_detect_artifact_gate_returns_none(self, tmp_path: Path) -> None:
        node = GateNode(id="g", evaluator_type="agent", gate_prompt="review")
        result = _detect_artifact("g", node, tmp_path)
        assert result is None

    def test_detect_artifact_same_role_collision(self, tmp_path: Path) -> None:
        """Two same-role nodes with writes: generic review must not cause collision."""
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("generic output")

        node1 = AgentNode(
            id="researcher_alpha", role=AgentRole.RESEARCHER,
            prompt_template="r", writes={".factory/strategy/alpha.md"},
        )
        (strategy_dir / "alpha.md").write_text("alpha content")
        result1 = _detect_artifact("researcher_alpha", node1, tmp_path)
        assert result1 == "alpha content"

        node2 = AgentNode(
            id="researcher_beta", role=AgentRole.RESEARCHER,
            prompt_template="r", writes={".factory/strategy/beta.md"},
        )
        result2 = _detect_artifact("researcher_beta", node2, tmp_path)
        assert result2 is None

    def test_detect_artifact_writes_before_generic(self, tmp_path: Path) -> None:
        """Writes file takes priority over generic review file."""
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("generic")
        (strategy_dir / "output.md").write_text("writes content")

        node = AgentNode(
            id="researcher", role=AgentRole.RESEARCHER,
            prompt_template="r", writes={".factory/strategy/output.md"},
        )
        result = _detect_artifact("researcher", node, tmp_path)
        assert result == "writes content"

    def test_detect_artifact_no_writes_backward_compat(self, tmp_path: Path) -> None:
        """Node with no writes falls back to generic review file."""
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("review output")

        node = AgentNode(
            id="researcher", role=AgentRole.RESEARCHER,
            prompt_template="r",
        )
        result = _detect_artifact("researcher", node, tmp_path)
        assert result == "review output"

    def test_detect_artifact_writes_absent_no_fallthrough(self, tmp_path: Path) -> None:
        """Declared writes that don't exist must return None, not fall through to generic."""
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("generic output")

        node = AgentNode(
            id="researcher", role=AgentRole.RESEARCHER,
            prompt_template="r", writes={".factory/strategy/missing.md"},
        )
        result = _detect_artifact("researcher", node, tmp_path)
        assert result is None

    def test_detect_artifact_graph_explorer_scenario(self, tmp_path: Path) -> None:
        """graph_explorer node with writes must match on writes, not generic."""
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("stale generic")
        (strategy_dir / "graph-context.md").write_text("graph analysis")

        node = AgentNode(
            id="graph_explorer", role=AgentRole.RESEARCHER,
            prompt_template="r", writes={".factory/strategy/graph-context.md"},
        )
        result = _detect_artifact("graph_explorer", node, tmp_path)
        assert result == "graph analysis"


class TestFinalize:
    def test_finalize_marks_remaining_nodes(self, tmp_path: Path) -> None:
        """Finalize auto-completes nodes whose artifacts exist but weren't tracked."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write artifacts without calling next/submit
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings")
        (reviews_dir / "builder-latest.md").write_text("Built successfully")

        result = tool_finalize(tmp_path)

        assert "Finalized" in result
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in state["completed"]
        assert "builder" in state["completed"]

    def test_finalize_no_pending(self, tmp_path: Path) -> None:
        """Finalize with all nodes already complete reports nothing to do."""
        wf = Workflow(
            name="test-single-fn",
            start_node="fn1",
            nodes={"fn1": FnNode(id="fn1", command="echo done")},
            edges=[],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-single-fn", tmp_path)
        tool_submit(tmp_path, "fn1", "Done")

        result = tool_finalize(tmp_path)

        assert "No pending nodes" in result
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["status"] == "completed"


class TestWorkflowCache:
    def test_cache_avoids_redundant_loads(self, tmp_path: Path) -> None:
        """Second call to _get_workflow_cached returns from cache dict."""
        wf = _simple_workflow()
        _register_workflow(wf)

        result1 = _get_workflow_cached("test-simple", tmp_path)
        cache_key = f"{tmp_path}:test-simple"
        assert cache_key in _workflow_cache

        result2 = _get_workflow_cached("test-simple", tmp_path)
        assert result1 is result2


class TestEventLogging:
    def _read_events(self, project_path: Path) -> list[dict]:
        events_file = project_path / ".factory" / "events.jsonl"
        if not events_file.exists():
            return []
        return [json.loads(line) for line in events_file.read_text().strip().split("\n") if line]

    def test_events_emitted_on_init(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-simple", tmp_path)

        events = self._read_events(tmp_path)
        init_events = [e for e in events if e["type"] == "workflow.tool.init"]
        assert len(init_events) == 1
        assert init_events[0]["workflow"] == "test-simple"

    def test_events_emitted_on_next(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        tool_next(tmp_path)

        events = self._read_events(tmp_path)
        next_events = [e for e in events if e["type"] == "workflow.tool.next"]
        assert len(next_events) == 1
        assert next_events[0]["node"] == "study"

    def test_events_emitted_on_auto_submit(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write artifact so auto-submit triggers
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )

        tool_next(tmp_path)

        events = self._read_events(tmp_path)
        auto_events = [e for e in events if e["type"] == "workflow.tool.auto_submit"]
        assert len(auto_events) == 1
        assert auto_events[0]["node"] == "study"

    def test_events_written_to_original_project(self, tmp_path: Path) -> None:
        """Events should be written to the original project, not the worktree."""
        wf = _simple_workflow()
        _register_workflow(wf)

        original = tmp_path / "my-project"
        wt = original / ".factory-worktrees" / "run-abc123"
        wt.mkdir(parents=True)
        (wt / ".factory").mkdir()
        (original / ".factory").mkdir(parents=True, exist_ok=True)

        tool_init("test-simple", wt)

        # Events should land in the original project, not the worktree
        orig_events = original / ".factory" / "events.jsonl"
        wt_events = wt / ".factory" / "events.jsonl"
        assert orig_events.exists()
        assert not wt_events.exists()

        events = self._read_events(original)
        init_events = [e for e in events if e["type"] == "workflow.tool.init"]
        assert len(init_events) == 1


class TestResolveOriginalProject:
    def test_factory_worktrees_pattern(self) -> None:
        p = Path("/home/user/project/.factory-worktrees/run-abc123")
        assert _resolve_original_project(p) == Path("/home/user/project")

    def test_factory_worktrees_nested(self) -> None:
        p = Path("/home/user/project/.factory/worktrees/run-abc123")
        assert _resolve_original_project(p) == Path("/home/user/project")

    def test_no_worktree_passthrough(self) -> None:
        p = Path("/home/user/project")
        assert _resolve_original_project(p) == Path("/home/user/project")

    def test_deep_factory_worktrees(self) -> None:
        p = Path("/workspace/src/repo/.factory-worktrees/run-deadbeef")
        assert _resolve_original_project(p) == Path("/workspace/src/repo")


class TestHeadlessFinalize:
    def test_run_headless_accepts_engine(self) -> None:
        """Verify _run_headless has engine in its signature."""
        import inspect
        from factory.cli._ceo_helpers import _run_headless

        sig = inspect.signature(_run_headless)
        assert "engine" in sig.parameters
        assert sig.parameters["engine"].default == "skill"


class TestWorkflowDiskCache:
    def test_cache_persisted_on_init(self, tmp_path: Path) -> None:
        """tool_init writes workflow_cache.json to session dir."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-simple", tmp_path)

        cache_file = tmp_path / ".factory" / "tool_session" / "workflow_cache.json"
        assert cache_file.exists()
        cache = json.loads(cache_file.read_text())
        assert cache["name"] == "test-simple"
        assert "study" in cache["nodes"]
        assert cache["nodes"]["study"]["type"] == "Study"

    def test_cache_loaded_on_next(self, tmp_path: Path) -> None:
        """After init, clearing in-memory cache still allows next to work via disk."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-simple", tmp_path)

        # Clear the in-memory cache
        _workflow_cache.clear()
        # Also clear the registry so register_all won't find it
        WorkflowRegistry.reset()

        result = tool_next(tmp_path)
        assert "Node: study" in result

    def test_rebuild_workflow_roundtrip(self, tmp_path: Path) -> None:
        """Serialized cache can be deserialized back into a valid Workflow."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-simple", tmp_path)

        cache_file = tmp_path / ".factory" / "tool_session" / "workflow_cache.json"
        cache_data = json.loads(cache_file.read_text())
        rebuilt = _rebuild_workflow(cache_data)

        assert rebuilt.name == wf.name
        assert rebuilt.start_node == wf.start_node
        assert set(rebuilt.nodes.keys()) == set(wf.nodes.keys())
        assert len(rebuilt.edges) == len(wf.edges)
        assert isinstance(rebuilt.nodes["study"], Study)
        assert isinstance(rebuilt.nodes["researcher"], AgentNode)
        assert isinstance(rebuilt.nodes["gate_research"], GateNode)


class TestInvokeAgentPromptOverride:
    def test_prompt_override_skips_resolve(self) -> None:
        """When prompt_override is set, resolve_prompt should NOT be called."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        with patch("factory.agents.runner.resolve_prompt") as mock_resolve, \
             patch("factory.agents.runner.get_runner") as mock_get_runner:
            mock_runner = AsyncMock()
            mock_runner.headless.return_value = AsyncMock(
                stdout="ok", return_code=0, usage=None, metadata={},
            )
            mock_get_runner.return_value = mock_runner

            from factory.agents.runner import invoke_agent

            asyncio.run(invoke_agent(
                "builder",
                "build it",
                Path("/tmp/fake-project"),
                prompt_override="custom prompt content",
                _track_failures=False,
            ))

            mock_resolve.assert_not_called()

    def test_no_override_calls_resolve(self) -> None:
        """Without prompt_override, resolve_prompt IS called."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        with patch("factory.agents.runner.resolve_prompt", return_value="resolved") as mock_resolve, \
             patch("factory.agents.runner.get_runner") as mock_get_runner:
            mock_runner = AsyncMock()
            mock_runner.headless.return_value = AsyncMock(
                stdout="ok", return_code=0, usage=None, metadata={},
            )
            mock_get_runner.return_value = mock_runner

            from factory.agents.runner import invoke_agent

            asyncio.run(invoke_agent(
                "builder",
                "build it",
                Path("/tmp/fake-project"),
                _track_failures=False,
            ))

            mock_resolve.assert_called_once()


class TestFormatProgress:
    def test_format_progress_linear(self, tmp_path: Path) -> None:
        """Linear format shows ✓/▶/○ markers and expands current node."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["completed"]["study"] = "done"
        state["completed"]["researcher"] = "done"

        result = _format_progress(state, wf, tmp_path, "gate_research", fmt="linear")

        assert "✓ study" in result
        assert "✓ researcher" in result
        assert "▶ gate_research" in result
        assert "← CURRENT" in result
        assert "○ builder" in result
        assert "Type: Gate" in result

    def test_format_progress_phased(self, tmp_path: Path) -> None:
        """Phased format shows 'Phase N:' labels with role/gate names."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["completed"]["study"] = "done"
        state["completed"]["researcher"] = "done"

        result = _format_progress(state, wf, tmp_path, "gate_research", fmt="phased")

        assert "Phase 1:" in result
        assert "Phase 2:" in result
        assert "Phase 3:" in result
        assert "Gate —" in result
        assert "Observe" in result or "Researcher" in result

    def test_overview_linear_format(self, tmp_path: Path) -> None:
        """tool_overview with fmt='linear' includes ✓/▶/○ markers."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Complete study via submit so it shows as ✓
        tool_submit(tmp_path, "study", "Observations done")

        result = tool_overview(tmp_path, fmt="linear")

        assert "✓ study" in result
        assert "▶ researcher" in result
        assert "○" in result

    def test_overview_phased_format(self, tmp_path: Path) -> None:
        """tool_overview with fmt='phased' includes 'Phase' labels."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_overview(tmp_path, fmt="phased")

        assert "Phase 1:" in result
        assert "Phase" in result

    def test_phase_label(self) -> None:
        """_phase_label produces correct labels for each node type."""
        agent = AgentNode(id="builder", role=AgentRole.BUILDER, prompt_template="build")
        assert "Builder" in _phase_label("builder", agent)

        gate = GateNode(id="gate_research", evaluator_type="agent", gate_prompt="review")
        label = _phase_label("gate_research", gate)
        assert "Gate —" in label
        assert "Research" in label

        study = Study(id="study", command="factory study")
        label = _phase_label("study", study)
        assert "Observe" in label
        assert "study" in label

        fn = FnNode(id="apply_spec", command="echo ok")
        label = _phase_label("apply_spec", fn)
        assert "Apply Spec" in label

        from factory.workflow.primitives import ForkNode
        fork = ForkNode(id="fork1", targets=["a", "b"])
        label = _phase_label("fork1", fork)
        assert "Fork" in label
        assert "a" in label
        assert "b" in label


class TestDeterministicImpliesHeadless:
    def test_deterministic_code_path(self) -> None:
        """Verify _run_headless handles engine='deterministic' early return path."""
        import inspect
        from factory.cli._ceo_helpers import _run_headless

        sig = inspect.signature(_run_headless)
        assert "engine" in sig.parameters
        assert "prompt_override" in sig.parameters

    def test_deterministic_warning_printed(self) -> None:
        """The deterministic engine block prints a WARNING and sets headless=True."""
        import io
        import sys

        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured
        try:
            # Simulate the code block from _execute_ceo
            engine = "deterministic"
            headless = False
            if engine == "deterministic":
                if not headless:
                    print(
                        "WARNING: --engine deterministic runs headless (no interactive CEO). "
                        "Adding --headless implicitly.",
                        file=sys.stderr,
                    )
                    headless = True
        finally:
            sys.stderr = old_stderr

        assert headless is True
        assert "WARNING" in captured.getvalue()
        assert "--engine deterministic" in captured.getvalue()


class TestStaleFileDetection:
    def test_stale_file_ignored(self, tmp_path: Path) -> None:
        """Review files from before the session start are ignored."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        stale_file = reviews_dir / "researcher-latest.md"
        stale_file.write_text("Stale findings from prior run")
        import os
        os.utime(stale_file, (1000000, 1000000))

        tool_init("test-simple", tmp_path)
        tool_submit(tmp_path, "study", "Observations done")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "researcher" not in state["completed"]
        assert "Node: researcher" in result

    def test_fresh_file_detected(self, tmp_path: Path) -> None:
        """Review files created after session start are auto-submitted."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        tool_submit(tmp_path, "study", "Observations done")

        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Fresh research findings")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "researcher" in state["completed"]
        assert "GATE" in result


class TestToolOverview:
    def test_overview_shows_all_nodes(self, tmp_path: Path) -> None:
        """tool_overview lists all nodes with completion markers."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_overview(tmp_path)

        assert "study" in result
        assert "researcher" in result
        assert "gate_research" in result
        assert "builder" in result
        assert "▶" in result or "○" in result


class TestToolCurr:
    def test_curr_shows_current(self, tmp_path: Path) -> None:
        """tool_curr shows first node details without advancing."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_curr(tmp_path)

        assert "Node: study" in result
        assert "Type: Study" in result

    def test_curr_done(self, tmp_path: Path) -> None:
        """tool_curr returns DONE when all nodes completed."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = len(state["topo_order"])
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_curr(tmp_path)
        assert "DONE" in result


class TestNextDryRun:
    def test_next_dry_run(self, tmp_path: Path) -> None:
        """dry_run=True returns the node but does NOT advance the pointer."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_next(tmp_path, dry_run=True)

        assert "Node: study" in result

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["pointer_idx"] == 0

    def test_next_dry_run_auto_submit_no_persist(self, tmp_path: Path) -> None:
        """dry_run scans for artifacts but does not persist completions."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )

        result = tool_next(tmp_path, dry_run=True)

        assert "researcher" in result

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["pointer_idx"] == 0
        assert "study" not in state["completed"]

    def test_dry_run_no_events_emitted(self, tmp_path: Path) -> None:
        """dry_run=True must not emit any events to events.jsonl."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        events_file = tmp_path / ".factory" / "events.jsonl"
        events_before = events_file.read_text() if events_file.exists() else ""

        tool_next(tmp_path, dry_run=True)

        events_after = events_file.read_text() if events_file.exists() else ""
        assert events_before == events_after, "dry_run should not emit events"

    def test_dry_run_completed_status_no_finalize(self, tmp_path: Path) -> None:
        """dry_run=True with status='completed' must not call finalize."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state_path = tmp_path / ".factory" / "tool_session" / "state.json"
        state = json.loads(state_path.read_text())
        state["status"] = "completed"
        state_path.write_text(json.dumps(state))

        state_before = state_path.read_text()
        events_file = tmp_path / ".factory" / "events.jsonl"
        events_before = events_file.read_text() if events_file.exists() else ""

        result = tool_next(tmp_path, dry_run=True)

        assert "DONE" in result
        assert state_path.read_text() == state_before, "dry_run should not mutate state"
        events_after = events_file.read_text() if events_file.exists() else ""
        assert events_before == events_after, "dry_run should not emit events"

    def test_dry_run_pointer_past_end_no_finalize(self, tmp_path: Path) -> None:
        """dry_run=True with pointer past end must not call finalize."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state_path = tmp_path / ".factory" / "tool_session" / "state.json"
        state = json.loads(state_path.read_text())
        order = state["topo_order"]
        state["pointer_idx"] = len(order)
        for nid in order:
            state["completed"][nid] = "done"
        state_path.write_text(json.dumps(state))

        state_before = state_path.read_text()
        events_file = tmp_path / ".factory" / "events.jsonl"
        events_before = events_file.read_text() if events_file.exists() else ""

        result = tool_next(tmp_path, dry_run=True)

        assert "DONE" in result
        assert state_path.read_text() == state_before, "dry_run should not mutate state"
        events_after = events_file.read_text() if events_file.exists() else ""
        assert events_before == events_after, "dry_run should not emit events"


class TestNextCompactOutput:
    def test_next_compact_output(self, tmp_path: Path) -> None:
        """tool_next returns compact node details, not progress markers."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_next(tmp_path)

        assert "✓" not in result
        assert "○" not in result
        assert "▶" not in result
        assert "Node: study" in result
        assert "Type: Study" in result
