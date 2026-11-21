"""Tests for loop context injection and feedback log in tool mode."""

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
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.tool import (
    _find_loop_context,
    _format_node_task,
    _load_state,
    _save_state,
    _workflow_cache,
    tool_init,
    tool_next,
    tool_submit,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    WorkflowRegistry.reset()
    _workflow_cache.clear()
    yield
    WorkflowRegistry.reset()
    _workflow_cache.clear()


def _register_workflow(wf: Workflow) -> None:
    """Register a workflow through the public API, so a later discovery keeps it."""
    WorkflowRegistry.register_callable(
        wf.name, lambda _wf=wf: _wf, source="builtin", description="test workflow"
    )


def _reloop_workflow() -> Workflow:
    """builder -> gate_qa -> (RELOOP) builder | (PROCEED) archivist."""
    return Workflow(
        name="test-reloop",
        start_node="builder",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build the project at {project_path}",
                reads={".factory/strategy/current.md"},
                writes={".factory/reviews/builder-latest.md"},
            ),
            "gate_qa": GateNode(
                id="gate_qa",
                evaluator_type="fn",
                evaluator_command="echo FAIL: tests broken",
                gate_prompt="Run QA checks on the builder output",
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
            Edge(source="builder", target="gate_qa"),
            Edge(source="gate_qa", target="archivist", condition=VerdictType.PROCEED),
            Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
        ],
    )


def _multi_gate_workflow() -> Workflow:
    """builder -> gate_build -> health_checker -> gate_qa -> (RELOOP) builder."""
    return Workflow(
        name="test-multi-gate",
        start_node="builder",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build at {project_path}",
                reads={".factory/strategy/current.md"},
                writes={".factory/reviews/builder-latest.md"},
            ),
            "gate_build": GateNode(
                id="gate_build",
                evaluator_type="agent",
                gate_prompt="Review build output",
                reads={".factory/reviews/builder-latest.md"},
            ),
            "health_checker": AgentNode(
                id="health_checker",
                role=AgentRole.HEALTH_CHECKER,
                prompt_template="Check health",
                writes={".factory/reviews/health-check.md"},
            ),
            "gate_qa": GateNode(
                id="gate_qa",
                evaluator_type="fn",
                evaluator_command="echo FAIL: qa issues",
                gate_prompt="Run QA verification",
                reads={".factory/reviews/health-check.md"},
            ),
            "archivist": AgentNode(
                id="archivist",
                role=AgentRole.ARCHIVIST,
                prompt_template="Archive",
                blocking=False,
            ),
        },
        edges=[
            Edge(source="builder", target="gate_build"),
            Edge(source="gate_build", target="health_checker", condition=VerdictType.PROCEED),
            Edge(source="gate_build", target="builder", condition=VerdictType.RELOOP),
            Edge(source="health_checker", target="gate_qa"),
            Edge(source="gate_qa", target="archivist", condition=VerdictType.PROCEED),
            Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
        ],
    )


class TestFindLoopContext:
    def test_not_a_reloop_target_returns_empty(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {},
            "feedback_log": {},
        }
        result = _find_loop_context("archivist", wf, state, tmp_path)
        assert result == ""

    def test_first_invocation_returns_topology_without_feedback(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {},
            "feedback_log": {},
        }
        result = _find_loop_context("builder", wf, state, tmp_path)
        assert "## LOOP CONTEXT" in result
        assert "0/3" in result
        assert "gate_qa" in result
        assert "Run QA checks" in result
        assert "Loop topology" in result
        assert "Feedback history" not in result

    def test_first_invocation_shows_zero_of_three(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 0},
            "feedback_log": {},
        }
        result = _find_loop_context("builder", wf, state, tmp_path)
        assert "## LOOP CONTEXT" in result
        assert "0/3" in result
        assert "FINAL ATTEMPT" not in result
        assert "Feedback history" not in result

    def test_single_gate_reloop_at_iteration_1(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {
                "builder": [{
                    "gate": "gate_qa",
                    "iteration": 1,
                    "feedback": "tests broken: 3 failures in test_auth.py",
                    "timestamp": 1000.0,
                }],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "## LOOP CONTEXT" in result
        assert "1/3" in result
        assert "gate_qa" in result
        assert "Run QA checks" in result
        assert "Loop topology" in result
        assert "builder" in result
        assert "Feedback history" in result
        assert "tests broken" in result
        assert "FINAL ATTEMPT" not in result

    def test_multiple_gates_most_recent_wins(self, tmp_path: Path) -> None:
        wf = _multi_gate_workflow()
        state = {
            "topo_order": ["builder", "gate_build", "health_checker", "gate_qa", "archivist"],
            "iteration_counts": {
                "gate_build->builder": 1,
                "gate_qa->builder": 1,
            },
            "feedback_log": {
                "builder": [
                    {
                        "gate": "gate_build",
                        "iteration": 1,
                        "feedback": "build review failed",
                        "timestamp": 1000.0,
                    },
                    {
                        "gate": "gate_qa",
                        "iteration": 1,
                        "feedback": "qa issues found",
                        "timestamp": 2000.0,
                    },
                ],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "## LOOP CONTEXT" in result
        assert "Triggered by: gate_qa" in result
        assert "qa issues found" in result

    def test_max_iteration_warning(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 3},
            "feedback_log": {
                "builder": [{
                    "gate": "gate_qa",
                    "iteration": 3,
                    "feedback": "still failing",
                    "timestamp": 1000.0,
                }],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "FINAL ATTEMPT" in result
        assert "3/3" in result

    def test_iterations_but_no_feedback_log(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {},
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "## LOOP CONTEXT" in result
        assert "1/3" in result
        assert "gate_qa" in result
        assert "Feedback history" not in result

    def test_loop_topology_includes_intermediate_nodes(self, tmp_path: Path) -> None:
        wf = _multi_gate_workflow()
        state = {
            "topo_order": ["builder", "gate_build", "health_checker", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {
                "builder": [{
                    "gate": "gate_qa",
                    "iteration": 1,
                    "feedback": "qa failed",
                    "timestamp": 1000.0,
                }],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "Loop topology" in result
        assert "**builder**" in result
        assert "**gate_build**" in result
        assert "**health_checker**" in result
        assert "**gate_qa**" in result

    def test_feedback_truncated_to_500_chars(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        long_feedback = "x" * 1000
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {
                "builder": [{
                    "gate": "gate_qa",
                    "iteration": 1,
                    "feedback": long_feedback,
                    "timestamp": 1000.0,
                }],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        feedback_section = result.split("### Feedback history")[1]
        line_with_feedback = [line for line in feedback_section.split("\n") if line.startswith("- [")][0]
        feedback_content = line_with_feedback.split("] ", 1)[1]
        assert len(feedback_content) <= 500

    def test_only_last_2_feedback_entries_shown(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 3},
            "feedback_log": {
                "builder": [
                    {"gate": "gate_qa", "iteration": 1, "feedback": "first failure", "timestamp": 1.0},
                    {"gate": "gate_qa", "iteration": 2, "feedback": "second failure", "timestamp": 2.0},
                    {"gate": "gate_qa", "iteration": 3, "feedback": "third failure", "timestamp": 3.0},
                ],
            },
        }
        result = _find_loop_context("builder", wf, state, tmp_path)

        assert "first failure" not in result
        assert "second failure" in result
        assert "third failure" in result


class TestFeedbackLog:
    def test_feedback_appended_on_fn_gate_reloop(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        result = tool_submit(tmp_path, "builder", "First attempt")
        assert result.startswith("RETRY")

        state = _load_state(tmp_path)
        assert "builder" in state["feedback_log"]
        entries = state["feedback_log"]["builder"]
        assert len(entries) == 1
        assert entries[0]["gate"] == "gate_qa"
        assert entries[0]["iteration"] == 1
        assert "FAIL" in entries[0]["feedback"]
        assert isinstance(entries[0]["timestamp"], float)

    def test_feedback_persists_across_save_load(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        state = _load_state(tmp_path)
        state["feedback_log"] = {
            "builder": [{
                "gate": "gate_qa",
                "iteration": 1,
                "feedback": "test feedback",
                "timestamp": 12345.0,
            }],
        }
        _save_state(tmp_path, state)

        reloaded = _load_state(tmp_path)
        assert reloaded["feedback_log"]["builder"][0]["feedback"] == "test feedback"
        assert reloaded["feedback_log"]["builder"][0]["timestamp"] == 12345.0

    def test_feedback_truncated_on_gate_output(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-long-feedback",
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
                    evaluator_command="python3 -c \"print('FAIL: ' + 'x' * 1000)\"",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_check"),
                Edge(source="gate_check", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-long-feedback", tmp_path)

        tool_submit(tmp_path, "builder", "attempt")

        state = _load_state(tmp_path)
        entries = state["feedback_log"]["builder"]
        assert len(entries[0]["feedback"]) <= 500

    def test_multiple_feedback_entries_preserved(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        tool_submit(tmp_path, "builder", "First attempt")

        state = _load_state(tmp_path)
        del state["completed"]["builder"]
        _save_state(tmp_path, state)

        tool_submit(tmp_path, "builder", "Second attempt")

        state = _load_state(tmp_path)
        entries = state["feedback_log"]["builder"]
        assert len(entries) == 2
        assert entries[0]["iteration"] == 1
        assert entries[1]["iteration"] == 2

    def test_ceo_retry_verdict_appends_feedback(self, tmp_path: Path) -> None:
        """When CEO submits RETRY for an agent gate, feedback is logged."""
        wf = Workflow(
            name="test-agent-gate",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_review": GateNode(
                    id="gate_review",
                    evaluator_type="agent",
                    gate_prompt="Review the build",
                    reads={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[
                Edge(source="builder", target="gate_review"),
                Edge(source="gate_review", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-agent-gate", tmp_path)

        state = _load_state(tmp_path)
        state["pointer_idx"] = 1
        state["completed"]["builder"] = "built"
        _save_state(tmp_path, state)

        tool_submit(
            tmp_path,
            "gate_review",
            'RETRY target=builder feedback="Missing test coverage for auth module"',
        )

        state = _load_state(tmp_path)
        assert "builder" in state["feedback_log"]
        entries = state["feedback_log"]["builder"]
        assert len(entries) == 1
        assert entries[0]["gate"] == "gate_review"
        assert "Missing test coverage" in entries[0]["feedback"]

    def test_feedback_log_initialized_in_state(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        state = _load_state(tmp_path)
        assert "feedback_log" in state
        assert state["feedback_log"] == {}


class TestFormatNodeTaskLoopContext:
    def test_loop_context_present_at_iteration_0(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {},
            "feedback_log": {},
        }
        result = _format_node_task("builder", wf.nodes["builder"], wf, state, tmp_path)
        assert "LOOP CONTEXT" in result
        assert "0/3" in result
        assert "gate_qa" in result
        assert "Feedback history" not in result

    def test_loop_context_appended_at_iteration_1(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {
                "builder": [{
                    "gate": "gate_qa",
                    "iteration": 1,
                    "feedback": "tests broken",
                    "timestamp": 1000.0,
                }],
            },
        }
        result = _format_node_task("builder", wf.nodes["builder"], wf, state, tmp_path)

        assert "Node: builder" in result
        assert "Type: Agent (builder)" in result
        assert "## LOOP CONTEXT" in result
        assert "1/3" in result
        assert "gate_qa" in result
        assert "tests broken" in result

    def test_loop_context_not_injected_for_non_reloop_node(self, tmp_path: Path) -> None:
        wf = _reloop_workflow()
        state = {
            "topo_order": ["builder", "gate_qa", "archivist"],
            "iteration_counts": {"gate_qa->builder": 1},
            "feedback_log": {},
        }
        result = _format_node_task("archivist", wf.nodes["archivist"], wf, state, tmp_path)
        assert "LOOP CONTEXT" not in result

    def test_integration_tool_next_includes_loop_context(self, tmp_path: Path) -> None:
        """Full integration: fn gate RELOOP -> tool_next returns builder with loop context."""
        wf = _reloop_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        result = tool_submit(tmp_path, "builder", "First attempt")
        assert result.startswith("RETRY")

        state = _load_state(tmp_path)
        del state["completed"]["builder"]
        del state["completed"]["gate_qa"]
        _save_state(tmp_path, state)

        review_file = tmp_path / ".factory" / "reviews" / "builder-latest.md"
        if review_file.exists():
            review_file.unlink()

        result = tool_next(tmp_path)

        assert "Node: builder" in result
        assert "## LOOP CONTEXT" in result
        assert "1/3" in result
        assert "gate_qa" in result


class TestLoopContextE2EComparison:
    """A/B comparison tests: verify loop context injection changes builder task prompts."""

    def _make_cli_app_workflow(self, name: str) -> Workflow:
        return Workflow(
            name=name,
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build the CLI app at {project_path}",
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-latest.md"},
                ),
                "gate_qa": GateNode(
                    id="gate_qa",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: lint errors",
                    gate_prompt="Check lint and tests pass",
                    reads={".factory/reviews/builder-latest.md"},
                ),
                "done": FnNode(id="done", command="echo done"),
            },
            edges=[
                Edge(source="builder", target="gate_qa"),
                Edge(source="gate_qa", target="done", condition=VerdictType.PROCEED),
                Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
            ],
        )

    def _make_web_app_workflow(self, name: str) -> Workflow:
        return Workflow(
            name=name,
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build the web app at {project_path}",
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-latest.md"},
                ),
                "health_checker": AgentNode(
                    id="health_checker",
                    role=AgentRole.HEALTH_CHECKER,
                    prompt_template="Check health",
                    writes={".factory/reviews/health-check.md"},
                ),
                "gate_qa": GateNode(
                    id="gate_qa",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: api tests broken",
                    gate_prompt="Verify API endpoints work",
                    reads={".factory/reviews/health-check.md"},
                ),
                "done": FnNode(id="done", command="echo done"),
            },
            edges=[
                Edge(source="builder", target="health_checker"),
                Edge(source="health_checker", target="gate_qa"),
                Edge(source="gate_qa", target="done", condition=VerdictType.PROCEED),
                Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
            ],
        )

    def _make_lib_workflow(self, name: str) -> Workflow:
        return Workflow(
            name=name,
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build the library at {project_path}",
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-latest.md"},
                ),
                "code_reviewer": AgentNode(
                    id="code_reviewer",
                    role=AgentRole.CODE_REVIEWER,
                    prompt_template="Review code",
                    writes={".factory/reviews/code-review.md"},
                ),
                "gate_review": GateNode(
                    id="gate_review",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: coverage below 80%",
                    gate_prompt="Check test coverage meets threshold",
                    reads={".factory/reviews/code-review.md"},
                ),
                "done": FnNode(id="done", command="echo done"),
            },
            edges=[
                Edge(source="builder", target="code_reviewer"),
                Edge(source="code_reviewer", target="gate_review"),
                Edge(source="gate_review", target="done", condition=VerdictType.PROCEED),
                Edge(source="gate_review", target="builder", condition=VerdictType.RELOOP),
            ],
        )

    def _simulate_reloop_cycle(
        self, wf: Workflow, tmp_path: Path, *, with_loop_context: bool,
    ) -> dict:
        """Simulate one RELOOP cycle and collect builder task prompts.

        Walks the workflow from builder through all intermediate nodes until
        a fn gate triggers RELOOP, then simulates CEO re-invocation of builder.
        Returns {prompts, reloop_count, has_loop_context, has_feedback}.
        """
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir(parents=True, exist_ok=True)
        tool_init(wf.name, tmp_path)

        prompts: list[str] = []
        reloop_count = 0

        result = tool_next(tmp_path)
        prompts.append(result)

        result = tool_submit(tmp_path, "builder", "attempt 1")

        if result.startswith("RETRY"):
            reloop_count += 1
        else:
            state = _load_state(tmp_path)
            order = state["topo_order"]
            idx = state["pointer_idx"]

            while idx < len(order):
                nid = order[idx]
                node = wf.nodes.get(nid)
                if isinstance(node, AgentNode):
                    if node.writes:
                        for wp in node.writes:
                            out = tmp_path / wp
                            out.parent.mkdir(parents=True, exist_ok=True)
                            out.write_text(f"{node.role.value} output")
                    else:
                        reviews_dir = tmp_path / ".factory" / "reviews"
                        reviews_dir.mkdir(parents=True, exist_ok=True)
                        role = node.role.value
                        (reviews_dir / f"{role}-latest.md").write_text(f"{role} output")
                    result = tool_next(tmp_path)
                    if result.startswith("RETRY"):
                        reloop_count += 1
                        break
                    state = _load_state(tmp_path)
                    idx = state["pointer_idx"]
                else:
                    break

        state = _load_state(tmp_path)
        for nid in list(state["completed"]):
            del state["completed"][nid]

        if not with_loop_context:
            state["iteration_counts"] = {}
            state["feedback_log"] = {}

        _save_state(tmp_path, state)

        reviews_dir = tmp_path / ".factory" / "reviews"
        if reviews_dir.exists():
            for f in reviews_dir.iterdir():
                if f.suffix == ".md":
                    f.unlink()

        result = tool_next(tmp_path)
        prompts.append(result)

        return {
            "prompts": prompts,
            "reloop_count": reloop_count,
            "has_loop_context": "LOOP CONTEXT" in prompts[-1],
            "has_feedback": "Feedback history" in prompts[-1],
        }

    def test_ab_comparison_cli_app(self, tmp_path: Path) -> None:
        """CLI app: both arms have topology; only with_ctx has feedback."""
        wf = self._make_cli_app_workflow("cli-app")

        without = self._simulate_reloop_cycle(
            wf, tmp_path / "cli-no-ctx", with_loop_context=False,
        )
        _workflow_cache.clear()
        WorkflowRegistry.reset()

        wf2 = self._make_cli_app_workflow("cli-app-ctx")
        with_ctx = self._simulate_reloop_cycle(
            wf2, tmp_path / "cli-with-ctx", with_loop_context=True,
        )

        assert without["has_loop_context"]
        assert not without["has_feedback"]
        assert with_ctx["has_loop_context"]
        assert with_ctx["has_feedback"]
        assert "lint" in with_ctx["prompts"][-1].lower()

    def test_ab_comparison_web_app(self, tmp_path: Path) -> None:
        """Web app: both arms have topology; only with_ctx has feedback."""
        wf = self._make_web_app_workflow("web-app")

        without = self._simulate_reloop_cycle(
            wf, tmp_path / "web-no-ctx", with_loop_context=False,
        )
        _workflow_cache.clear()
        WorkflowRegistry.reset()

        wf2 = self._make_web_app_workflow("web-app-ctx")
        with_ctx = self._simulate_reloop_cycle(
            wf2, tmp_path / "web-with-ctx", with_loop_context=True,
        )

        assert without["has_loop_context"]
        assert not without["has_feedback"]
        assert with_ctx["has_loop_context"]
        assert with_ctx["has_feedback"]
        assert "api" in with_ctx["prompts"][-1].lower()
        assert "health_checker" in with_ctx["prompts"][-1]

    def test_ab_comparison_library(self, tmp_path: Path) -> None:
        """Library: both arms have topology; only with_ctx has feedback."""
        wf = self._make_lib_workflow("lib")

        without = self._simulate_reloop_cycle(
            wf, tmp_path / "lib-no-ctx", with_loop_context=False,
        )
        _workflow_cache.clear()
        WorkflowRegistry.reset()

        wf2 = self._make_lib_workflow("lib-ctx")
        with_ctx = self._simulate_reloop_cycle(
            wf2, tmp_path / "lib-with-ctx", with_loop_context=True,
        )

        assert without["has_loop_context"]
        assert not without["has_feedback"]
        assert with_ctx["has_loop_context"]
        assert with_ctx["has_feedback"]
        assert "coverage" in with_ctx["prompts"][-1].lower()
        assert "code_reviewer" in with_ctx["prompts"][-1]

    def test_ab_report_generation(self, tmp_path: Path) -> None:
        """Generate a comparison report across all 3 test repos."""
        scenarios = [
            ("cli-app", self._make_cli_app_workflow),
            ("web-app", self._make_web_app_workflow),
            ("library", self._make_lib_workflow),
        ]

        report: dict[str, dict] = {}

        for name, factory_fn in scenarios:
            _workflow_cache.clear()
            WorkflowRegistry.reset()

            wf_no_ctx = factory_fn(f"{name}-no-ctx")
            result_no_ctx = self._simulate_reloop_cycle(
                wf_no_ctx, tmp_path / f"{name}-no-ctx", with_loop_context=False,
            )

            _workflow_cache.clear()
            WorkflowRegistry.reset()

            wf_with_ctx = factory_fn(f"{name}-with-ctx")
            result_with_ctx = self._simulate_reloop_cycle(
                wf_with_ctx, tmp_path / f"{name}-with-ctx", with_loop_context=True,
            )

            prompt_no_ctx = result_no_ctx["prompts"][-1]
            prompt_with_ctx = result_with_ctx["prompts"][-1]

            mentions_gate = any(
                kw in prompt_with_ctx.lower()
                for kw in ["gate", "qa", "check", "review", "coverage", "lint"]
            )

            report[name] = {
                "without_feedback": {
                    "prompt_length": len(prompt_no_ctx),
                    "has_loop_context": result_no_ctx["has_loop_context"],
                    "has_feedback": result_no_ctx["has_feedback"],
                    "reloop_count": result_no_ctx["reloop_count"],
                },
                "with_feedback": {
                    "prompt_length": len(prompt_with_ctx),
                    "has_loop_context": result_with_ctx["has_loop_context"],
                    "has_feedback": result_with_ctx["has_feedback"],
                    "reloop_count": result_with_ctx["reloop_count"],
                    "mentions_downstream_criteria": mentions_gate,
                },
            }

        report_path = tmp_path / "ab_comparison_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        for name, data in report.items():
            assert data["without_feedback"]["has_loop_context"], (
                f"{name}: prompt without feedback should still have LOOP CONTEXT topology"
            )
            assert not data["without_feedback"]["has_feedback"], (
                f"{name}: prompt without feedback should lack Feedback history"
            )
            assert data["with_feedback"]["has_loop_context"], (
                f"{name}: prompt WITH feedback should include LOOP CONTEXT"
            )
            assert data["with_feedback"]["has_feedback"], (
                f"{name}: prompt WITH feedback should include Feedback history"
            )
            assert data["with_feedback"]["mentions_downstream_criteria"], (
                f"{name}: prompt WITH feedback should mention downstream gate criteria"
            )
            assert data["with_feedback"]["prompt_length"] > data["without_feedback"]["prompt_length"], (
                f"{name}: prompt with feedback should be longer than without"
            )

        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert len(loaded) == 3
