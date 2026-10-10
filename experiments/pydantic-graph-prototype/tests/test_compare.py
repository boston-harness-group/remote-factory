"""Tests for the dual-engine comparison harness.

Verifies that both engines produce equivalent behavior for:
  - PROCEED path (sequential + gate + fork/join)
  - RELOOP path (gate routes back to builder)
  - HALT path (gate stops execution)
  - Max iterations (reloop exhaustion)
  - End-to-end comparison runner
"""

from pg_factory.compare import (
    build_current_engine_workflow,
    build_pydantic_graph,
    extract_active_nodes,
    extract_fork_children,
    extract_gate_verdicts,
    normalize_pg_events,
    run_comparison,
    simulate_current_engine,
)
from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState
from pg_factory.verdicts import HaltResult, VerdictType


# ── PROCEED path ─────────────────────────────────────────────────


async def test_proceed_active_nodes_match() -> None:
    """Both engines execute the same set of nodes on PROCEED."""
    result = await run_comparison([VerdictType.PROCEED])

    assert result["match"]["active_nodes"]
    assert result["match"]["gate_verdicts"]
    assert result["match"]["fork_children"]


async def test_proceed_gate_verdict() -> None:
    """Both engines record a single 'proceed' verdict."""
    result = await run_comparison([VerdictType.PROCEED])

    assert result["current_engine"]["gate_verdicts"] == ["proceed"]
    assert result["pydantic_graph"]["gate_verdicts"] == ["proceed"]


async def test_proceed_fork_children() -> None:
    """Both engines run all 3 QA agents in the fork/join."""
    result = await run_comparison([VerdictType.PROCEED])

    expected = {"health_checker", "code_reviewer", "adversarial_tester"}
    assert result["current_engine"]["fork_children"] == expected
    assert result["pydantic_graph"]["fork_children"] == expected


async def test_proceed_pg_result() -> None:
    """pydantic-graph returns HaltResult with reason 'qa_complete'."""
    result = await run_comparison([VerdictType.PROCEED])

    pg_result = result["pydantic_graph"]["result"]
    assert isinstance(pg_result, HaltResult)
    assert pg_result.reason == "qa_complete"


# ── RELOOP path ──────────────────────────────────────────────────


async def test_reloop_then_proceed_nodes_match() -> None:
    """RELOOP once then PROCEED — both engines execute builder twice."""
    result = await run_comparison([VerdictType.RELOOP, VerdictType.PROCEED])

    sim_active = result["current_engine"]["active_nodes"]
    pg_active = result["pydantic_graph"]["active_nodes"]

    assert sim_active.count("builder") == 2
    assert pg_active.count("builder") == 2
    assert result["match"]["active_nodes"]


async def test_reloop_then_proceed_verdicts() -> None:
    """RELOOP then PROCEED produces reloop + proceed verdict sequence."""
    result = await run_comparison([VerdictType.RELOOP, VerdictType.PROCEED])

    assert result["current_engine"]["gate_verdicts"] == ["reloop", "proceed"]
    assert result["pydantic_graph"]["gate_verdicts"] == ["reloop", "proceed"]


async def test_reloop_twice_then_proceed() -> None:
    """Two RELOOPs then PROCEED — builder executes 3 times total."""
    result = await run_comparison(
        [VerdictType.RELOOP, VerdictType.RELOOP, VerdictType.PROCEED]
    )

    sim_active = result["current_engine"]["active_nodes"]
    pg_active = result["pydantic_graph"]["active_nodes"]

    assert sim_active.count("builder") == 3
    assert pg_active.count("builder") == 3
    assert result["match"]["gate_verdicts"]
    assert result["match"]["fork_children"]


# ── HALT path ────────────────────────────────────────────────────


async def test_halt_stops_execution() -> None:
    """HALT stops execution before fork/join."""
    result = await run_comparison([VerdictType.HALT])

    assert result["current_engine"]["gate_verdicts"] == ["halt"]
    assert result["pydantic_graph"]["gate_verdicts"] == ["halt"]

    assert result["current_engine"]["fork_children"] == set()
    assert result["pydantic_graph"]["fork_children"] == set()


async def test_halt_pg_result() -> None:
    """pydantic-graph returns HaltResult with reason 'gate_halted'."""
    result = await run_comparison([VerdictType.HALT])

    pg_result = result["pydantic_graph"]["result"]
    assert isinstance(pg_result, HaltResult)
    assert pg_result.reason == "gate_halted"


# ── Max iterations ───────────────────────────────────────────────


async def test_max_iterations_halt() -> None:
    """Continuous RELOOP hits max_iterations and halts in both engines."""
    all_reloop = [VerdictType.RELOOP] * 10
    result = await run_comparison(all_reloop, max_iterations=2)

    sim_verdicts = result["current_engine"]["gate_verdicts"]
    pg_verdicts = result["pydantic_graph"]["gate_verdicts"]

    assert sim_verdicts[-1] == "halt"
    assert pg_verdicts[-1] == "halt"

    assert result["current_engine"]["fork_children"] == set()
    assert result["pydantic_graph"]["fork_children"] == set()


# ── Current engine simulation unit tests ─────────────────────────


def test_sim_workflow_has_all_nodes() -> None:
    """Simulated workflow has all 7 nodes."""
    wf = build_current_engine_workflow()
    expected = {
        "builder", "qa_gate", "fork_qa",
        "health_checker", "code_reviewer", "adversarial_tester",
        "join_qa",
    }
    assert set(wf.nodes.keys()) == expected


def test_sim_proceed_event_sequence() -> None:
    """PROCEED produces: builder execute, gate verdict, fork, 3 children, join."""
    wf = build_current_engine_workflow()
    events = simulate_current_engine(wf, lambda _: VerdictType.PROCEED)

    actions = [(e.node, e.action) for e in events]
    assert actions[0] == ("builder", "execute")
    assert actions[1] == ("qa_gate", "gate_verdict")
    assert actions[1 + 1] == ("fork_qa", "fork")

    child_executes = [(n, a) for n, a in actions if a == "execute" and n != "builder"]
    assert len(child_executes) == 3

    assert actions[-2] == ("fork_qa", "fork_join_complete")
    assert actions[-1] == ("join_qa", "join")


def test_sim_halt_no_fork() -> None:
    """HALT produces only builder execute + gate halt verdict."""
    wf = build_current_engine_workflow()
    events = simulate_current_engine(wf, lambda _: VerdictType.HALT)

    assert len(events) == 2
    assert events[0].action == "execute"
    assert events[1].detail["verdict"] == "halt"


# ── pydantic-graph unit tests ────────────────────────────────────


async def test_pg_proceed_events(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """pydantic-graph PROCEED records builder execute + gate verdict + children."""
    graph_builder, start = build_pydantic_graph(
        lambda _: VerdictType.PROCEED
    )
    graph = graph_builder.build()
    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "qa_complete"

    events = normalize_pg_events(factory_state)
    actions = [e.action for e in events]

    assert "execute" in actions
    assert "gate_verdict" in actions
    assert actions.count("child_completed") == 3
    assert "fork_join_complete" in actions


async def test_pg_halt_events(
    factory_state: FactoryState,
    factory_deps: FactoryDeps,
) -> None:
    """pydantic-graph HALT records builder execute + gate halt verdict only."""
    graph_builder, start = build_pydantic_graph(
        lambda _: VerdictType.HALT
    )
    graph = graph_builder.build()
    result = await graph.run(state=factory_state, deps=factory_deps, inputs=start)

    assert isinstance(result, HaltResult)
    assert result.reason == "gate_halted"

    events = normalize_pg_events(factory_state)
    assert len(events) == 2
    assert events[0].action == "execute"
    assert events[1].detail["verdict"] == "halt"


# ── Mermaid output ───────────────────────────────────────────────


async def test_mermaid_contains_all_nodes() -> None:
    """Mermaid diagram from pydantic-graph shows all workflow nodes."""
    result = await run_comparison([VerdictType.PROCEED])

    mermaid = result["mermaid"]
    assert "CompareBuilderNode" in mermaid
    assert "CompareQAGateNode" in mermaid
    assert "CompareQAForkJoinNode" in mermaid


async def test_mermaid_shows_gate_branching() -> None:
    """Mermaid diagram shows gate routing edges."""
    graph_builder, _ = build_pydantic_graph()
    graph = graph_builder.build()
    mermaid = graph.render()

    assert "CompareBuilderNode --> CompareQAGateNode" in mermaid
    assert "decision" in mermaid


# ── End-to-end ───────────────────────────────────────────────────


async def test_comparison_returns_all_fields() -> None:
    """run_comparison returns all expected fields."""
    result = await run_comparison([VerdictType.PROCEED])

    assert "current_engine" in result
    assert "pydantic_graph" in result
    assert "match" in result
    assert "mermaid" in result

    assert "events" in result["current_engine"]
    assert "active_nodes" in result["current_engine"]
    assert "gate_verdicts" in result["current_engine"]
    assert "fork_children" in result["current_engine"]

    assert "events" in result["pydantic_graph"]
    assert "result" in result["pydantic_graph"]
