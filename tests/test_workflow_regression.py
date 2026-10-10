"""Workflow regression tests — reproduction of executor and validation bugs.

Each test constructs the workflow state that existed when a bug shipped,
then proves the validator, executor, or FakeAgent catches it.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import subprocess as _sp

from factory.testing import FakeAgent
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataItem,
    DataNode,
    Edge,
    FnNode,
    JoinNode,
    Workflow)
from factory.workflow.validation import validate_workflow


def _init_git(path: Path) -> None:
    """Initialize a minimal git repo for DataNode worktree tests."""
    _sp.run(["git", "init", str(path)], capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "config", "user.email", "t@t"],
            capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "config", "user.name", "t"],
            capture_output=True, check=True)
    (path / "README.md").write_text("test\n")
    _sp.run(["git", "-C", str(path), "add", "."],
            capture_output=True, check=True)
    _sp.run(["git", "-C", str(path), "commit", "-m", "init"],
            capture_output=True, check=True)


class TestBug1534EmptyPromptTemplate:
    """#1534: Designer created AgentNodes with empty prompt_template.

    The Designer's _generate_prompt_template() returned "" for nodes it
    couldn't derive a prompt for. The old validator accepted this.
    Tests passed. Executor ran the node with an empty prompt -> garbage output.
    """

    def test_validator_rejects_empty_prompt(self):
        wf = Workflow(
            name="bug-1534-repro",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="",  # <-- the bug: empty prompt
                    reads=set(),
                    writes={"output.md"}),
            },
            edges=[],
            start_node="builder")
        issues = validate_workflow(wf)
        assert any("empty prompt_template" in i for i in issues), (
            f"Validator should reject empty prompt_template. Issues: {issues}"
        )

    def test_validator_rejects_whitespace_prompt(self):
        wf = Workflow(
            name="bug-1534-whitespace",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="   \n  ",  # whitespace-only
                    reads=set(),
                    writes={"output.md"}),
            },
            edges=[],
            start_node="builder")
        issues = validate_workflow(wf)
        assert any("empty prompt_template" in i for i in issues), (
            f"Validator should reject whitespace-only prompt. Issues: {issues}"
        )

    def test_validator_accepts_real_prompt(self):
        wf = Workflow(
            name="bug-1534-good",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build the feature described in the issue.",
                    reads=set(),
                    writes={"output.md"}),
            },
            edges=[],
            start_node="builder")
        issues = validate_workflow(wf)
        prompt_issues = [i for i in issues if "empty prompt_template" in i]
        assert not prompt_issues, f"Valid prompt should pass: {prompt_issues}"


class TestBug1535OrphanedSubgraph:
    """#1535: DataNode subgraph_entry pointed to a node not in the graph.

    The Designer's _rewire_data_nodes() replaced the subgraph_entry with
    template nodes, but the original subgraph nodes were still frozen.
    Result: DataNode pointed at nodes that didn't exist in the workflow.
    """

    def test_validator_rejects_missing_subgraph_entry(self):
        data_node = DataNode(
            id="documents",
            task_ref="some-task:Task",
            parallelism=1)
        wf = Workflow(
            name="bug-1535-repro",
            nodes={"documents": data_node},
            edges=[Edge(source="documents", target="generator")],  # <-- "generator" not in graph
            start_node="documents")
        issues = validate_workflow(wf)
        assert any(
            "generator" in i and ("not in nodes" in i or "not found" in i or "unreachable" in i)
            for i in issues
        ), f"Validator should catch missing subgraph entry. Issues: {issues}"


class TestBug1508MissingCompletedFiles:
    """#1508: current_item.json not in completed_files.

    DataNode wrote .factory/current_item.json before running the subgraph,
    but the sub-executor's completed_files set didn't include it. The
    subgraph entry node declared reads={".factory/current_item.json"} and
    timed out waiting for a file that already existed on disk.

    The FakeAgent enforces the writes contract: it writes exactly what the
    node declares. If the executor doesn't pass current_item.json through,
    the downstream node's reads won't be satisfied.
    """

    @pytest.mark.asyncio
    async def test_fake_agent_enforces_writes_contract(self, tmp_path: Path):
        generator = AgentNode(
            id="generator",
            role=AgentRole.BUILDER,
            prompt_template="Process the current item from .factory/current_item.json",
            reads={".factory/current_item.json"},
            writes={"document.md"},
            timeout=30)
        wf = Workflow(
            name="bug-1508-repro",
            nodes={"generator": generator},
            edges=[],
            start_node="generator")

        agent = FakeAgent(wf)
        await agent(
            role="builder",
            task="test",
            project_path=tmp_path,
            node_id="generator")

        assert agent.call_count == 1
        assert (tmp_path / "document.md").exists(), (
            "FakeAgent should write exactly the declared writes"
        )

    @pytest.mark.asyncio
    async def test_fake_agent_violate_writes_catches_missing_output(self, tmp_path: Path):
        generator = AgentNode(
            id="generator",
            role=AgentRole.BUILDER,
            prompt_template="Process the item",
            reads=set(),
            writes={"document.md"},
            timeout=30)
        wf = Workflow(
            name="bug-1508-violate",
            nodes={"generator": generator},
            edges=[],
            start_node="generator")

        agent = FakeAgent(wf, violate_writes=True)
        await agent(
            role="builder",
            task="test",
            project_path=tmp_path,
            node_id="generator")

        assert not (tmp_path / "document.md").exists(), (
            "violate_writes=True should NOT write declared files"
        )


class TestBug1541SplitBypass:
    """#1541: DataNode ignored allowed_instance_ids, bypassing train/val split.

    This is a semantic check the validator/FakeAgent can't catch generically.
    It needs a targeted behavioral test using the spy call log.
    """

    @pytest.mark.asyncio
    async def test_fake_agent_records_calls_for_targeted_assertions(self, tmp_path: Path):
        generator = AgentNode(
            id="generator",
            role=AgentRole.BUILDER,
            prompt_template="Process item",
            reads=set(),
            writes={"output.md"},
            timeout=30)
        wf = Workflow(
            name="bug-1541-repro",
            nodes={"generator": generator},
            edges=[],
            start_node="generator")

        agent = FakeAgent(wf)

        # Simulate 3 calls (like iterating over 3 instances)
        for i in range(3):
            await agent(
                role="builder",
                task=f"process item-{i}",
                project_path=tmp_path,
                node_id="generator")

        assert agent.call_count == 3
        tasks = [c.task for c in agent.calls]
        assert "process item-0" in tasks
        assert "process item-1" in tasks
        assert "process item-2" in tasks

        builder_calls = agent.get_calls_for_role("builder")
        assert len(builder_calls) == 3


class TestValidatorAtExecutorStart:
    """Verify the executor validates workflows at construction time."""

    def test_executor_rejects_invalid_workflow(self, tmp_path: Path):
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="invalid-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="",  # empty prompt
                    reads=set(),
                    writes=set()),
            },
            edges=[],
            start_node="builder")

        with pytest.raises(ValueError, match="empty prompt_template"):
            WorkflowExecutor(wf, tmp_path)

    def test_executor_accepts_valid_workflow(self, tmp_path: Path):
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="valid-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Do the thing.",
                    reads=set(),
                    writes=set()),
            },
            edges=[],
            start_node="builder")

        executor = WorkflowExecutor(wf, tmp_path, validate=True)
        assert executor is not None

    def test_executor_validate_false_skips_check(self, tmp_path: Path):
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="skip-validate",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="",  # empty but validate=False
                    reads=set(),
                    writes=set()),
            },
            edges=[],
            start_node="builder")

        executor = WorkflowExecutor(wf, tmp_path, validate=False)
        assert executor is not None


class TestInvokeAgentNodeId:
    """invoke_agent must accept node_id kwarg without TypeError.

    The executor passes node_id=node.id to the agent_fn. When using the
    default agent_fn (invoke_agent from runner.py), this would crash with
    TypeError if the parameter wasn't declared.
    """

    @pytest.mark.asyncio
    async def test_default_agent_fn_receives_node_id(self, tmp_path: Path):
        """Patch invoke_agent and verify it receives the node_id kwarg."""
        from factory.workflow.executor import WorkflowExecutor

        wf = Workflow(
            name="node-id-test",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Do the thing.",
                    reads=set(),
                    writes=set()),
            },
            edges=[],
            start_node="builder")

        captured_kwargs: dict = {}

        async def mock_invoke_agent(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return ("ok", 0)

        executor = WorkflowExecutor(
            wf, tmp_path, agent_fn=mock_invoke_agent, validate=False)
        await executor.execute()

        assert "node_id" in captured_kwargs, (
            "agent_fn should receive node_id kwarg from executor"
        )
        assert captured_kwargs["node_id"] == "builder"


class TestBug1568DataNodeImplicitCurrentItemWrite:
    """#1568: DataNode implicit current_item.json write not in validation.

    DataNode implicitly writes .factory/current_item.json before running its
    subgraph (executor.py L1012), but this implicit write is not declared in
    DataNode.writes. So _validate_data_dependencies flags the subgraph entry's
    read of current_item.json as unsatisfied, causing validation to fail at
    executor construction (since PR #1545 made validate=True the default).
    """

    def test_datanode_subgraph_entry_reading_current_item_passes_validation(self):
        """Subgraph entry reading current_item.json should NOT fail validation."""
        entry = AgentNode(
            id="processor",
            role=AgentRole.BUILDER,
            prompt_template="Process the current item from .factory/current_item.json",
            reads={".factory/current_item.json"},
            writes={"output.md"})
        data = DataNode(
            id="data",
            inline_items=[DataItem(id="item-1", prompt="first")],
            parallelism=1)
        wf = Workflow(
            name="bug-1568-repro",
            nodes={
                "data": data,
                "processor": entry,
                "_join_data": JoinNode(id="_join_data", sources=["processor"]),
            },
            edges=[
                Edge(source="data", target="processor"),
                Edge(source="processor", target="_join_data"),
            ],
            start_node="data")
        issues = validate_workflow(wf)
        current_item_issues = [
            i for i in issues if "current_item.json" in i
        ]
        assert not current_item_issues, (
            f"DataNode implicit write of current_item.json should satisfy "
            f"subgraph entry reads. Issues: {current_item_issues}"
        )

    def test_datanode_subgraph_entry_reading_other_file_still_fails(self):
        """Subgraph entry reading something else no predecessor writes SHOULD fail."""
        entry = AgentNode(
            id="processor",
            role=AgentRole.BUILDER,
            prompt_template="Process the data",
            reads={".factory/current_item.json", "nonexistent_input.md"},
            writes={"output.md"})
        data = DataNode(
            id="data",
            inline_items=[DataItem(id="item-1", prompt="first")],
            parallelism=1)
        wf = Workflow(
            name="bug-1568-other-read",
            nodes={
                "data": data,
                "processor": entry,
                "_join_data": JoinNode(id="_join_data", sources=["processor"]),
            },
            edges=[
                Edge(source="data", target="processor"),
                Edge(source="processor", target="_join_data"),
            ],
            start_node="data")
        issues = validate_workflow(wf)
        # current_item.json should be satisfied
        current_item_issues = [
            i for i in issues if "current_item.json" in i
        ]
        assert not current_item_issues, (
            f"current_item.json should be satisfied: {current_item_issues}"
        )
        # nonexistent_input.md should still fail
        other_issues = [
            i for i in issues if "nonexistent_input.md" in i
        ]
        assert other_issues, (
            "Reads not written by any predecessor should still fail validation"
        )


class TestDataNodeSwallowsSubgraphFailures:
    """DataNode error propagation gap: when ALL subgraph items fail,
    the DataNode still completed successfully and the workflow continued.

    Root cause: _execute_data() gathered item results but never checked
    whether any items failed. Even if every subgraph halted (e.g. because
    reads weren't satisfied), the DataNode emitted node.completed and
    followed edges.

    Fix: after gathering results, if ALL items have success=False, halt
    the DataNode. Partial failure (some items fail) is acceptable and
    logs a warning.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_all_items_failed_halts_datanode(self, tmp_path: Path):
        """DataNode with 2 inline items where researcher violates writes,
        causing builder to fail on reads → top-level halted=True."""
        from factory.workflow.executor import WorkflowExecutor

        # Build a DataNode with a 2-node subgraph:
        #   researcher (writes research.md) → builder (reads research.md)
        # FakeAgent(violate_writes=True) on researcher means research.md
        # is never written, so builder's reads aren't satisfied and the
        # subgraph halts.
        researcher = AgentNode(
            id="researcher",
            role=AgentRole.RESEARCHER,
            prompt_template="Research the item.",
            reads=set(),
            writes={"research.md"},
            timeout=10)
        builder = AgentNode(
            id="builder",
            role=AgentRole.BUILDER,
            prompt_template="Build from research.",
            reads={"research.md"},
            writes={"output.md"},
            timeout=10)
        data = DataNode(
            id="data",
            inline_items=[
                DataItem(id="item-1", prompt="first"),
                DataItem(id="item-2", prompt="second"),
            ],
            parallelism=1)
        wf = Workflow(
            name="datanode-failure-test",
            nodes={
                "data": data,
                "researcher": researcher,
                "builder": builder,
                "_join_data": JoinNode(id="_join_data", sources=["builder"]),
            },
            edges=[
                Edge(source="data", target="researcher"),
                Edge(source="researcher", target="builder"),
                Edge(source="builder", target="_join_data"),
            ],
            start_node="data")

        _init_git(tmp_path)
        agent = FakeAgent(wf, violate_writes=True)
        executor = WorkflowExecutor(
            wf, tmp_path, agent_fn=agent, validate=False, auto_write_outputs=False)

        # Patch _wait_for_reads max_wait to 0.5s so the test doesn't
        # wait 60s per item (the default read-wait timeout).
        async def fast_wait(self_inner, node):
            """_wait_for_reads with a 0.5s timeout instead of 60s."""
            if not node.reads:
                return
            poll_interval = 0.05
            max_wait = 0.5
            waited = 0.0
            while True:
                missing = node.reads - self_inner.completed_files
                if not missing:
                    return
                if waited >= max_wait:
                    self_inner.result.halted = True
                    self_inner.result.halt_reason = (
                        f"node '{node.id}' timed out waiting for reads: {sorted(missing)}"
                    )
                    return
                import asyncio
                await asyncio.sleep(poll_interval)
                waited += poll_interval

        with patch.object(WorkflowExecutor, '_wait_for_reads', fast_wait):
            result = await executor.execute()

        # In the fork/join model, sub-executor failures surface as
        # failed items (not errored) — the DataNode still completes.
        assert "data" in result.node_outputs
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2
        # Both items should have results (status is "failed" because
        # the sub-executor halted on reads and no task verified them)
        assert all(r["status"] in ("failed", "errored") for r in parsed)

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, tmp_path: Path):
        """DataNode with 2 items where only one fails → workflow continues."""
        from factory.workflow.executor import WorkflowExecutor

        node = AgentNode(
            id="worker",
            role=AgentRole.BUILDER,
            prompt_template="Process item.",
            reads=set(),
            writes={"output.md"},
            timeout=10)
        data = DataNode(
            id="data",
            inline_items=[
                DataItem(id="good-item", prompt="good"),
                DataItem(id="bad-item", prompt="bad"),
            ],
            parallelism=1)
        wf = Workflow(
            name="partial-failure-test",
            nodes={
                "data": data,
                "worker": node,
                "_join_data": JoinNode(id="_join_data", sources=["worker"]),
            },
            edges=[
                Edge(source="data", target="worker"),
                Edge(source="worker", target="_join_data"),
            ],
            start_node="data")

        call_count = 0

        def selective_behavior(role, task, project_path, **kwargs):
            nonlocal call_count
            call_count += 1
            # Second call raises to simulate failure
            if call_count == 2:
                raise RuntimeError("simulated item failure")
            return ("ok", 0)

        _init_git(tmp_path)
        agent = FakeAgent(wf, behavior=selective_behavior)
        executor = WorkflowExecutor(
            wf, tmp_path, agent_fn=agent, validate=False, auto_write_outputs=False)
        result = await executor.execute()

        assert result.success, (
            f"Partial failure should not halt workflow, got halt_reason: {result.halt_reason}"
        )
        assert "data" in result.node_outputs
        parsed = json.loads(result.node_outputs["data"])
        assert len(parsed) == 2


# ── Fix 4: ordinary workflow first-node reads don't fail validation ──


class TestOrdinaryWorkflowFirstNodeReads:
    """When runtime_inputs is empty (ordinary workflow), a first node
    with reads should NOT produce a validation error. Before the fix,
    removing the 'if not predecessors: continue' broke this."""

    def test_first_node_reads_external_file_passes_validation(self) -> None:
        from factory.workflow.validation import validate_workflow

        wf = Workflow(
            name="ordinary-first-read",
            nodes={
                "start": AgentNode(
                    id="start",
                    role=AgentRole.BUILDER,
                    prompt_template="Read the strategy",
                    reads={".factory/strategy/current.md"},
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="start",
        )
        issues = validate_workflow(wf)
        read_issues = [i for i in issues if "reads" in i and "no predecessor" in i]
        assert len(read_issues) == 0, (
            f"Ordinary workflow first-node reads should not fail: {read_issues}"
        )

    def test_data_workflow_first_node_reads_still_validated(self) -> None:
        """When runtime_inputs IS set (data workflow), a first node reading
        something not in runtime_inputs should still fail validation."""
        from factory.workflow.validation import validate_workflow

        wf = Workflow(
            name="data-first-read",
            nodes={
                "start": AgentNode(
                    id="start",
                    role=AgentRole.BUILDER,
                    prompt_template="Process item",
                    reads={".factory/current_item.json", "nonexistent.txt"},
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="start",
            runtime_inputs=frozenset({".factory/current_item.json"}),
        )
        issues = validate_workflow(wf)
        read_issues = [i for i in issues if "nonexistent.txt" in i]
        assert len(read_issues) == 1, (
            f"Data workflow should flag missing reads: {issues}"
        )


# ── Fix 5: _deep_copy_workflow preserves runtime_inputs ──────────


class TestDeepCopyPreservesRuntimeInputs:
    """_deep_copy_workflow must preserve runtime_inputs."""

    def test_deep_copy_preserves_runtime_inputs(self) -> None:
        from factory.outer_loop.mutations import _deep_copy_workflow

        wf = Workflow(
            name="ri-test",
            nodes={
                "a": FnNode(id="a", command="echo hello"),
            },
            edges=[],
            start_node="a",
            runtime_inputs=frozenset({".factory/current_item.json", "data.csv"}),
        )

        copied = _deep_copy_workflow(wf)
        assert copied.runtime_inputs == wf.runtime_inputs, (
            f"Expected {wf.runtime_inputs}, got {copied.runtime_inputs}"
        )

    def test_to_dict_from_dict_round_trip_preserves_runtime_inputs(self) -> None:
        wf = Workflow(
            name="ri-roundtrip",
            nodes={
                "a": FnNode(id="a", command="echo hi"),
            },
            edges=[],
            start_node="a",
            runtime_inputs=frozenset({".factory/current_item.json"}),
        )

        data = wf.to_dict()
        assert "runtime_inputs" in data

        restored = Workflow.from_dict(data)
        assert restored.runtime_inputs == wf.runtime_inputs
