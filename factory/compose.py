"""Mode × Task composition validation layer.

Validates that a workflow (mode) can run a task by checking capability
compatibility, then wires up a task-attached InnerLoop.

Imports from factory.task and factory.workflow.primitives directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from factory.task import CAPABILITY_ALIASES, Capability, VerifyResult

log = structlog.get_logger()


# ── IncompatibleCompositionError ─────────────────────────────────


class IncompatibleCompositionError(TypeError):
    """Raised when a mode cannot run a task due to missing capabilities."""

    def __init__(
        self,
        mode_name: str,
        task_name: str,
        mode_missing: set[str],
        task_missing: set[str] | None = None,
        hint: str = "",
    ) -> None:
        self.mode_name = mode_name
        self.task_name = task_name
        self.mode_missing = mode_missing
        self.task_missing = task_missing or set()
        self.hint = hint
        parts = [
            f"Mode '{mode_name}' cannot run task '{task_name}'.",
        ]
        if mode_missing:
            caps = ", ".join(sorted(mode_missing))
            parts.append(f"Mode is missing capabilities: {caps}")
        if self.task_missing:
            caps = ", ".join(sorted(self.task_missing))
            parts.append(f"Task is missing capabilities: {caps}")
        if hint:
            parts.append(f"Hint: {hint}")
        super().__init__(" ".join(parts))


# ── Task Protocol ────────────────────────────────────────────────


@runtime_checkable
class TaskProtocol(Protocol):
    """Runtime-checkable protocol for Task objects (four hooks)."""

    def instances(self) -> Any: ...
    def setup(self, instance: Any, workspace: Path) -> None: ...
    def prompt(self, instance: Any) -> str: ...
    def verify(self, instance: Any, workspace: Path) -> VerifyResult: ...
    def get_evaluator(self) -> Any: ...


# ── ModeCapabilities ─────────────────────────────────────────────


class ModeCapabilities:
    """Capabilities inferred from a Workflow's node structure."""

    __slots__ = ("provides",)

    def __init__(self, provides: frozenset[Capability]) -> None:
        self.provides = provides

    @classmethod
    def from_workflow(cls, workflow: Any) -> ModeCapabilities:
        """Infer capabilities from the workflow's node structure.

        A workflow with a builder AgentNode → CAN_MODIFY_CODE + HAS_BUILDER.
        A GateNode → HAS_QUALITY_GATE. A ForkNode → HAS_PARALLELISM. Etc.
        """
        from factory.workflow.primitives import (
            AgentNode,
            AgentRole,
            DataNode,
            FnNode,
            ForkNode,
            GateNode,
        )

        caps: set[Capability] = set()

        if not hasattr(workflow, "nodes"):
            return cls(frozenset(caps))

        for node in workflow.nodes.values():
            if isinstance(node, AgentNode):
                role = node.role
                if role == AgentRole.BUILDER:
                    caps.add(Capability.CAN_MODIFY_CODE)
                    caps.add(Capability.HAS_BUILDER)
                    caps.add(Capability.CAN_RUN_TESTS)
                    caps.add(Capability.CAN_RUN_SUBPROCESS)
                elif role == AgentRole.RESEARCHER:
                    caps.add(Capability.HAS_RESEARCHER)
                elif role == AgentRole.STRATEGIST:
                    caps.add(Capability.HAS_STRATEGIST)
                elif role == AgentRole.CODE_REVIEWER:
                    caps.add(Capability.HAS_CODE_REVIEW)
                elif role == AgentRole.ADVERSARIAL_TESTER:
                    caps.add(Capability.HAS_ADVERSARIAL_QA)
                elif role == AgentRole.ARCHIVIST:
                    caps.add(Capability.HAS_ARCHIVIST)
                elif role == AgentRole.HEALTH_CHECKER:
                    caps.add(Capability.HAS_HEALTH_CHECK)
                caps.add(Capability.CAN_GENERATE_PROMPTS)

            elif isinstance(node, GateNode):
                caps.add(Capability.HAS_QUALITY_GATE)

            elif isinstance(node, ForkNode):
                caps.add(Capability.HAS_PARALLELISM)

            elif isinstance(node, DataNode):
                caps.add(Capability.CAN_ITERATE)

            elif isinstance(node, FnNode):
                caps.add(Capability.CAN_RUN_SUBPROCESS)

        declared: frozenset[str] = getattr(workflow, "declared_capabilities", frozenset())
        if declared:
            for cap_str in declared:
                mapped = CAPABILITY_ALIASES.get(cap_str)
                if mapped is not None:
                    caps.add(mapped)
                else:
                    log.warning("unmapped_capability", capability=cap_str)

        return cls(frozenset(caps))


# ── TaskCapabilities ─────────────────────────────────────────────


class TaskCapabilities:
    """Capabilities a task requires from a mode, inferred from scoring."""

    __slots__ = ("requires",)

    def __init__(self, requires: frozenset[Capability]) -> None:
        self.requires = requires

    @classmethod
    def from_task(cls, task: Any) -> TaskCapabilities:
        """Infer required capabilities from a task's scoring contract."""
        from factory.task import ScoringContract

        constraints = getattr(task, "constraints", None)
        if constraints is None:
            defn = getattr(task, "definition", None)
            if defn is not None:
                constraints = defn.constraints

        raw_caps = getattr(constraints, "required_capabilities", None) if constraints is not None else None

        if raw_caps is not None:
            return cls(frozenset(raw_caps))

        caps: set[Capability] = set()

        scoring = getattr(task, "scoring", None)
        if scoring is None:
            defn = getattr(task, "definition", None)
            if defn is not None:
                scoring = defn.scoring

        if isinstance(scoring, ScoringContract) and scoring.method == "exit_code":
            caps.add(Capability.CAN_MODIFY_CODE)
            caps.add(Capability.CAN_RUN_TESTS)
            caps.add(Capability.HAS_BUILDER)

        return cls(frozenset(caps))


# ── validate_composition ─────────────────────────────────────────


def _workflow_has_data_node(workflow: Any) -> bool:
    """Check if a workflow contains at least one DataNode."""
    from factory.workflow.primitives import DataNode

    nodes = getattr(workflow, "nodes", {})
    return any(isinstance(n, DataNode) for n in nodes.values())


def validate_composition(workflow: Any, task: Any) -> None:
    """Validate that a workflow can run a task.

    Raises IncompatibleCompositionError if capabilities don't match.

    DataNode workflows are evaluation-only — they handle iteration
    internally and don't need build-pipeline capabilities like
    HAS_BUILDER, CAN_RUN_TESTS, or CAN_MODIFY_CODE.
    """
    mode_caps = ModeCapabilities.from_workflow(workflow)
    task_caps = TaskCapabilities.from_task(task)

    mode_missing = set(task_caps.requires) - set(mode_caps.provides)

    # DataNode workflows handle iteration internally and don't need
    # build-pipeline capabilities
    if mode_missing and _workflow_has_data_node(workflow):
        build_caps = {
            Capability.HAS_BUILDER,
            Capability.CAN_RUN_TESTS,
            Capability.CAN_MODIFY_CODE,
        }
        mode_missing -= build_caps

    mode_name = getattr(workflow, "name", "unknown")
    task_name = getattr(task, "name", "unknown")

    if mode_missing:
        raise IncompatibleCompositionError(
            mode_name=mode_name,
            task_name=task_name,
            mode_missing={str(c) for c in mode_missing},
            hint="Try a mode with builder/test capabilities (e.g. 'improve').",
        )


def check_mode_task_compat(
    workflow: Any, task: Any
) -> tuple[bool, frozenset[Capability]]:
    """Check mode-task compatibility without raising.

    Returns (compatible, missing_capabilities).
    """
    mode_caps = ModeCapabilities.from_workflow(workflow)
    task_caps = TaskCapabilities.from_task(task)
    missing = task_caps.requires - mode_caps.provides
    return (len(missing) == 0, missing)


# ── compose() public API ────────────────────────────────────────


def compose(workflow: Any, task: Any, project_dir: str | Path) -> Any:
    """Compose a workflow + task into a task-attached InnerLoop.

    InnerLoop.step() executes the task end-to-end when task is set:
    iterates task.instances(), runs setup → WorkflowExecutor → verify
    per instance, and aggregates scores via AggregateMethod.

    Known gap: capability compatibility is not re-validated after
    outer-loop mutations (e.g. NODE_REMOVE stripping the Builder).
    Mismatches should be scored as failures (score=0), not crashes.

    1. Protocol check — is this a valid Task?
    2. Composition validation — can this mode run this task?
    3. Derive evaluator from scoring contract
    4. Wire up InnerLoop

    Returns an InnerLoop instance.
    """
    if not isinstance(task, TaskProtocol):
        raise TypeError(
            f"Expected a Task with four hooks (instances, setup, prompt, verify), "
            f"got {type(task).__name__}"
        )

    validate_composition(workflow, task)

    evaluator = task.get_evaluator()

    from factory.inner_loop import InnerLoop

    inner_loop_config = None
    config_path = Path(project_dir) / '.factory' / 'config.json'
    if config_path.exists():
        import json
        try:
            fc = json.loads(config_path.read_text())
            if fc.get('inner_loop'):
                from factory.models import InnerLoopConfig
                inner_loop_config = InnerLoopConfig(**fc['inner_loop'])
        except (json.JSONDecodeError, OSError):
            pass

    mode_name = getattr(workflow, "name", "composed")
    return InnerLoop(
        project_dir=Path(project_dir),
        mode=mode_name,
        evaluator=evaluator,
        workflow=workflow,
        task=task,
        inner_loop_config=inner_loop_config,
    )
