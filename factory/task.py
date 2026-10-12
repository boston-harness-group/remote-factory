"""First-class Task abstraction with four-hook interface.

Defines what to evaluate (instances, setup, prompt, verify) independently
of modes, workflows, and the outer loop.

Architectural constraint: this module has ZERO module-level imports from
factory/workflow/, factory/outer_loop/, factory/agents/, or factory/compose.py.
"""

from __future__ import annotations

import re
import subprocess
import traceback
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

log = structlog.get_logger()

_SHELL_OPERATORS_RE = re.compile(r"&&|\|\||[;|]")


def _needs_shell(cmd: str) -> bool:
    """Return True if cmd contains shell operators that require shell=True."""
    return bool(_SHELL_OPERATORS_RE.search(cmd))


# ── Supporting Models ────────────────────────────────────────────


class TaskInstance(BaseModel):
    """Lightweight container yielded by Task.instances()."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    split: Literal["train", "val"] | None = None


class VerifyResult(BaseModel):
    """Standardised verification + scoring output from Task.verify()."""

    model_config = ConfigDict(strict=True, extra="forbid")

    passed: bool
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


# ── Scoring Contract ─────────────────────────────────────────────


class ScoringContract(BaseModel):
    """Unified scoring configuration.

    Two modes:
    - 'json': verify command outputs JSON with 'passed' (bool) and 'score' (float)
    - 'exit_code': binary pass/fail from subprocess exit code (exit 0 = 1.0, else 0.0)

    If the verify command outputs valid JSON with 'passed' and 'score' keys,
    those values are used regardless of the configured method.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    method: Literal["json", "exit_code"] = "exit_code"
    metric_path: str = "score"


# ── Capability StrEnum ───────────────────────────────────────────


class Capability(StrEnum):
    """Closed set of capabilities that modes provide and tasks require."""

    CAN_MODIFY_CODE = "can_modify_code"
    CAN_RUN_TESTS = "can_run_tests"
    HAS_BUILDER = "has_builder"
    HAS_RESEARCHER = "has_researcher"
    HAS_STRATEGIST = "has_strategist"
    HAS_QUALITY_GATE = "has_quality_gate"
    HAS_PARALLELISM = "has_parallelism"
    HAS_CODE_REVIEW = "has_code_review"
    HAS_ADVERSARIAL_QA = "has_adversarial_qa"
    HAS_ARCHIVIST = "has_archivist"
    CAN_GENERATE_PROMPTS = "can_generate_prompts"
    CAN_RUN_SUBPROCESS = "can_run_subprocess"
    CAN_ACCESS_NETWORK = "can_access_network"
    HAS_HEALTH_CHECK = "has_health_check"
    CAN_ITERATE = "can_iterate"


CAPABILITY_ALIASES: dict[str, Capability] = {
    "codebase-analysis": Capability.HAS_RESEARCHER,
    "code-generation": Capability.CAN_MODIFY_CODE,
    "health-check": Capability.HAS_HEALTH_CHECK,
    "code-review": Capability.HAS_CODE_REVIEW,
    "adversarial-qa": Capability.HAS_ADVERSARIAL_QA,
    "observation": Capability.HAS_RESEARCHER,
}


# ── Task Constraints ─────────────────────────────────────────────


class TaskConstraints(BaseModel):
    """Operational limits for task execution."""

    model_config = ConfigDict(strict=True, extra="forbid")

    timeout: int = 600
    max_retries: int = 1
    required_capabilities: list[Capability] | None = None

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def _coerce_capabilities(cls, v: object) -> list[Capability] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return [Capability(x) if isinstance(x, str) else x for x in v]
        return v  # type: ignore[return-value]


# ── Evaluator Reference ─────────────────────────────────────────


class EvaluatorRef(BaseModel):
    """Serialisable reference to an evaluator class."""

    model_config = ConfigDict(strict=True, extra="forbid")

    ref: str = ""  # "module.path:ClassName" or shorthand

    _SHORTHANDS: dict[str, str] = {
        "exit_code": "factory.outer_loop.evaluators.exit_code:ExitCodeEvaluator",
        "json": "factory.outer_loop.evaluators.json_evaluator:JSONEvaluator",
    }

    def resolve(self) -> Any:
        """Dynamically import and instantiate the evaluator."""
        import importlib

        target = self._SHORTHANDS.get(self.ref, self.ref)
        if ":" not in target:
            raise ValueError(
                f"Invalid evaluator ref {self.ref!r}. "
                f"Expected 'module:Class' or one of {sorted(self._SHORTHANDS)}"
            )
        module_path, class_name = target.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls()


# ── Task Reference ─────────────────────────────────────────────


class TaskRef(BaseModel):
    """Serialisable reference to a Task class via 'module.path:ClassName'."""

    model_config = ConfigDict(strict=True, extra="forbid")

    ref: str = ""

    def resolve(self) -> Task:
        """Dynamically import and return a Task instance.

        Split error handling (Django pattern): ImportError for module-not-found,
        separate ImportError for class-not-found in module.
        """
        import importlib

        if ":" not in self.ref:
            raise ValueError(
                f"Invalid task ref {self.ref!r}. "
                f"Expected 'module.path:ClassName' format."
            )
        module_path, class_name = self.ref.rsplit(":", 1)
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Could not import module {module_path!r} from task ref {self.ref!r}. "
                f"Ensure the package is installed (e.g. pip install -e <project>)."
            ) from exc
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(
                f"Module {module_path!r} has no attribute {class_name!r} "
                f"(from task ref {self.ref!r})."
            )
        if not (isinstance(cls, type) and issubclass(cls, Task)):
            raise TypeError(
                f"{self.ref!r} resolved to {cls!r}, which is not a Task subclass."
            )
        return cls()


# ── Instances / Setup / Prompt / Verify config sections ──────────


class InstancesConfig(BaseModel):
    """Configuration for the instances() hook (TOML section)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    format: str = "directory"
    source: str = ""
    holdout_ids: list[str] = Field(default_factory=list)


class SetupConfig(BaseModel):
    """Configuration for the setup() hook (TOML section)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    command: str = ""


class PromptConfig(BaseModel):
    """Configuration for the prompt() hook (TOML section)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = ""


class VerifyConfig(BaseModel):
    """Configuration for the verify() hook (TOML section)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    command: str = ""


# ── TaskDefinition (serialisable Pydantic model) ─────────────────


class TaskDefinition(BaseModel):
    """Serialisable, config-level representation of a Task.

    Contains TOML-friendly flat fields plus four-hook configuration
    sections.  ``from_toml()`` and ``to_task()`` bridge between config
    and live objects.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str = ""
    version: str = ""
    source: str = ""
    scoring: ScoringContract = Field(default_factory=ScoringContract)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    evaluator_ref: EvaluatorRef = Field(default_factory=EvaluatorRef)
    instances_config: InstancesConfig = Field(default_factory=InstancesConfig)
    setup_config: SetupConfig = Field(default_factory=SetupConfig)
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    verify_config: VerifyConfig = Field(default_factory=VerifyConfig)

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_toml(cls, path: str | Path) -> TaskDefinition:
        """Parse a TOML task definition file."""
        import tomllib

        raw = tomllib.loads(Path(path).read_text())

        task_section = raw.get("task", {})
        instances_section = raw.get("instances", {})
        setup_section = raw.get("setup", {})
        prompt_section = raw.get("prompt", {})
        verify_section = raw.get("verify", {})
        scoring_section = raw.get("scoring", {})
        constraints_section = raw.get("constraints", {})

        # Build scoring contract from [scoring] section
        method = scoring_section.get("method", "exit_code")
        if method in ("pytest", "exact_match"):
            method = "exit_code"
        if method not in ("json", "exit_code"):
            raise ValueError(f"Unknown scoring method: {method}")
        scoring = ScoringContract(
            method=method,
            metric_path=scoring_section.get("metric_path", "score"),
        )

        return cls(
            name=task_section.get("name", Path(path).stem),
            description=task_section.get("description", ""),
            version=task_section.get("version", ""),
            source=task_section.get("source", ""),
            scoring=scoring,
            constraints=TaskConstraints(
                timeout=constraints_section.get(
                    "timeout", scoring_section.get("timeout", 600)
                ),
                max_retries=constraints_section.get("max_retries", 1),
                required_capabilities=constraints_section.get(
                    "required_capabilities"
                ),
            ),
            evaluator_ref=EvaluatorRef(ref=scoring_section.get("evaluator_ref", "")),
            instances_config=InstancesConfig.model_validate(instances_section),
            setup_config=SetupConfig.model_validate(setup_section),
            prompt_config=PromptConfig.model_validate(prompt_section),
            verify_config=VerifyConfig.model_validate(verify_section),
        )

    def to_task(self) -> Task:
        """Convert this definition to a live Task with default hook impls."""
        return Task(definition=self)

    def get_evaluator(self) -> Any:
        """Derive the correct Evaluator from the scoring contract.

        Uses lazy imports to preserve task.py independence.
        """
        if self.evaluator_ref.ref:
            return self.evaluator_ref.resolve()

        if self.scoring.method == "json":
            from factory.outer_loop.evaluators.json_evaluator import JSONEvaluator

            return JSONEvaluator(metric_path=self.scoring.metric_path)

        from factory.outer_loop.evaluators.exit_code import ExitCodeEvaluator

        return ExitCodeEvaluator()


# ── Task base class (four-hook interface) ────────────────────────


@dataclass
class _RunResult:
    """Lightweight result of a shell command execution."""

    returncode: int
    stdout: str
    stderr: str
    score: float = 0.0


def _build_verify_details(
    scoring_name: str,
    result: _RunResult,
    passed: bool,
    **extra: Any,
) -> dict[str, Any]:
    """Build a consistent details dict for VerifyResult across all scoring branches."""
    details: dict[str, Any] = {
        "scoring_contract": scoring_name,
        "returncode": result.returncode,
        "stdout": result.stdout[:2000],
        "stderr": result.stderr[:2000],
    }
    details.update(extra)
    return details


class Task:
    """Base class implementing the four-hook interface.

    Default implementations use flat fields from an optional TaskDefinition.
    Subclasses override whichever hooks need real domain logic.
    """

    def __init__(self, definition: TaskDefinition | None = None) -> None:
        self._definition = definition or TaskDefinition(name=self._derive_name())

    # ── Introspection ────────────────────────────────────────────

    @property
    def definition(self) -> TaskDefinition:
        return self._definition

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def scoring(self) -> ScoringContract:
        return self._definition.scoring

    @property
    def constraints(self) -> TaskConstraints:
        return self._definition.constraints

    # ── Four hooks ───────────────────────────────────────────────

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[TaskInstance]:
        """Discover what to work on.

        Default: yield a single instance with id="default".
        Override for multi-instance tasks (directory scanning, API queries).

        Args:
            split: Filter instances by split assignment. "train" returns
                only training instances, "val" only validation, "all" returns
                everything. Default is "all" for backward compatibility.
        """
        raw = list(self._raw_instances())
        assigned = self._assign_splits(raw)
        yield from self._filter_by_split(assigned, split)

    def _raw_instances(self) -> Iterator[TaskInstance]:
        """Yield unfiltered instances from config. Subclasses may override."""
        cfg = self._definition.instances_config
        if cfg.source and cfg.format == "directory":
            source = Path(cfg.source)
            if source.is_dir():
                for subdir in sorted(source.iterdir()):
                    if subdir.is_dir():
                        yield TaskInstance(id=subdir.name, path=subdir)
                return
        yield TaskInstance(id="default")

    def _assign_splits(self, instances: list[TaskInstance]) -> list[TaskInstance]:
        """Assign split labels to instances that don't have one.

        Priority:
        1. If the instance already has a split set, keep it.
        2. If holdout_ids is configured, mark matching IDs as val.
        3. Otherwise (no-split fallback): all instances become train.
           Splits are opt-in only — no automatic partitioning.
        """
        holdout_ids = set(self._definition.instances_config.holdout_ids)

        result: list[TaskInstance] = []
        for inst in instances:
            if inst.split is not None:
                # Explicit split from subclass — keep as-is
                result.append(inst)
            elif holdout_ids and inst.id in holdout_ids:
                result.append(inst.model_copy(update={"split": "val"}))
            elif holdout_ids:
                result.append(inst.model_copy(update={"split": "train"}))
            else:
                # No-split fallback: everything is train
                result.append(inst.model_copy(update={"split": "train"}))
        return result

    @staticmethod
    def _filter_by_split(
        instances: list[TaskInstance],
        split: Literal["train", "val", "all"],
    ) -> Iterator[TaskInstance]:
        """Filter instances by split value."""
        for inst in instances:
            if split == "all" or inst.split == split:
                yield inst

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        """Prepare the environment for one instance.

        Default: run setup_config.command if set, else no-op.
        """
        cmd = self._definition.setup_config.command
        if not cmd:
            return
        expanded = cmd.replace("{instance_id}", instance.id)
        if instance.path is not None:
            expanded = expanded.replace("{instance_dir}", str(instance.path))
        self.shell(expanded, cwd=workspace)

    def prompt(self, instance: TaskInstance) -> str:
        """What should the agent do for this instance?

        Default: return prompt_config.text if set, else generic prompt.
        Substitutes {instance_id} and {instance_dir} placeholders (no quoting —
        prompt text is not shell-executed).
        """
        text = self._definition.prompt_config.text
        if text:
            expanded = text.replace("{instance_id}", instance.id)
            if instance.path is not None:
                expanded = expanded.replace("{instance_dir}", str(instance.path))
            return expanded
        return "Implement the feature. All tests must pass."

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Did it work? Returns unified VerifyResult with pass/fail + score.

        Default: run verify_config.command and parse output.
        Substitutes {instance_id} and {instance_dir} placeholders (shell-quoted
        via shlex.quote() for injection safety).
        If the command outputs valid JSON with 'passed' and 'score' keys, use those.
        Otherwise fall back to exit code scoring (exit 0 = pass/1.0).
        """
        import shlex as _shlex

        cmd = self._definition.verify_config.command
        if not cmd:
            return VerifyResult(passed=False, score=0.0)

        cmd = cmd.replace("{instance_id}", _shlex.quote(instance.id))
        if instance.path is not None:
            cmd = cmd.replace("{instance_dir}", _shlex.quote(str(instance.path)))

        result = self.shell(cmd, cwd=workspace)
        scoring = self._definition.scoring

        if scoring.method == "json":
            return self._parse_json_output(result, scoring.metric_path)

        # exit_code: try JSON first, fall back to exit code
        json_result = self._try_parse_json_output(result)
        if json_result is not None:
            return json_result

        passed = result.returncode == 0
        s = 1.0 if passed else 0.0
        return VerifyResult(
            passed=passed,
            score=s,
            details=_build_verify_details("exit_code", result, passed),
        )

    # ── Utilities ────────────────────────────────────────────────

    def shell(self, cmd: str, cwd: Path | None = None) -> _RunResult:
        """Run a shell command and return a _RunResult."""
        import shlex as _shlex

        try:
            if _needs_shell(cmd):
                argv: str | list[str] = cmd
                use_shell = True
            else:
                argv = _shlex.split(cmd)
                use_shell = False
            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=self._definition.constraints.timeout,
                shell=use_shell,
            )
            score = 0.0
            if proc.returncode == 0:
                score = 1.0
            return _RunResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                score=score,
            )
        except subprocess.TimeoutExpired:
            return _RunResult(returncode=-1, stdout="", stderr="timeout", score=0.0)
        except Exception as exc:
            log.error("task_run_failed", cmd=cmd, exc_info=True)
            return _RunResult(
                returncode=-1,
                stdout="",
                stderr=f"{exc}\n{traceback.format_exc()}",
                score=0.0,
            )

    def get_evaluator(self) -> Any:
        """Delegate to TaskDefinition.get_evaluator()."""
        return self._definition.get_evaluator()

    def to_definition(self) -> TaskDefinition:
        """Convert this Task to its serialisable TaskDefinition."""
        return self._definition

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_legacy(
        cls,
        name: str,
        test_command: str = "",
        test_format: str = "pytest",
        metric_path: str = "score",
        instance_format: str = "directory",
        prep_command: str = "",
    ) -> Task:
        """Construct a Task from legacy flat fields (backward compat)."""
        method: Literal["json", "exit_code"] = "exit_code"
        if test_format == "json":
            method = "json"

        defn = TaskDefinition(
            name=name,
            scoring=ScoringContract(method=method, metric_path=metric_path),
            instances_config=InstancesConfig(format=instance_format),
            setup_config=SetupConfig(command=prep_command),
            verify_config=VerifyConfig(command=test_command),
        )
        return cls(definition=defn)

    @classmethod
    def from_toml(cls, path: str | Path) -> Task:
        """Load a Task from a TOML file."""
        defn = TaskDefinition.from_toml(path)
        return cls(definition=defn)

    # ── Private helpers ──────────────────────────────────────────

    def _derive_name(self) -> str:
        """Derive a kebab-case name from the class name."""
        cls_name = type(self).__name__
        if cls_name == "Task":
            return "default"
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", cls_name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1).lower()

    @staticmethod
    def _parse_json_output(result: _RunResult, metric_path: str) -> VerifyResult:
        """Parse JSON stdout and extract metric at metric_path."""
        import json

        try:
            data = json.loads(result.stdout)
            if "passed" in data and "score" in data:
                passed = bool(data["passed"])
                score = min(max(float(data["score"]), 0.0), 1.0)
                return VerifyResult(
                    passed=passed,
                    score=score,
                    details=_build_verify_details("json", result, passed),
                )
            obj: Any = data
            for key in metric_path.split("."):
                obj = obj[key]
            score = float(obj)
            passed = score > 0
            return VerifyResult(
                passed=passed,
                score=min(max(score, 0.0), 1.0),
                details=_build_verify_details(
                    "json", result, passed,
                    metric_path=metric_path, raw_value=score,
                ),
            )
        except Exception:
            return VerifyResult(
                passed=False,
                score=0.0,
                details=_build_verify_details(
                    "json", result, False,
                    error="json_parse_failed",
                ),
            )

    @staticmethod
    def _try_parse_json_output(result: _RunResult) -> VerifyResult | None:
        """Try to parse JSON with 'passed' and 'score' keys. Return None if not valid."""
        import json

        try:
            data = json.loads(result.stdout)
            if isinstance(data, dict) and "passed" in data and "score" in data:
                passed = bool(data["passed"])
                score = min(max(float(data["score"]), 0.0), 1.0)
                return VerifyResult(
                    passed=passed,
                    score=score,
                    details=_build_verify_details("json", result, passed),
                )
        except Exception:
            pass
        return None


# ── SourceTask — wraps inline_items / source_path into a Task ────


class SourceTask(Task):
    """Task subclass for DataNodes with inline_items or source_path.

    Wraps the items from a DataNode source into proper TaskInstances.
    ``verify()`` is a no-op pass — scoring comes from the branch workflow.
    """

    def __init__(
        self,
        items: list[TaskInstance],
        *,
        name: str = "source-task",
    ) -> None:
        super().__init__(
            definition=TaskDefinition(name=name),
        )
        self._items = items

    def _raw_instances(self) -> Iterator[TaskInstance]:
        yield from self._items

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        pass

    def prompt(self, instance: TaskInstance) -> str:
        meta = instance.metadata or {}
        return meta.get("prompt", f"Process item {instance.id}")

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        return VerifyResult(passed=True, score=1.0)


# ── DefaultTask — single-instance Task whose verify runs test_command ──


class DefaultTask(Task):
    """Default single-instance Task for the no-task path.

    ``verify()`` runs the configured test_command.
    """

    def __init__(self, *, test_command: str = "", name: str = "default") -> None:
        defn = TaskDefinition(
            name=name,
            verify_config=VerifyConfig(command=test_command),
        )
        super().__init__(definition=defn)

    def _raw_instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(id="default")
