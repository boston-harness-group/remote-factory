"""Pydantic v2 strict models for the outer loop evolutionary search."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MutationType(str, Enum):
    """Types of graph mutation operators."""

    NODE_INSERT = "node_insert"
    NODE_REMOVE = "node_remove"
    EDGE_REDIRECT = "edge_redirect"
    PARALLELIZE = "parallelize"
    SERIALIZE = "serialize"
    PARAM_MUTATE = "param_mutate"
    PROMPT_MUTATE = "prompt_mutate"
    KNOB_MUTATE = "knob_mutate"


class MutationRecord(BaseModel):
    """Record of a single mutation applied to a workflow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    operator: MutationType
    target_node: str | None = None
    before: dict[str, object] = Field(default_factory=dict)
    after: dict[str, object] = Field(default_factory=dict)
    rationale: str = ""

    @field_validator("operator", mode="before")
    @classmethod
    def _coerce_operator(cls, v: object) -> MutationType:
        if isinstance(v, str):
            return MutationType(v)
        return v  # type: ignore[return-value]


class Individual(BaseModel):
    """A single candidate in the evolutionary population."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    workflow_data: dict[str, object]
    score: float | None = None
    val_score: float | None = None
    features: tuple[int, ...] = ()
    generation: int = 0
    parent_id: str | None = None
    mutation_record: MutationRecord | None = None
    cost_usd: float = 0.0

    @field_validator("features", mode="before")
    @classmethod
    def _coerce_features(cls, v: object) -> tuple[int, ...]:
        if isinstance(v, list):
            return tuple(v)
        return v  # type: ignore[return-value]


class HyperparameterRecord(BaseModel):
    """Per-generation evolutionary hyperparameters for Level 3 training data."""

    model_config = ConfigDict(strict=True, extra="forbid")

    generation: int
    mutation_rate: float
    population_size: int
    tournament_size: int
    designer_ratio: float
    operator_weights: dict[str, float] = Field(default_factory=dict)
    best_score: float = 0.0
    mean_score: float = 0.0
    diversity: float = 0.0
    novel_count: int = 0


class SwarmConfig(BaseModel):
    """Configuration for the evolutionary swarm search."""

    model_config = ConfigDict(strict=True, extra="forbid")

    benchmark: str
    budget: int
    population_size: int = 4
    tournament_size: int = 3
    rank_weighted_selection: bool = False
    mutation_rate: float = 0.3
    target_score: float | None = None
    frozen_node_ids: list[str] = Field(default_factory=list)
    mandatory_node_roles: list[str] = Field(default_factory=list)
    feature_axes: list[str] = Field(
        default_factory=lambda: ["depth", "fork_degree", "agent_count", "gate_count"]
    )
    mutation_strategy: str = "weighted_random"
    designer_count: int = 2
    training_instances: list[str] = Field(default_factory=list)
    plateau_window: int = 3
    plateau_threshold: float = 0.01
    diversity_floor: float = 0.2
    target_project: str = ""
    test_command: str = ""
    test_format: str = "pytest"
    metric_path: str = "score"
    seed_workflow: str = ""
    instance_format: str = "directory"
    prep_command: str = ""
    early_stop_unchanged: int = 3
    task_module: str = ""
    seed_workflow_module: str = ""
    execution_strategy: Literal["executor", "ceo-skill", "ceo-tool"] = "executor"

    # ── Task integration (not serialised — set at runtime) ───────

    _task: Any = None  # set via set_task(), not a Pydantic field

    def set_task(self, task: Any) -> None:
        """Attach a Task object at runtime (not serialised)."""
        object.__setattr__(self, "_task", task)

    def get_task(self) -> Any:
        """Return task via 3-tier precedence: set_task() > task_module > from_legacy().

        Uses lazy import to avoid circular dependency.
        """
        if self._task is not None:
            return self._task

        if self.task_module:
            from factory.task import TaskRef

            task = TaskRef(ref=self.task_module).resolve()
            object.__setattr__(self, "_task", task)
            return task

        from factory.task import Task

        task = Task.from_legacy(
            name=self.benchmark,
            test_command=self.test_command,
            test_format=self.test_format,
            metric_path=self.metric_path,
            instance_format=self.instance_format,
            prep_command=self.prep_command,
        )
        object.__setattr__(self, "_task", task)
        return task


class OuterLoopState(BaseModel):
    """Checkpoint state for the outer loop evolution."""

    model_config = ConfigDict(strict=True, extra="forbid")

    generation: int = 0
    total_evaluations: int = 0
    best_score: float = 0.0
    budget_remaining: int = 0
    convergence_reason: str | None = None
    score_trajectory: list[float] = Field(default_factory=list)
    hyperparameter_history: list[HyperparameterRecord] = Field(default_factory=list)


class GenerationSummary(BaseModel):
    """Summary of a single generation of evolution."""

    model_config = ConfigDict(strict=True, extra="forbid")

    generation: int
    population_size: int
    best_score: float
    mean_score: float
    diversity: float
    mutations_applied: list[MutationRecord] = Field(default_factory=list)
    novel_count: int = 0
    rejected_duplicates: int = 0
    holdout_score: float = 0.0
    hyperparameters: HyperparameterRecord | None = None


class EvalResult(BaseModel):
    """Result of evaluating a single workflow candidate."""

    model_config = ConfigDict(strict=True, extra="forbid")

    score: float
    benchmark_score: float = 0.0
    hygiene_score: float = 0.0
    cost_usd: float = 0.0
    complexity: float = 0.0
    details: dict[str, object] = Field(default_factory=dict)


class AuditResult(BaseModel):
    """Result of overfit detection on the best evolved workflow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    training_score: float
    holdout_score: float
    delta: float
    overfit_flag: bool
    details: str = ""


class OuterLoopResult(BaseModel):
    """Result of a complete outer loop evolutionary run."""

    model_config = ConfigDict(strict=True, extra="forbid")

    best_workflow_data: dict[str, object] = Field(default_factory=dict)
    best_score: float = 0.0
    val_score: float | None = None
    overfit_flag: bool = False
    trajectory: list[GenerationSummary] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    convergence_reason: str = ""
    generations_completed: int = 0
    total_evaluations: int = 0
    archive_size: int = 0
    total_candidates_evaluated: int = 0
    pareto_front: list[Individual] = Field(default_factory=list)
    hyperparameter_history: list[HyperparameterRecord] = Field(default_factory=list)
