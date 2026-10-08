"""All Pydantic v2 strict models for the remote factory."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── project state ─────────────────────────────────────────────────


class ProjectState(str, Enum):
    """The five possible states of a target project."""

    NO_REPO = "no_repo"
    REPO_INCOMPLETE = "incomplete"
    NO_FACTORY = "no_factory"
    EVALS_PENDING_REVIEW = "evals_pending_review"
    HAS_FACTORY = "has_factory"


# ── factory config ────────────────────────────────────────────────


class HypothesisBudget(BaseModel):
    """Backlog-first hypothesis budget — configurable per-project and per-run."""

    model_config = ConfigDict(strict=True, extra="forbid")

    min_growth: int = 2
    max_new: int = 2


class HardConstraint(BaseModel):
    """A user-defined constraint enforced at the code level via precheck.

    Each constraint has a shell command that must exit 0 for the constraint to pass.
    Non-zero exit = mandatory revert. The CEO cannot override this.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    check: str
    description: str = ""


class ProjectEvalDimension(BaseModel):
    """A user-defined project-specific eval dimension (e.g. benchmark accuracy, latency)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    command: str
    parse: Literal["json", "exit_code"] = "json"
    weight: float = 1.0
    timeout: float = 300.0
    description: str = ""


class EvalWeights(BaseModel):
    """Weight distribution across eval tiers: hygiene, growth, project."""

    model_config = ConfigDict(strict=True, extra="forbid")

    hygiene: float = 0.50
    growth: float = 0.50
    project: float = 0.0


class ResearchTarget(BaseModel):
    """Research mode target — defines the objective, metric, and how to measure it."""

    model_config = ConfigDict(strict=True, extra="forbid")

    objective: str
    metric: str
    target: float
    run_command: str
    result_path: str
    result_parser: Literal["json"] = "json"
    timeout: int = 3600

    def to_task(self) -> Any:
        """Bridge: convert ResearchTarget to a Task with JSON scoring.

        Uses lazy import to avoid circular dependency.
        """
        from factory.task import (
            PromptConfig,
            ScoringContract,
            Task,
            TaskConstraints,
            TaskDefinition,
            VerifyConfig,
        )

        defn = TaskDefinition(
            name=f"research-{self.metric}",
            description=self.objective,
            scoring=ScoringContract(method="json", metric_path=self.result_path),
            constraints=TaskConstraints(timeout=self.timeout),
            prompt_config=PromptConfig(text=self.objective),
            verify_config=VerifyConfig(command=self.run_command),
        )
        return Task(definition=defn)


class ItemStatus(str, Enum):
    """Status of a single item in a DataNode fork run."""

    ok = "ok"
    failed = "failed"
    errored = "errored"


class ItemResult(BaseModel):
    """Result of running one item through the DataNode fork branch."""

    model_config = ConfigDict(strict=True, extra="forbid")

    item_id: str
    split: str = "train"
    status: ItemStatus
    score: float = 0.0
    verify_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    cost: float = 0.0
    duration_s: float = 0.0

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v: object) -> ItemStatus:
        if isinstance(v, str):
            return ItemStatus(v)
        return v  # type: ignore[return-value]


class AggregateMethod(str, Enum):
    """How to aggregate multiple run metrics into a single value."""

    mean = "mean"
    median = "median"
    max = "max"
    all_pass = "all_pass"


class InnerLoopConfig(BaseModel):
    """Inner loop configuration — controls multi-run execution per cycle."""

    model_config = ConfigDict(strict=True, extra="forbid")

    runs_per_cycle: int = Field(ge=1, default=1)
    aggregate: AggregateMethod = AggregateMethod.mean
    plateau_threshold: int = 3
    max_inner_runs_per_cycle: int | None = None

    @field_validator("aggregate", mode="before")
    @classmethod
    def _coerce_aggregate(cls, v: object) -> AggregateMethod:
        if isinstance(v, str):
            return AggregateMethod(v)
        return v  # type: ignore[return-value]


class OuterLoopConfig(BaseModel):
    """Outer loop configuration — controls what happens when inner loop plateaus."""

    model_config = ConfigDict(strict=True, extra="forbid")

    max_outer_cycles: int | None = None
    inner_surfaces: list[str] = []
    outer_surfaces: list[str] = []


class CostBudgetConfig(BaseModel):
    """Per-project cost budget limits for research mode."""

    model_config = ConfigDict(strict=True, extra="forbid")

    max_per_cycle: float | None = None
    max_total: float | None = None


class RunStatus(str, Enum):
    """Possible outcomes of a research run."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class RunResult(BaseModel):
    """Result of executing a research run command."""

    model_config = ConfigDict(strict=True, extra="forbid")

    status: RunStatus
    metric_value: float
    duration_seconds: float
    artifacts_path: Path
    stdout: str
    stderr: str


class ResultParseError(Exception):
    """Raised when a result file cannot be parsed to extract the target metric."""


class TierWeights(BaseModel):
    """Sparse within-tier weight overrides for hygiene or growth dimensions.

    Only set the dimensions you want to override — unset fields (None) keep defaults.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    tests: float | None = None
    lint: float | None = None
    type_check: float | None = None
    coverage: float | None = None
    config_parser: float | None = None
    capability_surface: float | None = None
    experiment_diversity: float | None = None
    observability: float | None = None
    research_grounding: float | None = None
    factory_effectiveness: float | None = None
    spec_compliance: float | None = None


class ParallelConfig(BaseModel):
    """Parallel experiment execution configuration from factory.md."""

    model_config = ConfigDict(strict=True, extra="forbid")

    parallel_hypotheses: int = Field(default=1, ge=1, le=8)
    selection_strategy: Literal["best_score"] = "best_score"


class AdversarialComponent(BaseModel):
    """One side of an adversarial eval loop (generator or discriminator)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["generator", "discriminator"]
    eval_command: str
    metric_name: str
    threshold: float
    scope: list[str] = []
    timeout: float = 300.0


class AdversarialConfig(BaseModel):
    """GAN-style adversarial eval loop configuration from factory.md."""

    model_config = ConfigDict(strict=True, extra="forbid")

    generator: AdversarialComponent
    discriminator: AdversarialComponent
    hysteresis: int = 3
    max_rounds: int | None = None
    convergence_window: int = 5


class AdversarialPhaseRecord(BaseModel):
    """One entry in the adversarial phase history."""

    model_config = ConfigDict(strict=True, extra="forbid")

    round: int
    active_role: Literal["generator", "discriminator"]
    score: float
    metric_name: str
    timestamp: str
    switched: bool


class AdversarialState(BaseModel):
    """Persisted adversarial loop state at .factory/adversarial_state.json."""

    model_config = ConfigDict(strict=True, extra="forbid")

    active_role: Literal["generator", "discriminator"] = "generator"
    current_round: int = 0
    consecutive_above: int = 0
    generator_consecutive_above: int = 0
    discriminator_consecutive_above: int = 0
    converged: bool = False
    history: list[AdversarialPhaseRecord] = []


class FactoryConfig(BaseModel):
    """Machine-readable config stored at .factory/config.json."""

    model_config = ConfigDict(strict=True, extra="forbid")

    goal: str
    scope: list[str]
    guards: list[str]
    eval_command: str
    eval_threshold: float
    constraints: list[str]
    hypothesis_budget: HypothesisBudget = HypothesisBudget()
    target_branch: str = "main"
    smoke_test: str = ""
    project_eval: list[ProjectEvalDimension] = []
    eval_weights: EvalWeights = EvalWeights()
    research_target: ResearchTarget | None = None
    inner_loop: InnerLoopConfig | None = None
    outer_loop: OuterLoopConfig | None = None
    mutable_surfaces: list[str] = []
    fixed_surfaces: list[str] = []
    research_constraints: list[str] = []
    cost_budget: CostBudgetConfig | None = None
    hard_constraints: list[HardConstraint] = []
    eval_spec: list[str] = []
    hygiene_weights: TierWeights | None = None
    growth_weights: TierWeights | None = None
    adversarial: AdversarialConfig | None = None
    parallel: ParallelConfig | None = None
    clean_pr: bool = False
    clean_pr_include: list[str] = []
    clean_pr_exclude: list[str] = []
    test_timeout: int = Field(ge=1, default=600)


# ── eval ──────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Single eval output from one eval function."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    score: float
    weight: float
    passed: bool
    details: str


class CompositeScore(BaseModel):
    """Aggregated result from all evals + guards."""

    model_config = ConfigDict(strict=True, extra="forbid")

    total: float
    results: list[EvalResult]
    guard_violations: list[str]
    passed: bool


# ── eval discovery ────────────────────────────────────────────────


class EvalDimension(BaseModel):
    """One discovered or user-provided eval dimension."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    command: str
    weight: float
    parser: Literal["exit_code", "json", "regex"]
    regex_pattern: str | None = None
    description: str
    source: Literal["explicit", "discovered", "researched", "fallback"]


class EvalProfile(BaseModel):
    """Complete eval profile for a project, built by the Researcher agent."""

    model_config = ConfigDict(strict=True, extra="forbid")

    project_type: str
    dimensions: list[EvalDimension]
    tier: Literal["explicit", "discovered", "researched", "fallback"]
    confidence: float
    human_reviewed: bool = False


class DiscoveredEval(BaseModel):
    """An eval/benchmark script discovered during introspection."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    command: str
    source: str = "discovered"


class ProjectProfile(BaseModel):
    """Project metadata discovered during introspection."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    language: str
    framework: str | None = None
    project_type: str
    has_tests: bool
    has_linter: bool
    has_type_checker: bool
    has_ci: bool
    has_spec: bool = False
    test_command: str | None = None
    lint_command: str | None = None
    type_check_command: str | None = None
    package_manager: str | None = None
    discovered_evals: list[DiscoveredEval] = []


# ── experiments ───────────────────────────────────────────────────


class ExperimentRecord(BaseModel):
    """One row in results.tsv + the experiment directory."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: int
    timestamp: datetime
    hypothesis: str
    change_summary: str
    issue_number: int | None
    pr_number: int | None
    score_before: float | None
    score_after: float | None
    delta: float | None
    verdict: Literal["keep", "revert", "error", "superseded"]
    cost_usd: float | None
    notes: str
    research_citations: list[str] = []


# ── cross-project insights ───────────────────────────────────────


class HypothesisOutcome(BaseModel):
    """A hypothesis paired with its outcome, for cross-project pattern analysis."""

    model_config = ConfigDict(strict=True, extra="forbid")

    hypothesis: str
    verdict: Literal["keep", "revert", "error", "superseded"]
    category: str
    project: str
    delta: float | None = None


class ProjectSummary(BaseModel):
    """Summary of a factory-managed project's experiment history."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    experiment_count: int
    keep_count: int
    revert_count: int
    error_count: int
    keep_rate: float
    latest_score: float | None = None


class Pattern(BaseModel):
    """A recurring pattern discovered across projects."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str
    evidence: list[str]
    confidence: float


class CrossProjectInsights(BaseModel):
    """Aggregated cross-project analysis for the Strategist."""

    model_config = ConfigDict(strict=True, extra="forbid")

    projects: list[ProjectSummary]
    outcomes: list[HypothesisOutcome]
    category_stats: dict[str, dict[str, float]]
    winning_categories: list[str]
    losing_categories: list[str]
    patterns: list[Pattern]
    generated_at: datetime


# ── agent usage / token profiling ─────────────────────────────────


class AgentUsage(BaseModel):
    """Token usage data from a single agent invocation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    num_turns: int = 0
    model: str = ""


# ── session summary ──────────────────────────────────────────


class SessionSummary(BaseModel):
    """End-of-cycle summary: what was built, deferred, and needs human input."""

    model_config = ConfigDict(strict=True, extra="forbid")

    project_name: str
    generated_at: datetime
    mode: str
    experiments_kept: list[ExperimentRecord]
    experiments_reverted: list[ExperimentRecord]
    experiments_errored: list[ExperimentRecord]
    backlog_remaining: list[str]
    guard_violations: list[str]
    needs_human_input: list[str]
    score_start: float | None
    score_end: float | None
    total_cost_usd: float | None


# ── cycle state ──────────────────────────────────────────────────


class CycleState(BaseModel):
    """In-flight cycle state persisted at .factory/state/cycle.json.

    Ensures mode is preserved across CEO respawns within a single cycle.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    cycle_id: str
    started_at: datetime
    mode: str
    initial_prompt: str = ""
    respawns: int = 0
    runner_name: str | None = None
    claude_session_id: str | None = None


# ── ACE pipeline data ────────────────────────────────────────────


class AgentVerdict(BaseModel):
    """A CEO verdict on an agent's output, parsed from ceo-verdict-*.md files."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: str
    verdict: Literal["PROCEED", "REDIRECT", "ABORT"]
    rationale: str
    issues: list[str] = []
    experiment_id: int | None = None


class Observation(BaseModel):
    """A structured observation from the Archivist or Researcher."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source: str
    content: str
    timestamp: datetime
    project: str
    tags: list[str] = []


class PerformanceReport(BaseModel):
    """Per-project performance report aggregating verdicts and observations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    project_name: str
    generated_at: datetime
    total_experiments: int
    keep_count: int
    revert_count: int
    error_count: int
    keep_rate: float
    latest_score: float | None = None
    agent_verdicts: list[AgentVerdict] = []
    observations: list[Observation] = []
    verdict_patterns: dict[str, int] = {}


class ProjectEntry(BaseModel):
    """A single project entry in the global registry."""

    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    name: str
    registered_at: datetime
    last_experiment_at: datetime | None = None
    experiment_count: int = 0
    latest_score: float | None = None


class ProjectRegistry(BaseModel):
    """Global project registry at ~/.factory/registry.json."""

    model_config = ConfigDict(strict=True, extra="forbid")

    projects: list[ProjectEntry] = []
    updated_at: datetime


# ── refinement state ─────────────────────────────────────────────


class RefinementEntry(BaseModel):
    """One refinement in a post-cycle refinement session."""

    model_config = ConfigDict(strict=True, extra="forbid")

    sequence: int
    request: str
    started_at: str
    completed_at: str | None = None
    verdict: str | None = None


class RefinementState(BaseModel):
    """Tracks refinements within a single post-cycle session."""

    model_config = ConfigDict(strict=True, extra="forbid")

    entries: list[RefinementEntry] = []


# ── runner v2 ────────────────────────────────────────────────────


class AgentRunRequest(BaseModel):
    """Structured input for a runner invocation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    prompt_core: str = ""
    task: str
    cwd: Path
    timeout: float = 600.0
    model: str | None = None
    skip_permissions: bool = True
    role: str = "unknown"
    session_name: str | None = None
    session_id: str | None = None
    resume_session_id: str | None = None
    project_path: Path | None = None
    extras: dict[str, object] = {}


class AgentRunResult(BaseModel):
    """Structured output from a runner invocation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    stdout: str
    return_code: int
    usage: AgentUsage | None = None
    metadata: dict[str, object] = {}
