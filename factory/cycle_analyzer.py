"""CycleAnalyzer — reads .factory/ artifacts and produces structured records for outer-loop optimizers.

Assembles what happened in each inner-loop cycle: what agents ran, in what order,
what each produced, what the evaluator said, and whether it helped. Mode-agnostic —
works with evolve, improve, research, refine, or any experiment-producing workflow.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from factory.workflow.primitives import AgentNode, Workflow


@dataclass
class AgentStep:
    """One agent invocation within a cycle."""

    order: int
    role: str
    started_at: str
    duration_s: float
    cost_usd: float | None
    output_tokens: int | None
    succeeded: bool
    error: str | None = None
    node_id: str | None = None
    produced: list[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    """One experiment (hypothesis → build → eval → verdict)."""

    exp_id: int
    hypothesis: str | None
    verdict: str
    score_before: float | None
    score_after: float | None
    score_delta: float | None
    cost_usd: float
    duration_s: float
    agents: list[AgentStep] = field(default_factory=list)
    eval_artifacts: list[str] = field(default_factory=list)


@dataclass
class NodeTrace:
    """Maps a DAG node to its runtime artifact and event."""

    node_id: str
    node_type: str
    role: str | None
    declared_writes: set[str]
    declared_reads: set[str]
    artifact_exists: bool = False
    event: dict | None = None


@dataclass
class CycleRecord:
    """What an outer-loop optimizer sees after one inner-loop cycle."""

    cycle_number: int
    mode: str | None
    started_at: str | None
    ended_at: str | None
    duration_s: float

    score_start: float | None
    score_end: float | None
    score_delta: float | None
    score_trajectory: list[float] = field(default_factory=list)

    experiments: list[ExperimentRecord] = field(default_factory=list)
    kept: int = 0
    reverted: int = 0
    errored: int = 0
    keep_rate: float = 0.0

    total_cost_usd: float = 0.0
    cost_by_agent: dict[str, float] = field(default_factory=dict)

    consecutive_reverts: int = 0
    plateau_detected: bool = False
    stuck_detected: bool = False

    steps: list[AgentStep] = field(default_factory=list)
    eval_artifacts: list[str] = field(default_factory=list)
    node_trace: dict[str, NodeTrace] = field(default_factory=dict)

    frozen_nodes: list[str] = field(default_factory=list)
    mutable_node_ids: list[str] = field(default_factory=list)

    instance_results: list[dict] | None = None
    eval_details: dict[str, object] | None = None
    split: str | None = None

    @classmethod
    def from_run(
        cls,
        item_results: list[dict],
        *,
        aggregate: str = "mean",
        mode: str | None = None,
        duration_s: float = 0.0,
        cycle_number: int = 1,
    ) -> CycleRecord:
        """Build a CycleRecord from item results with configured aggregation.

        Scoring rules:
        - errored items are excluded from all aggregation
        - failed items count with their score
        - all_pass = 1.0 only if every non-errored score >= 1.0
        - if most items errored, the candidate is errored (score_end=None)
        """
        import statistics

        non_errored = [
            r for r in item_results if r.get("status") != "errored"
        ]
        errored_count = len(item_results) - len(non_errored)

        # If most items errored, the candidate is errored
        if len(non_errored) == 0 or errored_count > len(non_errored):
            return cls(
                cycle_number=cycle_number,
                mode=mode,
                started_at=None,
                ended_at=None,
                duration_s=duration_s,
                score_start=None,
                score_end=None,
                score_delta=None,
                errored=errored_count,
                instance_results=item_results,
            )

        scores = [float(r.get("score", 0.0)) for r in non_errored]

        if aggregate == "mean":
            score = statistics.mean(scores)
        elif aggregate == "median":
            score = statistics.median(scores)
        elif aggregate == "max":
            score = max(scores)
        elif aggregate == "all_pass":
            score = 1.0 if all(s >= 1.0 for s in scores) else 0.0
        else:
            score = statistics.mean(scores)

        return cls(
            cycle_number=cycle_number,
            mode=mode,
            started_at=None,
            ended_at=None,
            duration_s=duration_s,
            score_start=None,
            score_end=score,
            score_delta=None,
            errored=errored_count,
            instance_results=item_results,
        )


class CycleAnalyzer:
    """Reads .factory/ artifacts and produces structured CycleRecords."""

    def __init__(
        self,
        factory_dir: Path,
        workflow: Workflow | None = None,
        event_offset: int = 0,
        tsv_offset: int = 0,
    ) -> None:
        self.factory_dir = Path(factory_dir)
        self.workflow = workflow
        self._event_offset = event_offset
        self._tsv_offset = tsv_offset

    # ── Main API ──

    def analyze(self) -> list[CycleRecord]:
        events = self._parse_events()
        experiments = self._extract_experiments(events)
        steps = self._extract_agent_steps(events)
        scores = self._extract_scores(events)
        mode = self._detect_mode(events)

        self._enrich_from_results_tsv(experiments)
        self._add_missing_experiments_from_tsv(experiments)
        self._discover_eval_artifacts(experiments)

        tsv_scores = self._extract_scores_from_tsv()
        if len(tsv_scores) > len(scores):
            scores = tsv_scores

        record = CycleRecord(
            cycle_number=1,
            mode=mode,
            started_at=events[0]["timestamp"] if events else None,
            ended_at=events[-1]["timestamp"] if events else None,
            duration_s=self._compute_duration(events),
            score_start=scores[0] if scores else None,
            score_end=scores[-1] if scores else None,
            score_delta=(scores[-1] - scores[0]) if len(scores) >= 2 else None,
            score_trajectory=scores,
            experiments=experiments,
            kept=sum(1 for e in experiments if e.verdict == "keep"),
            reverted=sum(1 for e in experiments if e.verdict == "revert"),
            errored=sum(1 for e in experiments if e.verdict == "error"),
            steps=steps,
            total_cost_usd=sum(s.cost_usd or 0 for s in steps),
            cost_by_agent=self._cost_by_agent(steps),
        )
        total = record.kept + record.reverted + record.errored
        record.keep_rate = record.kept / total if total > 0 else 0.0
        record.consecutive_reverts = self._count_trailing_reverts(experiments)
        record.eval_artifacts = [a for e in experiments for a in e.eval_artifacts]

        if self.workflow:
            record.node_trace = self._build_node_trace(steps)

        return [record]

    def latest(self) -> CycleRecord | None:
        records = self.analyze()
        return records[-1] if records else None

    def trajectory(self) -> list[float]:
        records = self.analyze()
        return records[0].score_trajectory if records else []

    # ── Tier 1: events.jsonl ──

    def _parse_events(self) -> list[dict]:
        events_path = self.factory_dir / "events.jsonl"
        if not events_path.exists():
            return []
        events = []
        for idx, line in enumerate(events_path.read_text().splitlines()):
            if idx < self._event_offset:
                continue
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    if isinstance(e, dict) and "type" in e and "timestamp" in e:
                        events.append(e)
                except (json.JSONDecodeError, TypeError):
                    continue
        return events

    def _extract_experiments(self, events: list[dict]) -> list[ExperimentRecord]:
        begins: dict[int, int] = {}
        experiments: list[ExperimentRecord] = []

        for i, e in enumerate(events):
            if e["type"] == "experiment.begin":
                exp_id = e["data"].get("exp_id")
                if exp_id is not None:
                    begins[exp_id] = i
            elif e["type"] == "experiment.finalize":
                exp_id = e["data"].get("exp_id")
                if exp_id is None:
                    continue
                verdict = e["data"].get("verdict", "error")
                hypothesis = e["data"].get("hypothesis")
                begin_idx = begins.get(exp_id)

                begin_ts = (
                    events[begin_idx]["timestamp"] if begin_idx is not None else e["timestamp"]
                )
                end_ts = e["timestamp"]
                duration = self._ts_diff(begin_ts, end_ts)

                agents_in_exp: list[AgentStep] = []
                cost = 0.0
                if begin_idx is not None:
                    for j in range(begin_idx, i + 1):
                        ev = events[j]
                        if ev["type"] == "agent.completed":
                            c = ev["data"].get("total_cost_usd", 0) or 0
                            cost += c

                experiments.append(
                    ExperimentRecord(
                        exp_id=exp_id,
                        hypothesis=hypothesis,
                        verdict=verdict,
                        score_before=None,
                        score_after=None,
                        score_delta=None,
                        cost_usd=cost,
                        duration_s=duration,
                        agents=agents_in_exp,
                    )
                )

        return experiments

    def _extract_agent_steps(self, events: list[dict]) -> list[AgentStep]:
        pending: dict[str, list[dict]] = {}
        steps: list[AgentStep] = []
        order = 0

        for e in events:
            if e["type"] == "agent.started":
                role = e.get("agent", "unknown")
                pending.setdefault(role, []).append(e)

            elif e["type"] == "agent.completed":
                role = e.get("agent", "unknown")
                start_event = pending.get(role, [None]).pop(0) if pending.get(role) else None
                data = e.get("data", {})
                started_at = start_event["timestamp"] if start_event else e["timestamp"]
                duration = self._ts_diff(started_at, e["timestamp"])

                step = AgentStep(
                    order=order,
                    role=role,
                    started_at=started_at,
                    duration_s=duration,
                    cost_usd=data.get("total_cost_usd"),
                    output_tokens=data.get("output_tokens"),
                    succeeded=True,
                )
                if self.workflow:
                    step.node_id = self._match_node(role)
                    if step.node_id:
                        node = self.workflow.nodes[step.node_id]
                        step.produced = sorted(node.writes)

                steps.append(step)
                order += 1

            elif e["type"] == "agent.failed":
                role = e.get("agent", "unknown")
                start_event = pending.get(role, [None]).pop(0) if pending.get(role) else None
                data = e.get("data", {})
                started_at = start_event["timestamp"] if start_event else e["timestamp"]

                steps.append(
                    AgentStep(
                        order=order,
                        role=role,
                        started_at=started_at,
                        duration_s=self._ts_diff(started_at, e["timestamp"]),
                        cost_usd=None,
                        output_tokens=None,
                        succeeded=False,
                        error=data.get("stderr", data.get("error", "unknown")),
                    )
                )
                order += 1

        return steps

    def _extract_scores(self, events: list[dict]) -> list[float]:
        scores = []
        for e in events:
            if e["type"] == "eval.completed":
                composite = e["data"].get("composite")
                if composite is not None:
                    scores.append(float(composite))
        return scores

    def _detect_mode(self, events: list[dict]) -> str | None:
        if self.workflow:
            return self.workflow.name
        return None

    def _compute_duration(self, events: list[dict]) -> float:
        if len(events) < 2:
            return 0.0
        return self._ts_diff(events[0]["timestamp"], events[-1]["timestamp"])

    # ── Tier 2: results.tsv ──

    def _enrich_from_results_tsv(self, experiments: list[ExperimentRecord]) -> None:
        tsv_path = self.factory_dir / "results.tsv"
        if not tsv_path.exists():
            return
        rows: dict[int, dict[str, str]] = {}
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for data_idx, row in enumerate(reader):
                if data_idx < self._tsv_offset:
                    continue
                try:
                    rows[int(row["id"])] = row
                except (KeyError, ValueError):
                    continue

        for exp in experiments:
            tsv_row = rows.get(exp.exp_id)
            if not tsv_row:
                continue
            if not exp.hypothesis and tsv_row.get("hypothesis"):
                exp.hypothesis = tsv_row["hypothesis"]
            if tsv_row.get("score_before"):
                try:
                    exp.score_before = float(tsv_row["score_before"])
                except ValueError:
                    pass
            if tsv_row.get("score_after"):
                try:
                    exp.score_after = float(tsv_row["score_after"])
                except ValueError:
                    pass
            if exp.score_before is not None and exp.score_after is not None:
                exp.score_delta = exp.score_after - exp.score_before
            if tsv_row.get("verdict"):
                exp.verdict = tsv_row["verdict"]

    def _add_missing_experiments_from_tsv(self, experiments: list[ExperimentRecord]) -> None:
        """Add experiments that exist in results.tsv but not in events.jsonl."""
        tsv_path = self.factory_dir / "results.tsv"
        if not tsv_path.exists():
            return
        known_ids = {e.exp_id for e in experiments}
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for data_idx, row in enumerate(reader):
                if data_idx < self._tsv_offset:
                    continue
                try:
                    exp_id = int(row["id"])
                except (KeyError, ValueError):
                    continue
                if exp_id in known_ids:
                    continue
                score_before = score_after = score_delta = None
                try:
                    if row.get("score_before"):
                        score_before = float(row["score_before"])
                    if row.get("score_after"):
                        score_after = float(row["score_after"])
                    if score_before is not None and score_after is not None:
                        score_delta = score_after - score_before
                except ValueError:
                    pass
                cost = 0.0
                try:
                    if row.get("cost_usd"):
                        cost = float(row["cost_usd"])
                except ValueError:
                    pass
                experiments.append(
                    ExperimentRecord(
                        exp_id=exp_id,
                        hypothesis=row.get("hypothesis"),
                        verdict=row.get("verdict", "error"),
                        score_before=score_before,
                        score_after=score_after,
                        score_delta=score_delta,
                        cost_usd=cost,
                        duration_s=0,
                    )
                )
        experiments.sort(key=lambda e: e.exp_id)

    def _extract_scores_from_tsv(self) -> list[float]:
        """Build score trajectory from results.tsv score_after values."""
        tsv_path = self.factory_dir / "results.tsv"
        if not tsv_path.exists():
            return []
        scores: list[float] = []
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for data_idx, row in enumerate(reader):
                if data_idx < self._tsv_offset:
                    continue
                try:
                    if row.get("score_after"):
                        scores.append(float(row["score_after"]))
                except ValueError:
                    continue
        return scores

    # ── Tier 3: eval artifact discovery ──

    def _discover_eval_artifacts(self, experiments: list[ExperimentRecord]) -> None:
        exp_dir = self.factory_dir / "experiments"
        if not exp_dir.exists():
            return
        for exp in experiments:
            for dir_name in [str(exp.exp_id), f"{exp.exp_id:03d}"]:
                d = exp_dir / dir_name
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if f.name.startswith("eval") or f.name == "candidate.py":
                        exp.eval_artifacts.append(str(f))

    # ── Tier 4: DAG node mapping ──

    def _build_node_trace(self, steps: list[AgentStep]) -> dict[str, NodeTrace]:
        if not self.workflow:
            return {}
        trace: dict[str, NodeTrace] = {}
        step_by_role: dict[str, AgentStep] = {}
        for s in steps:
            step_by_role[s.role] = s

        for nid, node in self.workflow.nodes.items():
            role = getattr(node, "role", None)
            role_str = role.value if role else None
            nt = NodeTrace(
                node_id=nid,
                node_type=type(node).__name__,
                role=role_str,
                declared_writes=set(node.writes),
                declared_reads=set(node.reads),
            )
            if node.writes:
                nt.artifact_exists = any(
                    (self.factory_dir / w.removeprefix(".factory/")).exists()
                    or (self.factory_dir.parent / w.removeprefix("./")).exists()
                    for w in node.writes
                )
            step = step_by_role.get(role_str) if role_str else None
            if step:
                nt.event = {
                    "role": step.role,
                    "duration_s": step.duration_s,
                    "cost_usd": step.cost_usd,
                    "succeeded": step.succeeded,
                }
            trace[nid] = nt
        return trace

    def _match_node(self, role: str) -> str | None:
        if not self.workflow:
            return None
        for nid, node in self.workflow.nodes.items():
            if isinstance(node, AgentNode) and node.role.value == role:
                return nid
        return None

    # ── Helpers ──

    @staticmethod
    def _cost_by_agent(steps: list[AgentStep]) -> dict[str, float]:
        costs: dict[str, float] = {}
        for s in steps:
            if s.cost_usd:
                costs[s.role] = costs.get(s.role, 0) + s.cost_usd
        return costs

    @staticmethod
    def _count_trailing_reverts(experiments: list[ExperimentRecord]) -> int:
        count = 0
        for exp in reversed(experiments):
            if exp.verdict == "revert":
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ts_diff(start: str, end: str) -> float:
        from datetime import datetime

        fmt_options = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]
        s = e = None
        for fmt in fmt_options:
            try:
                s = datetime.strptime(start, fmt)
                break
            except ValueError:
                continue
        for fmt in fmt_options:
            try:
                e = datetime.strptime(end, fmt)
                break
            except ValueError:
                continue
        if s and e:
            return (e - s).total_seconds()
        return 0.0
