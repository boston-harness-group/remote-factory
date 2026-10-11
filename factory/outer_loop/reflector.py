"""Contrastive reflection engine for outer loop evolution.

Analyzes CycleRecord exhaust from winners vs losers to identify structural
differences that explain performance gaps. Produces a ReflectionReport with
failure patterns, success patterns, and informed mutation suggestions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from factory.cycle_analyzer import CycleRecord
from factory.outer_loop.models import MutationType

log = structlog.get_logger()

_PER_CANDIDATE_CHAR_BUDGET = 30000
_LLM_PAYLOAD_BUDGET = 8000
_VALID_LLM_OPERATORS = {t.value for t in MutationType}


@dataclass
class MutationSuggestion:
    """Typed mutation suggestion from reflection analysis."""

    operator: str
    target: str
    rationale: str
    value: str | None = None


@dataclass
class ReflectionReport:
    """Output of contrastive reflection analysis."""

    failure_patterns: list[str] = field(default_factory=list)
    success_patterns: list[str] = field(default_factory=list)
    mutation_suggestions: list[str] = field(default_factory=list)
    prompt_improvements: list[str] = field(default_factory=list)
    structural_recommendations: list[str] = field(default_factory=list)
    top_k_ids: list[str] = field(default_factory=list)
    bottom_k_ids: list[str] = field(default_factory=list)
    typed_suggestions: list[MutationSuggestion] = field(default_factory=list)


def _filter_suggestions(
    suggestions: list[MutationSuggestion],
    valid_node_ids: set[str],
) -> list[MutationSuggestion]:
    """Drop mutation suggestions targeting nodes that don't exist in the workflow.

    Keeps suggestions whose target is:
    - A known node ID
    - A known role name (agent roles like "builder")
    - A generic target like "any"
    - A knob name (contains no dots, doesn't look like a node ID)
    """
    # Operators that target knob names or roles, not node IDs
    _NON_NODE_OPERATORS = {"knob_mutate", "node_insert"}

    result: list[MutationSuggestion] = []
    for s in suggestions:
        if s.operator in _NON_NODE_OPERATORS:
            result.append(s)
        elif s.target in valid_node_ids or s.target == "any":
            result.append(s)
        else:
            log.debug(
                "reflector_dropped_suggestion",
                operator=s.operator,
                target=s.target,
                reason="target node not in workflow",
            )
    return result


class OuterLoopReflector:
    """Two-stage contrastive reflection on CycleRecord exhaust.

    Stage 1: Partition individuals into top-K and bottom-K by fitness.
    Stage 2: Compare their CycleRecords to identify causal structural differences.
    """

    def __init__(
        self,
        k: int = 2,
        project_dir: Path | None = None,
        *,
        llm_reflect: bool = False,
    ) -> None:
        self._k = k
        self._project_dir = project_dir
        self._llm_reflect_enabled = llm_reflect

    def reflect(
        self,
        records: list[tuple[str, float, CycleRecord | None]],
        generation: int = 0,
        knob_values_by_id: dict[str, dict[str, object]] | None = None,
        workflow_data_by_id: dict[str, dict] | None = None,
    ) -> ReflectionReport:
        """Analyze a generation's results via contrastive reflection.

        Args:
            records: list of (individual_id, fitness, CycleRecord|None) triples
            generation: current generation number
            knob_values_by_id: optional mapping of individual_id to knob_values
                dict. When provided, enables knob-contrastive analysis that
                identifies which knob settings correlate with high/low scores.
            workflow_data_by_id: optional mapping of individual_id to workflow
                data dict. When provided, includes workflow summaries with full
                prompts in the reflection context and detects identical prompts.

        Returns:
            ReflectionReport with patterns and suggestions
        """
        # Firewall assertion: reject holdout CycleRecords from reflection
        for _, _, rec in records:
            if rec is not None and getattr(rec, "split", None) == "val":
                raise RuntimeError(
                    "Validation CycleRecord passed to reflector — this violates "
                    "the train/val firewall. Validation data must never "
                    "reach the reflector."
                )

        valid = [(id_, score, rec) for id_, score, rec in records if rec is not None]
        if len(valid) < 2:
            log.warning("reflection_insufficient_data", count=len(valid))
            return ReflectionReport()

        valid.sort(key=lambda x: x[1], reverse=True)

        k = min(self._k, len(valid) // 2)
        if k < 1:
            k = 1

        top_k = valid[:k]
        bottom_k = valid[-k:]

        report = ReflectionReport(
            top_k_ids=[id_ for id_, _, _ in top_k],
            bottom_k_ids=[id_ for id_, _, _ in bottom_k],
        )

        self._extract_failure_patterns(bottom_k, report)
        self._extract_success_patterns(top_k, report)
        self._extract_eval_patterns(top_k, bottom_k, report)
        self._generate_mutation_suggestions(top_k, bottom_k, report)
        self._generate_structural_recommendations(top_k, bottom_k, report)
        if knob_values_by_id:
            self._extract_knob_patterns(valid, top_k, bottom_k, knob_values_by_id, report)

        if self._llm_reflect_enabled:
            self._llm_reflect(top_k, bottom_k, records, report,
                              workflow_data_by_id=workflow_data_by_id)

        # Collect real node IDs from CycleRecords and filter suggestions
        # to prevent the reflector from targeting invented node names.
        all_node_ids: set[str] = set()
        for _, _, rec in valid:
            if rec is not None:
                all_node_ids.update(rec.node_trace.keys())
                all_node_ids.update(rec.mutable_node_ids)
        if all_node_ids:
            report.typed_suggestions = _filter_suggestions(
                report.typed_suggestions, all_node_ids,
            )

        if self._project_dir:
            self._save_report(report, generation)

        log.info(
            "reflection_complete",
            generation=generation,
            failures=len(report.failure_patterns),
            successes=len(report.success_patterns),
            suggestions=len(report.mutation_suggestions),
        )
        return report

    def _extract_failure_patterns(
        self,
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
    ) -> None:
        for id_, score, rec in bottom_k:
            if rec is None:
                continue
            for step in rec.steps:
                if not step.succeeded:
                    report.failure_patterns.append(
                        f"Agent {step.role} failed in individual {id_[:8]} "
                        f"(score={score:.3f}): {step.error or 'unknown error'}"
                    )
            if rec.errored and rec.errored > 0:
                report.failure_patterns.append(
                    f"Individual {id_[:8]} had {rec.errored} errored experiments"
                )
            if rec.reverted > rec.kept:
                report.failure_patterns.append(
                    f"Individual {id_[:8]} had more reverts ({rec.reverted}) than keeps ({rec.kept})"
                )

    def _extract_success_patterns(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
    ) -> None:
        for id_, score, rec in top_k:
            if rec is None:
                continue
            successful_roles = [s.role for s in rec.steps if s.succeeded]
            if successful_roles:
                report.success_patterns.append(
                    f"Individual {id_[:8]} (score={score:.3f}) succeeded with "
                    f"agents: {', '.join(successful_roles)}"
                )
            if rec.kept > 0:
                report.success_patterns.append(
                    f"Individual {id_[:8]} kept {rec.kept} experiments"
                )

    def _extract_eval_patterns(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
    ) -> None:
        """Extract patterns from EvalResult.details stored on CycleRecords."""
        top_details = [
            (id_, score, rec.eval_details)
            for id_, score, rec in top_k
            if rec is not None and isinstance(rec.eval_details, dict)
        ]
        bottom_details = [
            (id_, score, rec.eval_details)
            for id_, score, rec in bottom_k
            if rec is not None and isinstance(rec.eval_details, dict)
        ]

        if not top_details and not bottom_details:
            return

        # --- Verify patterns ---
        for id_, score, details in bottom_details:
            verify = details.get("verify")
            if not isinstance(verify, dict):
                continue
            failed = verify.get("failed_count", 0)
            total = verify.get("verify_count", 0)
            if isinstance(failed, (int, float)) and isinstance(total, (int, float)) and failed > 0:
                report.failure_patterns.append(
                    f"Individual {id_[:8]} (score={score:.3f}) failed "
                    f"{int(failed)}/{int(total)} verify checks"
                )
                instances = verify.get("instance_results")
                if isinstance(instances, list):
                    for inst in instances:
                        if isinstance(inst, dict) and not inst.get("passed", True):
                            inst_details = inst.get("details")
                            if isinstance(inst_details, dict):
                                rc = inst_details.get("returncode")
                                if rc is not None and rc != 0:
                                    report.failure_patterns.append(
                                        f"Individual {id_[:8]} verify instance "
                                        f"{inst.get('index', '?')} failed with "
                                        f"returncode={rc}"
                                    )

        for id_, score, details in top_details:
            verify = details.get("verify")
            if not isinstance(verify, dict):
                continue
            passed = verify.get("passed_count", 0)
            total = verify.get("verify_count", 0)
            if isinstance(passed, (int, float)) and isinstance(total, (int, float)) and passed > 0:
                report.success_patterns.append(
                    f"Individual {id_[:8]} (score={score:.3f}) passed "
                    f"{int(passed)}/{int(total)} verify checks"
                )

        # --- Verify score comparison between top-K and bottom-K ---
        top_verify_scores: list[float] = []
        bottom_verify_scores: list[float] = []
        for _, _, details in top_details:
            verify = details.get("verify")
            if isinstance(verify, dict):
                instances = verify.get("instance_results")
                if isinstance(instances, list):
                    for inst in instances:
                        if isinstance(inst, dict) and isinstance(inst.get("score"), (int, float)):
                            top_verify_scores.append(float(inst["score"]))
        for _, _, details in bottom_details:
            verify = details.get("verify")
            if isinstance(verify, dict):
                instances = verify.get("instance_results")
                if isinstance(instances, list):
                    for inst in instances:
                        if isinstance(inst, dict) and isinstance(inst.get("score"), (int, float)):
                            bottom_verify_scores.append(float(inst["score"]))

        if top_verify_scores and bottom_verify_scores:
            top_avg = sum(top_verify_scores) / len(top_verify_scores)
            bottom_avg = sum(bottom_verify_scores) / len(bottom_verify_scores)
            if abs(top_avg - bottom_avg) > 0.05:
                msg = (
                    f"Bottom-K scored {bottom_avg:.2f} avg on verify while "
                    f"top-K scored {top_avg:.2f} — focus mutations on "
                    f"improving test/verify pass rate"
                )
                report.mutation_suggestions.append(msg)
                report.typed_suggestions.append(MutationSuggestion(
                    operator="prompt_mutate", target="any",
                    rationale=msg,
                ))

        # --- Test details patterns ---
        for id_, score, details in bottom_details:
            test_details = details.get("test_details")
            if not isinstance(test_details, dict):
                continue
            rc = test_details.get("returncode")
            if rc is not None and rc != 0:
                report.failure_patterns.append(
                    f"Individual {id_[:8]} (score={score:.3f}) tests failed "
                    f"with returncode={rc}"
                )
            failed_tests = test_details.get("failed")
            if isinstance(failed_tests, (int, float)) and failed_tests > 0:
                total_tests = test_details.get("total", "?")
                report.failure_patterns.append(
                    f"Individual {id_[:8]} had {int(failed_tests)}/{total_tests} "
                    f"test failures"
                )

        # --- Rejection/error patterns ---
        for id_, score, details in bottom_details:
            rejected = details.get("rejected")
            if isinstance(rejected, str):
                report.failure_patterns.append(
                    f"Individual {id_[:8]} was rejected: {rejected}"
                )
            error = details.get("error")
            if isinstance(error, str):
                report.failure_patterns.append(
                    f"Individual {id_[:8]} evaluation error: {error[:120]}"
                )

    def _generate_mutation_suggestions(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
    ) -> None:
        top_roles: set[str] = set()
        bottom_roles: set[str] = set()

        for _, _, rec in top_k:
            if rec:
                top_roles |= {s.role for s in rec.steps if s.succeeded}
        for _, _, rec in bottom_k:
            if rec:
                bottom_roles |= {s.role for s in rec.steps if s.succeeded}

        roles_in_top_not_bottom = top_roles - bottom_roles
        for role in roles_in_top_not_bottom:
            msg = f"NODE_INSERT: Add {role} agent — present in winners but not losers"
            report.mutation_suggestions.append(msg)
            report.typed_suggestions.append(MutationSuggestion(
                operator="node_insert", target=role, rationale=msg,
            ))

        roles_in_bottom_not_top = bottom_roles - top_roles
        for role in roles_in_bottom_not_top:
            msg = f"NODE_REMOVE: Consider removing {role} — present in losers but not winners"
            report.mutation_suggestions.append(msg)
            report.typed_suggestions.append(MutationSuggestion(
                operator="node_remove", target=role, rationale=msg,
            ))

        top_avg_steps = 0.0
        bottom_avg_steps = 0.0
        top_count = sum(1 for _, _, r in top_k if r)
        bottom_count = sum(1 for _, _, r in bottom_k if r)

        if top_count:
            top_avg_steps = sum(len(r.steps) for _, _, r in top_k if r) / top_count
        if bottom_count:
            bottom_avg_steps = sum(len(r.steps) for _, _, r in bottom_k if r) / bottom_count

        if top_avg_steps > bottom_avg_steps + 1:
            msg = (
                f"NODE_INSERT: Winners use more agents ({top_avg_steps:.1f} avg) "
                f"vs losers ({bottom_avg_steps:.1f} avg) — consider adding nodes"
            )
            report.mutation_suggestions.append(msg)
            report.typed_suggestions.append(MutationSuggestion(
                operator="node_insert", target="any", rationale=msg,
            ))
        elif bottom_avg_steps > top_avg_steps + 1:
            msg = (
                f"NODE_REMOVE: Losers use more agents ({bottom_avg_steps:.1f} avg) "
                f"vs winners ({top_avg_steps:.1f} avg) — consider removing nodes"
            )
            report.mutation_suggestions.append(msg)
            report.typed_suggestions.append(MutationSuggestion(
                operator="node_remove", target="any", rationale=msg,
            ))

    def _generate_structural_recommendations(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
    ) -> None:
        for _, score, rec in bottom_k:
            if rec is None:
                continue
            timeout_failures = [s for s in rec.steps if not s.succeeded and s.duration_s > 500]
            if timeout_failures:
                roles = ', '.join(s.role for s in timeout_failures)
                msg = f"PARAM_MUTATE: Increase timeout for agents that timed out ({roles})"
                report.structural_recommendations.append(msg)
                for s in timeout_failures:
                    report.typed_suggestions.append(MutationSuggestion(
                        operator="param_mutate", target=s.role, rationale=msg,
                    ))

        for _, score, rec in top_k:
            if rec is None:
                continue
            if rec.node_trace:
                parallel_nodes = [
                    nid for nid, nt in rec.node_trace.items()
                    if nt.node_type == "ForkNode"
                ]
                if parallel_nodes:
                    msg = (
                        "PARALLELIZE: Winners use parallel execution — "
                        "consider parallelizing independent agents"
                    )
                    report.structural_recommendations.append(msg)
                    report.typed_suggestions.append(MutationSuggestion(
                        operator="parallelize", target="any", rationale=msg,
                    ))
                    break

    def _extract_knob_patterns(
        self,
        all_sorted: Sequence[tuple[str, float, CycleRecord | None]],
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        knob_values_by_id: dict[str, dict[str, object]],
        report: ReflectionReport,
    ) -> None:
        """Contrastive analysis of knob values between winners and losers.

        For each knob, computes the average score per value across all
        individuals. Reports knobs where the best value significantly
        outperforms the worst, giving the optimizer a per-knob gradient.
        """
        all_knob_names: set[str] = set()
        for kv in knob_values_by_id.values():
            all_knob_names.update(kv.keys())

        for knob in sorted(all_knob_names):
            is_prompt = knob.startswith("_prompt_") or knob.startswith("prompt_")
            val_scores: dict[str, list[float]] = {}
            for id_, score, _ in all_sorted:
                kv = knob_values_by_id.get(id_, {})
                val = str(kv.get(knob, ""))
                if not val:
                    continue
                if is_prompt:
                    val = val[:100]
                val_scores.setdefault(val, []).append(score)

            if len(val_scores) < 2:
                continue

            avg_by_val = {v: sum(s) / len(s) for v, s in val_scores.items() if s}
            if not avg_by_val:
                continue

            best_val = max(avg_by_val, key=avg_by_val.get)  # type: ignore[arg-type]
            worst_val = min(avg_by_val, key=avg_by_val.get)  # type: ignore[arg-type]
            gap = avg_by_val[best_val] - avg_by_val[worst_val]

            if gap > 0:
                op = "PROMPT_MUTATE" if is_prompt else "KNOB_MUTATE"
                typed_op = "prompt_mutate" if is_prompt else "knob_mutate"
                display_best = best_val[:60] + "..." if len(best_val) > 60 else best_val
                display_worst = worst_val[:60] + "..." if len(worst_val) > 60 else worst_val
                msg = (
                    f"{op}: {knob}={display_best} "
                    f"(avg score {avg_by_val[best_val]:+.0f}) "
                    f"outperforms {knob}={display_worst} "
                    f"({avg_by_val[worst_val]:+.0f}) by {gap:.0f}"
                )
                report.mutation_suggestions.append(msg)
                report.typed_suggestions.append(MutationSuggestion(
                    operator=typed_op,
                    target=knob,
                    rationale=msg,
                    value=best_val,
                ))
                if is_prompt:
                    report.prompt_improvements.append(
                        f"Reinforce approach from prompt knob {knob} "
                        f"with value '{display_best}' which correlates with higher scores"
                    )

        # Top-K vs bottom-K: which knobs differ consistently?
        top_ids = {id_ for id_, _, _ in top_k}
        bottom_ids = {id_ for id_, _, _ in bottom_k}
        for knob in sorted(all_knob_names):
            top_vals = {str(knob_values_by_id.get(id_, {}).get(knob, "")) for id_ in top_ids}
            bottom_vals = {str(knob_values_by_id.get(id_, {}).get(knob, "")) for id_ in bottom_ids}
            top_vals.discard("")
            bottom_vals.discard("")
            if top_vals and bottom_vals and not top_vals & bottom_vals:
                report.success_patterns.append(
                    f"Top performers use {knob}={','.join(top_vals)}; "
                    f"bottom use {knob}={','.join(bottom_vals)}"
                )

    @staticmethod
    def _find_shared_failures(
        records: Sequence[tuple[str, float, CycleRecord | None]],
    ) -> list[str]:
        """Find failures common to ALL candidates across all items.

        Scans verify_details for keys like missing_sections, missing_terms
        where the same value appears in every candidate's results for that item.
        Returns lines like:
          'All candidates miss sections "Errors" on items: readme-cli, tutorial'
          'All candidates miss terms "flag" on items: readme-cli'
        """
        from collections import defaultdict

        item_failures: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        candidate_ids: set[str] = set()

        for cand_id, _, rec in records:
            if rec is None or not rec.instance_results:
                continue
            candidate_ids.add(cand_id)
            for item in rec.instance_results:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("item_id", "")
                vd = item.get("verify_details", {})
                if not isinstance(vd, dict):
                    continue
                for key in ("missing_sections", "missing_terms"):
                    vals = vd.get(key, [])
                    if isinstance(vals, list):
                        for v in vals:
                            item_failures[item_id][key][str(v)].add(cand_id)

        if not candidate_ids:
            return []

        n_candidates = len(candidate_ids)

        # Group by failure type and value across items
        # {(key, value): [item_ids where ALL candidates have this failure]}
        shared: dict[tuple[str, str], list[str]] = defaultdict(list)
        for item_id, keys in sorted(item_failures.items()):
            for key, values in sorted(keys.items()):
                for value, cands in sorted(values.items()):
                    if len(cands) == n_candidates:
                        shared[(key, value)].append(item_id)

        lines: list[str] = []
        for (key, value), item_ids in sorted(shared.items()):
            label = key.replace("missing_", "")  # 'sections' or 'terms'
            items_str = ", ".join(item_ids)
            lines.append(
                f'All candidates miss {label} "{value}" on items: {items_str}'
            )

        return lines

    @staticmethod
    def _format_workflow_summary(workflow_data: dict) -> str:
        """Format workflow_data dict into a WORKFLOW section for LLM context.

        For AgentNode: show id, type, role, model, timeout, max_iterations,
        and full prompt_template (indented, never truncated).
        For other nodes: just id and type.
        """
        lines: list[str] = ["  WORKFLOW:"]
        nodes = workflow_data.get("nodes", {})
        for nid, node in nodes.items():
            ntype = node.get("_type", "Unknown")
            if ntype == "AgentNode":
                role = node.get("role", "")
                model = node.get("model", "")
                timeout = node.get("timeout")
                max_iter = node.get("max_iterations", 1)
                parts = [f"role={role}"]
                if model:
                    parts.append(f"model={model}")
                if timeout is not None:
                    parts.append(f"timeout={timeout}")
                if max_iter != 1:
                    parts.append(f"max_turns={max_iter}")
                lines.append(f"    node: {nid} ({ntype}, {', '.join(parts)})")
                prompt = node.get("prompt_template", "")
                if prompt:
                    lines.append("      prompt_template: |")
                    for pline in prompt.splitlines():
                        lines.append(f"        {pline}")
            else:
                lines.append(f"    node: {nid} ({ntype})")
        knob_values = workflow_data.get("knob_values", {})
        if knob_values:
            lines.append(f"    knob_values: {knob_values}")
        return "\n".join(lines)

    @staticmethod
    def _collect_individual_details(
        id_: str, score: float, rec: CycleRecord | None,
        *, char_budget: int | None = None,
        workflow_data: dict | None = None,
    ) -> str:
        """Collect eval/verify details from one individual for LLM context.

        Uses a multi-line block format per item.  Never truncates inside a
        block.  When total output exceeds *char_budget* (default
        ``_PER_CANDIDATE_CHAR_BUDGET``), whole items are dropped — passed
        items first — and a summary line is appended.
        """
        budget = char_budget if char_budget is not None else _PER_CANDIDATE_CHAR_BUDGET
        header = f"ID: {id_[:8]}, score: {score:.3f}"
        if rec is None:
            return header

        lines: list[str] = [header]

        # --- Workflow summary (before item results) ---
        if workflow_data:
            wf_summary = OuterLoopReflector._format_workflow_summary(workflow_data)
            lines.append(wf_summary)

        # --- Instance blocks (single source: rec.instance_results) ---
        if rec.instance_results and isinstance(rec.instance_results, list):
            blocks: list[tuple[bool, str]] = []  # (passed, block_text)
            for inst in rec.instance_results:
                if not isinstance(inst, dict):
                    continue
                item_id = inst.get("item_id", "?")
                split = inst.get("split", "?")
                status = inst.get("status", "?")
                inst_score = inst.get("score", "?")
                passed = inst.get("passed", False)
                block_lines = [
                    f"  instance: item_id={item_id}, split={split}, "
                    f"status={status}, score={inst_score}, passed={passed}",
                ]
                error = inst.get("error")
                if error:
                    block_lines.append(f"    error: {error}")
                vd = inst.get("verify_details")
                if isinstance(vd, dict) and vd:
                    block_lines.append("    verify_details:")
                    for k, v in vd.items():
                        block_lines.append(f"      {k}: {v}")
                blocks.append((bool(passed), "\n".join(block_lines)))

            # Budget check: drop whole items (passed first) if over budget
            total_len = sum(len(header) + 1 + len(b) for _, b in blocks)
            if total_len > budget:
                # Sort: passed items first so they get dropped first
                indexed = list(enumerate(blocks))
                indexed.sort(key=lambda x: (not x[1][0], x[0]))
                kept_indices: set[int] = set()
                running = len(header) + 1
                # Reserve space for omission message
                omit_reserve = 60
                for orig_idx, (passed, text) in indexed:
                    if running + len(text) + 1 + omit_reserve <= budget:
                        kept_indices.add(orig_idx)
                        running += len(text) + 1
                omitted = len(blocks) - len(kept_indices)
                for orig_idx, (_, text) in enumerate(blocks):
                    if orig_idx in kept_indices:
                        lines.append(text)
                if omitted > 0:
                    lines.append(
                        f"  ({omitted} items omitted"
                        " — see runs/<run>/items/)"
                    )
            else:
                for _, text in blocks:
                    lines.append(text)

        # --- eval_details (legacy / extra keys) ---
        if rec.eval_details and isinstance(rec.eval_details, dict):
            for k, v in rec.eval_details.items():
                if k == 'verify':
                    # Render non-instance_results keys from verify dict
                    if isinstance(v, dict):
                        for vk, vv in v.items():
                            if vk == 'instance_results':
                                continue  # Already rendered above
                            lines.append(f"{vk}: {str(vv)[:200]}")
                    continue
                lines.append(f"{k}: {str(v)[:200]}")

        # --- Steps ---
        if rec.steps:
            roles = [
                f"{s.role}({'ok' if s.succeeded else 'FAIL'})"
                for s in rec.steps[:5]
            ]
            lines.append("agents: " + ", ".join(roles))

        # --- Experiments ---
        if rec.experiments:
            for exp in rec.experiments[:5]:
                line = f"exp{exp.exp_id}({exp.verdict}"
                if exp.score_delta is not None:
                    line += f" Δ={exp.score_delta:+.3f}"
                line += ")"
                if exp.hypothesis:
                    line += f" {exp.hypothesis[:200]}"
                candidate = "\n".join(lines + [line])
                if len(candidate) > budget:
                    break
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _collect_node_ids(
        records: Sequence[tuple[str, float, CycleRecord | None]],
    ) -> list[dict[str, str | None]]:
        """Collect unique node IDs and roles from all CycleRecords."""
        seen: dict[str, str | None] = {}
        for _, _, rec in records:
            if rec is None:
                continue
            for nid, nt in rec.node_trace.items():
                if nid not in seen:
                    seen[nid] = nt.role
            for step in rec.steps:
                if step.role and step.role not in seen:
                    seen[step.role] = step.role
        return [{"node_id": nid, "role": role} for nid, role in seen.items()]

    @staticmethod
    def _compare_prompts_across_candidates(
        workflow_data_by_id: dict[str, dict],
        all_ids: list[str],
    ) -> list[str]:
        """Compare prompt_templates across candidates.

        Returns lines describing candidates with identical prompts and
        what differs between them (params, knobs, timeout, etc.).
        """
        # Build a fingerprint → list of (id, node_params) mapping
        # Fingerprint = hash of all agent prompt_templates sorted by node id
        fingerprints: dict[str, list[tuple[str, dict]]] = {}
        for cid in all_ids:
            wf = workflow_data_by_id.get(cid, {})
            nodes = wf.get("nodes", {})
            agent_prompts: list[tuple[str, str]] = []
            for nid in sorted(nodes):
                node = nodes[nid]
                if node.get("_type") == "AgentNode":
                    agent_prompts.append((nid, node.get("prompt_template", "")))
            if not agent_prompts:
                continue
            fp = hashlib.sha256(
                json.dumps(agent_prompts, sort_keys=True).encode()
            ).hexdigest()[:16]
            fingerprints.setdefault(fp, []).append((cid, wf))

        result_lines: list[str] = []
        for _fp, group in fingerprints.items():
            if len(group) < 2:
                continue
            ids = [cid[:8] for cid, _ in group]
            # Find differences in non-prompt params
            diffs: set[str] = set()
            first_wf = group[0][1]
            first_nodes = first_wf.get("nodes", {})
            for _, other_wf in group[1:]:
                other_nodes = other_wf.get("nodes", {})
                for nid in set(first_nodes) | set(other_nodes):
                    fn = first_nodes.get(nid, {})
                    on = other_nodes.get(nid, {})
                    if fn.get("_type") == "AgentNode" or on.get("_type") == "AgentNode":
                        for key in ("model", "timeout", "max_iterations"):
                            if fn.get(key) != on.get(key):
                                diffs.add(key)
                first_knobs = first_wf.get("knob_values", {})
                other_knobs = other_wf.get("knob_values", {})
                if first_knobs != other_knobs:
                    for k in set(first_knobs) | set(other_knobs):
                        if first_knobs.get(k) != other_knobs.get(k):
                            diffs.add(f"knob:{k}")
            diff_desc = ", ".join(sorted(diffs)) if diffs else "run-to-run variance only"
            result_lines.append(
                f"IDENTICAL PROMPTS: Candidates {' and '.join(ids)} have identical "
                f"prompts; their score difference comes from {diff_desc}."
            )
        return result_lines

    def build_reflection_prompt(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        records: Sequence[tuple[str, float, CycleRecord | None]],
        workflow_data_by_id: dict[str, dict] | None = None,
    ) -> str:
        """Build the LLM reflection prompt without calling the LLM.

        Public so tests can inspect the prompt text directly.
        """
        per_individual = _PER_CANDIDATE_CHAR_BUDGET

        wf_by_id = workflow_data_by_id or {}

        top_details = []
        for id_, score, rec in top_k:
            top_details.append(
                self._collect_individual_details(
                    id_, score, rec, char_budget=per_individual,
                    workflow_data=wf_by_id.get(id_),
                )
            )
        bottom_details = []
        for id_, score, rec in bottom_k:
            bottom_details.append(
                self._collect_individual_details(
                    id_, score, rec, char_budget=per_individual,
                    workflow_data=wf_by_id.get(id_),
                )
            )

        node_info = self._collect_node_ids(list(top_k) + list(bottom_k))
        node_section = ""
        if node_info:
            node_lines = [f"  {n['node_id']} (role={n['role']})" for n in node_info]
            node_section = (
                "\n\nAVAILABLE WORKFLOW NODES:\n"
                + "\n".join(node_lines)
            )

        payload = (
            "TOP-PERFORMING CANDIDATES:\n"
            + "\n".join(f"  {d}" for d in top_details)
            + "\n\nBOTTOM-PERFORMING CANDIDATES:\n"
            + "\n".join(f"  {d}" for d in bottom_details)
            + node_section
        )

        # Compare prompts across candidates and append identical-prompt lines
        if wf_by_id:
            all_ids = [id_ for id_, _, _ in top_k] + [id_ for id_, _, _ in bottom_k]
            identical_lines = self._compare_prompts_across_candidates(wf_by_id, all_ids)
            if identical_lines:
                payload += "\n\n" + "\n".join(identical_lines)

        # Shared failures across all candidates
        all_records = list(top_k) + list(bottom_k)
        shared_failures = self._find_shared_failures(all_records)
        if shared_failures:
            payload += "\n\nFAILURES COMMON TO ALL CANDIDATES:\n"
            payload += "\n".join(f"  - {f}" for f in shared_failures)

        operators_list = ", ".join(sorted(_VALID_LLM_OPERATORS))
        return (
            "Analyze these verification results from an evolutionary search. "
            "Here are the details from the top-performing candidates and the "
            "bottom-performing candidates.\n\n"
            "Before attributing a score difference to a prompt change, check "
            "whether the prompts actually differ. Candidates with identical "
            "prompts have their score difference explained by parameter changes "
            "or run-to-run variance, not prompt quality.\n\n"
            f"{payload}\n\n"
            "Identify what distinguishes successful from unsuccessful candidates. "
            "Produce concrete improvement advice — specific changes to agent "
            "prompts, parameter choices, or strategies that would move bottom "
            "candidates toward top candidate behavior.\n\n"
            "Also list failures common to ALL candidates — these are systemic gaps "
            "that no candidate has solved yet, and should be the highest priority "
            "for prompt improvements.\n\n"
            "Output a JSON object with three fields:\n"
            '  "prompt_improvements": list of concrete advice strings\n'
            '  "failure_patterns": list of identified failure mode strings\n'
            '  "mutation_suggestions": list of objects, each with:\n'
            '    "operator": one of [' + operators_list + ']\n'
            '    "target": the node_id or knob name to mutate\n'
            '    "rationale": why this mutation would help\n'
            '    "value": (optional) suggested new value for knob_mutate\n'
            "  Operator guide:\n"
            "    prompt_mutate — change an agent's prompt (target = node_id)\n"
            "    knob_mutate — change a tunable parameter (target = knob name, include value)\n"
            "    param_mutate — change agent params like timeout/model (target = node_id)\n"
            "    node_insert — add a new agent node (target = role name)\n"
            "    node_remove — remove an agent node (target = node_id or role)\n"
            "  For each improvement you identify, also specify which mutation "
            "operator and target would implement it. Be specific — name the "
            "actual node or knob to change.\n\n"
            "Output ONLY the JSON object."
        )

    def _llm_reflect(
        self,
        top_k: Sequence[tuple[str, float, CycleRecord | None]],
        bottom_k: Sequence[tuple[str, float, CycleRecord | None]],
        records: list[tuple[str, float, CycleRecord | None]],
        report: ReflectionReport,
        workflow_data_by_id: dict[str, dict] | None = None,
    ) -> None:
        """LLM-based contrastive reflection: one call per generation."""
        prompt = self.build_reflection_prompt(
            top_k, bottom_k, records,
            workflow_data_by_id=workflow_data_by_id,
        )

        from factory.runners.claude import _claude_bin, _claude_model, _cli_error_text

        try:
            cmd = [_claude_bin(), "--model", _claude_model(),
                   "--append-system-prompt", "Output only valid JSON.",
                   "--output-format", "text"]
            data: dict[str, object] | None = None
            for attempt in range(3):
                proc = subprocess.run(
                    cmd, input=prompt,
                    capture_output=True, text=True, timeout=120,
                )
                raw = proc.stdout.strip()
                if _cli_error_text(raw):
                    # CLI-level auth/model failure (e.g. a 403 on a
                    # single-model gateway): retrying the same command
                    # won't help — bail out immediately. Transient errors
                    # (429/529, network blips) exit nonzero with empty or
                    # non-JSON output and fall through to the retry below.
                    log.warning(
                        "llm_reflect_cli_error",
                        returncode=proc.returncode, head=raw[:120],
                    )
                    return
                if not raw or "{" not in raw:
                    if attempt < 2:
                        log.info("llm_reflect_retry", attempt=attempt + 1, raw_head=raw[:80] if raw else "EMPTY")
                        time.sleep(2 ** attempt)
                        continue
                    log.info("llm_reflect_empty_response", attempts=3)
                    return
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    if attempt < 2:
                        log.info("llm_reflect_retry", attempt=attempt + 1, error="json_decode")
                        time.sleep(2 ** attempt)
                        continue
                    raise
                break
            if not isinstance(data, dict):
                log.warning("llm_reflect_non_dict_json", type=type(data).__name__)
                return
            improvements = data.get("prompt_improvements", [])
            if isinstance(improvements, list):
                for item in improvements:
                    if isinstance(item, str) and item.strip():
                        report.prompt_improvements.append(item.strip())
            failures = data.get("failure_patterns", [])
            if isinstance(failures, list):
                for item in failures:
                    if isinstance(item, str) and item.strip():
                        report.failure_patterns.append(item.strip())
            mut_suggestions = data.get("mutation_suggestions", [])
            if isinstance(mut_suggestions, list):
                for item in mut_suggestions:
                    if not isinstance(item, dict):
                        continue
                    op = item.get("operator")
                    target = item.get("target")
                    rationale = item.get("rationale")
                    if not isinstance(op, str) or not isinstance(target, str):
                        continue
                    if op not in _VALID_LLM_OPERATORS:
                        log.debug("llm_reflect_invalid_operator", operator=op)
                        continue
                    if not isinstance(rationale, str):
                        rationale = ""
                    raw_value = item.get("value")
                    value = str(raw_value) if raw_value is not None else None
                    report.typed_suggestions.append(MutationSuggestion(
                        operator=op,
                        target=target,
                        rationale=rationale,
                        value=value,
                    ))
            log.info(
                "llm_reflect_complete",
                prompt_improvements=len(report.prompt_improvements),
                failure_patterns_added=len(failures) if isinstance(failures, list) else 0,
                llm_typed_suggestions=len(report.typed_suggestions),
            )
        except subprocess.TimeoutExpired:
            log.warning("llm_reflect_timeout")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            log.warning("llm_reflect_error", error=str(exc))
        except FileNotFoundError:
            log.warning("llm_reflect_claude_not_found")

    def _save_report(self, report: ReflectionReport, generation: int) -> None:
        if not self._project_dir:
            return
        reflect_dir = self._project_dir / ".factory" / "outer_loop" / "reflections"
        reflect_dir.mkdir(parents=True, exist_ok=True)

        report_data = {
            "generation": generation,
            "failure_patterns": report.failure_patterns,
            "success_patterns": report.success_patterns,
            "mutation_suggestions": report.mutation_suggestions,
            "prompt_improvements": report.prompt_improvements,
            "structural_recommendations": report.structural_recommendations,
            "top_k_ids": report.top_k_ids,
            "bottom_k_ids": report.bottom_k_ids,
            "typed_suggestions": [
                {
                    "operator": ts.operator,
                    "target": ts.target,
                    "rationale": ts.rationale,
                    "value": ts.value,
                }
                for ts in report.typed_suggestions
            ],
        }
        path = reflect_dir / f"gen{generation}.json"
        path.write_text(json.dumps(report_data, indent=2))

        md_path = reflect_dir / f"gen{generation}.md"
        lines = [f"# Reflection — Generation {generation}\n"]
        if report.failure_patterns:
            lines.append("## Failure Patterns")
            for p in report.failure_patterns:
                lines.append(f"- {p}")
            lines.append("")
        if report.success_patterns:
            lines.append("## Success Patterns")
            for p in report.success_patterns:
                lines.append(f"- {p}")
            lines.append("")
        if report.mutation_suggestions:
            lines.append("## Mutation Suggestions")
            for s in report.mutation_suggestions:
                lines.append(f"- {s}")
            lines.append("")
        if report.structural_recommendations:
            lines.append("## Structural Recommendations")
            for r in report.structural_recommendations:
                lines.append(f"- {r}")
            lines.append("")
        if report.prompt_improvements:
            lines.append("## Prompt Improvements")
            for p in report.prompt_improvements:
                lines.append(f"- {p}")
            lines.append("")
        if report.typed_suggestions:
            lines.append("## Typed Suggestions")
            for ts in report.typed_suggestions:
                val_part = f" value={ts.value}" if ts.value else ""
                lines.append(f"- [{ts.operator}] {ts.target}{val_part}: {ts.rationale}")
        md_path.write_text("\n".join(lines) + "\n")
