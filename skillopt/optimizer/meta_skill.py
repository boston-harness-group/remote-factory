"""Optimizer-side meta-skill memory for cross-epoch optimization guidance.

Maintains a compact optimizer-facing memory distilled from adjacent-epoch
skill comparisons. Does NOT modify the target skill document. Instead, it
produces guidance meant to improve future optimizer behavior when proposing,
merging, and ranking edits.
"""
from __future__ import annotations

import json
import logging
import os
import traceback

from skillopt.prompts import load_prompt

log = logging.getLogger(__name__)

MAX_META_SKILL_TOKENS = 3000
RECENCY_WINDOW = 3


def format_meta_skill_context(meta_skill_content: str) -> str:
    """Render optimizer memory into a prompt-ready context block.

    Returns an empty string when *meta_skill_content* is empty so that callers
    can unconditionally prepend the result without checking.
    """
    content = (meta_skill_content or "").strip()
    if not content:
        return ""
    content = _enforce_token_cap(content)
    return (
        "## Optimizer Meta Skill\n"
        "This is optimizer-side memory distilled from prior epoch transitions in "
        "this environment. Use it to improve how you propose, merge, and rank "
        "skill edits. Prefer it when the current evidence is ambiguous, but do "
        "not force it if the current trajectories clearly contradict it.\n\n"
        f"{content}"
    )


def run_meta_skill(
    prev_skill: str,
    curr_skill: str,
    comparison_pairs: list[dict],
    *,
    prev_meta_skill_content: str = "",
    system_prompt: str | None = None,
    chat_fn: object | None = None,
) -> dict | None:
    """Produce updated optimizer-side meta-skill from adjacent epochs.

    Parameters
    ----------
    prev_skill:
        The last-step skill from the previous epoch.
    curr_skill:
        The last-step skill from the current epoch.
    comparison_pairs:
        Longitudinal comparison pairs (same tasks, two skill versions).
    prev_meta_skill_content:
        Previous epoch's meta-skill content (empty for first generation).
    system_prompt:
        Override the default meta-skill system prompt.
    chat_fn:
        Callable ``(system, user, **kw) -> (response_text, metadata)``.
        When ``None`` the function returns ``None`` (dry-run / testing).
    """
    actual_system = system_prompt if system_prompt is not None else load_prompt("meta_skill")

    prev_meta_section = (
        prev_meta_skill_content.strip()
        if prev_meta_skill_content and prev_meta_skill_content.strip()
        else "(No previous optimizer meta skill — this is the first update.)"
    )

    comparison_text = _format_comparison_text(comparison_pairs)
    user = (
        f"## Previous Epoch Last-Step Skill\n{prev_skill}\n\n"
        f"## Current Epoch Last-Step Skill\n{curr_skill}\n\n"
        f"## Previous Optimizer Meta Skill\n"
        f"The following optimizer memory was available during the current epoch. "
        f"Reflect on whether it improved or harmed the quality of edits.\n\n"
        f"{prev_meta_section}\n\n"
        f"## Longitudinal Comparison (same tasks, two last-step skills)\n"
        f"{comparison_text}"
    )

    if chat_fn is None:
        return None

    try:
        response, _ = chat_fn(
            system=actual_system,
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="meta_skill",
        )
        result = _extract_json(response)
        if result and result.get("meta_skill_content"):
            content = str(result["meta_skill_content"]).strip()
            content = _enforce_token_cap(content)
            return {
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_skill_content": content,
            }
    except Exception:
        traceback.print_exc()

    return None


def load_meta_skill_content(out_root: str, epoch: int) -> str:
    """Load meta-skill content from a previous epoch's result file.

    Searches the last ``RECENCY_WINDOW`` epochs and returns the most recent
    content found. Returns empty string if nothing is found.
    """
    if epoch <= 0:
        return ""
    start_epoch = max(1, epoch - RECENCY_WINDOW + 1)
    for e in range(epoch, start_epoch - 1, -1):
        path = os.path.join(
            out_root, "meta_skill", f"epoch_{e:02d}", "meta_skill_result.json",
        )
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                result = json.load(f)
            content = str(result.get("meta_skill_content", "")).strip()
            if content:
                return content
        except Exception:
            continue
    return ""


def validate_deployment_gate(skill_content: str) -> list[str]:
    """Check final skill for meta-skill leakage markers.

    Returns a list of warnings (empty if clean).
    """
    markers = ["Optimizer Meta Skill", "optimizer memory", "meta_skill"]
    warnings: list[str] = []
    for marker in markers:
        if marker in skill_content:
            warnings.append(
                f"Deployment gate: meta-skill marker '{marker}' found in final SKILL.md"
            )
    return warnings


def should_generate_meta_skill(
    epoch: int,
    score_delta: float | None,
) -> bool:
    """Determine whether meta-skill generation should run for this epoch.

    Requires epoch >= 2 and a non-negative score delta.
    """
    if epoch < 2:
        return False
    if score_delta is not None and score_delta < 0:
        return False
    return True


def _enforce_token_cap(content: str) -> str:
    """Truncate content to approximate token budget."""
    approx_tokens = len(content) // 4
    if approx_tokens <= MAX_META_SKILL_TOKENS:
        return content
    char_limit = MAX_META_SKILL_TOKENS * 4
    return content[:char_limit]


def _format_comparison_text(pairs: list[dict]) -> str:
    """Format comparison pairs into human-readable text for the optimizer."""
    if not pairs:
        return "(No comparison data available.)"

    by_cat: dict[str, list[dict]] = {
        "regressed": [],
        "persistent_fail": [],
        "improved": [],
        "stable_success": [],
    }
    for p in pairs:
        by_cat.setdefault(p.get("category", ""), []).append(p)

    total = len(pairs)
    parts = [
        f"Total samples: {total}\n"
        f"- Improved (wrong->right): {len(by_cat['improved'])}\n"
        f"- Regressed (right->wrong): {len(by_cat['regressed'])}\n"
        f"- Persistent failures (wrong->wrong): {len(by_cat['persistent_fail'])}\n"
        f"- Stable successes (right->right): {len(by_cat['stable_success'])}\n"
    ]

    for cat_key, label in [
        ("regressed", "Regressions (right->wrong)"),
        ("persistent_fail", "Persistent Failures (wrong->wrong)"),
        ("improved", "Improvements (wrong->right)"),
        ("stable_success", "Stable Successes (right->right)"),
    ]:
        entries = by_cat[cat_key]
        if not entries:
            parts.append(f"### {label}\n(none)\n")
            continue
        lines = [f"### {label}"]
        for e in entries:
            task_id = e.get("id", "?")
            task_desc = e.get("task", "")
            lines.append(f"- Task {task_id}: {task_desc}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a text response."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if depth != 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
