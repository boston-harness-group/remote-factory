"""Reflect-stage meta-skill injection.

Provides reflect wrapper functions that accept ``meta_skill_context`` and
prepend it to optimizer prompts in-memory. The on-disk analyst prompt
templates (analyst_error.md, analyst_success.md) are NEVER modified.

The meta-skill context is prepended to the user message before trajectories,
so the optimizer sees accumulated cross-epoch guidance before analyzing the
current batch.
"""
from __future__ import annotations

import logging

from skillopt.optimizer.meta_skill import format_meta_skill_context

log = logging.getLogger(__name__)


def reflect_on_errors(
    skill_content: str,
    failed_items: list[dict],
    prediction_dir: str,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    edit_budget: int = 4,
    chat_fn: object | None = None,
) -> dict | None:
    """Analyze failed trajectories with optional meta-skill context.

    Wraps the error analyst call, prepending meta-skill guidance to the
    user prompt in-memory. Does NOT modify any prompt files on disk.

    Parameters
    ----------
    skill_content:
        Current skill document text.
    failed_items:
        Rollout result dicts for failed trajectories.
    prediction_dir:
        Path to predictions directory with conversation files.
    system_prompt:
        Custom system prompt override.
    step_buffer_context:
        Summary of previous steps in this epoch.
    meta_skill_context:
        Raw meta-skill content to prepend to the optimizer prompt.
    edit_budget:
        Maximum number of edits to propose.
    chat_fn:
        Callable ``(system, user, **kw) -> (response_text, metadata)``.
    """
    if not failed_items:
        return None

    user = f"## Current Skill\n{skill_content}\n\n"
    user += f"## Edit Budget\nProduce at most L={edit_budget} edits.\n\n"

    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"

    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"

    user += f"## Failed Trajectories ({len(failed_items)} total)\n"
    user += _format_items_summary(failed_items)

    if chat_fn is None:
        return None

    try:
        response, _ = chat_fn(
            system=system_prompt or "",
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="analyst",
        )
        from skillopt.optimizer.meta_skill import _extract_json
        result = _extract_json(response)
        if result:
            result["source_type"] = "failure"
            return result
    except Exception:
        log.exception("reflect_on_errors failed")

    return None


def reflect_on_successes(
    skill_content: str,
    success_items: list[dict],
    prediction_dir: str,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    edit_budget: int = 4,
    chat_fn: object | None = None,
) -> dict | None:
    """Analyze successful trajectories with optional meta-skill context.

    Same pattern as ``reflect_on_errors`` but for success trajectories.
    Meta-skill context is prepended in-memory only.
    """
    if not success_items:
        return None

    user = f"## Current Skill\n{skill_content}\n\n"
    user += f"## Edit Budget\nProduce at most L={edit_budget} edits.\n\n"

    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"

    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"

    user += f"## Successful Trajectories ({len(success_items)} total)\n"
    user += _format_items_summary(success_items)

    if chat_fn is None:
        return None

    try:
        response, _ = chat_fn(
            system=system_prompt or "",
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="analyst",
        )
        from skillopt.optimizer.meta_skill import _extract_json
        result = _extract_json(response)
        if result:
            result["source_type"] = "success"
            return result
    except Exception:
        log.exception("reflect_on_successes failed")

    return None


def reflect_and_merge(
    skill_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    *,
    meta_skill_context: str = "",
    chat_fn: object | None = None,
) -> dict | None:
    """Merge failure and success patches with meta-skill context.

    Meta-skill context is prepended to the merge prompt in-memory only.
    """
    user = f"## Current Skill\n{skill_content}\n\n"

    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"

    user += f"## Failure Patches ({len(failure_patches)} total)\n"
    for i, patch in enumerate(failure_patches, 1):
        user += f"### Patch {i}\n{_format_patch(patch)}\n\n"

    user += f"## Success Patches ({len(success_patches)} total)\n"
    for i, patch in enumerate(success_patches, 1):
        user += f"### Patch {i}\n{_format_patch(patch)}\n\n"

    if chat_fn is None:
        return None

    try:
        response, _ = chat_fn(
            system="",
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="merge",
        )
        from skillopt.optimizer.meta_skill import _extract_json
        return _extract_json(response)
    except Exception:
        log.exception("reflect_and_merge failed")

    return None


def _format_items_summary(items: list[dict]) -> str:
    """Format rollout items into a compact summary."""
    parts: list[str] = []
    for item in items:
        task_id = item.get("id", "?")
        task_desc = item.get("task_description", item.get("instruction", ""))
        fail_reason = item.get("fail_reason", "")
        line = f"- Task {task_id}: {task_desc}"
        if fail_reason:
            line += f" (reason: {fail_reason})"
        parts.append(line)
    return "\n".join(parts)


def _format_patch(patch: dict) -> str:
    """Format a single patch dict into readable text."""
    import json
    return json.dumps(patch, indent=2, ensure_ascii=False)
