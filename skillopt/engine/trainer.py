"""Trainer-level meta-skill integration hooks.

Provides epoch-boundary and epoch-start hooks for integrating optimizer-side
meta-skill memory into the ReflACT training loop. These functions are called
by the main trainer at specific lifecycle points.

Epoch lifecycle with meta-skill:

1. **Epoch start** — ``load_active_meta_skill()`` loads previous epoch's
   meta-skill content so it can be passed to reflect calls.
2. **Training steps** — ``meta_skill_context`` is passed to each
   ``adapter.reflect()`` call, which injects it into optimizer prompts.
3. **Epoch end** (after slow update) — ``generate_epoch_meta_skill()``
   produces updated meta-skill from adjacent-epoch comparison.
"""
from __future__ import annotations

import json
import logging
import os

from skillopt.optimizer.meta_skill import (
    load_meta_skill_content,
    run_meta_skill,
    should_generate_meta_skill,
)

log = logging.getLogger(__name__)


def load_active_meta_skill(
    out_root: str,
    epoch: int,
    *,
    use_meta_skill: bool = False,
) -> str:
    """Load the active meta-skill for the current epoch.

    Called at the start of each epoch. Returns the meta-skill content from
    the previous epoch, or empty string if disabled or unavailable.
    """
    if not use_meta_skill:
        return ""
    content = load_meta_skill_content(out_root, epoch - 1)
    if content:
        log.info(
            "meta_skill.loaded",
            epoch=epoch,
            source_epoch=epoch - 1,
            chars=len(content),
        )
    return content


def generate_epoch_meta_skill(
    out_root: str,
    epoch: int,
    prev_skill: str,
    curr_skill: str,
    comparison_pairs: list[dict],
    *,
    score_delta: float | None = None,
    chat_fn: object | None = None,
) -> dict | None:
    """Generate meta-skill at epoch boundary.

    Called after the slow update (if any) at the end of each epoch.
    Handles resume safety, first-epoch skip, and score-delta conditioning.

    Returns the meta-skill result dict, or None if skipped/failed.
    """
    meta_skill_dir = os.path.join(out_root, "meta_skill", f"epoch_{epoch:02d}")
    done_path = os.path.join(meta_skill_dir, "meta_skill_result.json")
    os.makedirs(meta_skill_dir, exist_ok=True)

    if os.path.exists(done_path):
        log.info("meta_skill.resume", epoch=epoch, status="already_done")
        with open(done_path) as f:
            return json.load(f)

    if epoch == 1:
        sentinel = {"action": "skip_first_epoch", "epoch": epoch}
        with open(done_path, "w") as f:
            json.dump(sentinel, f, indent=2, ensure_ascii=False)
        log.info("meta_skill.skip", epoch=epoch, reason="first_epoch")
        return sentinel

    if not should_generate_meta_skill(epoch, score_delta):
        sentinel = {
            "action": "skip_negative_delta",
            "epoch": epoch,
            "score_delta": score_delta,
        }
        with open(done_path, "w") as f:
            json.dump(sentinel, f, indent=2, ensure_ascii=False)
        log.info(
            "meta_skill.skip",
            epoch=epoch,
            reason="negative_delta",
            delta=score_delta,
        )
        return sentinel

    prev_meta_skill = load_meta_skill_content(out_root, epoch - 1)

    result = run_meta_skill(
        prev_skill=prev_skill,
        curr_skill=curr_skill,
        comparison_pairs=comparison_pairs,
        prev_meta_skill_content=prev_meta_skill,
        chat_fn=chat_fn,
    )

    if result and result.get("meta_skill_content"):
        result["action"] = "write_meta_skill"
        result["epoch"] = epoch
        with open(done_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(
            "meta_skill.generated",
            epoch=epoch,
            chars=len(result["meta_skill_content"]),
        )
        return result

    fallback = {
        "action": "generation_failed",
        "epoch": epoch,
    }
    with open(done_path, "w") as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)
    log.warning("meta_skill.failed", epoch=epoch)
    return None
