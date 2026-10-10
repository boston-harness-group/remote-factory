"""Prompt loading utilities for SkillOpt.

Prompts are stored as ``.md`` files in this directory and loaded at runtime.
"""
from __future__ import annotations

import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template by name from the prompts directory.

    Returns the contents of ``skillopt/prompts/{name}.md``.
    Raises ``FileNotFoundError`` if the file does not exist.
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if path in _cache:
        return _cache[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _cache[path] = content
    return content


def clear_cache() -> None:
    """Clear the prompt file cache (useful for testing)."""
    _cache.clear()
