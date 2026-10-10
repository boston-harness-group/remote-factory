"""Runner abstraction layer for CLI backends."""

from __future__ import annotations

from pathlib import Path

from factory.runners._stream import should_stream, stream_subprocess
from factory.runners.claude import ClaudeRunner
from factory.runners.codex import CodexRunner
from factory.runners.protocol import Runner, RunnerMeta

import structlog

log = structlog.get_logger()

__all__ = [
    "Runner",
    "RunnerMeta",
    "ClaudeRunner",
    "CodexRunner",
    "get_runner",
    "get_available_runners",
    "get_runner_choices",
    "should_stream",
    "stream_subprocess",
]

_RUNNERS: dict[str, type[Runner]] = {
    "claude": ClaudeRunner,  # type: ignore[dict-item]
    "codex": CodexRunner,  # type: ignore[dict-item]
}


def get_available_runners() -> dict[str, type[Runner]]:
    """Return all registered runners (built-in + entry-point plugins)."""
    _load_entrypoint_runners()
    return dict(_RUNNERS)


def get_runner(name: str | None = None, project_path: Path | None = None) -> Runner:
    """Get a runner by name.

    Resolution order:
    1. Explicit name argument
    2. FACTORY_RUNNER environment variable
    3. Default to "claude"
    """
    from factory.user_config import resolve

    _load_entrypoint_runners()

    resolved = (
        resolve("runner", cli_value=name, env_var="FACTORY_RUNNER", default="claude") or "claude"
    )
    resolved = resolved.lower().strip()

    if resolved not in _RUNNERS:
        available = ", ".join(_RUNNERS.keys())
        raise ValueError(f"Unknown runner '{resolved}'. Available: {available}")

    return _RUNNERS[resolved]()


def get_runner_choices() -> list[str]:
    """Return sorted list of all available runner names for CLI choices."""
    return sorted(get_available_runners().keys())


def get_all_runner_meta() -> list[RunnerMeta]:
    """Return RunnerMeta for all registered runners."""
    result = []
    for runner_cls in get_available_runners().values():
        try:
            result.append(runner_cls.metadata())  # type: ignore[union-attr]
        except (AttributeError, TypeError):
            pass
    return result


_entrypoints_loaded = False


def _load_entrypoint_runners() -> None:
    """Discover and load runners registered via entry_points."""
    global _entrypoints_loaded
    if _entrypoints_loaded:
        return
    _entrypoints_loaded = True

    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="factory.runners")
        for ep in eps:
            if ep.name not in _RUNNERS:
                try:
                    runner_class = ep.load()
                    _RUNNERS[ep.name] = runner_class
                except Exception:
                    log.debug("runner_plugin_load_failed", name=ep.name, exc_info=True)
    except Exception:
        pass
