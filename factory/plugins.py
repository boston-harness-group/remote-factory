"""Plugin system — discover and load pip-installable factory extensions via entry points."""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import structlog

log = structlog.get_logger()

ENTRY_POINT_GROUP = "factory.plugins"

BUILTIN_COMMANDS: frozenset[str] = frozenset({
    "ace", "ace-stats", "adversarial-state", "agent", "archive",
    "backfill-archive", "backfill-citations", "backlog-add", "backlog-list",
    "backlog-remove", "baseline", "begin", "ceo", "checkpoint", "clean-pr",
    "config", "contained", "dashboard", "deferred-list", "deferred-remove",
    "detect", "diff", "digest", "discover", "emit", "eval", "explain",
    "export", "finalize", "graph", "guard", "history", "home", "init",
    "insights", "install", "leakage-check", "log", "mempalace", "message",
    "notify", "plugins", "precheck", "profile", "refactory", "refine-begin",
    "refine-complete", "refine-status", "registry-list", "report-update",
    "research", "resume", "review", "run", "runners", "self-update",
    "serve-mcp", "spec", "status", "study", "summary", "tmux",
    "tmux-capture", "tmux-ls", "tmux-stop", "usage", "validate-research",
    "vault-init", "workflow",
})


@dataclass
class CommandSpec:
    handler: Callable[..., int]
    help: str
    add_arguments: Callable[..., None] | None = None


@dataclass
class PluginLoadResult:
    name: str
    status: Literal["loaded", "skipped", "failed"]
    reason: str | None = None
    version: str | None = None


@dataclass
class PluginRegistry:
    commands: dict[str, CommandSpec] = field(default_factory=dict)
    modes: list[str] = field(default_factory=list)
    agent_roles: list[str] = field(default_factory=list)
    ceo_pre_hooks: list[Callable[..., Any]] = field(default_factory=list)
    workflow_search_paths: list[str] = field(default_factory=list)
    parser_extensions: dict[str, list[Callable[[argparse.ArgumentParser], None]]] = field(
        default_factory=dict
    )

    def add_commands(self, commands: dict[str, CommandSpec]) -> None:
        for name, spec in commands.items():
            if name in BUILTIN_COMMANDS:
                log.warning("plugin_command_collision_builtin", command=name, action="skipped")
                continue
            if name in self.commands:
                log.warning("plugin_command_collision", command=name, action="keeping_first")
                continue
            self.commands[name] = spec

    def add_modes(self, modes: list[str]) -> None:
        from factory.cli._helpers import CEO_MODES

        for mode in modes:
            if mode in CEO_MODES:
                log.warning("plugin_mode_collision_builtin", mode=mode, action="skipped")
                continue
            if mode in self.modes:
                log.warning("plugin_mode_collision", mode=mode, action="keeping_first")
                continue
            self.modes.append(mode)

    def add_agent_roles(self, roles: list[str]) -> None:
        from factory.cli._parser_groups import BUILTIN_AGENT_ROLES
        from factory.workflow.primitives import register_agent_role

        for role in roles:
            if role in BUILTIN_AGENT_ROLES:
                log.warning("plugin_agent_role_collision_builtin", role=role, action="skipped")
                continue
            if role in self.agent_roles:
                log.warning("plugin_agent_role_collision", role=role, action="keeping_first")
                continue
            try:
                register_agent_role(role)
            except ValueError as exc:
                log.warning(
                    "plugin_agent_role_registration_failed", role=role, error=str(exc)
                )
                continue
            self.agent_roles.append(role)

    def add_ceo_pre_hook(self, hook: Callable[..., Any]) -> None:
        self.ceo_pre_hooks.append(hook)

    def add_parser_extensions(
        self, extensions: dict[str, Callable[[argparse.ArgumentParser], None]]
    ) -> None:
        for name, func in extensions.items():
            self.parser_extensions.setdefault(name, []).append(func)

    def add_workflow_search_path(self, path: str) -> None:
        self.workflow_search_paths.append(path)


_registry: PluginRegistry | None = None
_results: list[PluginLoadResult] | None = None


def load_plugins(registry: PluginRegistry | None = None) -> list[PluginLoadResult]:
    """Discover and load plugins from the ``factory.plugins`` entry point group.

    Uses three-tier error isolation: discovery → load → validation.
    Sorted by distribution name for deterministic order.
    """
    global _registry, _results

    if registry is None:
        registry = PluginRegistry()

    eps = importlib.metadata.entry_points()
    group_eps = eps.get(ENTRY_POINT_GROUP, []) if isinstance(eps, dict) else eps.select(group=ENTRY_POINT_GROUP)
    sorted_eps = sorted(group_eps, key=lambda ep: (ep.dist.name if ep.dist else ep.name))

    results: list[PluginLoadResult] = []

    for ep in sorted_eps:
        dist_name = ep.dist.name if ep.dist else ep.name
        dist_version = ep.dist.version if ep.dist else None

        # Tier 1: Load the entry point
        try:
            factory_plugin = ep.load()
        except Exception as exc:
            log.warning("plugin_import_failed", plugin=dist_name, error=str(exc))
            results.append(PluginLoadResult(
                name=dist_name, status="failed",
                reason=f"Import error: {exc}", version=dist_version,
            ))
            continue

        # Tier 2: Validate it's callable
        if not callable(factory_plugin):
            log.warning("plugin_not_callable", plugin=dist_name)
            results.append(PluginLoadResult(
                name=dist_name, status="failed",
                reason="Entry point is not callable", version=dist_version,
            ))
            continue

        # Tier 3: Call the registration function
        try:
            factory_plugin(registry)
        except Exception as exc:
            log.warning("plugin_registration_failed", plugin=dist_name, error=str(exc))
            results.append(PluginLoadResult(
                name=dist_name, status="failed",
                reason=f"Registration error: {exc}", version=dist_version,
            ))
            continue

        results.append(PluginLoadResult(
            name=dist_name, status="loaded", version=dist_version,
        ))

    _registry = registry
    _results = results
    _warn_missing_role_prompts(registry)
    return results


def _warn_missing_role_prompts(registry: PluginRegistry) -> None:
    """Warn when a plugin-registered agent role has no resolvable prompt.

    Roles resolve through the three-tier lookup in
    ``factory.agents.runner.resolve_prompt``: project override
    (``<project>/.factory/agents/<role>.md``), user-global
    (``~/.factory/agents/prompts/<role>.md``), factory default
    (``factory/agents/prompts/<role>.md``). A plugin role has no factory
    default, so a broken install (e.g. prompt data files missing from the
    built wheel) only surfaces as a FileNotFoundError at invocation time —
    after the CEO has already started. This check catches it at load time;
    a project override still satisfies the role, so the warning names that
    escape hatch.
    """
    if not registry.agent_roles:
        return

    from pathlib import Path

    factory_prompts = Path(__file__).parent / "agents" / "prompts"
    user_prompts = Path.home() / ".factory" / "agents" / "prompts"

    for role in registry.agent_roles:
        if (factory_prompts / f"{role}.md").exists():
            continue
        if (user_prompts / f"{role}.md").exists():
            continue
        log.warning(
            "plugin_agent_role_prompt_missing",
            role=role,
            expected_user=str(user_prompts / f"{role}.md"),
            hint="a project override at .factory/agents/<role>.md also works",
        )


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        load_plugins(_registry)
    return _registry


def get_results() -> list[PluginLoadResult]:
    global _results
    if _results is None:
        get_registry()
    return _results or []
