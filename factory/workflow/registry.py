"""Workflow registry for discovering and loading contributed workflows.

Follows the same search-path pattern as sdg_hub's FlowRegistry:
register directories, auto-discover workflow files within them.

A workflow file is any .py file containing:
  - A `meta` dict with at least `name` and `description`
  - A `workflow()` function returning a Workflow object
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from factory.workflow.primitives import Workflow

log = structlog.get_logger()

# Discovery priority per source label. Higher rank wins when two sources
# provide a workflow with the same name. Unknown labels (e.g. a plugin
# appending to WorkflowRegistry._search_paths directly) rank as "plugin".
_SOURCE_PRIORITY: dict[str, int] = {
    "builtin": 0,
    "user": 1,
    "plugin": 2,
    "project": 3,
}


def _source_priority(source: str) -> int:
    """Rank of a source label. Higher wins a name collision."""
    return _SOURCE_PRIORITY.get(source, _SOURCE_PRIORITY["plugin"])


@dataclass
class WorkflowEntry:
    """A discovered workflow in the registry."""

    name: str
    description: str
    path: str
    source: str  # "builtin", "user", "plugin", "project"
    _workflow_fn: Any = field(default=None, repr=False)


class WorkflowRegistry:
    """Registry for discovering contributed workflows.

    Search paths are scanned for .py files with a `meta` dict and
    `workflow()` function. Built-in workflows from `definitions.py`
    are always available as the lowest-priority source.
    """

    _entries: dict[str, WorkflowEntry] = {}
    _callables: dict[str, WorkflowEntry] = {}  # register_callable() entries, kept across discover()
    _search_paths: list[tuple[str, str]] = []  # (path, source_label)
    _initialized: bool = False
    _discovered: bool = False
    _discovered_project: Path | None = None

    @classmethod
    def reset(cls) -> None:
        """Reset registry state. Useful for testing."""
        cls._entries.clear()
        cls._callables.clear()
        cls._search_paths.clear()
        cls._initialized = False
        cls._discovered = False
        cls._discovered_project = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Register default search paths on first access."""
        if cls._initialized:
            return

        # User-global workflows
        user_dir = Path.home() / ".factory" / "workflows"
        if user_dir.is_dir():
            cls._search_paths.append((str(user_dir), "user"))
            log.debug("workflow_registry.search_path", path=str(user_dir), source="user")

        cls._initialized = True

    @classmethod
    def discover(cls, project_path: Path | None = None) -> dict[str, WorkflowEntry]:
        """Discover all workflows from search paths + built-ins.

        Priority (highest wins on a name collision):
          1. Project-local (.factory/workflows/ in *project_path*)
          2. Plugin-registered paths (PluginRegistry.workflow_search_paths,
             plus any paths on WorkflowRegistry._search_paths)
          3. User-global (~/.factory/workflows/)
          4. Built-in workflows (definitions.py)

        Parameters
        ----------
        project_path : Path, optional
            If provided, also searches .factory/workflows/ in this project.

        Returns
        -------
        dict[str, WorkflowEntry]
            Name → entry mapping, resolved per the priority above.
        """
        cls._ensure_initialized()
        cls._entries.clear()

        # Built-in workflows (lowest priority)
        cls._load_builtins()

        # User-global workflows
        for search_path, source in cls._search_paths:
            if source == "user":
                cls._discover_in_directory(search_path, source)

        # Plugin-registered paths: consumed natively from PluginRegistry so
        # plugins never need to touch WorkflowRegistry._search_paths.
        for plugin_path in cls._plugin_search_paths():
            cls._discover_in_directory(plugin_path, "plugin")

        # Legacy/direct registrations on _search_paths (non-user labels)
        for search_path, source in cls._search_paths:
            if source not in ("user",):
                cls._discover_in_directory(search_path, source)

        # Callables registered by plugins (e.g. Package compositions)
        for name, entry in cls._callables.items():
            prev = cls._entries.get(name)
            if prev is None or _source_priority(entry.source) > _source_priority(prev.source):
                cls._entries[name] = entry

        # Project-local workflows (highest priority)
        if project_path:
            project_wf_dir = project_path / ".factory" / "workflows"
            if project_wf_dir.is_dir():
                cls._discover_in_directory(str(project_wf_dir), "project")

        cls._warn_mode_drift()

        cls._discovered = True
        cls._discovered_project = project_path
        log.info("workflow_registry.discovered", count=len(cls._entries))
        return cls._entries

    @classmethod
    def _discover_if_needed(cls, project_path: Path | None) -> None:
        """Discover on first use, and again when asked about a different project."""
        if not cls._discovered or (
            project_path is not None and project_path != cls._discovered_project
        ):
            cls.discover(project_path)

    @classmethod
    def _warn_mode_drift(cls) -> None:
        """Log consistency warnings between plugin modes and discovered workflows.

        A plugin mode with no discovered workflow of the same name usually means
        a typo or a renamed file — without a warning it surfaces only as a
        silent fallback to the default improve loop. A plugin workflow that is
        not declared as a mode is legal (subgraph libraries, composed
        packages), so that direction is info-level only.
        """
        try:
            from factory.plugins import get_registry

            plugin_modes = set(get_registry().modes)
        except Exception as exc:
            log.debug("workflow_registry.plugin_modes_unavailable", error=str(exc))
            return
        if not plugin_modes:
            return

        for mode in sorted(plugin_modes):
            if mode not in cls._entries:
                log.warning(
                    "workflow_registry.mode_without_workflow",
                    mode=mode,
                    action="ceo_falls_back_to_improve_loop",
                )
        for name, entry in cls._entries.items():
            if entry.source == "plugin" and name not in plugin_modes:
                log.info(
                    "workflow_registry.plugin_workflow_not_a_mode",
                    name=name,
                )

    @classmethod
    def _plugin_search_paths(cls) -> list[str]:
        """Workflow search paths registered by loaded plugins, if any."""
        try:
            from factory.plugins import get_registry

            return list(get_registry().workflow_search_paths)
        except Exception as exc:
            log.debug("workflow_registry.plugin_paths_unavailable", error=str(exc))
            return []

    @classmethod
    def register_callable(
        cls,
        name: str,
        fn: Any,
        *,
        source: str = "plugin",
        description: str = "",
    ) -> None:
        """Register a workflow callable (e.g. from a Package composition).

        The callable is stored lazily — it is only invoked when
        get_workflow() is called for this name.
        """
        entry = WorkflowEntry(
            name=name,
            description=description or f"Composed mode: {name}",
            path=f"<{source}>",
            source=source,
            _workflow_fn=fn,
        )
        cls._callables[name] = entry
        cls._entries[name] = entry

    @classmethod
    def _load_builtins(cls) -> None:
        """Load built-in workflows from definitions.py.

        Uses _get_builtin_registry() so that contributed-workflow modules
        are NOT imported at discovery time.  The callable is stored but
        NOT invoked — the Workflow object is only constructed when
        get_workflow() is called for that specific name.
        """
        from factory.workflow.definitions import _get_builtin_registry

        for name, fn in _get_builtin_registry().items():
            cls._entries[name] = WorkflowEntry(
                name=name,
                description=_get_builtin_description(name),
                path="<builtin>",
                source="builtin",
                _workflow_fn=fn,
            )

    @classmethod
    def _discover_in_directory(cls, directory: str, source: str) -> None:
        """Discover workflow files in a directory."""
        path = Path(directory)
        if not path.is_dir():
            return

        for py_file in sorted(path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                meta, workflow_fn = _load_workflow_file(py_file)
                name = meta["name"]
                priority = _source_priority(source)
                prev = cls._entries.get(name)
                if prev is not None:
                    if priority <= _source_priority(prev.source):
                        # Lower-priority source cannot shadow a higher one.
                        log.info(
                            "workflow_registry.shadow_skipped",
                            name=name,
                            new_source=source,
                            kept_source=prev.source,
                        )
                        continue
                    log.warning(
                        "workflow_registry.shadow",
                        name=name,
                        new_source=source,
                        old_source=prev.source,
                    )
                cls._entries[name] = WorkflowEntry(
                    name=name,
                    description=meta.get("description", ""),
                    path=str(py_file),
                    source=source,
                    _workflow_fn=workflow_fn,
                )
                log.debug(
                    "workflow_registry.loaded",
                    name=name,
                    path=str(py_file),
                    source=source,
                )
            except Exception as exc:
                log.debug("workflow_registry.skip", path=str(py_file), reason=str(exc))

    @classmethod
    def get_workflow(cls, name: str, project_path: Path | None = None) -> Workflow | None:
        """Get a workflow by name, discovering if needed.

        Returns None if not found.
        """
        cls._discover_if_needed(project_path)

        entry = cls._entries.get(name)
        if entry is None:
            return None

        if entry._workflow_fn is None:
            return None

        return entry._workflow_fn()

    @classmethod
    def list_workflows(cls, project_path: Path | None = None) -> list[WorkflowEntry]:
        """List all discovered workflows."""
        cls._discover_if_needed(project_path)
        return sorted(cls._entries.values(), key=lambda e: (e.source != "builtin", e.name))


def _load_workflow_file(path: Path) -> tuple[dict[str, Any], Any]:
    """Load a workflow .py file and extract meta + workflow function.

    Raises ValueError if the file doesn't have the required exports.
    """
    spec = importlib.util.spec_from_file_location(f"factory_workflow_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise ValueError(f"Failed to load {path}: {exc}") from exc

    meta = getattr(module, "meta", None)
    workflow_fn = getattr(module, "workflow", None)

    # Clean up sys.modules — we only need the extracted objects
    sys.modules.pop(spec.name, None)

    if not isinstance(meta, dict) or "name" not in meta:
        raise ValueError(f"{path} missing 'meta' dict with 'name' key")

    if not callable(workflow_fn):
        raise ValueError(f"{path} missing 'workflow()' function")

    return meta, workflow_fn


def _get_builtin_description(name: str) -> str:
    """Get description for a built-in workflow from WORKFLOW_META."""
    from factory.workflow.skill_export import WORKFLOW_META

    meta = WORKFLOW_META.get(name, {})
    return str(meta.get("description", f"Built-in {name} workflow"))
