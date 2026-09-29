"""CLI parser construction and main dispatch."""

from __future__ import annotations

import argparse
import sys

from factory.cli._helpers import _load_env_local


_REFACTORY_AGENT_COMMANDS: frozenset[str] = frozenset(
    {
        "ceo",
        "run",
        "tmux",
        "tmux-ls",
        "tmux-stop",
        "tmux-capture",
        "discover",
        "init",
        "detect",
        "eval",
        "history",
        "study",
        "status",
        "backlog-list",
        "backlog-add",
        "checkpoint",
        "resume",
        "ace",
        "ace-stats",
    }
)


_COMMAND_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Entry Points",
        [
            "ceo",
            "run",
            "tmux",
            "tmux-ls",
            "tmux-capture",
            "tmux-stop",
            "refactory",
            "contained",
            "dashboard",
            "agent",
        ],
    ),
    ("Project Setup", ["home", "detect", "discover", "init"]),
    (
        "Experiment Lifecycle",
        [
            "begin",
            "finalize",
            "guard",
            "precheck",
            "log",
            "emit",
            "review",
        ],
    ),
    (
        "Project Intelligence",
        [
            "eval",
            "history",
            "study",
            "status",
            "summary",
            "diff",
            "explain",
            "export",
            "research",
            "insights",
            "report-update",
            "baseline",
            "clean-pr",
            "spec",
            "adversarial-state",
        ],
    ),
    (
        "Backlog & Refinement",
        [
            "backlog-add",
            "backlog-list",
            "backlog-remove",
            "deferred-list",
            "deferred-remove",
            "refine-status",
            "refine-begin",
            "refine-complete",
            "message",
        ],
    ),
    (
        "Knowledge & Archive",
        [
            "archive",
            "vault-init",
            "backfill-citations",
            "backfill-archive",
        ],
    ),
    ("Self-Evolution", ["ace", "ace-stats", "digest", "workflow", "graph", "mempalace", "outer-loop", "task"]),
    (
        "Configuration",
        [
            "config",
            "profile",
            "install",
            "self-update",
            "runners",
            "plugins",
            "usage",
            "serve-mcp",
        ],
    ),
    (
        "Validation & Recovery",
        [
            "leakage-check",
            "validate-research",
            "check-patch-targets",
            "checkpoint",
            "resume",
            "notify",
            "registry-list",
        ],
    ),
]


class _GroupedHelpParser(argparse.ArgumentParser):
    """ArgumentParser that renders subcommands in labelled groups."""

    def format_help(self) -> str:
        if self._subparsers is None:
            return super().format_help()

        sub_action: argparse._SubParsersAction | None = None  # type: ignore[type-arg]
        for action in self._subparsers._group_actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_action = action
                break

        if sub_action is None:
            return super().format_help()

        parts = [f"usage: {self.prog} [-h] <command> ...\n"]
        if self.description:
            parts.append(f"{self.description}\n")

        help_map: dict[str, str] = {}
        for sub_act in sub_action._choices_actions:
            help_map[sub_act.dest] = sub_act.help or ""

        refactory_filter = "--refactory-agent" in sys.argv

        grouped_cmds: set[str] = set()
        for group_name, cmds in _COMMAND_GROUPS:
            lines = []
            for cmd in cmds:
                if cmd in sub_action._name_parser_map and cmd in help_map:
                    if refactory_filter and cmd not in _REFACTORY_AGENT_COMMANDS:
                        continue
                    lines.append(f"  {cmd:25s}{help_map[cmd]}")
                    grouped_cmds.add(cmd)
            if lines:
                parts.append(f"\n{group_name}:\n" + "\n".join(lines))

        if not refactory_filter:
            ungrouped = [
                c for c in help_map if c not in grouped_cmds and c in sub_action._name_parser_map
            ]
            if ungrouped:
                lines = [f"  {cmd:25s}{help_map[cmd]}" for cmd in ungrouped]
                parts.append("\nOther:\n" + "\n".join(lines))

        parts.append("")
        return "\n".join(parts)


def _cmd_plugins(args: argparse.Namespace) -> int:
    """List discovered plugins and their registered extensions."""
    import dataclasses
    import json

    from factory.plugins import get_registry, get_results

    results = get_results()
    registry = get_registry()

    if getattr(args, "json", False) if hasattr(args, "json") else False:
        data = [dataclasses.asdict(r) for r in results]
        print(json.dumps(data, indent=2))
        return 0

    if not results:
        print("No plugins discovered.")
        return 0

    for r in results:
        ver = f" v{r.version}" if r.version else ""
        line = f"  {r.name}{ver}: {r.status}"
        if r.reason:
            line += f" ({r.reason})"
        print(line)

    if registry.commands:
        print(f"\nRegistered commands: {', '.join(sorted(registry.commands))}")
    if registry.modes:
        print(f"Registered modes: {', '.join(registry.modes)}")
    if registry.agent_roles:
        print(f"Registered agent roles: {', '.join(registry.agent_roles)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    from factory.cli._parser_groups import (
        add_archive_parsers,
        add_backlog_refinement_parsers,
        add_configuration_parsers,
        add_entry_point_parsers,
        add_experiment_lifecycle_parsers,
        add_project_intelligence_parsers,
        add_project_setup_parsers,
        add_self_evolution_parsers,
        add_validation_recovery_parsers,
    )

    from importlib.metadata import version as pkg_version

    parser = _GroupedHelpParser(
        prog="factory",
        description="Remote Factory — domain-agnostic multi-agent software evolution loop",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"remote-factory {pkg_version('remote-factory')}",
    )
    parser.add_argument(
        "--refactory-agent",
        action="store_true",
        help="Show only commands used by the re:factory agent",
    )
    sub = parser.add_subparsers(dest="command")

    add_project_setup_parsers(sub)
    add_experiment_lifecycle_parsers(sub)
    add_project_intelligence_parsers(sub)
    add_backlog_refinement_parsers(sub)
    add_archive_parsers(sub)
    add_self_evolution_parsers(sub)
    add_configuration_parsers(sub)
    add_validation_recovery_parsers(sub)
    add_entry_point_parsers(sub)

    # ── plugin commands ──────────────────────────────────────────
    p_plugins = sub.add_parser("plugins", help="List discovered plugins and their extensions")
    p_plugins.add_argument("--json", action="store_true", default=False, help="Machine-readable JSON output")

    from factory.plugins import PluginRegistry, load_plugins

    _plugin_registry = PluginRegistry()
    load_plugins(_plugin_registry)

    for cmd_name, spec in _plugin_registry.commands.items():
        p_plugin = sub.add_parser(cmd_name, help=spec.help)
        if spec.add_arguments is not None:
            spec.add_arguments(p_plugin)
        p_plugin.set_defaults(_plugin_handler=spec.handler)

    # ── plugin parser extensions ────────────────────────────────
    sub_action: argparse._SubParsersAction | None = None  # type: ignore[type-arg]
    if parser._subparsers is not None:
        for action in parser._subparsers._group_actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_action = action
                break

    if sub_action is not None:
        import structlog as _structlog

        _ext_log = _structlog.get_logger()
        for ext_name, ext_fns in _plugin_registry.parser_extensions.items():
            ext_parser = sub_action._name_parser_map.get(ext_name)
            if ext_parser is None:
                _ext_log.warning("plugin_parser_extension_no_target", subcommand=ext_name)
                continue
            for ext_fn in ext_fns:
                ext_fn(ext_parser)

    # graph — code knowledge graph operations
    graph_parser = sub.add_parser("graph", help="Code knowledge graph via graphify")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")
    p_graph_extract = graph_sub.add_parser("extract", help="Extract a code knowledge graph")
    p_graph_extract.add_argument("path", help="Path to the project")
    p_graph_update = graph_sub.add_parser("update", help="Incrementally update the knowledge graph")
    p_graph_update.add_argument("path", help="Path to the project")
    p_graph_status = graph_sub.add_parser("status", help="Show graph freshness and stats")
    p_graph_status.add_argument("path", help="Path to the project")
    p_graph_query = graph_sub.add_parser("query", help="BFS traversal of the knowledge graph")
    p_graph_query.add_argument("path", help="Path to the project")
    p_graph_query.add_argument("question", help="Natural-language query for graph traversal")
    p_graph_query.add_argument("--depth", type=int, default=2, help="BFS depth (default: 2)")
    p_graph_explain = graph_sub.add_parser("explain", help="Explain a node and its neighbors")
    p_graph_explain.add_argument("path", help="Path to the project")
    p_graph_explain.add_argument("node", help="Node name or label to explain")
    p_graph_path = graph_sub.add_parser("path", help="Shortest path between two nodes")
    p_graph_path.add_argument("path", help="Path to the project")
    p_graph_path.add_argument("source", help="Source node name")
    p_graph_path.add_argument("target", help="Target node name")

    # mempalace — MemPalace operations (read, write, browse)
    mp = sub.add_parser("mempalace", help="MemPalace operations (read, write, browse)")
    mp_sub = mp.add_subparsers(dest="mempalace_action", required=True)

    mp_read = mp_sub.add_parser("read", help="Read MemPalace context for a project")
    mp_read.add_argument("project_path", help="Path to the project")
    mp_read.add_argument("--task-hint", help="Task context for targeted retrieval")

    mp_write = mp_sub.add_parser("write", help="Write project data to MemPalace")
    mp_write.add_argument("project_path", help="Path to the project")

    mp_browse = mp_sub.add_parser("browse", help="Browse palace hierarchy: wings → rooms → drawers")
    mp_browse.add_argument("project_path", help="Path to the project")
    mp_browse.add_argument("--wing", help="Filter to a specific wing")
    mp_browse.add_argument("--room", help="Filter to a specific room (requires --wing)")
    mp_browse.add_argument("--drawer", help="Show full content of a specific drawer by ID")
    mp_browse.add_argument("--all", action="store_true", help="Show all wings (default: only this project's wing)")

    # task — task management (create, validate, list)
    from factory.cli.task import add_task_parser
    add_task_parser(sub)

    # outer-loop — evolutionary workflow search
    from factory.cli.outer_loop import add_outer_loop_parser
    add_outer_loop_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env_local()
    parser = build_parser()
    args = parser.parse_args(argv)

    import factory.cli as _cli

    if not args.command:
        if sys.stdin.isatty() and sys.stderr.isatty():
            return _cli.cmd_refactory(args)
        parser.print_help()
        return 1

    handlers = {
        "home": _cli.cmd_home,
        "detect": _cli.cmd_detect,
        "discover": _cli.cmd_discover,
        "init": _cli.cmd_init,
        "eval": _cli.cmd_eval,
        "guard": _cli.cmd_guard,
        "begin": _cli.cmd_begin,
        "finalize": _cli.cmd_finalize,
        "history": _cli.cmd_history,
        "notify": _cli.cmd_notify,
        "study": _cli.cmd_study,
        "backlog-remove": _cli.cmd_backlog_remove,
        "deferred-remove": _cli.cmd_backlog_remove,
        "backlog-list": _cli.cmd_backlog_list,
        "deferred-list": _cli.cmd_backlog_list,
        "backlog-add": _cli.cmd_backlog_add,
        "status": _cli.cmd_status,
        "summary": _cli.cmd_summary,
        "research": _cli.cmd_research,
        "backfill-citations": _cli.cmd_backfill_citations,
        "backfill-archive": _cli.cmd_backfill_archive,
        "diff": _cli.cmd_diff,
        "explain": _cli.cmd_explain,
        "export": _cli.cmd_export,
        "insights": _cli.cmd_insights,
        "report-update": _cli.cmd_report_update,
        "registry-list": _cli.cmd_registry_list,
        "ace": _cli.cmd_ace,
        "ace-stats": _cli.cmd_ace_stats,
        "digest": _cli.cmd_digest,
        "archive": _cli.cmd_archive,
        "precheck": _cli.cmd_precheck,
        "clean-pr": _cli.cmd_clean_pr,
        "baseline": _cli.cmd_baseline,
        "leakage-check": _cli.cmd_leakage_check,
        "validate-research": _cli.cmd_validate_research,
        "adversarial-state": _cli.cmd_adversarial_state,
        "refine-status": _cli.cmd_refine_status,
        "refine-begin": _cli.cmd_refine_begin,
        "refine-complete": _cli.cmd_refine_complete,
        "review": _cli.cmd_review,
        "checkpoint": _cli.cmd_checkpoint,
        "resume": _cli.cmd_resume,
        "log": _cli.cmd_log,
        "vault-init": _cli.cmd_vault_init,
        "message": _cli.cmd_message,
        "self-update": _cli.cmd_self_update,
        "install": _cli.cmd_install,
        "serve-mcp": _cli.cmd_serve_mcp,
        "dashboard": _cli.cmd_dashboard,
        "config": _cli.cmd_config,
        "profile": _cli.cmd_profile,
        "emit": _cli.cmd_emit,
        "usage": _cli.cmd_usage,
        "runners": _cli.cmd_runners_list,
        "agent": _cli.cmd_agent,
        "ceo": _cli.cmd_ceo,
        "run": _cli.cmd_run,
        "tmux": _cli.cmd_tmux,
        "tmux-ls": _cli.cmd_tmux_ls,
        "tmux-capture": _cli.cmd_tmux_capture,
        "tmux-stop": _cli.cmd_tmux_stop,
        "refactory": _cli.cmd_refactory,
        "contained": _cli.cmd_contained,
        "spec": lambda a: {
            "generate": _cli.cmd_spec_generate,
            "validate": _cli.cmd_spec_validate,
            "scope": _cli.cmd_spec_scope,
            "update": _cli.cmd_spec_update,
            "apply-diff": _cli.cmd_spec_apply_diff,
            "impact": _cli.cmd_spec_impact,
        }.get(
            str(getattr(a, "spec_command", "")),
            lambda args: (
                print("Usage: factory spec {generate,validate,scope,update,apply-diff,impact}") or 1
            ),
        )(a),
        "workflow": lambda a: __import__(
            "factory.workflow.cli", fromlist=["cmd_workflow"]
        ).cmd_workflow(a),
        "plugins": _cmd_plugins,
        "task": lambda a: __import__(
            "factory.cli.task", fromlist=["cmd_task"]
        ).cmd_task(a),
        "outer-loop": lambda a: __import__(
            "factory.cli.outer_loop", fromlist=["cmd_outer_loop"]
        ).cmd_outer_loop(a),
        "check-patch-targets": _cli.cmd_check_patch_targets,
        "mempalace": _cli.cmd_mempalace,
        "graph": lambda a: {
            "extract": _cli.cmd_graph_extract,
            "update": _cli.cmd_graph_update,
            "status": _cli.cmd_graph_status,
            "query": _cli.cmd_graph_query,
            "explain": _cli.cmd_graph_explain,
            "path": _cli.cmd_graph_path,
        }.get(
            str(getattr(a, "graph_command", "")),
            lambda args: print("Usage: factory graph {extract,update,status,query,explain,path}") or 1,
        )(a),
    }

    handler = handlers.get(args.command)
    if handler is None:
        handler = getattr(args, "_plugin_handler", None)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
