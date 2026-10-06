"""CLI entry point for the factory — argparse subcommands wrapping library functions."""

from __future__ import annotations

from factory.cli._helpers import CEO_MODES as CEO_MODES
from factory.cli._helpers import RUN_MODES as RUN_MODES
from factory.cli._main import build_parser as build_parser
from factory.cli._main import main as main
from factory.cli.admin import (
    cmd_config as cmd_config,
    cmd_detect as cmd_detect,
    cmd_discover as cmd_discover,
    cmd_emit as cmd_emit,
    cmd_home as cmd_home,
    cmd_init as cmd_init,
    cmd_install as cmd_install,
    cmd_log as cmd_log,
    cmd_notify as cmd_notify,
    cmd_profile as cmd_profile,
    cmd_self_update as cmd_self_update,
    cmd_study as cmd_study,
    cmd_usage as cmd_usage,
)
from factory.cli.agents import (
    cmd_ace as cmd_ace,
    cmd_ace_stats as cmd_ace_stats,
    cmd_agent as cmd_agent,
    cmd_runners_list as cmd_runners_list,
)
from factory.cli.backlog import (
    cmd_backlog_add as cmd_backlog_add,
    cmd_backlog_list as cmd_backlog_list,
    cmd_backlog_remove as cmd_backlog_remove,
)
from factory.cli._tmux_commands import (
    cmd_tmux as cmd_tmux,
    cmd_tmux_capture as cmd_tmux_capture,
    cmd_tmux_ls as cmd_tmux_ls,
    cmd_tmux_stop as cmd_tmux_stop,
)
from factory.cli.mempalace import cmd_mempalace as cmd_mempalace
from factory.cli.contained import (
    cmd_contained as cmd_contained,
)
from factory.cli.ceo import (
    cmd_ceo as cmd_ceo,
    cmd_refactory as cmd_refactory,
)
from factory.cli.run import (
    cmd_run as cmd_run,
)
from factory.cli.graph import (
    cmd_graph_explain as cmd_graph_explain,
    cmd_graph_extract as cmd_graph_extract,
    cmd_graph_path as cmd_graph_path,
    cmd_graph_query as cmd_graph_query,
    cmd_graph_status as cmd_graph_status,
    cmd_graph_update as cmd_graph_update,
)
from factory.cli.eval_cmds import (
    cmd_adversarial_state as cmd_adversarial_state,
    cmd_baseline as cmd_baseline,
    cmd_eval as cmd_eval,
    cmd_guard as cmd_guard,
    cmd_precheck as cmd_precheck,
)
from factory.cli.infra import (
    cmd_archive as cmd_archive,
    cmd_backfill_archive as cmd_backfill_archive,
    cmd_checkpoint as cmd_checkpoint,
    cmd_dashboard as cmd_dashboard,
    cmd_resume as cmd_resume,
    cmd_serve_mcp as cmd_serve_mcp,
    cmd_vault_init as cmd_vault_init,
)
from factory.cli.registry import (
    cmd_digest as cmd_digest,
    cmd_insights as cmd_insights,
    cmd_registry_list as cmd_registry_list,
    cmd_report_update as cmd_report_update,
)
from factory.cli.research import (
    cmd_backfill_citations as cmd_backfill_citations,
    cmd_leakage_check as cmd_leakage_check,
    cmd_research as cmd_research,
    cmd_validate_research as cmd_validate_research,
)
from factory.cli.review import (
    cmd_clean_pr as cmd_clean_pr,
    cmd_refine_begin as cmd_refine_begin,
    cmd_refine_complete as cmd_refine_complete,
    cmd_refine_status as cmd_refine_status,
    cmd_review as cmd_review,
)
from factory.cli.spec import (
    cmd_spec_apply_diff as cmd_spec_apply_diff,
    cmd_spec_generate as cmd_spec_generate,
    cmd_spec_impact as cmd_spec_impact,
    cmd_spec_scope as cmd_spec_scope,
    cmd_spec_update as cmd_spec_update,
    cmd_spec_validate as cmd_spec_validate,
)
from factory.cli.testing import (
    cmd_check_patch_targets as cmd_check_patch_targets,
)
from factory.cli.store import (
    cmd_begin as cmd_begin,
    cmd_diff as cmd_diff,
    cmd_explain as cmd_explain,
    cmd_export as cmd_export,
    cmd_finalize as cmd_finalize,
    cmd_history as cmd_history,
    cmd_message as cmd_message,
    cmd_status as cmd_status,
    cmd_summary as cmd_summary,
)


if __name__ == "__main__":
    raise SystemExit(main())
