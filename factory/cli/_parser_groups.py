"""Argparse subcommand group builders — extracted from _main.build_parser()."""
from __future__ import annotations

import argparse

BUILTIN_AGENT_ROLES: frozenset[str] = frozenset({
    "researcher", "strategist", "builder",
    "health_checker", "code_reviewer", "adversarial_tester",
    "archivist", "ceo", "failure_analyst", "refiner",
})


def add_project_setup_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sub.add_parser("home", help="Print factory installation root directory")

    p = sub.add_parser("detect", help="Print project state")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("discover", help="Introspect project and generate eval profile")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("init", help="Create .factory/ or reparse factory.md")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--reparse", action="store_true", help="Reparse existing factory.md")


def add_experiment_lifecycle_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("eval", help="Run project evals, print JSON CompositeScore")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--skip-project-eval", action="store_true", default=False,
                    help="Skip user-defined project eval dimensions (run only hygiene + growth)")
    p.add_argument("--targeted", action="store_true", default=False,
                    help="Use graph-based targeted test selection (falls back to full suite)")

    p = sub.add_parser("guard", help="Check guard rules, print violations or 'clean'")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--baseline", required=True, help="Baseline commit SHA")
    p.add_argument("--check-scope", action="store_true", help="Also check file scope")
    p.add_argument("--check-surfaces", action="store_true",
                    help="Also check fixed surface constraints (research mode)")

    p = sub.add_parser("begin", help="Start experiment, print ID")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--hypothesis", required=True, help="Experiment hypothesis text")

    p = sub.add_parser("finalize", help="Finalize experiment with verdict")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--id", required=True, type=int, help="Experiment ID")
    p.add_argument("--verdict", required=True, choices=["keep", "revert", "error"],
                    help="Experiment verdict")
    p.add_argument("--hypothesis", default=None, help="Hypothesis text")
    p.add_argument("--summary", default=None, help="Change summary")
    p.add_argument("--cost", default=None, type=float, help="Cost in USD")
    p.add_argument("--issue", default=None, type=int, help="GitHub issue number")
    p.add_argument("--pr", default=None, type=int, help="GitHub PR number")
    p.add_argument("--notes", default=None, help="Additional notes")
    p.add_argument("--score-before", type=float, default=None, help="Eval score before change")
    p.add_argument("--score-after", type=float, default=None, help="Eval score after change")
    p.add_argument("--force", action="store_true", default=False,
                    help="Bypass precheck gate (for pre-existing failures)")

    p = sub.add_parser("precheck", help="Run hard precheck gate before keep/revert decision")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--score-before", type=float, default=None, help="Eval score before change")
    p.add_argument("--score-after", type=float, default=None, help="Eval score after change")
    p.add_argument("--hypothesis", default=None, help="Current experiment hypothesis")
    p.add_argument("--baseline", default=None, help="Baseline commit SHA for scope check")
    p.add_argument("--similarity-threshold", type=float, default=0.6,
                    help="Similarity threshold for anti-pattern detection (default: 0.6)")

    p = sub.add_parser("log", help="Append a structured event to .factory/events.jsonl")
    p.add_argument("path", help="Path to the project")
    p.add_argument("event_type", help="Event type (e.g. phase.research.completed)")
    p.add_argument("--data", help="JSON data payload")
    p.add_argument("--agent", help="Agent name to attribute the event to")

    p = sub.add_parser("emit", help="Emit a structured event to .factory/events.jsonl")
    p.add_argument("event_type", help="Event type (e.g. agent.started, agent.completed)")
    p.add_argument("--agent", default=None, help="Agent role name")
    p.add_argument("--project", default=".", help="Project path")
    p.add_argument("--data", default=None, help="JSON string of additional event data")

    p = sub.add_parser("review", help="Format and post a structured review on a GitHub PR")
    p.add_argument("--verdict", required=True, choices=["keep", "revert", "KEEP", "REVERT"],
                    help="Review verdict")
    p.add_argument("--reason", default=None, help="One-sentence reason for the verdict")
    p.add_argument("--score-before", type=float, default=None, help="Score before change")
    p.add_argument("--score-after", type=float, default=None, help="Score after change")
    p.add_argument("--threshold", type=float, default=0.8, help="Eval threshold")
    p.add_argument("--guards", default=None,
                    help="Guard results as 'check:PASS,check:FAIL' pairs")
    p.add_argument("--precheck-summary", default=None, help="Precheck gate output summary")
    p.add_argument("--code-notes", default=None,
                    help="Code review notes separated by | (pipe)")
    p.add_argument("--experiment-id", type=int, default=None, help="Experiment ID")
    p.add_argument("--hypothesis", default=None, help="Experiment hypothesis text")
    p.add_argument("--pr", type=int, default=None, help="PR number to post review on")
    p.add_argument("--repo", default=None, help="GitHub repo (owner/name) for the PR")
    p.add_argument("--qa-body-file", default=None,
                    help="Path to file containing QA analysis to include in review")
    p.add_argument("--dry-run", action="store_true", default=False,
                    help="Print review without posting")


def add_project_intelligence_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("history", help="Print formatted experiment history table")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("study", help="Read interaction logs and write observations")
    p.add_argument("path", help="Path to the project")
    p.add_argument(
        "--projects-dir", default=None,
        help="Directory containing factory-managed projects for cross-project insights",
    )
    p.add_argument(
        "--focus", default=None,
        help="Targeted mode: filter observations to a single backlog item",
    )

    p = sub.add_parser("status", help="Print project status summary")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("summary", help="Generate end-of-session summary report")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("leakage-check", help="Scan text for ground truth leakage against fixed surfaces")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--text", default=None, help="Text to scan for leakage (hypothesis, strategy, etc.)")
    p.add_argument("--text-file", default=None, help="Path to file containing text to scan (safer for large diffs)")
    p.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium",
                    help="Sensitivity level (default: medium)")

    p = sub.add_parser("validate-research", help="Validate research mode configuration for ground truth isolation")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("research", help="Print research citation index for experiments")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("diff", help="Compare two experiments side-by-side")
    p.add_argument("path", help="Path to the project")
    p.add_argument("id_a", type=int, help="First experiment ID")
    p.add_argument("id_b", type=int, help="Second experiment ID")

    p = sub.add_parser("explain", help="Explain a single experiment with FEEC analysis")
    p.add_argument("path", help="Path to the project")
    p.add_argument("id", type=int, help="Experiment ID")

    p = sub.add_parser("export", help="Export complete project snapshot as JSON to stdout")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("insights", help="Cross-project analysis of experiment histories")
    p.add_argument("path", help="Path to the project (insights.md written here)")
    p.add_argument(
        "--projects-dir", default=None,
        help="Directory containing factory-managed projects (default: from registry or ~/factory-projects)",
    )

    p = sub.add_parser("report-update", help="Generate performance report for a project")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("clean-pr", help="Strip non-essential artifacts from a PR diff")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--exp", type=int, default=None, help="Experiment ID (archives full diff before stripping)")

    p = sub.add_parser("baseline", help="Fetch stored eval baseline from eval-data branch")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--commit", default=None,
                    help="Commit SHA to look up (default: git merge-base HEAD <target-branch>)")

    p = sub.add_parser("adversarial-state", help="Inspect or reset adversarial eval loop state")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--reset", action="store_true", default=False,
                    help="Reset adversarial state to defaults")

    spec_parser = sub.add_parser("spec", help="Repo spec generation and analysis")
    spec_sub = spec_parser.add_subparsers(dest="spec_command")
    p_spec_gen = spec_sub.add_parser("generate", help="Generate a repo spec for a project")
    p_spec_gen.add_argument("path", help="Path to the project")
    p_spec_val = spec_sub.add_parser("validate", help="Validate a repo spec against the project")
    p_spec_val.add_argument("path", help="Path to the project")
    p_spec_scope = spec_sub.add_parser("scope", help="Scope a diff against the repo spec")
    p_spec_scope.add_argument("path", help="Path to the project")
    p_spec_scope.add_argument("--experiment", type=int, default=None, help="Experiment ID to scope")
    p_spec_update = spec_sub.add_parser("update", help="Update the repo spec from recent changes")
    p_spec_update.add_argument("path", help="Path to the project")
    p_spec_apply_diff = spec_sub.add_parser("apply-diff", help="Apply SPEC Diff from strategy to SPEC.md")
    p_spec_apply_diff.add_argument("path", help="Path to the project")
    p_spec_apply_diff.add_argument("--strategy", default=None,
                                    help="Path to strategy file (default: .factory/strategy/current.md)")
    p_spec_impact = spec_sub.add_parser("impact", help="Show impact subgraph for a module")
    p_spec_impact.add_argument("module", help="Module name to query")
    p_spec_impact.add_argument("--project", required=True, help="Path to the project")


def add_backlog_refinement_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("backlog-remove", aliases=["deferred-remove"], help="Remove a completed backlog item")
    p.add_argument("path", help="Path to the project")
    p.add_argument("item", help="Exact text of the backlog item to remove")

    p = sub.add_parser("backlog-list", aliases=["deferred-list"], help="List pending backlog items")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("backlog-add", help="Add a new item to the backlog")
    p.add_argument("path", help="Path to the project")
    p.add_argument("item", help="Text of the backlog item to add")

    p = sub.add_parser("refine-status", help="Print refinement state and regrounding output")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("refine-begin", help="Record a new refinement and emit regrounding output")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--request", required=True, help="Summary of the user's refinement request")

    p = sub.add_parser("refine-complete", help="Complete the current refinement with a verdict")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--verdict", required=True, choices=["keep", "revert", "error", "tier3_exit"],
                    help="Refinement verdict")

    p = sub.add_parser("message", help="Send a message to the CEO for the next cycle")
    p.add_argument("path", help="Path to the project")
    p.add_argument("text", help="Message text")


def add_archive_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("backfill-citations", help="Extract citations from experiment text into citations.json")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("backfill-archive", help="Generate archive notes for experiments missing from archive")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("archive", help="Write experiment notes to Obsidian vault")
    p.add_argument("path", help="Path to the project")

    sub.add_parser("vault-init", help="Create the factory Obsidian vault")


def add_self_evolution_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("ace", help="Run ACE self-improvement on agent playbooks")
    p.add_argument("path", help="Path to the project")
    p.add_argument(
        "--projects-dir", default=None,
        help="Directory containing factory-managed projects (default: from registry or ~/factory-projects)",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print candidates without writing playbooks",
    )

    sub.add_parser("ace-stats", help="Print playbook item counters for all roles")

    p = sub.add_parser("digest", help="Summarize recent factory activity across projects")
    p.add_argument("--date", default=None, help="Show activity for a specific date (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")


def add_configuration_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sub.add_parser("self-update", help="Upgrade the factory CLI to the latest version")

    p = sub.add_parser("install", help="Install Factory agents as CLI agents (~/.claude/agents/)")
    p.add_argument(
        "--role",
        default=None,
        help="Install only a specific agent role (default: all)",
    )

    p = sub.add_parser("usage", help="Show per-agent token usage and cost breakdown")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--json", action="store_true", default=False,
                    help="Output as JSON instead of table")

    runners_parser = sub.add_parser("runners", help="Manage factory runners")
    runners_sub = runners_parser.add_subparsers(dest="runners_command")
    p_runners_list = runners_sub.add_parser("list", help="List all registered runners")
    p_runners_list.add_argument("--json", action="store_true", default=False,
                                help="Output as JSON")

    sub.add_parser("serve-mcp", help="Start the Factory MCP stdio server")

    p = sub.add_parser("dashboard", help="Launch the live Factory dashboard")
    p.add_argument(
        "--projects-dir", default="~/factory-projects",
        help="Directory containing factory-managed projects (default: ~/factory-projects)",
    )
    p.add_argument("--port", type=int, default=8420, help="Server port (default: 8420)")
    p.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")

    config_parser = sub.add_parser("config", help="Manage ~/.factory/config.toml")
    config_sub = config_parser.add_subparsers(dest="config_command")
    p_show = config_sub.add_parser("show", help="Show resolved config (secrets masked)")
    p_show.add_argument("--reveal", action="store_true", default=False,
                        help="Show full secret values instead of masking")
    config_sub.add_parser("edit", help="Open config.toml in $EDITOR")
    config_sub.add_parser("migrate", help="Create starter config.toml from current env vars")

    profile_parser = sub.add_parser("profile", help="Manage the user profile at ~/.factory/profile.md")
    profile_sub = profile_parser.add_subparsers(dest="profile_command")
    p_build = profile_sub.add_parser("build", help="Collect evidence and synthesize user profile")
    p_build.add_argument("paths", nargs="*", default=None,
                         help="Project paths to collect evidence from (default: all registered)")
    p_build.add_argument("--dry-run", action="store_true", default=False,
                         help="Print collected evidence without running LLM synthesis")
    p_build.add_argument("--runner", default=None,
                         help="CLI backend to use for synthesis")
    profile_sub.add_parser("show", help="Print the current user profile")


def add_validation_recovery_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("notify", help="Send Telegram digest")
    p.add_argument("path", help="Path to the project")

    p = sub.add_parser("checkpoint", help="Show or save a CEO checkpoint for crash-resilient resume")
    p.add_argument("path", help="Path to the project")
    ckpt_action = p.add_mutually_exclusive_group()
    ckpt_action.add_argument("--save", action="store_true", default=False, help="Save a checkpoint")
    ckpt_action.add_argument("--clear", action="store_true", default=False,
                              help="Clear the checkpoint file")
    p.add_argument("--mode", default=None, help="CEO mode (e.g. improve, build)")
    p.add_argument("--experiment", type=int, default=None, help="Active experiment ID")
    p.add_argument("--completed", default=None,
                    help="Comma-separated list of completed agent roles")
    p.add_argument("--pending", default=None,
                    help="Comma-separated list of pending agent roles")
    p.add_argument("--scores", default=None,
                    help="JSON dict of eval scores (e.g. '{\"tests\": 0.9}')")
    p.add_argument("--hypothesis", default=None, help="Current hypothesis text")
    p.add_argument("--completed-hypotheses", default=None,
                    help="Comma-separated list of completed experiment IDs (e.g. '1,2,3')")

    p = sub.add_parser("resume", help="Resume a CEO session via Claude --resume")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--model", help="Model override for the resumed session")

    sub.add_parser("registry-list", help="List all registered factory-managed projects")

    p = sub.add_parser(
        "check-patch-targets",
        help="Check a test file for over-mocking (advisory)",
    )
    p.add_argument("test_file", help="Path to the test file to analyse")
    p.add_argument(
        "--project-name",
        default=None,
        help="Project package name (auto-detected from pyproject.toml if omitted)",
    )


def add_entry_point_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("agent", help="Invoke a specialist agent with a task")
    p.add_argument("role",
                    help="Agent role to invoke (built-in or plugin-registered)")
    p.add_argument("--task", required=True, help="Task description for the agent")
    p.add_argument("--project", required=True, help="Path to the project")
    p.add_argument("--timeout", type=float, default=600.0,
                    help="Timeout in seconds (default: 600)")
    p.add_argument("--model", default=None,
                    help="Claude model for agent subprocess (default: FACTORY_MODEL env var, or claude CLI default)")
    p.add_argument("--runner", default=None,
                    help="CLI backend to use (default: FACTORY_RUNNER env var, or 'claude')")
    p.add_argument("--profile", default=None,
                    help="Credential profile from ~/.factory/config.toml")
    p.add_argument("--use-profile", action="store_true", default=False,
                    help="Inject user profile (~/.factory/profile.md) into the agent prompt")
    p.add_argument("--tmux-persist", action="store_true", default=False,
                    help="Run agent interactively in a tmux window instead of headless (claude only)")
    p.add_argument("--bg", action="store_true", default=False,
                    help="Dispatch agent as a background session via claude agent view (claude only)")
    p.add_argument("--review-tag", default=None,
                    help="Tag for distinct review output files (writes <role>-<tag>-latest.md)")
    p.add_argument("--parent-session", default=None,
                    help="Parent session ID for linking specialist sessions to a CEO cycle session")

    p = sub.add_parser("ceo", help="Launch the Factory CEO agent (interactive by default)")
    p.add_argument("path", nargs="?", default=None,
                    help="Project path, GitHub URL, idea file path, or prompt. "
                         "In design mode, pass a raw idea string")
    p.add_argument(
        "--prompt", default=None,
        help="Path to a prompt/spec file (absolute or relative to project). "
             "Loaded as the build spec into .factory/strategy/current.md",
    )
    p.add_argument(
        "--mode",
        metavar="MODE",
        default="auto",
        help="Operating mode. Built-in: auto, design, create. "
             "Project-local: project:<name> (loads from .factory/workflows/<name>.py)",
    )
    p.add_argument(
        "--focus", default=None,
        help="Target a specific item: backlog name ('dashboard UI'), issue number (42), "
             "URL (https://github.com/o/r/issues/42), or shorthand (owner/repo#42). "
             "Issue refs are auto-detected and fetched via gh/glab CLI",
    )
    p.add_argument(
        "--dir", default=None,
        help="Working directory name for the new project (overrides auto-derived name from prompt or idea file). "
             "Ignored when pointing at an existing directory or GitHub URL.",
    )
    p.add_argument(
        "--headless", action="store_true", default=False,
        help="Run in pipe mode (non-interactive) instead of foreground",
    )
    p.add_argument(
        "--discover-only", action="store_true", default=False,
        help="Only run discovery and review — do not chain into improve",
    )
    p.add_argument(
        "--no-github", action="store_true", default=False,
        help="Disable GitHub operations (issue creation, PR posting, cloning)",
    )
    p.add_argument("--min-growth", type=int, default=None,
                    help="Minimum guaranteed growth hypotheses (default: 2)")
    p.add_argument("--max-new", type=int, default=None,
                    help="Max new items added to backlog per cycle (default: 2)")
    p.add_argument("--branch", default=None,
                    help="Target branch for PRs (default: from factory.md, fallback: main)")
    p.add_argument("--model", default=None,
                    help="Claude model for agent subprocesses (default: FACTORY_MODEL env var, or claude CLI default)")
    p.add_argument("--runner", default=None,
                    help="CLI backend to use (default: FACTORY_RUNNER env var, or 'claude')")
    p.add_argument("--profile", default=None,
                    help="Credential profile from ~/.factory/config.toml")
    p.add_argument(
        "--refine", default=None, metavar="REQUEST",
        help="Refinement mode: classify and implement a user-directed change. "
             "Mutually exclusive with --mode design, --mode research, --mode meta, --prompt, --focus",
    )
    p.add_argument("--use-profile", action="store_true", default=False,
                    help="Inject user profile (~/.factory/profile.md) into agent prompts")
    clean_pr_group = p.add_mutually_exclusive_group()
    clean_pr_group.add_argument("--clean-pr", action="store_true", default=None, dest="clean_pr",
                                help="Enable clean PR mode: strip non-essential artifacts before PR")
    clean_pr_group.add_argument("--no-clean-pr", action="store_false", dest="clean_pr",
                                help="Disable clean PR mode")
    p.add_argument("--tmux-persist", action="store_true", default=False,
                    help="Run agent interactively in a tmux window instead of headless (claude only)")
    p.add_argument("--bg", action="store_true", default=False,
                    help="Dispatch agent as a background session via claude agent view (claude only)")
    p.add_argument("--bg-agents", action="store_true", default=False,
                    help="Background sub-agents (via FACTORY_BG=1) while CEO runs in foreground")
    p.add_argument("--pr", type=int, default=None,
                    help="PR number for --mode review or --mode deep-qa (required when mode=review or mode=deep-qa)")
    p.add_argument("--repo", default=None,
                    help="Repository (owner/repo) for --mode review or --mode deep-qa (optional, defaults to current repo)")
    p.add_argument("--run-id", default=None, dest="run_id",
                    help="Use a specific run ID (e.g., UUID from external orchestrator). "
                         "First 8 chars are used for worktree naming")
    p.add_argument("--no-worktree", action="store_true", default=False, dest="no_worktree",
                    help="Run directly in the project directory without creating a worktree "
                         "(useful for testing in-flight branch changes)")
    p.add_argument("--overwrite", default=None, metavar="TEXT",
                    help="Natural-language directive to mutate the workflow for this session "
                         "(e.g. 'skip adversarial testing', 'add a lint step after build')")
    p.add_argument("--auto-approve", action="store_true", default=False,
                    help="Auto-approve user gates in design mode (skip interactive strategy review)")
    p.add_argument("--from-plan", default=None, metavar="PLAN_SOURCE", dest="from_plan",
                    help="Load an existing plan into design mode instead of running research. "
                         "Accepts a local file path, GitHub issue URL, issue number, or fuzzy search string. "
                         "Requires --mode design; mutually exclusive with --focus and --prompt")
    p.add_argument("--just-plan", action="store_true", default=False, dest="just_plan",
                    help="Plan-only mode: research + strategy + GitHub publishing, NO implementation. "
                         "Requires --mode design. Mutually exclusive with --from-plan and --prompt.")
    p.add_argument("--plugin", action="store_true", default=False,
                    help="Generate mode as a standalone pip-installable plugin package. "
                         "Requires --mode create. Output includes pyproject.toml with "
                         "factory.plugins entry point registration.")
    p.add_argument("--folder", default=None, metavar="PATH",
                    help="Output directory for plugin package (default: ./<mode-name>-plugin). "
                         "Only used with --plugin.")
    p.add_argument("--data-node", action="store_true", default=False, dest="data_node",
                    help="Wrap the created workflow in a DataNode for per-item data iteration. "
                         "Builder designs a per-item subgraph; DataNode wrapping is added "
                         "mechanically after. Requires --mode create.")
    p.add_argument("--task", default=None, metavar="TASK_REF", dest="task_ref",
                    help="Task class reference (name or module:Class format) providing "
                         "setup/verify hooks. Injects Task context (fields, sample prompt, "
                         "scoring contract) into the Builder directive. "
                         "Works with or without --data-node.")
    p.add_argument("--engine", choices=["skill", "tool", "deterministic"], default="skill",
                    help="Execution engine: skill (CEO follows SKILL.md, default), "
                         "tool (CEO drives via factory workflow tool commands), "
                         "deterministic (headless WorkflowExecutor, no CEO)")

    p = sub.add_parser("run", help="Run factory cycle (delegates to CEO agent)")
    p.add_argument("path", help="Project path, GitHub URL, idea file path, or prompt")
    p.add_argument(
        "--prompt", default=None,
        help="Path to a prompt/spec file (absolute or relative to project). "
             "Loaded as the build spec into .factory/strategy/current.md",
    )
    p.add_argument(
        "--mode",
        metavar="MODE",
        default="auto",
        help="Operating mode. Built-in: auto, design, create. "
             "Project-local: project:<name>",
    )
    p.add_argument(
        "--focus", default=None,
        help="Target a specific item: backlog name ('dashboard UI'), issue number (42), "
             "URL (https://github.com/o/r/issues/42), or shorthand (owner/repo#42). "
             "Issue refs are auto-detected and fetched via gh/glab CLI",
    )
    p.add_argument(
        "--discover-only", action="store_true", default=False,
        help="Only run discovery and review — do not chain into improve",
    )
    p.add_argument(
        "--no-github", action="store_true", default=False,
        help="Disable GitHub operations (issue creation, PR posting, cloning)",
    )
    p.add_argument(
        "--loop", action="store_true", default=False,
        help="Enable heartbeat mode: run continuously with sleep between cycles",
    )
    p.add_argument(
        "--interval", type=int, default=1800,
        help="Seconds to sleep between cycles (default: 1800)",
    )
    p.add_argument(
        "--max-cycles", type=int, default=None,
        help="Maximum number of cycles (default: unlimited)",
    )
    p.add_argument("--min-growth", type=int, default=None,
                    help="Minimum guaranteed growth hypotheses (default: 2)")
    p.add_argument("--max-new", type=int, default=None,
                    help="Max new items added to backlog per cycle (default: 2)")
    p.add_argument("--branch", default=None,
                    help="Target branch for PRs (default: from factory.md, fallback: main)")
    p.add_argument("--model", default=None,
                    help="Claude model for agent subprocesses (default: FACTORY_MODEL env var, or claude CLI default)")
    p.add_argument("--runner", default=None,
                    help="CLI backend to use (default: FACTORY_RUNNER env var, or 'claude')")
    p.add_argument("--profile", default=None,
                    help="Credential profile from ~/.factory/config.toml")
    p.add_argument("--use-profile", action="store_true", default=False,
                    help="Inject user profile (~/.factory/profile.md) into agent prompts")
    run_clean_pr_group = p.add_mutually_exclusive_group()
    run_clean_pr_group.add_argument("--clean-pr", action="store_true", default=None, dest="clean_pr",
                                    help="Enable clean PR mode: strip non-essential artifacts before PR")
    run_clean_pr_group.add_argument("--no-clean-pr", action="store_false", dest="clean_pr",
                                    help="Disable clean PR mode")
    p.add_argument("--tmux-persist", action="store_true", default=False,
                    help="Run agent interactively in a tmux window instead of headless (claude only)")
    p.add_argument("--bg", action="store_true", default=False,
                    help="Dispatch agent as a background session via claude agent view (claude only)")
    p.add_argument("--bg-agents", action="store_true", default=False,
                    help="Background sub-agents (via FACTORY_BG=1) while CEO runs in foreground")
    p.add_argument("--run-id", default=None, dest="run_id",
                    help="Use a specific run ID (e.g., UUID from external orchestrator). "
                         "First 8 chars are used for worktree naming")
    p.add_argument("--no-worktree", action="store_true", default=False, dest="no_worktree",
                    help="Run directly in the project directory without creating a worktree "
                         "(useful for testing in-flight branch changes)")
    p.add_argument("--engine", choices=["skill", "tool", "deterministic"], default="skill",
                    help="Execution engine: skill (CEO follows SKILL.md, default), "
                         "tool (CEO drives via factory workflow tool commands), "
                         "deterministic (headless WorkflowExecutor, no CEO)")
    p.add_argument("--overwrite", default=None, metavar="TEXT",
                    help="Natural-language directive to mutate the workflow for this session")
    p.add_argument("--auto-approve", action="store_true", default=False,
                    help="Auto-approve user gates in design mode (skip interactive strategy review)")

    p = sub.add_parser("tmux", help="Launch factory run in a detached tmux session")
    p.add_argument("path", help="Path to the project")
    p.add_argument("--session", default=None, help="Custom tmux session name")
    p.add_argument(
        "--mode",
        metavar="MODE",
        default="auto",
        help="Run mode (default: auto, respects in-flight cycle)",
    )
    p.add_argument("--loop", action="store_true", default=False, help="Enable loop mode")
    p.add_argument("--interval", type=int, default=1800, help="Loop interval in seconds")
    p.add_argument("--max-cycles", type=int, default=None, help="Max cycles for loop mode")
    p.add_argument("--attach", action="store_true", default=False,
                    help="Attach to session after creating")
    p.add_argument(
        "--no-github", action="store_true", default=False,
        help="Disable GitHub operations (issue creation, PR posting, cloning)",
    )
    p.add_argument("--model", default=None,
                    help="Claude model for agent subprocesses (default: FACTORY_MODEL env var, or claude CLI default)")
    p.add_argument("--runner", default=None,
                    help="CLI backend to use (default: FACTORY_RUNNER env var, or 'claude')")
    p.add_argument("--profile", default=None,
                    help="Credential profile from ~/.factory/config.toml")
    p.add_argument(
        "--focus", default=None,
        help="Target a specific item: backlog name, issue number, URL, or shorthand",
    )
    p.add_argument(
        "--refine", default=None, metavar="REQUEST",
        help="Refinement mode: classify and implement a user-directed change",
    )
    tmux_clean_pr = p.add_mutually_exclusive_group()
    tmux_clean_pr.add_argument("--clean-pr", action="store_true", default=None, dest="clean_pr",
                                help="Enable clean PR mode")
    tmux_clean_pr.add_argument("--no-clean-pr", action="store_false", dest="clean_pr",
                                help="Disable clean PR mode")
    p.add_argument(
        "--prompt", default=None,
        help="Path to a prompt/spec file",
    )
    p.add_argument("--branch", default=None,
                    help="Target branch for PRs")
    p.add_argument("--min-growth", type=int, default=None,
                    help="Minimum guaranteed growth hypotheses")
    p.add_argument("--max-new", type=int, default=None,
                    help="Max new items added to backlog per cycle")
    p.add_argument("--discover-only", action="store_true", default=False,
                    help="Only run discovery and review — do not chain into improve")
    p.add_argument("--bg-agents", action="store_true", default=False,
                    help="Background sub-agents (via FACTORY_BG=1) while CEO runs in foreground")
    p.add_argument("--tmux-persist", action="store_true", default=False,
                    help="Run agent interactively in a tmux window instead of headless (claude only)")
    p.add_argument("--use-profile", action="store_true", default=False,
                    help="Inject user profile (~/.factory/profile.md) into agent prompts")
    p.add_argument("--engine", choices=["skill", "tool", "deterministic"], default="skill",
                    help="Execution engine: skill (CEO follows SKILL.md, default), "
                         "tool (CEO drives via factory workflow tool commands), "
                         "deterministic (headless WorkflowExecutor, no CEO)")
    p.add_argument("--overwrite", default=None, metavar="TEXT",
                    help="Natural-language directive to mutate the workflow for this session")

    p = sub.add_parser("tmux-ls", help="List running factory tmux sessions")
    p.add_argument("--json", action="store_true", default=False, dest="json_output",
                    help="Output as JSON array for programmatic consumption")

    p = sub.add_parser("tmux-capture", help="Capture recent output from a factory tmux session")
    p.add_argument("path", nargs="?", default=None, help="Project path (derives session name)")
    p.add_argument("--session", default=None, help="Session name to capture from")
    p.add_argument("--lines", type=int, default=-100, help="Number of lines to capture (default: -100)")

    p = sub.add_parser("tmux-stop", help="Stop factory tmux session(s)")
    p.add_argument("--session", default=None, help="Session name to stop")
    p.add_argument("--path", default=None, help="Project path (derives session name)")
    p.add_argument("--all", action="store_true", default=False, dest="stop_all",
                    help="Stop ALL factory tmux sessions (required when no --session/--path given)")
    p.add_argument("--force", action="store_true", default=False,
                    help="Force-kill a session even if it's not in the factory registry")

    p = sub.add_parser("refactory", help="Launch the re:factory persistent supervisor agent")
    p.add_argument("path", nargs="?", default=None,
                    help="Project directory (default: current working directory)")
    p.add_argument("--reset", action="store_true", default=False,
                    help="Reset session (new session ID, fresh start)")
    p.add_argument("--model", default=None,
                    help="Claude model override")
    p.add_argument("--loop", action="store_true", default=False,
                    help="Enable workflow-tune loop: adds /workflow-tune skill for iterative tuning")

    from factory.cli.contained import build_contained_parser
    build_contained_parser(sub)

    from factory.workflow.cli import add_workflow_parser
    add_workflow_parser(sub)  # type: ignore[arg-type]
