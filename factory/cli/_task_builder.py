"""Build the CEO agent task string from mode and optional context."""

from __future__ import annotations

import re as _re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factory.messages import Message


def _slug(desc: str) -> str:
    slug = _re.sub(r"[^a-z0-9]+", "-", desc.lower().strip())
    return slug.strip("-")[:40]


def _mode_suffix(mode: str, discover_only: bool) -> str:
    _SIMPLE_MODE_SUFFIXES = {
        "design": (
            "\n\nRun Design mode: the universal entry point. Design mode handles build, "
            "discover, and improve workflows inline via its conditional gates. "
            "Follow the step-by-step playbook in your system prompt above."
        ),
        "meta": (
            "\n\nRun Meta mode: full self-improvement. First, run the complete Improve loop "
            "on this project (experiments, keep/revert decisions). Then run ACE playbook "
            "evolution for all agent roles using cross-project experiment data. "
            "The full step-by-step playbook is in your system prompt above."
        ),
        "research": (
            "\n\nRun Research mode: the project has a research target defined in factory.md. "
            "Read the research_target from config.json to understand the objective, metric, "
            "target value, and run command. Each cycle: form a hypothesis to improve the "
            "metric, implement the change within mutable_surfaces only (leave fixed_surfaces "
            "untouched), run the research command, compare results against the target, and "
            "make a keep/revert decision. Respect research_constraints and cost_budget. "
            "The full step-by-step playbook is in your system prompt above."
        ),
        "create": (
            "\n\nRun Create mode: this mode creates a new factory mode (workflow + skill + "
            "CLI wiring + tests) from the user's description above. "
            "The full step-by-step playbook is in your system prompt above."
        ),
        "founder": (
            "\n\nRun Founder mode: rapid prototyping — one hypothesis, one build, "
            "minimal verification. Pick the highest-leverage idea, prototype it fast, "
            "run tests once. No research, no code review, no adversarial QA, no eval "
            "scoring. Record the experiment and stop. This is NOT production-quality — "
            "run --mode improve afterward to harden what works. "
            "The full step-by-step playbook is in your system prompt above."
        ),
        "deep-research": (
            "\n\nRun Deep Research mode: single-agent iterative research with coverage checking. "
            "The workflow runs Study → deep_researcher (single agent with internal iteration) → "
            "CEO coverage gate. "
            "The researcher performs multiple rounds of WebSearch/WebFetch internally, "
            "following an inside-out protocol: internal project state first, then external "
            "search shaped by internal findings. Includes faithfulness checks every iteration. "
            "The coverage gate is a safety net — it should almost always PROCEED. "
            "The mode outputs only research-combined.md. "
            "If --focus is provided, it defines the research topic. Otherwise, research the "
            "project's domain broadly. Terminal mode — does not chain to build or improve. "
            "The full step-by-step playbook is in your system prompt above."
        ),
        "study": (
            "\n\nRun Study mode: analyze the codebase structure and dependency graph. "
            "Update the code knowledge graph, then run factory study for observations "
            "with structural graph context included. "
            "Terminal mode — does not chain to other modes."
        ),
    }
    if mode == "discover":
        if discover_only:
            return (
                "\n\nRun Discover mode: introspect the project, auto-detect eval dimensions, "
                "and generate the eval harness. Then complete Review mode to initialize the "
                "factory. Do NOT run the Improve loop."
            )
        return (
            "\n\nRun Discover mode: introspect the project, auto-detect eval dimensions, "
            "and generate the eval harness. Then complete Review mode: verify the eval "
            "harness works, mark as reviewed, and initialize the factory. "
            "After initialization, proceed to Improve mode for one experiment cycle."
        )
    if mode in _SIMPLE_MODE_SUFFIXES:
        return _SIMPLE_MODE_SUFFIXES[mode]
    return (
        f"\n\nRun {mode} mode. Follow the step-by-step playbook in your system prompt "
        f"exactly as written — do not add additional steps, research, or ceremony "
        f"beyond what the playbook describes."
    )


def _append_focus_directive(
    focus: str | None,
    mode: str,
    create_description: str | None,
    issue_numbers: list[int] | None,
    issue_urls: list[str] | None,
    issue_number: int | None,
    issue_url: str | None,
) -> str:
    if not focus or create_description or mode == "deep-research":
        return ""
    _issue_numbers = issue_numbers or []
    _issue_urls = issue_urls or []
    result = f"\n\n## Focus Directive (Targeted Mode)\n\nTarget: {focus}\n\n"
    if _issue_numbers:
        issue_labels = []
        for i, num in enumerate(_issue_numbers):
            label = f"#{num}"
            if i < len(_issue_urls) and _issue_urls[i]:
                label += f" ({_issue_urls[i]})"
            issue_labels.append(label)
        result += (
            f"These targets are from issues {', '.join(issue_labels)}. "
            f"All issue specs have been written to `.factory/strategy/current.md`. "
            f"Read it for the complete requirements.\n\n"
        )
    elif issue_number:
        issue_label = f"#{issue_number}"
        if issue_url:
            issue_label += f" ({issue_url})"
        result += (
            f"This target is from issue {issue_label}. "
            f"The full issue spec has been written to `.factory/strategy/current.md`. "
            f"Read it for the complete requirements.\n\n"
        )
    result += (
        "Single-item mode. This target has been added to the backlog. "
        "The Strategist must generate exactly ONE hypothesis for this item. "
        "No other hypotheses this cycle — no additional backlog clearing, no new items.\n"
        "After this single experiment completes (keep or revert), skip to final archival. "
        "Do not loop back for more hypotheses.\n"
    )
    if _issue_numbers:
        nums_str = ", ".join(f"#{n}" for n in _issue_numbers)
        finalize_flags = " ".join(f"--issue {n}" for n in _issue_numbers)
        result += (
            f"\n## Issue Tracking\n\n"
            f"This cycle is working on issues {nums_str}. "
            f"When finalizing, pass `{finalize_flags}` to `factory finalize`."
        )
    elif issue_number:
        result += (
            f"\n## Issue Tracking\n\n"
            f"This cycle is working on issue #{issue_number}. "
            f"When finalizing, pass `--issue {issue_number}` to `factory finalize`."
        )
    return result


def _append_deep_research_topic(task: str, focus: str) -> str:
    return task + (
        f"\n\n## Research Topic\n\n"
        f"**Topic:** {focus}\n\n"
        f"Focus all research on this specific topic. The deep researcher investigates "
        f"this topic using the inside-out protocol: internal project context first, "
        f"then targeted external search. The coverage gate evaluates completeness "
        f"against this topic.\n"
    )


def _build_ceo_task(
    project_path: Path,
    mode: str,
    context: str | None = None,
    focus: str | None = None,
    prompt_file: str | None = None,
    min_growth: int | None = None,
    max_new: int | None = None,
    branch: str | None = None,
    discover_only: bool = False,
    no_github: bool = False,
    design_idea: str | None = None,
    design_existing: bool = False,
    research_ideation: str | None = None,
    messages: list[Message] | None = None,
    issue_number: int | None = None,
    issue_url: str | None = None,
    issue_numbers: list[int] | None = None,
    issue_urls: list[str] | None = None,
    refine_request: str | None = None,
    clean_pr: bool = False,
    display_mode: str | None = None,
    create_description: str | None = None,
    update_existing_mode: str | None = None,
    plugin_mode: bool = False,
    plugin_folder: str | None = None,
    data_node: bool = False,
    task_ref: str | None = None,
    task_context: dict | None = None,
    from_plan: str | None = None,
    from_plan_feedback: list[str] | None = None,
    just_plan: bool = False,
    auto_approve: bool = False,
) -> str:
    """Build the CEO agent task string from mode and optional context."""
    shown_mode = display_mode if display_mode is not None else mode
    task = f"Project: {project_path}\nMode: {shown_mode}"

    if messages:
        task += "\n\n## User Messages\n"
        task += "The user has sent the following directives. Treat these as HIGH PRIORITY:\n\n"
        for msg in messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            task += f"**[{ts}]** {msg.text}\n\n"

    if from_plan:
        task += (
            "\n\n## Plan Loop (From Existing Plan)\n\n"
            "An existing plan has been loaded via `--from-plan`.\n"
            "The plan content is at `.factory/strategy/current.md`.\n\n"
            "**Skip the Research phase.** But DO run the Strategist in reconciliation mode.\n\n"
        )
        if from_plan_feedback:
            task += (
                "Thread feedback exists (saved at `.factory/strategy/thread-feedback.md`):\n\n"
                "1. Read the plan at `.factory/strategy/current.md`\n"
                "2. Read the thread feedback at `.factory/strategy/thread-feedback.md`\n"
                "3. Run the Strategist with task: "
                "'Reconcile this plan with the following thread feedback. "
                "Update the plan to address the feedback. "
                "Write the reconciled plan to .factory/strategy/current.md.'\n"
                "4. Present the RECONCILED plan to the user for approval\n"
                "5. On approval → proceed to Builder\n\n"
            )
        else:
            task += (
                "No thread feedback exists.\n\n"
                "1. Read the plan at `.factory/strategy/current.md`\n"
                "2. Present it to the user for approval (no Strategist needed)\n"
                "3. On approval → proceed to Builder\n\n"
            )
        task += (
            "Do NOT run parallel researchers. Do NOT regenerate the plan from scratch. "
            "The plan content has already been resolved and persisted.\n"
        )
    elif just_plan:
        task += (
            "\n\n## Plan Loop (Just Plan)\n\n"
            "**just_plan: true**\n\n"
            "Run the full Plan mode workflow: research + strategy + approval + GitHub publish.\n\n"
            "1. Check for prior plans (GitHub issues with plan label, .factory/archive/)\n"
            "2. Run 3 parallel researchers (domain, practices, constraints)\n"
            "3. CEO review gate\n"
            "4. Strategist synthesizes phased plan\n"
            "5. Single user approval gate: Keep this plan?\n"
            "6. On approval: publish to GitHub + seed backlog\n\n"
            "Terminal mode — do NOT transition to build or improve.\n"
            "\n### Post-Approval: GitHub Publish (MANDATORY)\n\n"
            "After the user approves the plan, you MUST:\n\n"
            "1. Create the plan label if it does not exist: "
            '`gh label create plan --description "Approved plan" --color 0366d6 --force`\n'
            "2. If --focus targets a GitHub issue number, post the plan as a comment on that issue "
            "and add the plan label:\n"
            "   - `gh issue comment <NUMBER> --body-file .factory/strategy/current.md`\n"
            "   - `gh issue edit <NUMBER> --add-label plan`\n"
            "3. Otherwise, create a new issue with the plan label:\n"
            '   - `gh issue create --title "Plan: <focus>" --body-file .factory/strategy/current.md --label plan`\n'
            "4. Seed the backlog: extract phase headers from current.md and append to backlog.md\n\n"
            "Do NOT skip this step. Do NOT exit without publishing.\n"
        )
    elif design_existing and mode == "design-v2":
        task += "\n\n## Plan Loop (Interactive)\n\n"
        task += "**existing_project: true**\n\n"
        task += (
            f"You are in design-v2 mode on an existing project at `{project_path}`.\n"
            "Follow the design-v2 SKILL.md playbook: Research Director, "
            "Strategy Director, Synthesize, Design Doc, then user approval.\n"
        )
        if focus:
            task += (
                f"\n**Focus topic (from --focus):** {focus}\n\n"
                f"The user wants to discuss this specific topic. Use it to seed the "
                f"research and spec, but be open to the user redirecting.\n"
            )
        else:
            task += (
                "\nNo specific topic was provided. Study the project broadly — "
                "look at the backlog, eval scores, open issues, and recent history — "
                "then present your findings and recommendations.\n"
            )
    elif design_existing:
        task += (
            f"\n\n## Plan Loop (Interactive)\n\n"
            f"**existing_project: true**\n\n"
            f"You are in interactive planning mode on an **existing project** "
            f"at `{project_path}`.\n\n"
            f"Run the Plan Loop (P0-P3) with interactive approval. Research the "
            f"project (local study + external best practices), synthesize an "
            f"improvement spec through user feedback. After you approve the plan "
            f"at the strategy gate, the workflow continues to implementation "
            f"automatically.\n\n"
        )
        if focus:
            task += (
                f"**Focus topic (from --focus):** {focus}\n\n"
                f"The user wants to discuss this specific topic. Use it to seed the "
                f"research and spec, but be open to the user redirecting.\n"
            )
        else:
            task += (
                "No specific topic was provided. Study the project broadly — "
                "look at the backlog, eval scores, open issues, and recent history — "
                "then present your findings and recommendations.\n"
            )
    elif design_idea and mode == "design-v2":
        task += (
            f"\n\n## Plan Loop (Interactive)\n\n"
            f"**Raw idea from user:** {design_idea}\n\n"
            f"You are in design-v2 mode with a new idea.\n"
            f"Follow the design-v2 SKILL.md playbook: Research Director, "
            f"Strategy Director, Synthesize, Design Doc, then user approval.\n\n"
            f"After you approve the plan at the strategy gate, persist it to "
            f".factory/strategy/current.md — the workflow continues to "
            f"implementation automatically.\n"
        )
    elif design_idea:
        task += (
            f"\n\n## Plan Loop (Interactive)\n\n"
            f"**Raw idea from user:** {design_idea}\n\n"
            f"Run the Plan Loop (P0-P3) with interactive approval. "
            f"Research the space, synthesize a build plan, and refine it "
            f"through user feedback before building.\n\n"
            f"After you approve the plan at the strategy gate, persist it to "
            f".factory/strategy/current.md — the workflow continues to "
            f"implementation automatically.\n"
        )

    if research_ideation:
        task += (
            f"\n\n## Plan Loop (Interactive)\n\n"
            f"**Raw idea from user:** {research_ideation}\n\n"
            f"**research_project: true**\n\n"
            f"Run the Plan Loop (P0-P3) with interactive approval. "
            f"This is a research project — the Strategist MUST collect research configuration:\n"
            f"- Research Target (objective, metric, target value, run_command, result_path)\n"
            f"- Mutable Surfaces (files the Builder can modify)\n"
            f"- Fixed Surfaces (ground truth / eval files that must never be touched)\n"
            f"- Research Constraints (additional rules)\n"
            f"- Cost Budget (optional)\n\n"
            f"After the user approves, persist the spec AND the research "
            f"config to .factory/strategy/current.md, then proceed to Build mode. "
            f"During Review mode (factory.md creation), populate the research sections "
            f"from the approved spec.\n"
        )

    if create_description and update_existing_mode:
        task += (
            f"\n\n## Create Mode (Update Existing Mode)\n\n"
            f"**Target mode:** {update_existing_mode}\n"
            f"**Requested changes:** {create_description}\n\n"
            f"You are updating an EXISTING factory workflow mode, not creating a new one.\n\n"
            f"**Before making any changes:**\n"
            f"1. Read the existing workflow definition: `factory workflow show {update_existing_mode}`\n"
            f"2. Read the current SKILL.md: `cat skills/workflow-{update_existing_mode}/SKILL.md`\n"
            f"3. Understand the current behavior before modifying it.\n\n"
            f"**After implementing changes, verify ALL 20 registration points:**\n"
            f"1. `factory workflow validate {update_existing_mode}` passes (exit 0)\n"
            f"2. `factory workflow show {update_existing_mode}` reflects the changes\n"
            f"3. `factory workflow export-skills --verify` succeeds\n"
            f"4. SKILL.md under skills/workflow-{update_existing_mode}/ is regenerated\n"
            f"5. WORKFLOW_META description in skill_export.py is still accurate\n"
            f"6. CLI help text (factory ceo --help) still lists the mode correctly\n"
            f"7. register_all() entry still resolves\n"
            f"8. CycleState.mode Literal in models.py still includes the mode\n"
            f"9. CEO_MODES and RUN_MODES in _helpers.py still include the mode\n"
            f"10. CEO prompt (ceo.md) mode detection table is still correct\n"
            f"11. All existing tests for this mode still pass\n"
            f"12. No import errors in any factory module\n"
            f"13. __all__ in definitions.py still exports the workflow function\n"
            f"14. factory/workflow/registry.py resolves the mode\n"
            f"15. factory/skill_cache.py will auto-invalidate (no action needed, but verify)\n"
            f"16. CLAUDE.md mentions the mode correctly\n"
            f"17. workflow/README.md references are accurate\n"
            f"18. Trigger function still returns True for the correct context\n"
            f"19. Start node is still valid and reachable from all edges\n\n"
            f"Follow the Create workflow playbook in skills/workflow-create/SKILL.md.\n"
        )
    elif create_description and plugin_mode:
        folder = plugin_folder if plugin_folder else f"./{_slug(create_description)}-plugin"
        task += (
            f"\n\n## Create Mode (Plugin Package)\n\n"
            f"**Mode description from user:**\n{create_description}\n\n"
            f"**plugin_mode:** true\n"
            f"**output_folder:** {folder}\n\n"
            f"You are creating a PLUGIN workflow — a standalone pip-installable package.\n\n"
            f"**Package structure:**\n"
            f"```\n"
            f"{folder}/\n"
            f"├── pyproject.toml         # Package metadata + entry point\n"
            f"├── README.md              # Installation and usage instructions\n"
            f"└── <mode_name>.py         # Workflow definition + registration\n"
            f"```\n\n"
            f"**pyproject.toml requirements:**\n"
            f"- Build system: hatchling\n"
            f"- Package name: `factory-<mode-name>-workflow`\n"
            f"- Version: `0.1.0`\n"
            f"- `requires-python = '>=3.11'`\n"
            f"- Dependencies: `['remote-factory']` (no version pin)\n"
            f"- Entry point group: `[project.entry-points.'factory.plugins']`\n"
            f"- Entry point value: `<mode_name> = '<mode_name>:register_plugin'`\n\n"
            f"**Workflow file (`<mode_name>.py`) requirements:**\n"
            f"- `meta` dict with `name` and `description` keys\n"
            f"- `workflow()` function returning a `Workflow` object\n"
            f"- `register_plugin(registry)` function that calls:\n"
            f"  - `registry.add_modes([meta['name']])`\n"
            f"  - `registry.add_workflow_search_path(str(Path(__file__).parent))`\n"
            f"- Only import from `factory.workflow.primitives` and stdlib\n"
            f"- NO imports from other factory internals\n\n"
            f"**README.md content:**\n"
            f"- Project description\n"
            f"- Installation: `pip install -e {folder}/`\n"
            f"- Usage: `factory ceo /path/to/project --mode <mode-name>`\n"
            f"- Verification: `factory workflow list`, `factory workflow validate <name>`\n\n"
            f"**Verification steps (Builder MUST run all):**\n"
            f"1. Create the output directory: `mkdir -p {folder}`\n"
            f"2. Write `pyproject.toml` with correct entry point\n"
            f"3. Write `<mode_name>.py` with `meta` + `workflow()` + `register_plugin()`\n"
            f"4. Write `README.md` with installation instructions\n"
            f"5. Install locally: `pip install -e {folder}/`\n"
            f"6. Verify discovery: `factory workflow list` (should show the new mode)\n"
            f"7. Validate graph: `factory workflow validate <mode-name>`\n"
            f"8. Clean up: `pip uninstall -y factory-<mode-name>-workflow`\n\n"
            f"**Constraints:**\n"
            f"- Do NOT modify `factory/workflow/definitions.py` or any upstream factory files\n"
            f"- Do NOT use `src/` layout — flat layout with workflow `.py` at package root\n"
            f"- Do NOT `git init` the output directory\n"
            f"- Do NOT commit the plugin to the factory repo or open a PR — "
            f"the plugin package stays in the output directory as a standalone artifact\n"
            f"- Do NOT include a `tests/` directory (users can add their own later)\n\n"
            f"Follow the Create workflow playbook in skills/workflow-create/SKILL.md.\n"
        )

    elif create_description:
        task += (
            f"\n\n## Create Mode (New Portable Workflow)\n\n"
            f"**Mode description from user:**\n{create_description}\n\n"
            f"**output_folder:** .factory/workflows/\n\n"
            f"You are in Create mode — creating a new portable workflow.\n\n"
            f"Follow the Create workflow playbook in your system prompt:\n"
            f"1. Research existing workflow patterns and the user's intent\n"
            f"2. Synthesize a complete workflow specification\n"
            f"3. Present the spec to the user for interactive approval\n"
            f"4. Implement: write the workflow file to the **output_folder** above\n"
            f"5. QA verification (graph validates, SKILL.md generates)\n"
            f"6. Open PR for review\n\n"
            f"The workflow file goes in the **output_folder** specified above. "
            f"Do NOT modify factory/workflow/definitions.py, register_all(), "
            f"WORKFLOW_META, or CLI wiring — portable workflows are auto-discovered "
            f"by the registry.\n"
        )

    # --- DataNode directive: Builder creates workflow WITH DataNode ---
    if create_description and data_node:
        task += (
            "\n\n## DataNode Directive\n\n"
            "**data_node:** true\n\n"
            "Your workflow MUST include a **DataNode** + **JoinNode** connected by "
            "real edges.  The DataNode forks one branch per data item; the JoinNode "
            "waits for all branches.\n\n"
            "**How to build the workflow:**\n"
            "1. Design your per-item processing pipeline (the branch) — these are "
            "the nodes that process ONE data item\n"
            "2. Identify the branch entry node (first node) and exit node (last node)\n"
            "3. Create a DataNode with `id='data'` (no source — late-bound at runtime)\n"
            "4. Create a JoinNode with `sources=[<exit_node_id>]`\n"
            "5. Add edges: DataNode → entry, ..., exit → JoinNode\n"
            "6. Set the DataNode as the workflow's `start_node`\n\n"
            "**Example structure:**\n"
            "```python\n"
            "from factory.workflow.primitives import (\n"
            "    DataNode, JoinNode, AgentNode, Edge, Workflow\n"
            ")\n\n"
            "nodes = {\n"
            "    'data': DataNode(id='data'),\n"
            "    'process': AgentNode(id='process', role=AgentRole.BUILDER, ...),\n"
            "    'evaluate': AgentNode(id='evaluate', role=AgentRole.HEALTH_CHECKER, ...),\n"
            "    'join': JoinNode(id='join', sources=['evaluate']),\n"
            "}\n"
            "edges = [\n"
            "    Edge(source='data', target='process'),\n"
            "    Edge(source='process', target='evaluate'),\n"
            "    Edge(source='evaluate', target='join'),\n"
            "]\n"
            "workflow = Workflow(name='...', nodes=nodes, edges=edges, start_node='data')\n"
            "```\n\n"
            "**How branch nodes access the current item:**\n"
            "Before each branch invocation, the runtime writes the current "
            "item to `.factory/current_item.json`. Your agent nodes MUST read this file "
            "to get the item data.\n\n"
            "The JSON file has this structure::\n"
            "```json\n"
            "{\n"
            '  "id": "item-001",\n'
            '  "path": "/data/items/item-001" or null,\n'
            '  "metadata": {"key": "value", ...},\n'
            '  "prompt": "The full prompt text for this item"\n'
            "}\n"
            "```\n\n"
            "**Constraints:**\n"
            "- The DataNode MUST be the start_node\n"
            "- Use real edges from DataNode → branch entry, branch exit → JoinNode\n"
            "- Do NOT set source_path, task_ref, or inline_items on the DataNode — "
            "the data source is late-bound (resolved at runtime)\n"
            "- The branch must have exactly one entry node and one exit node\n"
            "- Do NOT hardcode data file paths\n"
        )

    # --- Task directive: Builder receives Task contract context ---
    if create_description and task_context:
        tc = task_context
        task += (
            "\n\n## Task Directive\n\n"
            f"**task_ref:** {tc.get('ref_string', task_ref or '(unknown)')}\n\n"
            f"A Task class provides custom setup/verify hooks for this workflow.\n\n"
            f"**Task:** {tc.get('name', 'unknown')}\n"
        )
        if tc.get("description"):
            task += f"**Description:** {tc['description']}\n"
        task += "\n"

        # Field names from instance metadata
        if tc.get("field_names"):
            fields_str = ", ".join(f"`{f}`" for f in tc["field_names"])
            task += (
                f"**Instance fields:** {fields_str}\n"
                "Each data instance provides these metadata fields to your workflow.\n\n"
            )

        # Sample prompt output
        if tc.get("sample_prompt"):
            # Truncate very long prompts
            sample = tc["sample_prompt"]
            if len(sample) > 1500:
                sample = sample[:1500] + "\n... (truncated)"
            task += (
                "**Sample prompt (what the agent will receive per item):**\n"
                f"```\n{sample}\n```\n\n"
            )

        # Scoring contract
        scoring_method = tc.get("scoring_method", "exit_code")
        task += "**Scoring contract:**\n"
        if scoring_method == "json":
            metric_path = tc.get("scoring_metric_path", "score")
            task += (
                f"- Method: JSON scoring — your workflow's output is scored by "
                f"extracting `{metric_path}` from JSON output\n"
            )
        else:
            task += (
                "- Method: exit_code scoring — exit code 0 = pass, non-zero = fail\n"
            )
        if tc.get("verify_command"):
            task += f"- Verify command: `{tc['verify_command']}`\n"
        if tc.get("timeout"):
            task += f"- Timeout: {tc['timeout']}s\n"
        task += "\n"

        task += (
            "**Your responsibility:**\n"
            "- Design workflow nodes that process the Task's prompt output\n"
            "- Ensure your workflow's output is compatible with the Task's verify hook\n\n"
            "**NOT your responsibility (handled by the Task class):**\n"
            "- Setup logic (Task.setup() runs before your workflow per item)\n"
            "- Verification logic (Task.verify() runs after your workflow per item)\n"
            "- Instance generation (Task.instances() provides data items)\n"
        )

    if prompt_file:
        task += (
            f"\n\n## Directive\n\n"
            f"The user has provided a specific prompt file (`{prompt_file}`) as the build spec. "
            f"This is your primary instruction — read it at `.factory/strategy/current.md` and "
            f"execute exactly what it describes. Do not infer or improvise beyond what the prompt asks for."
        )

    if mode == "deep-research" and focus:
        task = _append_deep_research_topic(task, focus)

    task += _append_focus_directive(
        focus,
        mode,
        create_description,
        issue_numbers,
        issue_urls,
        issue_number,
        issue_url,
    )

    if branch:
        task += (
            f"\n\n## Branch Override\n\n"
            f"Target branch for all PRs and merges: `{branch}`\n"
            f"The Builder should create experiment branches from `{branch}` and "
            f"target PRs against `{branch}`. After revert, checkout `{branch}` instead of main.\n"
        )

    if any(v is not None for v in (min_growth, max_new)):
        budget_lines = ["\n\n## Budget Override\n"]
        budget_lines.append("The user has overridden the hypothesis budget for this run:")
        if min_growth is not None:
            budget_lines.append(f"- **min_growth:** {min_growth} (guaranteed growth hypotheses)")
        if max_new is not None:
            budget_lines.append(
                f"- **max_new:** {max_new} (max new items added to backlog per cycle)"
            )
        budget_lines.append("")
        budget_lines.append(
            "Pass these overrides to the Strategist. They take precedence over "
            "factory.md defaults and study-computed values."
        )
        task += "\n".join(budget_lines)

    if context:
        task += f"\n\n## Project Specification\n\n{context}"

    task += _mode_suffix(mode, discover_only)

    if no_github:
        task += (
            "\n\n## GitHub Operations Disabled\n\n"
            "The user has passed --no-github. Do NOT:\n"
            "- Create issues on GitHub\n"
            "- Create or post pull requests\n"
            "- Push to remote repositories\n"
            "- Clone from GitHub URLs\n\n"
            "Work locally only. When a GitHub operation would normally occur, "
            "skip it and note what was skipped in the experiment log."
        )

    if refine_request:
        task += (
            f"\n\n## Refinement Mode\n\n"
            f"**User's refinement request:** {refine_request}\n\n"
            f"You are in Refinement mode. Follow the `Mode: Refine` section in your "
            f"system prompt. The pipeline is:\n\n"
            f"1. Spawn the Refiner agent to classify and scope the request\n"
            f"2. If Tier 3 → exit, tell user to use full Improve mode\n"
            f"3. Begin experiment, create GitHub issue from Refiner's scoped task\n"
            f"4. Spawn Builder with the Refiner's task description\n"
            f"5. Run the FULL review pipeline (2d-review through 2h-final) — identical to Improve mode\n"
            f"6. Keep/revert verdict + finalize\n"
            f"7. Archivist (single batch)\n\n"
            f"Do NOT skip the review pipeline. Do NOT abbreviate any step.\n"
        )

    if auto_approve:
        task += (
            "\n\n## Auto-Approve Mode\n\n"
            "auto_approve: true\n\n"
            "At user approval gates (like gate_strategy), you act as the user:\n"
            "1. Read the plan at .factory/strategy/current.md\n"
            "2. Read the user's original intent at .factory/strategy/user-intent.md\n"
            "3. Compare: does the plan match what the user asked for?\n"
            "4. If YES: approve and proceed (say \"Approved\" and continue to the next step)\n"
            "5. If NO: provide specific feedback about what's missing or wrong, then reloop\n\n"
            "You are the CEO acting on behalf of the user. Apply judgment — approve good plans, "
            "reject bad ones. Do NOT blindly approve everything. Do NOT wait for human input.\n"
        )

    if clean_pr:
        task += (
            "\n\n## Clean PR Mode\n\n"
            "Clean PR mode is ACTIVE. After the final review gate (2h-final), "
            "run step 2i-clean before marking the PR ready:\n\n"
            "```bash\n"
            "factory clean-pr $PROJECT_PATH --exp $EXP_ID\n"
            "```\n\n"
            "This strips non-essential artifacts (eval scripts, benchmarks, .factory files) "
            "from the PR while preserving the full diff in the experiment archive. "
            "If stripping breaks tests, fall back to the full diff.\n"
        )

    return task
