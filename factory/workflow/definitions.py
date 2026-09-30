"""Workflow definitions as Python functions returning Workflow objects.

W₁: Build Mode
W₂: Design Mode (= W₁ with user gate at strategy approval)
W₉: Create Mode (meta-mode for creating new factory modes)
W₁₃: Spec Generate Mode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    ArtifactCheck,
    Edge,
    FnNode,
    ProjectState,
    ForkNode,
    GateNode,
    JoinNode,
    Study,
    VerdictType,
    Workflow,
)

# Re-export for test convenience
__all__ = [
    "DOC_FRESHNESS_GATE_PROMPT",
    "_GRAPH_EXPLORER_PROMPT",
    "_graph_explorer_prompt",
    "ResearcherConfig",
    "_deep_qa_subgraph",
    "_get_builtin_registry",
    "_research_subgraph",
    "_study_subgraph",
    "build_workflow",
    "create_workflow",
    "design_workflow",
    "register_all",
    "spec_generate_workflow",
]

DOC_FRESHNESS_GATE_PROMPT = (
    "Check the PR diff for documentation freshness. "
    "If public APIs, CLI commands, configuration options, "
    "or architecture were changed or added, corresponding documentation "
    "(README.md, CLAUDE.md, docstrings, --help text, or doc/ files) "
    "MUST be updated. PROCEED if docs are current or no doc-worthy changes "
    "exist. RELOOP to builder if documentation is stale — specify exactly "
    "which changes need doc updates."
)


# ── Study subgraph helper ───────────────────────────────────────


_GRAPH_EXPLORER_PROMPT = (
    "Explore the project's code knowledge graph to build structural understanding. "
    "Read .factory/strategy/observations.md for focus context.\n\n"
    "**Step 0 — detect graph availability:** Your working directory is already "
    "the project root. The graph file lives at `{project_path}/graph.json` "
    "(NOT inside `.factory/`). "
    "Run this smoke check FIRST — use a relative path since your CWD is the "
    "project root: "
    "`test -f graph.json && echo 'GRAPH AVAILABLE' || echo 'NO GRAPH'` — "
    "if the output says GRAPH AVAILABLE, proceed with the graph commands below. "
    "If the output says NO GRAPH, skip to the fallback section.\n\n"
    "**If the graph IS available:**\n"
    '1. Run `factory graph query "{project_path}" "<focus from observations>" --depth 2` '
    "to find relevant nodes\n"
    '2. Run `factory graph explain "{project_path}" "<key node>"` on the most important '
    "nodes to understand their connections and dependencies\n"
    '3. Run `factory graph path "{project_path}" "<A>" "<B>"` to trace dependency paths '
    "between key components\n"
    "4. Write structured findings to .factory/strategy/graph-context.md covering: "
    "key modules and their relationships, dependency paths, architectural layers, "
    "entry points and hotspots\n\n"
    "**If the graph is NOT available**, fall back to direct file exploration:\n"
    "1. Use `find . -name '*.py' | head -50` to discover source files\n"
    "2. Use `grep -rn 'class \\|def ' --include='*.py' | head -100` to map functions and classes\n"
    "3. Use `grep -rn 'import ' --include='*.py' | head -100` to trace dependencies\n"
    "4. Write the same structured findings to .factory/strategy/graph-context.md"
)


def _graph_explorer_prompt(focus: str | None = None) -> str:
    """Return the graph_explorer prompt, optionally scoped to *focus*."""
    if not focus:
        return _GRAPH_EXPLORER_PROMPT
    return (
        f"Focus your exploration on: {focus}\n\n"
        "Explore the project's code knowledge graph targeting the area above. "
        "Read .factory/strategy/observations.md for additional context.\n\n"
        "If graphify is installed and graph.json exists:\n"
        f'1. Run `factory graph query "{focus}" --depth 2` to find relevant nodes\n'
        '2. Run `factory graph explain "<key node>"` on the most important nodes to understand '
        "their connections and dependencies\n"
        '3. Run `factory graph path "<A>" "<B>"` to trace dependency paths between key components\n'
        "4. Write structured findings to .factory/strategy/graph-context.md covering: "
        "key modules and their relationships, dependency paths, architectural layers, "
        "entry points and hotspots\n\n"
        "If graphify is NOT installed or graph.json is missing, fall back to direct file exploration:\n"
        "1. Use `find . -name '*.py' | head -50` to discover source files\n"
        "2. Use `grep -rn 'class \\|def ' --include='*.py' | head -100` to map functions and classes\n"
        "3. Use `grep -rn 'import ' --include='*.py' | head -100` to trace dependencies\n"
        "4. Write the same structured findings to .factory/strategy/graph-context.md"
    )


def _study_subgraph(
    *,
    focus: str | None = None,
) -> tuple[dict[str, Any], list[Edge]]:
    """Return (nodes, internal_edges) for the graph-powered study chain.

    Four nodes run sequentially:

        graph_update → study → graph_explorer → concat_study

    The caller wires the entry edge (→ graph_update) and exit edge
    (concat_study →) into the surrounding workflow.
    """
    nodes: dict[str, Any] = {}

    nodes["graph_update"] = FnNode(
        id="graph_update",
        command="factory graph update {project_path}",
        notes="Extract or incrementally update the code knowledge graph before study.",
        writes={"graph.json"},
    )

    nodes["study"] = Study(
        id="study",
        command="factory study {project_path}",
        writes={".factory/strategy/observations.md"},
        focus=focus,
    )

    nodes["graph_explorer"] = AgentNode(
        id="graph_explorer",
        role=AgentRole.RESEARCHER,
        prompt_template=_graph_explorer_prompt(focus),
        reads={".factory/strategy/observations.md"},
        writes={".factory/strategy/graph-context.md"},
    )

    nodes["concat_study"] = FnNode(
        id="concat_study",
        command=(
            "cat {project_path}/.factory/strategy/observations.md"
            " {project_path}/.factory/strategy/graph-context.md"
            " > {project_path}/.factory/strategy/study-combined.md"
        ),
        reads={".factory/strategy/observations.md", ".factory/strategy/graph-context.md"},
        writes={".factory/strategy/study-combined.md"},
    )

    internal_edges = [
        Edge(source="graph_update", target="study"),
        Edge(source="study", target="graph_explorer"),
        Edge(source="graph_explorer", target="concat_study"),
    ]

    return nodes, internal_edges


# ── Deep-QA subgraph helper ─────────────────────────────────────


def _deep_qa_subgraph(
    *,
    code_reviewer_extra: str = "",
    adversarial_extra: str = "",
) -> tuple[dict[str, Any], list[Edge]]:
    """Return (nodes, internal_edges) for the parallel deep-qa verification subgraph.

    Three specialist agents run in parallel via fork/join:

        fork_qa → [health_checker, code_reviewer, adversarial_tester] → join_qa

    Agent prompts live in their role .md files; prompt_template is only set
    when a workflow passes extra context via code_reviewer_extra / adversarial_extra.
    The caller wires the entry edge (→ fork_qa) and the exit edge
    (join_qa →) into the surrounding workflow.
    """
    nodes: dict[str, Any] = {}

    nodes["health_checker"] = AgentNode(
        id="health_checker",
        role=AgentRole.HEALTH_CHECKER,
        prompt_template="Run health checks on the builder output.",
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/health-check.md"},
    )

    nodes["code_reviewer"] = AgentNode(
        id="code_reviewer",
        role=AgentRole.CODE_REVIEWER,
        prompt_template=code_reviewer_extra or "Review the builder output for code quality.",
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/code-review.md"},
    )

    nodes["adversarial_tester"] = AgentNode(
        id="adversarial_tester",
        role=AgentRole.ADVERSARIAL_TESTER,
        timeout=1800,
        prompt_template=adversarial_extra or "Run adversarial tests on the builder output.",
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/adversarial-qa.md"},
    )

    nodes["fork_qa"] = ForkNode(
        id="fork_qa",
        targets=["health_checker", "code_reviewer", "adversarial_tester"],
    )

    nodes["join_qa"] = JoinNode(
        id="join_qa",
        sources=["health_checker", "code_reviewer", "adversarial_tester"],
        reads={
            ".factory/reviews/health-check.md",
            ".factory/reviews/code-review.md",
            ".factory/reviews/adversarial-qa.md",
        },
    )

    internal_edges = [
        Edge(source="fork_qa", target="join_qa"),
    ]

    return nodes, internal_edges


# ── Research subgraph helper ───────────────────────────────────


@dataclass(frozen=True)
class ResearcherConfig:
    """Configuration for a single researcher in a parallel research fork."""

    id: str
    prompt_template: str
    post_check_min_size: int | None = None


def _research_subgraph(
    *,
    researchers: list[ResearcherConfig],
    gate_prompt: str,
) -> tuple[dict[str, Any], list[Edge]]:
    """Return (nodes, internal_edges) for the fork/join research subgraph.

    Three parallel researcher agents run behind a fork, converge at a join,
    and pass through a CEO gate:

        fork_research → researcher_{id}... → join_research → gate_research

    The caller wires the exit edges (gate_research → next PROCEED,
    gate_research → fork_research RELOOP) into the surrounding workflow.
    """
    researcher_ids = [f"researcher_{r.id}" for r in researchers]
    nodes: dict[str, Any] = {}

    nodes["fork_research"] = ForkNode(
        id="fork_research",
        targets=researcher_ids,
    )

    for r in researchers:
        rid = f"researcher_{r.id}"
        write_path = f".factory/strategy/research-{r.id}.md"
        kwargs: dict[str, Any] = {
            "id": rid,
            "role": AgentRole.RESEARCHER,
            "prompt_template": r.prompt_template,
            "writes": {write_path},
        }
        if r.post_check_min_size is not None:
            kwargs["post_checks"] = [
                ArtifactCheck(path=write_path, must_exist=True, min_size=r.post_check_min_size)
            ]
        nodes[rid] = AgentNode(**kwargs)

    nodes["join_research"] = JoinNode(
        id="join_research",
        sources=researcher_ids,
    )

    nodes["gate_research"] = GateNode(
        id="gate_research",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=gate_prompt,
        reads={f".factory/strategy/research-{r.id}.md" for r in researchers},
    )

    internal_edges = [
        *[Edge(source="fork_research", target=rid) for rid in researcher_ids],
        *[Edge(source=rid, target="join_research") for rid in researcher_ids],
        Edge(source="join_research", target="gate_research"),
    ]

    return nodes, internal_edges


# ── W₁: Build Mode ──────────────────────────────────────────────


def build_workflow() -> Workflow:
    """W₁: Build Mode — new project from idea/spec.

    Fork(3 researchers) → Join → CEO gate → Strategist → CEO gate →
    Archivist(async) → Builder → CEO gate → deep-QA → gate_qa(max 3) →
    Precheck gate → Archivist(async)
    """
    nodes: dict[str, Any] = {}
    edges: list[Edge] = []

    # Research subgraph: fork → 3 researchers → join → CEO gate
    _BUILD_RESEARCHERS = [
        ResearcherConfig(
            id="similar",
            prompt_template=(
                "Similar projects research. "
                "Read .factory/strategy/study-combined.md for project context "
                "(observations + structural graph analysis). "
                "Search the web for similar projects, existing solutions, and prior art. "
                "Analyze their strengths, weaknesses, and market positioning. "
                "Check .factory/archive/ for prior knowledge on similar builds. "
                "Write findings to .factory/strategy/research-similar.md covering: "
                "similar projects found (with links), what they do well and what's missing, "
                "differentiation opportunities."
            ),
            post_check_min_size=50,
        ),
        ResearcherConfig(
            id="techstack",
            prompt_template=(
                "Tech stack research. "
                "Read .factory/strategy/study-combined.md for project context "
                "(observations + structural graph analysis). "
                "Identify the best technology stack for this type of project. "
                "Find architecture patterns and best practices. "
                "Evaluate framework/library options with trade-offs. "
                "Write findings to .factory/strategy/research-techstack.md covering: "
                "recommended tech stack with rationale, architecture patterns, "
                "framework comparisons."
            ),
            post_check_min_size=50,
        ),
        ResearcherConfig(
            id="pitfalls",
            prompt_template=(
                "Pitfalls and scope research. "
                "Read .factory/strategy/study-combined.md for project context "
                "(observations + structural graph analysis). "
                "Identify potential pitfalls and common mistakes for this type of project. "
                "Research MVP scope best practices. "
                "Check .factory/archive/ for lessons from past builds. "
                "Write findings to .factory/strategy/research-pitfalls.md covering: "
                "potential pitfalls to avoid, MVP scope recommendation, "
                "lessons from similar past builds."
            ),
            post_check_min_size=50,
        ),
    ]
    r_nodes, r_edges = _research_subgraph(
        researchers=_BUILD_RESEARCHERS,
        gate_prompt=(
            "Is the research relevant? Does it cover the technology landscape adequately? "
            "Check for gaps in similar projects, tech stack analysis, and pitfall coverage."
        ),
    )
    nodes.update(r_nodes)

    # Strategist
    nodes["strategist"] = AgentNode(
        id="strategist",
        role=AgentRole.STRATEGIST,
        prompt_template=(
            "Synthesize a project specification from study and research. "
            "If .factory/strategy/study-combined.md exists, read it for project observations "
            "and structural graph analysis. "
            "Read ALL research files at .factory/strategy/research-similar.md, "
            "research-techstack.md, and research-pitfalls.md. "
            "Produce a complete phased build plan. Phase 1 must be project scaffold + eval harness. "
            "Every Phase must have substantive What/Why/Expected impact fields. "
            "Build EVERYTHING in this pass. Only defer items requiring human intervention. "
            "Write the plan to .factory/strategy/current.md."
        ),
        reads={
            ".factory/strategy/research-similar.md",
            ".factory/strategy/research-techstack.md",
            ".factory/strategy/research-pitfalls.md",
        },
        writes={".factory/strategy/current.md"},
        post_checks=[
            ArtifactCheck(
                path=".factory/strategy/current.md",
                must_exist=True,
                min_size=200,
                must_contain=["### Phase 1", "### Architecture"],
            )
        ],
    )

    # CEO gate on strategy quality — HARD GATE
    nodes["gate_strategy"] = GateNode(
        id="gate_strategy",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "HARD GATE — Builder MUST NOT start until approved. Check: "
            "1) Depth: every hypothesis has Category/What/Why/Expected impact. "
            "2) Research grounding: architecture and rationale cite research findings. "
            "3) Buildability: a Builder could implement each phase without clarifying questions. "
            "4) Phase 1 is scaffold + eval harness. "
            "5) Deferred section only contains items requiring human intervention. "
            "Write PLAN APPROVED in verdict if all checks pass."
        ),
        reads={".factory/strategy/current.md"},
    )

    # Archivist (async, non-blocking)
    nodes["archivist_plan"] = AgentNode(
        id="archivist_plan",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the approved research and strategy.",
        reads={".factory/strategy/current.md"},
        writes={".factory/archive/plan.md"},
        blocking=False,
    )

    # Per-phase: Builder → CEO gate → deep-QA → gate_qa(max 3) → Precheck → Archivist(async)
    nodes["builder"] = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        prompt_template=(
            "Implement the next phase from .factory/strategy/current.md. "
            "Read the CEO's plan approval at .factory/reviews/ceo-verdict-strategist.md. "
            "Read CLAUDE.md and factory.md if they exist. "
            "Implement exactly what the current phase describes. Run tests. "
            "Commit changes and open a draft PR."
        ),
        reads={".factory/strategy/current.md"},
        writes={".factory/reviews/builder-latest.md"},
        post_checks=[
            ArtifactCheck(
                path=".factory/reviews/builder-latest.md",
                must_exist=True,
                min_size=500,
                must_contain=["commit"],
            )
        ],
    )

    nodes["gate_build"] = GateNode(
        id="gate_build",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Read builder output. Check git log and diff. "
            "Does the work match the plan for this phase? "
            "If the Builder opened a PR, read it. "
            "REDIRECT if off-scope or missed key requirements."
        ),
        reads={".factory/reviews/builder-latest.md"},
    )

    # Deep-QA subgraph replaces monolithic QA
    dq_nodes, dq_edges = _deep_qa_subgraph()
    nodes.update(dq_nodes)

    nodes["gate_qa"] = GateNode(
        id="gate_qa",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Review QA results. PROCEED if all checks pass. "
            "RELOOP to builder (max 3 iterations) if issues found."
        ),
        reads={
            ".factory/reviews/health-check.md",
            ".factory/reviews/code-review.md",
            ".factory/reviews/adversarial-qa.md",
        },
    )

    nodes["gate_doc_freshness"] = GateNode(
        id="gate_doc_freshness",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=DOC_FRESHNESS_GATE_PROMPT,
        reads={".factory/reviews/adversarial-qa.md"},
    )

    nodes["gate_precheck"] = GateNode(
        id="gate_precheck",
        evaluator_type="fn",
        evaluator_command="factory precheck {project_path} --score-before 0 --score-after 0",
        reads={".factory/reviews/adversarial-qa.md"},
    )

    nodes["archivist_build"] = AgentNode(
        id="archivist_build",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the build phase results.",
        reads={".factory/reviews/adversarial-qa.md"},
        writes={".factory/archive/build.md"},
        blocking=False,
    )

    nodes["spec_generate"] = FnNode(
        id="spec_generate",
        command="factory workflow run spec-generate {project_path}",
        notes="Generate the project specification via the gated spec-generate workflow. Runs non-blocking after archival.",
        blocking=False,
    )

    # Edges
    edges = [
        # Research subgraph internal edges
        *r_edges,
        # Research gate → strategist (proceed) or back to researchers (reloop)
        Edge(source="gate_research", target="strategist", condition=VerdictType.PROCEED),
        Edge(source="gate_research", target="fork_research", condition=VerdictType.RELOOP),
        # Strategist → strategy gate
        Edge(source="strategist", target="gate_strategy"),
        # Strategy gate → archivist (proceed) or back (reloop)
        Edge(source="gate_strategy", target="archivist_plan", condition=VerdictType.PROCEED),
        Edge(source="gate_strategy", target="strategist", condition=VerdictType.RELOOP),
        # Archivist → builder
        Edge(source="archivist_plan", target="builder"),
        # Builder → build gate
        Edge(source="builder", target="gate_build"),
        # Build gate → deep-qa (proceed) or builder (reloop)
        Edge(source="gate_build", target="fork_qa", condition=VerdictType.PROCEED),
        Edge(source="gate_build", target="builder", condition=VerdictType.RELOOP),
        # Deep-QA internal edges
        *dq_edges,
        # adversarial_tester → gate_qa
        Edge(source="join_qa", target="gate_qa"),
        # gate_qa → doc freshness (proceed) or builder (reloop, max 3)
        Edge(source="gate_qa", target="gate_doc_freshness", condition=VerdictType.PROCEED),
        Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
        # Doc freshness → precheck (proceed) or builder (reloop)
        Edge(source="gate_doc_freshness", target="gate_precheck", condition=VerdictType.PROCEED),
        Edge(source="gate_doc_freshness", target="builder", condition=VerdictType.RELOOP),
        # Precheck → archivist (proceed) or halt → archivist (error handling)
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.PROCEED),
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.HALT),
        # Archivist → spec generate (non-blocking)
        Edge(source="archivist_build", target="spec_generate"),
    ]

    def trigger(state: ProjectState, ctx: dict[str, Any]) -> bool:
        return state in {ProjectState.NO_REPO, ProjectState.REPO_INCOMPLETE}

    return Workflow(
        name="build",
        nodes=nodes,
        edges=edges,
        start_node="fork_research",
        trigger=trigger,
    )


# ── W₂: Design Mode ─────────────────────────────────────────────


def design_workflow(just_plan: bool = False) -> Workflow:
    """W₂: Design Mode — W₁ with user gate at strategy approval.

    W₂ = W₁[gate_strategy ← GateNode(user), +gate_has_factory, +study]

    Existing projects (HAS_FACTORY) route through study before research.
    New/partial projects route through discover → study → fork_research.

    When just_plan=True, the workflow is truncated after strategy approval:
    prior plan check → research → strategy → user gate → publish → seed backlog.
    No builder, QA, or archivist nodes. Terminal mode.
    """
    wf = build_workflow()

    # Conditional entry: existing projects get study, new projects skip it
    wf.nodes["gate_has_factory"] = GateNode(
        id="gate_has_factory",
        evaluator_type="fn",
        evaluator_command=(
            'python3 -c "'
            "from pathlib import Path; "
            'exists = Path("{project_path}/.factory/config.json").exists(); '
            'print("PROCEED" if exists else "HALT")'
            '"'
        ),
    )

    wf.nodes["discover"] = FnNode(
        id="discover",
        command="factory discover {project_path}",
        writes={".factory/eval_profile.json"},
    )

    # Study subgraph: graph_update → study
    s_nodes, s_edges = _study_subgraph()
    wf.nodes.update(s_nodes)

    # Researchers and strategist read study-combined.md produced by study
    for nid in ("researcher_similar", "researcher_techstack", "researcher_pitfalls", "strategist"):
        node = wf.nodes[nid]
        wf.nodes[nid] = node.model_copy(
            update={"reads": (node.reads or set()) | {".factory/strategy/study-combined.md"}},
        )

    # Bootstrap nodes: create factory.md + config.json when missing
    wf.nodes["gate_factory_md_exists"] = GateNode(
        id="gate_factory_md_exists",
        evaluator_type="fn",
        evaluator_command=(
            'python3 -c "'
            "from pathlib import Path; "
            'exists = Path("{project_path}/factory.md").exists(); '
            'print("PROCEED" if exists else "HALT")'
            '"'
        ),
    )

    wf.nodes["create_factory_md"] = AgentNode(
        id="create_factory_md",
        role=AgentRole.CEO,
        prompt_template=(
            "Create factory.md from template. "
            "Copy the factory config template to the project root. "
            "Fill in: Goal, Scope, Guards, Eval command, Threshold, and Smoke Test. "
            "If .factory/eval_spec.json exists, populate the Eval Spec section. "
            "If .factory/strategy/current.md has a Research Configuration section, "
            "populate research sections (Research Target, Mutable/Fixed Surfaces, etc.)."
        ),
        reads={".factory/eval_profile.json"},
        writes={"factory.md"},
    )

    wf.nodes["factory_init"] = FnNode(
        id="factory_init",
        command="factory init {project_path}",
        notes="Parse factory.md and generate .factory/config.json. Must run after factory.md is created.",
        reads={"factory.md"},
        writes={".factory/config.json"},
    )

    wf.edges.extend(
        [
            *s_edges,
            Edge(source="gate_has_factory", target="graph_update", condition=VerdictType.PROCEED),
            Edge(source="gate_has_factory", target="discover", condition=VerdictType.HALT),
            Edge(source="discover", target="gate_factory_md_exists"),
            Edge(source="gate_factory_md_exists", target="factory_init", condition=VerdictType.PROCEED),
            Edge(source="gate_factory_md_exists", target="create_factory_md", condition=VerdictType.HALT),
            Edge(source="create_factory_md", target="factory_init"),
            Edge(source="factory_init", target="graph_update"),
            Edge(source="concat_study", target="fork_research"),
        ]
    )

    wf.start_node = "gate_has_factory"

    wf.nodes["gate_strategy"] = GateNode(
        id="gate_strategy",
        evaluator_type="user",
        reads={".factory/strategy/current.md"},
    )

    wf.name = "design"

    if just_plan:
        # ── Prior plan detection (prepend before fork_research) ──

        wf.nodes["check_prior_plans"] = GateNode(
            id="check_prior_plans",
            evaluator_type="fn",
            evaluator_command=(
                ': > "{project_path}/.factory/strategy/prior-plans.md"; '
                'if [ -n "$FOCUS" ]; then '
                "  if gh auth status >/dev/null 2>&1 && git remote -v 2>/dev/null | grep -q .; then "
                '    gh issue list --label plan --search "$FOCUS" --json number,title,url '
                '      --jq ".[] | \\"#\\(.number) \\(.title) — \\(.url)\\"" '
                '      > "{project_path}/.factory/strategy/prior-plans.md" 2>/dev/null || true; '
                "  fi; "
                '  if [ ! -s "{project_path}/.factory/strategy/prior-plans.md" ]; then '
                '    grep -Frl "$FOCUS" "{project_path}/.factory/archive/" --include="plan-*.md" '
                '      >> "{project_path}/.factory/strategy/prior-plans.md" 2>/dev/null || true; '
                "  fi; "
                "fi; "
                '[ -s "{project_path}/.factory/strategy/prior-plans.md" ]'
            ),
            gate_prompt=(
                "Check GitHub issues with plan label and .factory/archive/ for prior plans "
                "matching the focus keywords. Write matching results to .factory/strategy/prior-plans.md "
                "(GitHub issue URLs or local file paths). "
                "PROCEED if matches exist (file is non-empty), HALT if no matches (skip to fresh research)."
            ),
            writes={".factory/strategy/prior-plans.md"},
        )

        wf.nodes["gate_prior_plans"] = GateNode(
            id="gate_prior_plans",
            evaluator_type="user",
            gate_prompt=(
                "Prior plan(s) found matching this topic. "
                "Present the matching plans from .factory/strategy/prior-plans.md to the user. "
                "If one match: ask 'Found a prior plan on this topic. Continue this plan or start fresh?' "
                "If multiple matches: list them and let user pick which to continue, or start fresh. "
                "The selected prior plan (if any) will be passed as context to researchers and strategist."
            ),
            reads={".factory/strategy/prior-plans.md"},
        )

        # ── Plan publishing nodes (after gate_strategy) ──

        wf.nodes["publish_github"] = FnNode(
            id="publish_github",
            command=(
                "bash -c '"
                "set -e; "
                'echo "none" > "{project_path}/.factory/strategy/github-issue-ref.txt"; '
                "if ! gh auth status >/dev/null 2>&1; then "
                '  echo "SKIP: gh not authenticated — plan saved locally only"; exit 0; '
                "fi; "
                "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
                '  echo "SKIP: not inside a git repository"; exit 0; '
                "fi; "
                "if ! git remote -v 2>/dev/null | grep -q .; then "
                '  SLUG=$(basename "{project_path}"); '
                '  echo "Creating GitHub repository: $SLUG..."; '
                '  if gh repo create "$SLUG" --public --source=. --remote=origin --push 2>&1; then '
                '    REPO_URL=$(gh repo view "$SLUG" --json url -q .url 2>/dev/null || echo ""); '
                '    echo "GitHub repository created: ${REPO_URL:-$SLUG}"; '
                '  elif gh repo view "$SLUG" >/dev/null 2>&1; then '
                '    echo "Repository $SLUG already exists on GitHub, linking as remote..."; '
                '    REMOTE_URL=$(gh repo view "$SLUG" --json sshUrl -q .sshUrl 2>/dev/null || '
                '      gh repo view "$SLUG" --json url -q .url); '
                '    git remote add origin "$REMOTE_URL" 2>/dev/null || true; '
                "    git push -u origin HEAD 2>/dev/null || true; "
                "  else "
                '    echo "SKIP: could not create GitHub repo — plan saved locally only"; exit 0; '
                "  fi; "
                "fi; "
                'gh label create plan --description "Approved plan" --color 0366d6 --force 2>/dev/null || true; '
                'FOCUS="${FOCUS:-}"; '
                'ISSUE_NUM=""; '
                'if echo "$FOCUS" | grep -qE "^[0-9]+$"; then '
                '  ISSUE_NUM="$FOCUS"; '
                'elif echo "$FOCUS" | grep -qoE "#([0-9]+)"; then '
                '  ISSUE_NUM=$(echo "$FOCUS" | grep -oE "[0-9]+" | tail -1); '
                "fi; "
                'if [ -n "$ISSUE_NUM" ]; then '
                '  gh issue comment "$ISSUE_NUM" --body-file "{project_path}/.factory/strategy/current.md"; '
                '  gh issue edit "$ISSUE_NUM" --add-label plan; '
                '  echo "$ISSUE_NUM" > "{project_path}/.factory/strategy/github-issue-ref.txt"; '
                '  echo "Plan posted to issue #$ISSUE_NUM"; '
                "else "
                '  TITLE="Plan: ${FOCUS:-project}"; '
                '  ISSUE_URL=$(gh issue create --title "$TITLE" --body-file "{project_path}/.factory/strategy/current.md" --label plan); '
                '  ISSUE_NUM=$(echo "$ISSUE_URL" | grep -oE "[0-9]+$"); '
                '  echo "$ISSUE_NUM" > "{project_path}/.factory/strategy/github-issue-ref.txt"; '
                '  echo "Created plan issue: $ISSUE_URL"; '
                "fi"
                "'"
            ),
            reads={".factory/strategy/current.md"},
            writes={".factory/strategy/github-issue-ref.txt"},
            notes=(
                "Publishes the approved plan to a GitHub issue. If no git remote exists, "
                "auto-creates a public GitHub repository via 'gh repo create --public "
                "--source=. --remote=origin --push'. If the repo name already exists on "
                "GitHub, links it as a remote instead. After ensuring a remote exists, "
                "publishes the plan: if --focus is an issue number, posts as a comment; "
                "otherwise creates a new issue titled 'Plan: <focus>'. "
                "Writes the issue number to github-issue-ref.txt for downstream use by "
                "seed_backlog. Graceful degradation: if gh is not authenticated, not in "
                "a git repo, or repo creation fails, writes 'none' and exits cleanly."
            ),
        )

        wf.nodes["seed_backlog"] = FnNode(
            id="seed_backlog",
            command=(
                'python3 -c "'
                "import re, os; "
                "project = '{project_path}'; "
                "plan = open(f'{project}/.factory/strategy/current.md').read(); "
                "ref_file = f'{project}/.factory/strategy/github-issue-ref.txt'; "
                "issue_num = open(ref_file).read().strip() if os.path.exists(ref_file) else 'none'; "
                "ref = f'(see #{issue_num})' if issue_num != 'none' else '(see .factory/strategy/current.md)'; "
                "phases = re.findall(r'### Phase \\d+:.*', plan); "
                "backlog_path = f'{project}/.factory/strategy/backlog.md'; "
                "items = '\\n'.join(f'- [ ] {p[4:]} {ref}' for p in phases); "
                "open(backlog_path, 'a').write('\\n' + items + '\\n') if items else None; "
                "print(f'Seeded {len(phases)} backlog items from plan')"
                '"'
            ),
            reads={".factory/strategy/current.md", ".factory/strategy/github-issue-ref.txt"},
            writes={".factory/strategy/backlog.md"},
            notes=(
                "Extracts phase headers from the approved plan at current.md and appends them "
                "as backlog items to backlog.md. References GitHub issue number if publish_github "
                "ran (reads github-issue-ref.txt), otherwise references current.md. "
                "Example: '- [ ] Phase 1: Set up auth middleware (see #42)'"
            ),
        )

        # ── Remove build-phase nodes that are unreachable in plan mode ──
        build_phase_nodes = {
            "archivist_plan",
            "builder",
            "gate_build",
            "fork_qa",
            "health_checker",
            "code_reviewer",
            "adversarial_tester",
            "join_qa",
            "gate_qa",
            "gate_doc_freshness",
            "gate_precheck",
            "archivist_build",
            "spec_generate",
        }
        for node_id in build_phase_nodes:
            wf.nodes.pop(node_id, None)

        # ── Filter out edges referencing removed build-phase nodes ──
        removed = build_phase_nodes
        wf.edges = [e for e in wf.edges if e.source not in removed and e.target not in removed]

        # Replace concat_study → fork_research with concat_study → check_prior_plans
        wf.edges = [
            e for e in wf.edges if not (e.source == "concat_study" and e.target == "fork_research")
        ]

        # Add plan-specific edges
        wf.edges.extend(
            [
                Edge(source="concat_study", target="check_prior_plans"),
                Edge(
                    source="check_prior_plans",
                    target="gate_prior_plans",
                    condition=VerdictType.PROCEED,
                ),
                Edge(
                    source="check_prior_plans", target="fork_research", condition=VerdictType.HALT
                ),
                Edge(
                    source="gate_prior_plans", target="fork_research", condition=VerdictType.PROCEED
                ),
                Edge(
                    source="gate_strategy", target="publish_github", condition=VerdictType.PROCEED
                ),
                Edge(source="publish_github", target="seed_backlog"),
            ]
        )

        wf.name = "plan"
        wf.start_node = "gate_has_factory"
        wf.terminal = True

        def plan_trigger(state: ProjectState, ctx: dict[str, Any]) -> bool:
            return ctx.get("just_plan") is True

        wf.trigger = plan_trigger
        return wf

    wf.terminal = True

    def trigger(state: ProjectState, ctx: dict[str, Any]) -> bool:
        return state in {
            ProjectState.NO_REPO,
            ProjectState.REPO_INCOMPLETE,
            ProjectState.HAS_FACTORY,
        } and ctx.get("interactive", False)

    wf.trigger = trigger
    return wf



# ── W₉: Create Mode ──────────────────────────────────────────────


def create_workflow() -> Workflow:
    """W₉: Create Mode — meta-mode for creating new factory modes.

    Takes a user description and produces a fully working workflow definition,
    SKILL.md, CLI wiring, and tests.

    Fork(3 researchers) → Join → CEO gate → Strategist → User gate →
    Archivist(async) → Builder → CEO gate → deep-QA → gate_qa(max 3) →
    Precheck gate → Archivist(async)
    """
    nodes: dict[str, Any] = {}
    edges: list[Edge] = []

    # Research subgraph: fork → 3 researchers → join → CEO gate
    _CREATE_RESEARCHERS = [
        ResearcherConfig(
            id="existing",
            prompt_template=(
                "Existing workflow analysis. "
                "If the CEO task includes '## Create Mode (Update Existing Mode)', read the "
                "**Target mode:** field and focus your analysis on that specific mode's workflow "
                "definition via `factory workflow show <target_mode>`. Document its current node "
                "sequences, gate logic, edge wiring, trigger function, and reads/writes. Also read "
                "its SKILL.md at skills/workflow-<target_mode>/SKILL.md for the generated playbook. "
                "Otherwise, read factory/workflow/definitions.py and analyze all existing workflow "
                "definitions (build, design, create, spec-generate). "
                "Document common patterns: node sequences, gate conventions, fork/join patterns, "
                "archivist placement, edge wiring, trigger functions, reads/writes declarations. "
                "Read factory/workflow/primitives.py for available node types and their fields. "
                "Read factory/workflow/skill_export.py for WORKFLOW_META format. "
                "Write findings to .factory/strategy/research-existing.md covering: "
                "node type usage patterns, common subgraphs (builder→gate→qa→gate loop), "
                "trigger function conventions, data flow patterns."
            ),
        ),
        ResearcherConfig(
            id="intent",
            prompt_template=(
                "Mode description analysis. "
                "Read the user's mode description from the CEO task. "
                "If the CEO task includes '## Create Mode (Plugin Package)', parse the "
                "**output_folder** and plugin-specific constraints (standalone package, "
                "entry point registration, no upstream modifications). Structure the plugin "
                "packaging requirements: pyproject.toml entry point, workflow file layout, "
                "register_plugin() function pattern, installation and verification steps. "
                "Write findings to .factory/strategy/research-intent.md covering: "
                "structured requirements, packaging needs, workflow node candidates. "
                "Otherwise, if the CEO task includes '## Create Mode (Update Existing Mode)', "
                "parse the **Requested changes:** field and structure the requested modifications "
                "against the existing mode's current behavior. Identify which nodes, edges, "
                "prompts, or gates need to change and which must remain untouched. "
                "Otherwise, parse and structure the description into a new workflow specification: "
                "- Purpose and trigger conditions "
                "- Agent roles needed (which specialists) "
                "- Gate logic (user vs agent vs fn evaluators) "
                "- Data flow (what files are read/written) "
                "- Interactive vs headless requirements "
                "- Input format (text, file, drawing, flow) "
                "Write findings to .factory/strategy/research-intent.md covering: "
                "structured requirements, node candidates, suggested graph topology."
            ),
        ),
        ResearcherConfig(
            id="practices",
            prompt_template=(
                "Workflow design best practices. "
                "Search the web for workflow and pipeline design patterns relevant "
                "to the described mode. Look for: DAG design patterns, agent orchestration "
                "patterns, quality gate strategies, error recovery approaches. "
                "Check .factory/archive/ for lessons from past mode creation or workflow changes. "
                "Write findings to .factory/strategy/research-practices.md covering: "
                "relevant design patterns, pitfalls to avoid, testing strategies."
            ),
        ),
    ]
    r_nodes, r_edges = _research_subgraph(
        researchers=_CREATE_RESEARCHERS,
        gate_prompt=(
            "Are the existing workflow patterns well-documented? "
            "Is the user's intent clearly structured into workflow requirements? "
            "Are best practices relevant to this type of mode? Any gaps?"
        ),
    )
    nodes.update(r_nodes)

    # Strategist synthesizes workflow specification
    nodes["strategist"] = AgentNode(
        id="strategist",
        role=AgentRole.STRATEGIST,
        prompt_template=(
            "Synthesize a workflow specification. "
            "Read ALL tagged research files at .factory/strategy/research-*.md. "
            "If the CEO task includes '## Create Mode (Update Existing Mode)', produce a "
            "change spec describing modifications to the existing workflow: which nodes/edges/"
            "prompts/gates to modify, what to add or remove, and a diff-oriented implementation "
            "plan. Include the 20-point verification checklist from the CEO task. Do NOT produce "
            "a complete new workflow definition — describe changes to the existing one. "
            "If the CEO task includes '## Create Mode (New Portable Workflow)', the spec "
            "MUST instruct writing to the output_folder from the CEO task. The spec MUST NOT "
            "include CLI wiring changes, definitions.py modifications, or WORKFLOW_META entries. "
            "Portable workflows are auto-discovered by the registry — no registration needed. "
            "Produce a specification for a portable workflow including: "
            "1) Python code for the workflow function (nodes dict, edges list, trigger) "
            "2) meta dict with name and description "
            "3) Test cases (graph validation, skill export, trigger function) "
            "4) Node details: for each node, specify id, type, role, prompt_template, reads, writes "
            "5) Edge details: for each edge, specify source, target, condition "
            "6) Interactive vs headless behavior "
            "Otherwise, produce a complete specification for a new factory mode including: "
            "1) Python code for the workflow function (nodes dict, edges list, trigger) "
            "2) WORKFLOW_META entry (description, argument_hint) "
            "3) CLI wiring changes (build_parser mode choices, cmd_ceo routing, _build_ceo_task section) "
            "4) Test cases (graph validation, skill export, trigger function, registration) "
            "5) Node details: for each node, specify id, type, role, prompt_template, reads, writes "
            "6) Edge details: for each edge, specify source, target, condition "
            "7) Interactive vs headless behavior "
            "Follow conventions from existing workflows — use the same patterns for "
            "builder→gate→QA→gate loops, archivist placement, and research forks. "
            "Write the specification to .factory/strategy/current.md."
        ),
        reads={
            ".factory/strategy/research-existing.md",
            ".factory/strategy/research-intent.md",
            ".factory/strategy/research-practices.md",
        },
        writes={".factory/strategy/current.md"},
    )

    # User gate for workflow spec approval — interactive
    nodes["gate_strategy"] = GateNode(
        id="gate_strategy",
        evaluator_type="user",
        reads={".factory/strategy/current.md"},
    )

    # Archivist (async, non-blocking)
    nodes["archivist_plan"] = AgentNode(
        id="archivist_plan",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the approved workflow specification for the new mode.",
        reads={".factory/strategy/current.md"},
        writes={".factory/archive/create-plan.md"},
        blocking=False,
    )

    # Builder implements everything
    nodes["builder"] = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        timeout=1800,
        prompt_template=(
            "Implement the workflow changes from the approved specification. "
            "Read the approved spec at .factory/strategy/current.md. "
            "Read CLAUDE.md for project conventions. "
            "If the CEO task includes '## Create Mode (Plugin Package)', follow the "
            "PLUGIN checklist: "
            "1) Read **output_folder** from the CEO task "
            "2) Create the output directory: mkdir -p <output_folder> "
            "3) Write pyproject.toml with: "
            "   name factory-<mode-name>-workflow, version 0.1.0, "
            "   build-system hatchling, requires-python >=3.11, "
            "   dependencies [remote-factory], "
            "   entry point [factory.plugins] <mode-name> = '<mode_name>:register_plugin' "
            "4) Write <mode_name>.py with: "
            "   meta dict (name, description), "
            "   workflow() function returning a Workflow object, "
            "   register_plugin(registry) calling registry.add_modes() and "
            "   registry.add_workflow_search_path(str(Path(__file__).parent)) "
            "5) Write README.md with installation and usage "
            "6) Test: pip install -e <output_folder>/ "
            "7) Verify: factory workflow list shows the mode "
            "8) Validate: factory workflow validate <mode-name> "
            "9) Clean up: pip uninstall -y factory-<mode-name>-workflow "
            "The plugin package stays in the output directory — do NOT commit it "
            "to the factory repo or open a PR. It is a standalone artifact. "
            "Do NOT modify factory/workflow/definitions.py or register_all(). "
            "Otherwise, if the CEO task includes '## Create Mode (Update Existing Mode)', "
            "follow the update checklist: modify the existing workflow function in "
            "definitions.py, verify the register_all() entry still resolves, update "
            "WORKFLOW_META if needed, verify all 20 registration points from the CEO task, "
            "run factory workflow validate <name>, regenerate SKILL.md via factory workflow "
            "export-skills, update tests, run pytest and ruff check. "
            "Otherwise, if the CEO task includes '## Create Mode (New Portable Workflow)', "
            "follow the portable workflow checklist: "
            "1) Read **output_folder** from the CEO task "
            "2) Create the output directory: mkdir -p <output_folder> "
            "3) Write the workflow file to <output_folder>/<name>.py "
            "4) The file must contain a `meta` dict with `name` and `description` keys, "
            "and a `workflow()` function returning a Workflow object "
            "5) Only import from factory.workflow.primitives and stdlib — no other factory internals "
            "6) Do NOT modify factory/workflow/definitions.py, register_all(), WORKFLOW_META, "
            "or CLI wiring — the workflow registry discovers the output_folder automatically "
            "7) Run factory workflow validate <name> --project-path $PROJECT_PATH to verify the graph "
            "8) Run factory workflow export-skills --project-path $PROJECT_PATH to generate the SKILL.md "
            "9) Write tests in tests/ "
            "10) Run pytest and ruff check to verify "
            "Commit changes and open a draft PR."
        ),
        reads={".factory/strategy/current.md"},
        writes={".factory/reviews/builder-latest.md"},
    )

    # CEO gate on build
    nodes["gate_build"] = GateNode(
        id="gate_build",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Read builder output and PR diff. Does work match the approved spec? "
            "For plugin packages: verify output directory contains pyproject.toml, "
            "workflow .py with meta + workflow() + register_plugin(), and README.md. "
            "Verify NO upstream factory files were modified. "
            "For new modes: verify workflow file exists at .factory/workflows/<name>.py "
            "with meta dict and workflow() function, NOT patched into definitions.py. "
            "For existing mode updates: verify definitions.py changes are correct. "
            "Tests written. REDIRECT if any component is missing."
        ),
        reads={".factory/reviews/builder-latest.md"},
    )

    # Deep-QA verification (replaces monolithic QA)
    dq_nodes, dq_edges = _deep_qa_subgraph(
        adversarial_extra=(
            "**Plugin mode check:** If the CEO task includes '## Create Mode "
            "(Plugin Package)', verify the plugin package structure: "
            "1) Output directory exists at the specified output_folder path. "
            "2) pyproject.toml exists with [project.entry-points.'factory.plugins'] section. "
            "3) Workflow .py file has meta dict + workflow() + register_plugin() function. "
            "4) README.md documents installation and usage. "
            "5) Run: pip install -e <folder>/ (must succeed). "
            "6) Run: factory workflow list (must show the new mode). "
            "7) Run: factory workflow validate <mode-name> (must pass). "
            "8) Run: pip uninstall -y factory-<mode-name>-workflow (cleanup). "
            "Verify NO upstream factory files were modified (definitions.py, register_all, etc). "
            "**Portable workflow check:** If the CEO task includes '## Create Mode "
            "(New Portable Workflow)', verify the workflow was written to the output_folder "
            "from the CEO task (default: .factory/workflows/). "
            "FAIL if any files were written to factory/workflow/contributed/ or if "
            "definitions.py was modified. "
            "Run: factory workflow validate <name> --project-path $PROJECT_PATH, "
            "factory workflow show <name> --project-path $PROJECT_PATH. "
            "Verify SKILL.md generated under skills/workflow-<name>/. "
            "Check workflow handles both interactive and headless paths."
        ),
    )
    nodes.update(dq_nodes)

    # CEO gate on QA (max 3 iterations)
    nodes["gate_qa"] = GateNode(
        id="gate_qa",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Review QA results for the new mode. PROCEED if all checks pass: "
            "workflow validates, SKILL.md generated, tests pass, CLI recognizes mode. "
            "RELOOP to builder (max 3 iterations) if issues found."
        ),
        reads={
            ".factory/reviews/health-check.md",
            ".factory/reviews/code-review.md",
            ".factory/reviews/adversarial-qa.md",
        },
    )

    nodes["gate_doc_freshness"] = GateNode(
        id="gate_doc_freshness",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=DOC_FRESHNESS_GATE_PROMPT,
        reads={".factory/reviews/adversarial-qa.md"},
    )

    # Precheck gate
    nodes["gate_precheck"] = GateNode(
        id="gate_precheck",
        evaluator_type="fn",
        evaluator_command="factory precheck {project_path} --score-before 0 --score-after 0",
        reads={".factory/reviews/adversarial-qa.md"},
    )

    # Archivist (async)
    nodes["archivist_build"] = AgentNode(
        id="archivist_build",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the new mode build results and learnings.",
        reads={".factory/reviews/adversarial-qa.md"},
        writes={".factory/archive/create-build.md"},
        blocking=False,
    )

    # Edges
    edges = [
        # Research subgraph internal edges
        *r_edges,
        # Research gate
        Edge(source="gate_research", target="strategist", condition=VerdictType.PROCEED),
        Edge(source="gate_research", target="fork_research", condition=VerdictType.RELOOP),
        # Strategist → user gate
        Edge(source="strategist", target="gate_strategy"),
        # User gate
        Edge(source="gate_strategy", target="archivist_plan", condition=VerdictType.PROCEED),
        Edge(source="gate_strategy", target="strategist", condition=VerdictType.RELOOP),
        # Archivist → builder
        Edge(source="archivist_plan", target="builder"),
        # Builder → build gate
        Edge(source="builder", target="gate_build"),
        # Build gate → deep-qa (proceed) or builder (reloop)
        Edge(source="gate_build", target="fork_qa", condition=VerdictType.PROCEED),
        Edge(source="gate_build", target="builder", condition=VerdictType.RELOOP),
        # Deep-QA internal edges
        *dq_edges,
        # adversarial_tester → gate_qa
        Edge(source="join_qa", target="gate_qa"),
        # gate_qa → doc freshness (proceed) or builder (reloop)
        Edge(source="gate_qa", target="gate_doc_freshness", condition=VerdictType.PROCEED),
        Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
        # Doc freshness → precheck (proceed) or builder (reloop)
        Edge(source="gate_doc_freshness", target="gate_precheck", condition=VerdictType.PROCEED),
        Edge(source="gate_doc_freshness", target="builder", condition=VerdictType.RELOOP),
        # Precheck → archivist (proceed) or halt → archivist (error handling)
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.PROCEED),
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.HALT),
    ]

    def trigger(state: ProjectState, ctx: dict[str, Any]) -> bool:
        return ctx.get("mode") == "create"

    return Workflow(
        name="create",
        nodes=nodes,
        edges=edges,
        start_node="fork_research",
        trigger=trigger,
    )



# ── W₁₃: Spec Generate Mode ────────────────────────────────────


def spec_generate_workflow() -> Workflow:
    """W₁₃: Spec Generate — extract behavioral spec, annotate, validate.

    extract → gate_extract → annotate → gate_annotate →
    validate → gate_validate → done
    """
    nodes: dict[str, Any] = {}
    edges: list[Edge] = []

    # Graphify extraction — produces graph.json (local AST, no LLM cost)
    nodes["extract"] = FnNode(
        id="extract",
        command="factory graph extract {project_path}",
        notes="Run graphify to extract a code knowledge graph from the project source.",
        writes={"graph.json"},
    )

    # CEO gate — check extraction quality
    nodes["gate_extract"] = GateNode(
        id="gate_extract",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Check that graph.json was produced. "
            "Verify it contains nodes and edges. "
            "PROCEED if the graph was extracted successfully. RELOOP if missing or empty."
        ),
        reads={"graph.json"},
    )

    # Researcher annotation — reads graph.json directly, produces SPEC.md
    nodes["annotate"] = AgentNode(
        id="annotate",
        role=AgentRole.RESEARCHER,
        prompt_template=(
            "Read the code knowledge graph at graph.json. "
            "Read the spec_annotator prompt at factory/agents/prompts/spec_annotator.md. "
            "Produce a two-tier behavioral spec with RFC 2119 normative language. "
            "Use [[graph:...]] reference links for granular module details. "
            "Write output to SPEC.md in the project root."
        ),
        reads={"graph.json"},
        writes={"SPEC.md"},
    )

    # CEO gate — check annotation quality and section completeness
    nodes["gate_annotate"] = GateNode(
        id="gate_annotate",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Review the annotated spec at SPEC.md. "
            "Check: do module behavioral contracts match the actual code? "
            "Does the spec use RFC 2119 normative language (MUST/SHOULD/MAY)? "
            "Are there scoring tables (there should NOT be)? "
            "SECTION COMPLETENESS CHECK — verify ALL of the following sections are present "
            "and non-empty: "
            " Problem Statement, "
            " Goals and Non-Goals (including.1 Goals.2 Non-Goals.3 Design Philosophy), "
            " Project Identity, "
            " Technical Stack, "
            " Architecture Overview, "
            " Domain Model, "
            " State Machines and Lifecycles, "
            " Module Specifications, "
            " Shared Contracts, "
            " Configuration Specification, "
            " Entry Points, "
            " Failure Model and Recovery, "
            " Security and Safety, "
            " Test and Validation Matrix, "
            " Extension Points, "
            " Implementation Checklist, "
            "Appendix A: Reference Algorithms. "
            "RELOOP if ANY section is missing or empty. "
            "PROCEED only if ALL 16 sections + Appendix A are present and non-empty."
        ),
        reads={"SPEC.md"},
    )

    # Validation — run automated consistency checks
    nodes["validate"] = FnNode(
        id="validate",
        command="factory spec validate {project_path}",
        notes="Run automated consistency checks on the annotated SPEC.md. Must run after annotation is CEO-approved.",
        reads={"SPEC.md"},
        writes={".factory/spec_validation.md"},
    )

    # Final quality gate
    nodes["gate_validate"] = GateNode(
        id="gate_validate",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Final quality gate for the repo spec. "
            "Read SPEC.md. Is it complete, well-structured, "
            "and under 24K tokens? PROCEED to finish."
        ),
        reads={"SPEC.md"},
    )

    edges = [
        # Extract → gate
        Edge(source="extract", target="gate_extract"),
        Edge(source="gate_extract", target="annotate", condition=VerdictType.PROCEED),
        Edge(source="gate_extract", target="extract", condition=VerdictType.RELOOP),
        # Annotate → gate
        Edge(source="annotate", target="gate_annotate"),
        Edge(source="gate_annotate", target="validate", condition=VerdictType.PROCEED),
        Edge(source="gate_annotate", target="annotate", condition=VerdictType.RELOOP),
        # Validate → gate
        Edge(source="validate", target="gate_validate"),
    ]

    return Workflow(
        name="spec-generate",
        nodes=nodes,
        edges=edges,
        start_node="extract",
        trigger=None,
    )


# ── Registry ─────────────────────────────────────────────────────

_BUILTIN_REGISTRY: dict[str, Any] | None = None


def _get_builtin_registry() -> dict[str, Any]:
    """Return the lazy-callable registry, building it on first access."""
    global _BUILTIN_REGISTRY
    if _BUILTIN_REGISTRY is not None:
        return _BUILTIN_REGISTRY
    _BUILTIN_REGISTRY = {
        "design": design_workflow,
        "create": create_workflow,
        "spec-generate": spec_generate_workflow,
        "swebench": lambda: __import__(
            "factory.workflow.contributed.swebench", fromlist=["workflow"]
        ).workflow(),
        "legacybench": lambda: __import__(
            "factory.workflow.contributed.legacybench", fromlist=["workflow"]
        ).workflow(),
        "featurebench": lambda: __import__(
            "factory.workflow.contributed.featurebench", fromlist=["workflow"]
        ).workflow(),
        "programbench": lambda: __import__(
            "factory.workflow.contributed.programbench", fromlist=["workflow"]
        ).workflow(),
        "terminalbench": lambda: __import__(
            "factory.workflow.contributed.terminalbench", fromlist=["workflow"]
        ).workflow(),
        "tomswe": lambda: __import__(
            "factory.workflow.contributed.tomswe", fromlist=["workflow"]
        ).workflow(),
        "salitrap": lambda: __import__(
            "factory.workflow.contributed.salitrap", fromlist=["workflow"]
        ).workflow(),
        "swebenchifyhard": lambda: __import__(
            "factory.workflow.contributed.swebenchifyhard", fromlist=["workflow"]
        ).workflow(),
        "mini-swebench": lambda: __import__(
            "factory.workflow.contributed.mini_swebench", fromlist=["workflow"]
        ).workflow(),
        "devopsgym": lambda: __import__(
            "factory.workflow.contributed.devopsgym", fromlist=["workflow"]
        ).workflow(),
        "outer-loop": lambda: __import__(
            "factory.workflow.contributed.outer_loop", fromlist=["workflow"]
        ).workflow(),
        "design-v2": lambda: __import__(
            "factory.workflow.contributed.design_v2", fromlist=["workflow"]
        ).workflow(),
        "create-v2": lambda: __import__(
            "factory.workflow.contributed.create_v2", fromlist=["workflow"]
        ).workflow(),
    }
    return _BUILTIN_REGISTRY


def register_all() -> dict[str, Workflow]:
    """Build and return all workflow definitions.

    Uses _get_builtin_registry() internally — each callable is invoked
    to construct the Workflow object.  Kept for backward compatibility.
    """
    registry = _get_builtin_registry()
    return {name: fn() for name, fn in registry.items()}
