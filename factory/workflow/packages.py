"""Reusable workflow Packages for mode composition.

Each function returns a Package that can be composed with Sequential,
Parallel, Loop, and Conditional operators. Modes become compositions
of these building blocks rather than monolithic graph definitions.

Example:
    build_mode = Sequential(
        study_package(),
        research_package(researchers=BUILD_RESEARCHERS),
        strategy_package(),
        build_package(),
        qa_package(),
    )
"""

from __future__ import annotations

from factory.workflow.package import (
    Package,
    Port,
    Sequential,
    StateContract,
)
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    ArtifactCheck,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    NodeType,
    Study,
    VerdictType,
    Workflow,
)

from factory.workflow.definitions import (
    DOC_FRESHNESS_GATE_PROMPT,
    ResearcherConfig,
    _graph_explorer_prompt,
)


# ── Study Package ──────────────────────────────────────────────


def study_package(*, focus: str | None = None) -> Package:
    """Graph-powered codebase study: graph_update → study → graph_explorer → concat."""
    graph_update = FnNode(
        id="graph_update",
        command="factory graph update {project_path}",
        writes={"graph.json"},
        notes="Extract or incrementally update the code knowledge graph before study.",
    )
    study = Study(
        id="study",
        command="factory study {project_path}",
        writes={".factory/strategy/observations.md"},
        focus=focus,
    )
    graph_explorer = AgentNode(
        id="graph_explorer",
        role=AgentRole.RESEARCHER,
        prompt_template=_graph_explorer_prompt(focus),
        reads={".factory/strategy/observations.md"},
        writes={".factory/strategy/graph-context.md"},
    )
    concat_study = FnNode(
        id="concat_study",
        command=(
            "cat {project_path}/.factory/strategy/observations.md"
            " {project_path}/.factory/strategy/graph-context.md"
            " > {project_path}/.factory/strategy/study-combined.md"
        ),
        reads={".factory/strategy/observations.md", ".factory/strategy/graph-context.md"},
        writes={".factory/strategy/study-combined.md"},
    )

    nodes: dict[str, NodeType] = {
        "graph_update": graph_update,
        "study": study,
        "graph_explorer": graph_explorer,
        "concat_study": concat_study,
    }
    edges = [
        Edge(source="graph_update", target="study"),
        Edge(source="study", target="graph_explorer"),
        Edge(source="graph_explorer", target="concat_study"),
    ]

    return Package(
        name="study",
        version="1.0.0",
        description="Graph-powered codebase study and analysis",
        outputs=[
            Port(name="study", artifact_path=".factory/strategy/study-combined.md"),
        ],
        contract=StateContract(
            produces=frozenset({"study_complete"}),
        ),
        graph=Workflow(
            name="study", nodes=nodes, edges=edges, start_node="graph_update",
        ),
        entry_node="graph_update",
        exit_node="concat_study",
    )


# ── Research Package ───────────────────────────────────────────


def research_package(
    *,
    researchers: list[ResearcherConfig],
    gate_prompt: str,
    study_read: str | None = ".factory/strategy/study-combined.md",
) -> Package:
    """Parallel research with CEO gate: fork → N researchers → join → gate.

    ``study_read`` is injected into every researcher's ``reads`` so the study
    output flows into research.  Set to ``None`` when composing without a
    preceding study phase.
    """
    researcher_ids = [f"researcher_{r.id}" for r in researchers]
    nodes: dict[str, NodeType] = {}

    nodes["fork_research"] = ForkNode(
        id="fork_research", targets=researcher_ids,
    )

    research_outputs = []
    for r in researchers:
        rid = f"researcher_{r.id}"
        write_path = f".factory/strategy/research-{r.id}.md"
        reads: set[str] = set()
        if study_read:
            reads.add(study_read)
        kwargs: dict = {
            "id": rid,
            "role": AgentRole.RESEARCHER,
            "prompt_template": r.prompt_template,
            "writes": {write_path},
            "reads": reads,
        }
        if r.post_check_min_size is not None:
            kwargs["post_checks"] = [
                ArtifactCheck(path=write_path, must_exist=True, min_size=r.post_check_min_size)
            ]
        nodes[rid] = AgentNode(**kwargs)
        research_outputs.append(
            Port(name=f"research-{r.id}", artifact_path=write_path),
        )

    nodes["join_research"] = JoinNode(
        id="join_research", sources=researcher_ids,
    )
    nodes["gate_research"] = GateNode(
        id="gate_research",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=gate_prompt,
        reads={f".factory/strategy/research-{r.id}.md" for r in researchers},
    )

    edges = [
        *[Edge(source="fork_research", target=rid) for rid in researcher_ids],
        *[Edge(source=rid, target="join_research") for rid in researcher_ids],
        Edge(source="join_research", target="gate_research"),
        Edge(source="gate_research", target="fork_research", condition=VerdictType.RELOOP),
    ]

    return Package(
        name="research",
        version="1.0.0",
        description="Parallel research with CEO quality gate",
        inputs=[
            Port(name="study", artifact_path=".factory/strategy/study-combined.md"),
        ],
        outputs=research_outputs,
        contract=StateContract(
            requires=frozenset({"study_complete"}),
            produces=frozenset({"research_complete"}),
        ),
        graph=Workflow(
            name="research", nodes=nodes, edges=edges, start_node="fork_research",
        ),
        entry_node="fork_research",
        exit_node="gate_research",
    )


# ── Strategy Package ──────────────────────────────────────────


def strategy_package(
    *,
    research_reads: set[str] | None = None,
    strategist_prompt: str = "",
    gate_prompt: str = "",
    gate_type: str = "agent",
    study_read: str | None = ".factory/strategy/study-combined.md",
) -> Package:
    """Strategist → CEO gate → archivist (async).

    ``study_read`` is added to the strategist's reads so the study output
    flows into strategy synthesis.  Set to ``None`` when composing without a
    preceding study phase.
    """
    reads = research_reads or {
        ".factory/strategy/research-similar.md",
        ".factory/strategy/research-techstack.md",
        ".factory/strategy/research-pitfalls.md",
    }
    if study_read:
        reads = set(reads) | {study_read}

    if not strategist_prompt:
        strategist_prompt = (
            "Synthesize a project specification from study and research. "
            "If .factory/strategy/study-combined.md exists, read it for project "
            "observations and structural graph analysis. "
            "Read ALL research files at .factory/strategy/research-similar.md, "
            "research-techstack.md, and research-pitfalls.md. "
            "Produce a complete phased build plan. Phase 1 must be project scaffold + eval harness. "
            "Every Phase must have substantive What/Why/Expected impact fields and an acceptance criteria table per hypothesis. "
            "Build EVERYTHING in this pass. Only defer items requiring human intervention. "
            "Write the plan to .factory/strategy/current.md."
        )

    if not gate_prompt:
        gate_prompt = (
            "HARD GATE — Builder MUST NOT start until approved. Check: "
            "1) Depth: every hypothesis has Category/What/Why/Expected impact and acceptance criteria. "
            "2) Research grounding: architecture and rationale cite research findings. "
            "3) Buildability: a Builder could implement each phase without clarifying questions. "
            "4) Phase 1 is scaffold + eval harness. "
            "Write PLAN APPROVED in verdict if all checks pass."
        )

    strategist = AgentNode(
        id="strategist",
        role=AgentRole.STRATEGIST,
        prompt_template=strategist_prompt,
        reads=reads,
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
    gate_kwargs: dict = {
        "id": "gate_strategy",
        "evaluator_type": gate_type,
        "reads": {".factory/strategy/current.md"},
    }
    if gate_type == "agent":
        gate_kwargs["evaluator_role"] = AgentRole.CEO
        gate_kwargs["gate_prompt"] = gate_prompt
    gate_strategy = GateNode(**gate_kwargs)
    archivist_plan = AgentNode(
        id="archivist_plan",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the approved research and strategy.",
        reads={".factory/strategy/current.md"},
        writes={".factory/archive/plan.md"},
        blocking=False,
    )

    nodes: dict[str, NodeType] = {
        "strategist": strategist,
        "gate_strategy": gate_strategy,
        "archivist_plan": archivist_plan,
    }
    edges = [
        Edge(source="strategist", target="gate_strategy"),
        Edge(source="gate_strategy", target="archivist_plan", condition=VerdictType.PROCEED),
        Edge(source="gate_strategy", target="strategist", condition=VerdictType.RELOOP),
    ]

    return Package(
        name="strategy",
        version="1.0.0",
        description="Strategy synthesis with CEO approval gate",
        inputs=[Port(name="research", artifact_path=".factory/strategy/research-similar.md")],
        outputs=[Port(name="strategy", artifact_path=".factory/strategy/current.md")],
        contract=StateContract(
            requires=frozenset({"research_complete"}),
            produces=frozenset({"strategy_complete"}),
        ),
        graph=Workflow(
            name="strategy", nodes=nodes, edges=edges, start_node="strategist",
        ),
        entry_node="strategist",
        exit_node="archivist_plan",
    )


# ── Build Package ─────────────────────────────────────────────


def build_package() -> Package:
    """Builder → CEO gate, with reloop on redirect."""
    builder = AgentNode(
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
    gate_build = GateNode(
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

    nodes: dict[str, NodeType] = {"builder": builder, "gate_build": gate_build}
    edges = [
        Edge(source="builder", target="gate_build"),
        Edge(source="gate_build", target="builder", condition=VerdictType.RELOOP),
    ]

    return Package(
        name="build",
        version="1.0.0",
        description="Builder with CEO review gate",
        inputs=[Port(name="strategy", artifact_path=".factory/strategy/current.md")],
        outputs=[Port(name="build", artifact_path=".factory/reviews/builder-latest.md")],
        contract=StateContract(
            requires=frozenset({"strategy_complete"}),
            produces=frozenset({"build_complete"}),
        ),
        graph=Workflow(
            name="build", nodes=nodes, edges=edges, start_node="builder",
        ),
        entry_node="builder",
        exit_node="gate_build",
    )


# ── QA Package ────────────────────────────────────────────────


def qa_package() -> Package:
    """Deep QA: fork → [health, code_review, adversarial] → join → gates."""
    health_checker = AgentNode(
        id="health_checker",
        role=AgentRole.HEALTH_CHECKER,
        prompt_template="Run health checks on the builder output.",
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/health-check.md"},
    )
    code_reviewer = AgentNode(
        id="code_reviewer",
        role=AgentRole.CODE_REVIEWER,
        prompt_template="Review the builder output for code quality.",
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/code-review.md"},
    )
    adversarial_tester = AgentNode(
        id="adversarial_tester",
        role=AgentRole.ADVERSARIAL_TESTER,
        prompt_template="Run adversarial tests on the builder output.",
        timeout=1800,
        reads={".factory/reviews/builder-latest.md", ".factory/strategy/current.md"},
        writes={".factory/reviews/adversarial-qa.md"},
    )
    fork_qa = ForkNode(id="fork_qa", targets=["health_checker", "code_reviewer", "adversarial_tester"])
    join_qa = JoinNode(
        id="join_qa",
        sources=["health_checker", "code_reviewer", "adversarial_tester"],
        reads={
            ".factory/reviews/health-check.md",
            ".factory/reviews/code-review.md",
            ".factory/reviews/adversarial-qa.md",
        },
    )
    gate_qa = GateNode(
        id="gate_qa",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt="Review QA results. PROCEED if all checks pass. RELOOP to builder (max 3 iterations) if issues found.",
        reads={
            ".factory/reviews/health-check.md",
            ".factory/reviews/code-review.md",
            ".factory/reviews/adversarial-qa.md",
        },
    )
    gate_doc = GateNode(
        id="gate_doc_freshness",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=DOC_FRESHNESS_GATE_PROMPT,
        reads={".factory/reviews/adversarial-qa.md"},
    )
    gate_precheck = GateNode(
        id="gate_precheck",
        evaluator_type="fn",
        evaluator_command="factory precheck {project_path} --score-before 0 --score-after 0",
        reads={".factory/reviews/adversarial-qa.md"},
    )
    archivist_build = AgentNode(
        id="archivist_build",
        role=AgentRole.ARCHIVIST,
        prompt_template="Archive the build phase results.",
        reads={".factory/reviews/adversarial-qa.md"},
        writes={".factory/archive/build.md"},
        blocking=False,
    )

    spec_generate = FnNode(
        id="spec_generate",
        command="factory workflow run spec-generate {project_path}",
        blocking=False,
        notes="Generate the project specification via the gated spec-generate workflow. Runs non-blocking after archival.",
    )

    nodes: dict[str, NodeType] = {
        "fork_qa": fork_qa,
        "health_checker": health_checker,
        "code_reviewer": code_reviewer,
        "adversarial_tester": adversarial_tester,
        "join_qa": join_qa,
        "gate_qa": gate_qa,
        "gate_doc_freshness": gate_doc,
        "gate_precheck": gate_precheck,
        "archivist_build": archivist_build,
        "spec_generate": spec_generate,
    }
    edges = [
        Edge(source="fork_qa", target="health_checker"),
        Edge(source="fork_qa", target="code_reviewer"),
        Edge(source="fork_qa", target="adversarial_tester"),
        Edge(source="health_checker", target="join_qa"),
        Edge(source="code_reviewer", target="join_qa"),
        Edge(source="adversarial_tester", target="join_qa"),
        Edge(source="join_qa", target="gate_qa"),
        Edge(source="gate_qa", target="gate_doc_freshness", condition=VerdictType.PROCEED),
        Edge(source="gate_doc_freshness", target="gate_precheck", condition=VerdictType.PROCEED),
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.PROCEED),
        Edge(source="gate_precheck", target="archivist_build", condition=VerdictType.HALT),
        Edge(source="archivist_build", target="spec_generate"),
    ]

    return Package(
        name="qa",
        version="1.0.0",
        description="Deep QA: health check, code review, adversarial testing",
        inputs=[Port(name="build", artifact_path=".factory/reviews/builder-latest.md")],
        outputs=[Port(name="qa", artifact_path=".factory/reviews/adversarial-qa.md")],
        contract=StateContract(
            requires=frozenset({"build_complete"}),
            produces=frozenset({"qa_complete"}),
        ),
        graph=Workflow(
            name="qa", nodes=nodes, edges=edges, start_node="fork_qa",
        ),
        entry_node="fork_qa",
        exit_node="spec_generate",
    )


# ── Mode compositions ─────────────────────────────────────────

BUILD_RESEARCHERS = [
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


def discovery_package() -> Package:
    """Bootstrap: detect project state, create factory.md + config.json if needed."""
    gate_has_factory = GateNode(
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
    discover = FnNode(
        id="discover",
        command="factory discover {project_path}",
        writes={".factory/eval_profile.json"},
    )
    gate_factory_md = GateNode(
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
    create_factory_md = AgentNode(
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
    factory_init = FnNode(
        id="factory_init",
        command="factory init {project_path}",
        reads={"factory.md"},
        writes={".factory/config.json"},
        notes="Parse factory.md and generate .factory/config.json. Must run after factory.md is created.",
    )

    # Bootstrap path: discover → gate_factory_md → [create_factory_md →] factory_init
    bootstrap_nodes: dict[str, NodeType] = {
        "discover": discover,
        "gate_factory_md_exists": gate_factory_md,
        "create_factory_md": create_factory_md,
        "factory_init": factory_init,
    }
    bootstrap_edges = [
        Edge(source="discover", target="gate_factory_md_exists"),
        Edge(source="gate_factory_md_exists", target="factory_init", condition=VerdictType.PROCEED),
        Edge(source="gate_factory_md_exists", target="create_factory_md", condition=VerdictType.HALT),
        Edge(source="create_factory_md", target="factory_init"),
    ]
    # All nodes in one graph: gate → [bootstrap path or skip]
    skip_node = FnNode(
        id="skip_bootstrap",
        command='echo "Factory config exists, skipping bootstrap"',
    )

    all_nodes: dict[str, NodeType] = {
        "gate_has_factory": gate_has_factory,
        "skip_bootstrap": skip_node,
        **bootstrap_nodes,
    }
    all_edges = [
        Edge(source="gate_has_factory", target="skip_bootstrap", condition=VerdictType.PROCEED),
        Edge(source="gate_has_factory", target="discover", condition=VerdictType.HALT),
        *bootstrap_edges,
    ]

    return Package(
        name="discovery",
        version="1.0.0",
        description="Detect project state, bootstrap factory.md + config.json if needed",
        outputs=[Port(name="config", artifact_path=".factory/config.json")],
        contract=StateContract(produces=frozenset({"discovery_complete"})),
        graph=Workflow(
            name="discovery", nodes=all_nodes,
            edges=all_edges, start_node="gate_has_factory",
        ),
        entry_node="gate_has_factory",
        exit_node="skip_bootstrap",
    )


# Cross-package back-edges: QA gates reloop to the builder (which lives in an
# earlier package). Sequential can only add forward bridge edges, so these
# reloops are declared explicitly via extra_edges. Mirrors the monolithic
# build_workflow's "gate_qa → builder (max 3)" and doc-freshness reloop.
_QA_RELOOP_EDGES = [
    Edge(source="gate_qa", target="builder", condition=VerdictType.RELOOP),
    Edge(source="gate_doc_freshness", target="builder", condition=VerdictType.RELOOP),
]


def build_mode(*, focus: str | None = None) -> Package:
    """Build mode as a Package composition.

    study → research → strategy → build → qa
    """
    return Sequential(
        study_package(focus=focus),
        research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt=(
                "Is the research relevant? Does it cover the technology landscape adequately? "
                "Check for gaps in similar projects, tech stack analysis, and pitfall coverage."
            ),
        ),
        strategy_package(),
        build_package(),
        qa_package(),
        name="build",
        extra_edges=_QA_RELOOP_EDGES,
        terminal=True,
    )


def design_mode(*, focus: str | None = None) -> Package:
    """Design mode: discovery → study → research → strategy (user gate) → build → qa.

    Like build mode but with:
    - Discovery/bootstrap for new projects
    - User gate on strategy (instead of CEO gate)
    """
    return Sequential(
        discovery_package(),
        study_package(focus=focus),
        research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt=(
                "Is the research relevant? Does it cover the technology landscape adequately? "
                "Check for gaps in similar projects, tech stack analysis, and pitfall coverage."
            ),
        ),
        strategy_package(
            gate_type="user",
        ),
        build_package(),
        qa_package(),
        name="design",
        extra_edges=_QA_RELOOP_EDGES,
        terminal=True,
    )


def design_with_frontend_mode(*, focus: str | None = None) -> Package:
    """Design mode + frontend discovery.

    Demonstrates Package composition: inject a frontend-specific
    discovery step into the standard design pipeline.
    """
    frontend_discovery = AgentNode(
        id="frontend_discovery",
        role=AgentRole.RESEARCHER,
        prompt_template=(
            "Discover the project's frontend design system. "
            "Find: design tokens (colors, spacing, typography), "
            "component library (React/Vue/Svelte components), "
            "layout patterns, data fetching conventions, "
            "and styling approach (CSS modules, Tailwind, styled-components). "
            "Write a structured design system reference to "
            ".factory/strategy/design-system.md."
        ),
        reads={".factory/strategy/study-combined.md"},
        writes={".factory/strategy/design-system.md"},
    )
    frontend_pkg = Package(
        name="frontend-discovery",
        version="1.0.0",
        description="Discover frontend design system tokens, components, patterns",
        inputs=[Port(name="study", artifact_path=".factory/strategy/study-combined.md")],
        outputs=[Port(name="design-system", artifact_path=".factory/strategy/design-system.md")],
        contract=StateContract(
            requires=frozenset({"study_complete"}),
            produces=frozenset({"frontend_discovery_complete"}),
        ),
        graph=Workflow(
            name="frontend-discovery",
            nodes={"frontend_discovery": frontend_discovery},
            edges=[], start_node="frontend_discovery",
        ),
        entry_node="frontend_discovery",
        exit_node="frontend_discovery",
    )

    return Sequential(
        discovery_package(),
        study_package(focus=focus),
        frontend_pkg,
        research_package(
            researchers=BUILD_RESEARCHERS,
            gate_prompt=(
                "Is the research relevant? Does the frontend design system analysis "
                "cover tokens, components, and patterns adequately?"
            ),
        ),
        strategy_package(
            research_reads={
                ".factory/strategy/research-similar.md",
                ".factory/strategy/research-techstack.md",
                ".factory/strategy/research-pitfalls.md",
                ".factory/strategy/design-system.md",
            },
            gate_type="user",
        ),
        build_package(),
        qa_package(),
        name="frontend-design-mode",
    )


# ── Plugin registration ───────────────────────────────────────


def register(registry: object) -> None:
    """Plugin entry point — registers composed modes with factory.

    Called by factory's plugin loader when this module is installed
    as a factory.plugins entry point. Adds composed modes to the
    WorkflowRegistry so they're available via `factory ceo --mode <name>`.
    """
    from factory.workflow.registry import WorkflowRegistry

    _COMPOSED_MODES = {
        "build-composed": lambda: build_mode().compile(),
        "design-composed": lambda: design_mode().compile(),
        "frontend-design": lambda: design_with_frontend_mode().compile(),
    }

    for name, fn in _COMPOSED_MODES.items():
        WorkflowRegistry.register_callable(name, fn, source="package")

    if hasattr(registry, "add_modes"):
        registry.add_modes(list(_COMPOSED_MODES.keys()))
