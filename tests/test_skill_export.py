"""Tests for factory/workflow/skill_export.py."""

from pathlib import Path

import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    Study,
    SubgraphForkNode,
    VerdictType,
    Workflow,
)
from factory.workflow.skill_export import (
    _agent_to_instruction,
    _fn_to_instruction,
    _fork_to_instruction,
    _gate_to_checkpoint,
    _study_to_instruction,
    export_all_skills,
    validate_skill,
    workflow_to_skill_md,
)


# ── helpers ──────────────────────────────────────────────────────


def _make_agent(
    id: str,
    role: AgentRole = AgentRole.BUILDER,
    *,
    blocking: bool = True,
    prompt: str = "",
    reads: set[str] | None = None,
    writes: set[str] | None = None,
    timeout: int | None = None,
) -> AgentNode:
    return AgentNode(
        id=id,
        role=role,
        blocking=blocking,
        prompt_template=prompt,
        reads=reads or set(),
        writes=writes or set(),
        timeout=timeout,
    )


def _minimal_workflow(
    name: str = "test",
    nodes: dict | None = None,
    edges: list[Edge] | None = None,
    start: str | None = None,
) -> Workflow:
    if nodes is None:
        n = _make_agent("builder", AgentRole.BUILDER)
        nodes = {n.id: n}
    if edges is None:
        edges = []
    return Workflow(
        name=name,
        nodes=nodes,
        edges=edges,
        start_node=start or next(iter(nodes)),
    )


# ── _agent_to_instruction ───────────────────────────────────────


class TestAgentToInstruction:
    def test_blocking_agent_no_ampersand(self) -> None:
        node = _make_agent("builder", blocking=True)
        wf = _minimal_workflow(nodes={"builder": node}, start="builder")
        result = _agent_to_instruction(node, wf)
        assert " &" not in result

    def test_nonblocking_agent_has_ampersand(self) -> None:
        node = _make_agent("archivist", AgentRole.ARCHIVIST, blocking=False)
        wf = _minimal_workflow(nodes={"archivist": node}, start="archivist")
        result = _agent_to_instruction(node, wf)
        assert " &" in result
        assert "fire-and-forget" in result

    def test_parallel_flag_forces_ampersand(self) -> None:
        node = _make_agent("researcher_a", AgentRole.RESEARCHER, blocking=True)
        wf = _minimal_workflow(nodes={"researcher_a": node}, start="researcher_a")
        result = _agent_to_instruction(node, wf, is_parallel=True)
        assert " &" in result

    def test_parallel_researcher_gets_review_tag(self) -> None:
        node = _make_agent("researcher_web", AgentRole.RESEARCHER)
        wf = _minimal_workflow(nodes={"researcher_web": node}, start="researcher_web")
        result = _agent_to_instruction(node, wf, is_parallel=True)
        assert "--review-tag web" in result

    def test_archivist_gets_haiku_model(self) -> None:
        node = _make_agent("archivist", AgentRole.ARCHIVIST)
        wf = _minimal_workflow(nodes={"archivist": node}, start="archivist")
        result = _agent_to_instruction(node, wf)
        assert "--model haiku" in result

    def test_reads_and_writes_in_prompt(self) -> None:
        node = _make_agent(
            "builder",
            reads={"observations.md"},
            writes={"changes.diff"},
        )
        wf = _minimal_workflow(nodes={"builder": node}, start="builder")
        result = _agent_to_instruction(node, wf)
        assert "observations.md" in result
        assert "changes.diff" in result

    def test_emits_timeout_slot(self) -> None:
        node = _make_agent("builder")
        wf = _minimal_workflow(nodes={"builder": node}, start="builder")
        result = _agent_to_instruction(node, wf)
        assert "{{timeout_builder::" in result

    def test_emits_task_prompt_slot(self) -> None:
        node = _make_agent("builder", prompt="Build the thing.")
        wf = _minimal_workflow(nodes={"builder": node}, start="builder")
        result = _agent_to_instruction(node, wf)
        assert "{{task_prompt_builder::" in result

    def test_emits_annotation_comments(self) -> None:
        node = _make_agent("builder")
        wf = _minimal_workflow(nodes={"builder": node}, start="builder")
        result = _agent_to_instruction(node, wf)
        assert "<!-- node: AgentNode id=builder" in result
        assert "<!-- reads:" in result
        assert "<!-- writes:" in result
        assert "<!-- edges:" in result


# ── _fn_to_instruction ──────────────────────────────────────────


class TestFnToInstruction:
    def test_basic_command(self) -> None:
        fn = FnNode(id="fn_eval", command="factory eval {project_path}")
        wf = _minimal_workflow(nodes={"fn_eval": fn}, start="fn_eval")
        result = _fn_to_instruction(fn, wf)
        assert "$PROJECT_PATH" in result
        assert "<!-- node: FnNode id=fn_eval" in result

    def test_template_placeholder_gets_slot(self) -> None:
        fn = FnNode(
            id="fn_finalize", command="factory review --verdict $VERDICT --project {project_path}"
        )
        wf = _minimal_workflow(nodes={"fn_finalize": fn}, start="fn_finalize")
        result = _fn_to_instruction(fn, wf)
        assert "{{finalize_command_fn_finalize::" in result

    def test_fn_node_notes_rendered(self) -> None:
        fn = FnNode(
            id="fn_begin",
            command="factory begin {project_path}",
            notes="Open a new experiment for the current hypothesis.",
        )
        wf = _minimal_workflow(nodes={"fn_begin": fn}, start="fn_begin")
        result = _fn_to_instruction(fn, wf)
        assert "Open a new experiment for the current hypothesis." in result
        idx_notes = result.index("Open a new experiment")
        idx_bash = result.index("```bash")
        assert idx_notes < idx_bash, "Notes must appear before the bash command block"

    def test_fn_node_empty_notes(self) -> None:
        fn = FnNode(id="fn_eval", command="factory eval {project_path}")
        wf = _minimal_workflow(nodes={"fn_eval": fn}, start="fn_eval")
        result = _fn_to_instruction(fn, wf)
        lines_before_bash = result.split("```bash")[0]
        non_annotation_lines = [
            line
            for line in lines_before_bash.strip().split("\n")
            if line.strip() and not line.strip().startswith("<!--")
        ]
        assert non_annotation_lines == [], "Empty notes should produce no prose before bash block"

    def test_reads_writes_annotations(self) -> None:
        fn = FnNode(
            id="fn_score",
            command="factory eval {project_path}",
            reads={"eval_profile.json"},
            writes={"eval_after.json"},
        )
        wf = _minimal_workflow(nodes={"fn_score": fn}, start="fn_score")
        result = _fn_to_instruction(fn, wf)
        assert "<!-- reads: eval_profile.json -->" in result
        assert "<!-- writes: eval_after.json -->" in result


# ── _fork_to_instruction ────────────────────────────────────────


class TestForkToInstruction:
    def test_fork_targets_get_ampersand(self) -> None:
        """Critical: fork target agents must run with & for parallel execution."""
        r1 = _make_agent("researcher_a", AgentRole.RESEARCHER, blocking=True)
        r2 = _make_agent("researcher_b", AgentRole.RESEARCHER, blocking=True)
        fork = ForkNode(id="fork_research", targets=["researcher_a", "researcher_b"])
        wf = _minimal_workflow(
            nodes={"fork_research": fork, "researcher_a": r1, "researcher_b": r2},
            start="fork_research",
        )
        result = _fork_to_instruction(fork, wf)
        assert result.count(" &") >= 2, "Each fork target must have '&' suffix"
        assert "wait" in result

    def test_fork_header_shows_agent_count(self) -> None:
        r1 = _make_agent("researcher_a", AgentRole.RESEARCHER)
        r2 = _make_agent("researcher_b", AgentRole.RESEARCHER)
        r3 = _make_agent("researcher_c", AgentRole.RESEARCHER)
        fork = ForkNode(
            id="fork_research",
            targets=["researcher_a", "researcher_b", "researcher_c"],
        )
        wf = _minimal_workflow(
            nodes={
                "fork_research": fork,
                "researcher_a": r1,
                "researcher_b": r2,
                "researcher_c": r3,
            },
            start="fork_research",
        )
        result = _fork_to_instruction(fork, wf)
        assert "3 agents" in result

    def test_fork_skips_non_agent_targets(self) -> None:
        fn = FnNode(id="fn_eval", command="factory eval {project_path}")
        fork = ForkNode(id="fork_mixed", targets=["fn_eval"])
        wf = _minimal_workflow(
            nodes={"fork_mixed": fork, "fn_eval": fn},
            start="fork_mixed",
        )
        result = _fork_to_instruction(fork, wf)
        assert "factory agent" not in result
        assert "wait" in result

    def test_fork_includes_timeout_guidance_when_needed(self) -> None:
        """When parallel agents have timeout > 120s, emit timeout guidance for Bash tool."""
        r1 = _make_agent("researcher_a", AgentRole.RESEARCHER, timeout=600)
        r2 = _make_agent("researcher_b", AgentRole.RESEARCHER, timeout=600)
        fork = ForkNode(id="fork_research", targets=["researcher_a", "researcher_b"])
        wf = _minimal_workflow(
            nodes={"fork_research": fork, "researcher_a": r1, "researcher_b": r2},
            start="fork_research",
        )
        result = _fork_to_instruction(fork, wf)
        assert "Important:" in result
        assert "single" in result.lower()
        assert "Bash tool" in result
        assert "600 seconds" in result

    def test_fork_omits_timeout_guidance_when_not_needed(self) -> None:
        """When all parallel agents have timeout <= 120s, no timeout guidance needed."""
        r1 = _make_agent("researcher_a", AgentRole.RESEARCHER, timeout=100)
        r2 = _make_agent("researcher_b", AgentRole.RESEARCHER, timeout=120)
        fork = ForkNode(id="fork_research", targets=["researcher_a", "researcher_b"])
        wf = _minimal_workflow(
            nodes={"fork_research": fork, "researcher_a": r1, "researcher_b": r2},
            start="fork_research",
        )
        result = _fork_to_instruction(fork, wf)
        assert "Important:" not in result
        assert "Bash tool" not in result


# ── _gate_to_checkpoint ─────────────────────────────────────────


class TestGateToCheckpoint:
    def test_user_gate(self) -> None:
        gate = GateNode(id="gate_strategy", evaluator_type="user")
        wf = _minimal_workflow(nodes={"gate_strategy": gate}, start="gate_strategy")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "User Approval" in result
        assert "Do NOT self-approve" in result
        assert "MUST wait for the user" in result

    def test_user_gate_anti_self_approval(self) -> None:
        gate = GateNode(id="gate_approval", evaluator_type="user")
        wf = _minimal_workflow(nodes={"gate_approval": gate}, start="gate_approval")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "Do NOT self-approve" in result
        assert "MUST wait for the user" in result
        assert "Do you approve this plan" in result
        assert "Do NOT write a verdict file" in result
        assert "CEO Review" not in result

    def test_fn_gate_with_command(self) -> None:
        gate = GateNode(
            id="gate_eval",
            evaluator_type="fn",
            evaluator_command="factory eval {project_path}",
        )
        wf = _minimal_workflow(nodes={"gate_eval": gate}, start="gate_eval")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "Automated" in result
        assert "$PROJECT_PATH" in result

    def test_agent_gate_with_reads(self) -> None:
        gate = GateNode(
            id="gate_review",
            evaluator_type="agent",
            reads={"reviews/health-check.md"},
            gate_prompt="Assess quality.",
        )
        wf = _minimal_workflow(nodes={"gate_review": gate}, start="gate_review")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "CEO Review" in result
        assert "health-check.md" in result
        assert "Assess quality" in result

    def test_reloop_edges_shown(self) -> None:
        gate = GateNode(id="gate_build")
        builder = _make_agent("builder")
        reloop = Edge(
            source="gate_build",
            target="builder",
            condition=VerdictType.RELOOP,
        )
        wf = _minimal_workflow(
            nodes={"gate_build": gate, "builder": builder},
            edges=[reloop],
            start="gate_build",
        )
        result = _gate_to_checkpoint(gate, [reloop], wf)
        assert "RELOOP" in result
        assert "builder" in result

    def test_reloop_budget_prefers_gate_cap(self) -> None:
        # A generated guard gate enforces its own max_iterations at runtime; the rendered
        # skill must show that number, not the RELOOP target's unrelated default (3).
        gate = GateNode(id="guard_draft_gate", evaluator_type="fn", max_iterations=2)
        draft = _make_agent("draft")
        reloop = Edge(
            source="guard_draft_gate",
            target="draft",
            condition=VerdictType.RELOOP,
        )
        wf = _minimal_workflow(
            nodes={"guard_draft_gate": gate, "draft": draft},
            edges=[reloop],
            start="guard_draft_gate",
        )
        result = _gate_to_checkpoint(gate, [reloop], wf)
        assert "{{max_iterations_guard_draft_gate::2}}" in result

    def test_emits_gate_prompt_slot(self) -> None:
        gate = GateNode(
            id="gate_review",
            evaluator_type="agent",
            gate_prompt="Check quality.",
        )
        wf = _minimal_workflow(nodes={"gate_review": gate}, start="gate_review")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "{{gate_prompt_gate_review::" in result

    def test_fn_gate_includes_mandatory_wait(self) -> None:
        gate = GateNode(
            id="gate_precheck",
            evaluator_type="fn",
            evaluator_command="factory precheck {project_path}",
        )
        wf = _minimal_workflow(nodes={"gate_precheck": gate}, start="gate_precheck")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "MANDATORY" in result
        assert "Do NOT run agents in parallel" in result

    def test_emits_annotation_comments(self) -> None:
        gate = GateNode(
            id="gate_review",
            evaluator_type="agent",
            gate_prompt="Check.",
        )
        wf = _minimal_workflow(nodes={"gate_review": gate}, start="gate_review")
        result = _gate_to_checkpoint(gate, [], wf)
        assert "<!-- gate: GateNode id=gate_review" in result


# ── workflow_to_skill_md ─────────────────────────────────────────


class TestWorkflowToSkillMd:
    def test_generates_valid_frontmatter(self) -> None:
        wf = _minimal_workflow(name="alpha")
        result = workflow_to_skill_md(wf)
        assert result.startswith("---")
        assert "name: workflow-alpha" in result
        assert "description:" in result

    def test_contains_arguments_placeholder(self) -> None:
        wf = _minimal_workflow()
        result = workflow_to_skill_md(wf)
        assert "$ARGUMENTS" in result

    def test_phases_numbered_sequentially(self) -> None:
        n1 = _make_agent("researcher", AgentRole.RESEARCHER)
        n2 = _make_agent("builder", AgentRole.BUILDER)
        wf = _minimal_workflow(
            nodes={"researcher": n1, "builder": n2},
            edges=[Edge(source="researcher", target="builder")],
            start="researcher",
        )
        result = workflow_to_skill_md(wf)
        assert "Phase 1:" in result
        assert "Phase 2:" in result

    def test_fork_targets_excluded_from_standalone_phases(self) -> None:
        r1 = _make_agent("researcher_a", AgentRole.RESEARCHER)
        r2 = _make_agent("researcher_b", AgentRole.RESEARCHER)
        fork = ForkNode(id="fork_research", targets=["researcher_a", "researcher_b"])
        join = JoinNode(id="join_research", sources=["researcher_a", "researcher_b"])
        builder = _make_agent("builder", AgentRole.BUILDER)
        wf = _minimal_workflow(
            nodes={
                "fork_research": fork,
                "researcher_a": r1,
                "researcher_b": r2,
                "join_research": join,
                "builder": builder,
            },
            edges=[
                Edge(source="fork_research", target="researcher_a"),
                Edge(source="fork_research", target="researcher_b"),
                Edge(source="researcher_a", target="join_research"),
                Edge(source="researcher_b", target="join_research"),
                Edge(source="join_research", target="builder"),
            ],
            start="fork_research",
        )
        result = workflow_to_skill_md(wf)
        lines = result.split("\n")
        phase_lines = [line for line in lines if line.startswith("## Phase")]
        phase_titles = [line.lower() for line in phase_lines]
        researcher_standalone = [
            t for t in phase_titles if "researcher" in t and "parallel" not in t
        ]
        assert len(researcher_standalone) == 0, (
            "Fork targets should not appear as standalone phases"
        )

    def test_study_node_generates_observe_phase(self) -> None:
        study = Study(
            id="study",
            command="factory study {project_path}",
            focus="performance",
        )
        wf = _minimal_workflow(nodes={"study": study}, start="study")
        result = workflow_to_skill_md(wf)
        assert "Observe" in result
        assert "--focus" in result


# ── export_all_skills ────────────────────────────────────────────


class TestExportAllSkills:
    def test_creates_skill_files(self, tmp_path: Path) -> None:
        builder = _make_agent("builder", AgentRole.BUILDER)
        wf = _minimal_workflow(name="test_wf", nodes={"builder": builder})
        paths = export_all_skills(tmp_path, workflows={"test_wf": wf})
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "SKILL.md"
        assert "workflow-test_wf" in str(paths[0].parent)

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        wf1 = _minimal_workflow(name="alpha")
        wf2 = _minimal_workflow(name="beta")
        paths = export_all_skills(tmp_path, workflows={"alpha": wf1, "beta": wf2})
        assert len(paths) == 2
        dirs = {p.parent.name for p in paths}
        assert "workflow-alpha" in dirs
        assert "workflow-beta" in dirs

    def test_written_content_passes_validation(self, tmp_path: Path) -> None:
        wf = _minimal_workflow(name="alpha")
        paths = export_all_skills(tmp_path, workflows={"alpha": wf})
        content = paths[0].read_text()
        issues = validate_skill(content)
        assert issues == [], f"Validation issues: {issues}"


# ── validate_skill ───────────────────────────────────────────────


class TestValidateSkill:
    def test_valid_skill_no_issues(self) -> None:
        content = (
            "---\nname: workflow-build\n"
            'description: "Build things."\n'
            "disable-model-invocation: true\n"
            "---\n\n# Build\nDo stuff.\n"
        )
        assert validate_skill(content) == []

    def test_missing_frontmatter(self) -> None:
        issues = validate_skill("# No frontmatter\nJust text.")
        assert any("frontmatter" in i.lower() for i in issues)

    def test_malformed_frontmatter(self) -> None:
        issues = validate_skill("---\nname: test\nno closing marker")
        assert any("malformed" in i.lower() for i in issues)

    def test_missing_name(self) -> None:
        content = '---\ndescription: "test"\n---\nbody'
        issues = validate_skill(content)
        assert any("name" in i.lower() for i in issues)

    def test_missing_description(self) -> None:
        content = "---\nname: workflow-test\n---\nbody"
        issues = validate_skill(content)
        assert any("description" in i.lower() for i in issues)

    def test_invalid_name_format(self) -> None:
        content = '---\nname: INVALID NAME\ndescription: "x"\n---\nbody'
        issues = validate_skill(content)
        assert any("kebab" in i.lower() for i in issues)

    def test_oversized_body(self) -> None:
        body = "\n".join(f"line {i}" for i in range(1300))
        content = f'---\nname: workflow-test\ndescription: "x"\n---\n{body}'
        issues = validate_skill(content)
        assert any("1200" in i for i in issues)


# ── real workflow skill generation ──────────────────────────────


class TestRealWorkflowSkills:
    """Tests that real workflow definitions produce valid, exportable skills."""

    def test_all_registered_skills_exported(self, tmp_path: Path) -> None:
        from factory.workflow.definitions import register_all

        workflows = register_all()
        paths = export_all_skills(tmp_path, workflows=workflows)
        assert len(paths) == len(workflows), f"Expected {len(workflows)} skills, got {len(paths)}"
        dirs = {p.parent.name for p in paths}
        expected = {f"workflow-{name}" for name in workflows}
        assert dirs == expected, f"Missing: {expected - dirs}"
        for p in paths:
            content = p.read_text()
            issues = validate_skill(content)
            assert issues == [], f"{p.parent.name} validation: {issues}"


# ── QA phase enforcement ──────────────────────────────────────────


def _workflows_with_builder() -> list[str]:
    """Return names of workflows containing a Builder AgentNode.

    Excludes workflows with SubgraphForkNode — QA runs inside the subgraph,
    not in the top-level skill prose.
    """
    from factory.workflow.definitions import register_all

    names = []
    for name, wf in register_all().items():
        if wf.terminal:
            continue
        has_builder = any(
            isinstance(n, AgentNode) and n.role == AgentRole.BUILDER for n in wf.nodes.values()
        )
        has_subgraph_fork = any(isinstance(n, SubgraphForkNode) for n in wf.nodes.values())
        if has_builder and not has_subgraph_fork:
            names.append(name)
    return sorted(names)


class TestSkillQaEnforcement:
    """Every workflow with a Builder must include QA verification in its exported SKILL.md."""

    @pytest.mark.parametrize("workflow_name", _workflows_with_builder())
    def test_builder_workflow_has_qa_in_skill(self, workflow_name: str) -> None:
        from factory.workflow.definitions import register_all

        wf = register_all()[workflow_name]
        content = workflow_to_skill_md(wf)
        qa_roles = {"health_checker", "code_reviewer", "adversarial_tester"}
        has_qa = any(f"factory agent {role}" in content for role in qa_roles)
        assert has_qa, (
            f"workflow-{workflow_name} SKILL.md is missing any QA agent invocation "
            f"(health_checker, code_reviewer, or adversarial_tester)"
        )


# ── _study_to_instruction focus threading ──────────────────────


class TestStudyToInstructionFocus:
    def test_with_focus(self) -> None:
        study = Study(
            id="study",
            command="factory study {project_path}",
            focus="auth",
        )
        wf = _minimal_workflow(nodes={"study": study}, start="study")
        result = _study_to_instruction(study, wf)
        assert '--focus "auth"' in result

    def test_without_focus_has_ceo_hint(self) -> None:
        study = Study(
            id="study",
            command="factory study {project_path}",
        )
        wf = _minimal_workflow(nodes={"study": study}, start="study")
        result = _study_to_instruction(study, wf)
        assert '--focus "auth"' not in result
        assert "focus directive" in result
        assert '--focus "<your focus topic>"' in result
