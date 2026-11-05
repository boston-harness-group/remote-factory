"""Offline unit/contract tests for agent-scoped guardrails."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from factory.workflow.executor import WorkflowExecutor
from factory.workflow.guardrails import (
    DEFAULT_AUDIT_HINT,
    GUARD_ARTIFACT_ROOT,
    GUARDRAILS_HEADER,
    GUARD_SLUG_RE,
    MAX_RULES_PAYLOAD_BYTES,
    Guard,
    _add_stage_nodes,
    _agent_segment,
    _check_command,
    _reset_command,
    _unique_slugs,
    expand_agent_guards,
    guard_dir,
    guard_rules_payload,
    guard_slug,
    with_guards,
)
from factory.workflow.package import Package
from factory.workflow.primitives import (
    MAX_GUARD_TEXT_LEN,
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    NodeType,
    SubgraphForkNode,
    VerdictType,
    Workflow,
)
from factory.workflow.validation import validate_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
AGGREGATOR = "factory.workflow.guardrails_aggregate"
RULE_A = "no egress outside the sandbox"
RULE_B = "cite every test run"


def _guard(rule: str = RULE_A, audit_hint: str = "") -> Guard:
    return Guard(rule=rule, audit_hint=audit_hint)


def _agent(node_id: str = "draft", **fields: Any) -> AgentNode:
    data: dict[str, Any] = {"role": AgentRole.BUILDER, "prompt_template": f"{node_id} task"}
    data.update(fields)
    return AgentNode(id=node_id, **data)


def _guarded(
    node_id: str = "draft",
    *,
    guards: list[Guard] | None = None,
    max_iterations: int = 3,
    **fields: Any,
) -> AgentNode:
    attached = guards if guards is not None else [_guard(), _guard(RULE_B)]
    return with_guards(_agent(node_id, **fields), attached, max_iterations=max_iterations)


def _toy(
    *,
    guards: list[Guard] | None = None,
    max_iterations: int = 3,
    successors: int = 1,
) -> Workflow:
    nodes: dict[str, Any] = {
        "draft": _guarded(guards=guards, max_iterations=max_iterations),
    }
    edges: list[Edge] = []
    if successors >= 1:
        nodes["publish"] = _agent("publish")
        edges.append(Edge(source="draft", target="publish"))
    if successors >= 2:
        nodes["publish_two"] = _agent("publish_two")
        edges.append(Edge(source="draft", target="publish_two"))
    return Workflow(name="toy-guard", nodes=nodes, edges=edges, start_node="draft")


def _stage_ids(expanded: Workflow, agent_id: str = "draft") -> set[str]:
    segment = _agent_segment(agent_id)
    return {nid for nid in expanded.nodes if nid.startswith(f"guard_{segment}_")}


def _package(graph: Workflow | None = None) -> Package:
    return Package(
        name="toy-guard",
        graph=graph if graph is not None else _toy(),
        entry_node="draft",
        exit_node="publish",
    )


def _run_aggregator(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-m", AGGREGATOR, *args],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _stage(project: Path, segment: str = "draft") -> Path:
    return project / GUARD_ARTIFACT_ROOT / segment


def _run_reset_raw(project: Path, raw_payload: str, segment: str = "draft") -> Any:
    return _run_aggregator(project, "reset", str(project), segment, raw_payload)


def _run_reset(
    project: Path, payload: dict[str, str], segment: str = "draft"
) -> Any:
    return _run_reset_raw(project, json.dumps(payload), segment)


def _run_check(project: Path, slugs: list[str], segment: str = "draft") -> Any:
    return _run_aggregator(project, "check", str(project), segment, *slugs)


def _prepare_stage(
    project: Path, payload: dict[str, str] | None = None, segment: str = "draft"
) -> Path:
    data = payload if payload is not None else {"alpha": "alpha rule", "beta": "beta rule"}
    proc = _run_reset(project, data, segment)
    assert proc.returncode == 0, proc.stderr
    return _stage(project, segment)


def _token(stage: Path) -> str:
    return (stage / "token").read_text().strip()


def _write_verdict(
    stage: Path,
    slug: str,
    token: str,
    *,
    violated: bool = False,
    evidence: str = "checked git status",
    fix: str = "",
) -> None:
    verdict = {
        "guard": slug,
        "token": token,
        "violated": violated,
        "evidence": evidence,
        "fix": fix,
    }
    (stage / f"{slug}.json").write_text(json.dumps(verdict))


def _stdout_line(proc: Any) -> str:
    lines = proc.stdout.splitlines()
    assert lines, f"no stdout; stderr={proc.stderr!r}"
    return lines[0]


def _parsing_executor(project: Path) -> WorkflowExecutor:
    workflow = Workflow(
        name="fn-gate-parsing",
        nodes={
            "draft": _agent(),
            "gate": GateNode(
                id="gate", evaluator_type="fn", evaluator_command="true", max_iterations=2
            ),
        },
        edges=[
            Edge(source="draft", target="gate"),
            Edge(source="gate", target="draft", condition=VerdictType.RELOOP),
        ],
        start_node="draft",
    )
    return WorkflowExecutor(workflow, project, validate=False)


class TestGuardModel:
    def test_defaults(self) -> None:
        guard = Guard(rule="x")
        assert guard.rule == "x"
        assert guard.audit_hint == ""
        assert Guard(rule="rule", audit_hint="hint").audit_hint == "hint"

    def test_rule_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            Guard()  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="non-empty"):
            Guard(rule="   ")
        with pytest.raises(ValidationError):
            Guard(rule="")

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Guard(rule="x", owner="builder")  # type: ignore[call-arg]

    def test_plain_json_braces_accepted(self) -> None:
        guard = Guard(rule='write {"violated": false} to the verdict file')
        assert '"violated": false' in guard.rule

    @pytest.mark.parametrize("field", ["rule", "audit_hint"])
    def test_text_length_bounds(self, field: str) -> None:
        at_limit = "x" * MAX_GUARD_TEXT_LEN
        kwargs = {"rule": "ok", field: at_limit}
        assert len(getattr(Guard(**kwargs), field)) == MAX_GUARD_TEXT_LEN
        with pytest.raises(ValidationError) as exc_info:
            Guard(**{"rule": "ok", field: "x" * (MAX_GUARD_TEXT_LEN + 1)})
        message = str(exc_info.value)
        assert field in message
        assert str(MAX_GUARD_TEXT_LEN) in message

    def test_public_import_path(self) -> None:
        from factory.workflow import primitives

        assert Guard is primitives.Guard

    def test_no_executor_import(self) -> None:
        code = (
            "import sys, factory.workflow.guardrails; "
            "assert 'factory.workflow.executor' not in sys.modules, sorted(sys.modules)"
        )
        proc = subprocess.run(
            [PYTHON, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr


class TestAgentNodeGuardFields:
    def test_defaults(self) -> None:
        node = _agent()
        assert node.guards == []
        assert node.guard_max_iterations == 3

    def test_max_iterations_ge_one(self) -> None:
        with pytest.raises(ValidationError):
            _agent(guard_max_iterations=0)

    def test_model_dump_reconstruct(self) -> None:
        node = _guarded(max_iterations=2)
        rebuilt = type(node)(**node.model_dump())
        assert [guard.rule for guard in rebuilt.guards] == [RULE_A, RULE_B]
        assert rebuilt.guard_max_iterations == 2


class TestSerialization:
    def test_roundtrip_preserves_guards(self) -> None:
        workflow = _toy()
        first = workflow.to_dict()
        loaded = Workflow.from_dict(first)
        assert [guard.model_dump() for guard in loaded.nodes["draft"].guards] == [
            guard.model_dump() for guard in workflow.nodes["draft"].guards
        ]
        assert loaded.nodes["draft"].guard_max_iterations == 3
        second = loaded.to_dict()
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_legacy_payload_without_guard_keys(self) -> None:
        data = _toy().to_dict()
        for node_data in data["nodes"].values():
            node_data.pop("guards", None)
            node_data.pop("guard_max_iterations", None)
        loaded = Workflow.from_dict(data)
        draft = loaded.nodes["draft"]
        assert isinstance(draft, AgentNode)
        assert draft.guards == []
        assert draft.guard_max_iterations == 3


class TestWithGuards:
    def test_sets_fields(self) -> None:
        attached = with_guards(_agent(), [_guard()], max_iterations=2)
        assert attached.id == "draft"
        assert [guard.rule for guard in attached.guards] == [RULE_A]
        assert attached.guard_max_iterations == 2

    def test_returns_copy_and_replaces(self) -> None:
        caller_guards = [_guard()]
        node = _guarded(guards=[_guard(RULE_A), _guard(RULE_B)])
        attached = with_guards(node, caller_guards, max_iterations=2)

        assert node.guard_max_iterations == 3
        assert len(node.guards) == 2
        assert len(attached.guards) == 1
        caller_guards.append(_guard("third rule"))
        caller_guards[0].rule = "mutated by the caller"
        assert len(attached.guards) == 1
        assert attached.guards[0].rule == RULE_A

    def test_rejects_non_agent_and_bad_budget(self) -> None:
        with pytest.raises(TypeError, match="expects an AgentNode"):
            with_guards(FnNode(id="f", command="echo"), [_guard()])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_iterations"):
            with_guards(_agent(), [_guard()], max_iterations=0)

    def test_empty_guards_is_unguarded(self) -> None:
        node = with_guards(_agent(), [])
        assert node.guards == []
        workflow = Workflow(
            name="plain", nodes={node.id: node}, edges=[], start_node=node.id
        )
        before = workflow.to_dict()
        expanded = expand_agent_guards(workflow)
        assert expanded.to_dict() == before


class TestGuardSlug:
    @pytest.mark.parametrize(
        "rule",
        [
            "No egress, outside the sandbox!",
            "---",
            "  Mixed CASE / punctuation: (a) [b] {c} <d>  ",
            "ünïcödé rüle with ümlauts",
            "a" * 120,
        ],
    )
    def test_charset(self, rule: str) -> None:
        slug = guard_slug(rule)
        assert GUARD_SLUG_RE.fullmatch(slug)
        assert slug and not slug.endswith("_")

    def test_length_and_word_boundary(self) -> None:
        long_words = " ".join(["word"] * 30)
        slug = guard_slug(long_words)
        assert len(slug) <= 40
        assert not slug.endswith("_")
        assert slug.split("_") == ["word"] * len(slug.split("_"))
        truncated = guard_slug("q" * 120)
        assert len(truncated) == 40
        assert GUARD_SLUG_RE.fullmatch(truncated)
        assert guard_slug("!!!") == "guard"

    def test_deterministic(self) -> None:
        rule = "No egress, outside the sandbox!"
        assert guard_slug(rule) == guard_slug(rule)
        code = (
            "from factory.workflow.guardrails import guard_slug; "
            "print(guard_slug('No egress, outside the sandbox!'))"
        )
        proc = subprocess.run(
            [PYTHON, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == guard_slug(rule)

    def test_unique_slugs_disambiguate(self) -> None:
        guards = [_guard(RULE_A), _guard(RULE_A), _guard(RULE_A)]
        slugs = _unique_slugs(guards)
        assert slugs == [guard_slug(RULE_A), f"{guard_slug(RULE_A)}_2", f"{guard_slug(RULE_A)}_3"]

    def test_unique_slugs_reserve_structural_suffixes(self) -> None:
        for suffix in ("reset", "fork", "join", "gate"):
            assert _unique_slugs([_guard(suffix)]) == [f"{suffix}_2"]
        guards = [_guard("fork"), _guard("fork"), _guard("gate")]
        assert _unique_slugs(guards) == _unique_slugs(guards) == ["fork_2", "fork_3", "gate_2"]


class TestExpansionShape:
    def test_node_and_edge_counts(self) -> None:
        one = expand_agent_guards(_toy(successors=1))
        assert len(one.nodes) == 2 + 4 + 2
        assert len(one.edges) == 6
        two = expand_agent_guards(_toy(successors=2))
        assert len(two.nodes) == 3 + 4 + 2
        assert len(two.edges) == 6 + (2 - 1)

    @pytest.mark.parametrize("suffix", ["reset", "fork", "join", "gate"])
    def test_reserved_stage_suffix_slug_never_overwrites(self, suffix: str) -> None:
        expanded = expand_agent_guards(_toy(guards=[_guard(suffix)]))
        segment = _agent_segment("draft")
        checker_id = f"guard_{segment}_{suffix}_2"
        checker = expanded.nodes[checker_id]
        assert isinstance(checker, AgentNode)
        reset = expanded.nodes[f"guard_{segment}_reset"]
        fork = expanded.nodes[f"guard_{segment}_fork"]
        join = expanded.nodes[f"guard_{segment}_join"]
        gate = expanded.nodes[f"guard_{segment}_gate"]
        assert isinstance(reset, FnNode)
        assert isinstance(fork, ForkNode)
        assert isinstance(join, JoinNode)
        assert isinstance(gate, GateNode)
        assert fork.targets == join.sources == [checker_id]
        assert len(expanded.nodes) == 2 + 4 + 1
        assert validate_workflow(expanded) == []
        assert expand_agent_guards(expanded).to_dict() == expanded.to_dict()

    @pytest.mark.parametrize("budget", [2, 7])
    def test_gate_carries_guard_budget(self, budget: int) -> None:
        expanded = expand_agent_guards(_toy(max_iterations=budget))
        gate = expanded.nodes[f"guard_{_agent_segment('draft')}_gate"]
        assert isinstance(gate, GateNode)
        assert gate.max_iterations == budget

    def test_ids_namespaced(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        slugs = _unique_slugs(_toy().nodes["draft"].guards)
        expected = {
            f"guard_{segment}_reset",
            f"guard_{segment}_fork",
            f"guard_{segment}_join",
            f"guard_{segment}_gate",
            *[f"guard_{segment}_{slug}" for slug in slugs],
        }
        assert _stage_ids(expanded) == expected
        assert all(GUARD_SLUG_RE.fullmatch(nid) for nid in expected)

    def test_fork_join_wiring(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        slugs = _unique_slugs(expanded.nodes["draft"].guards)
        checker_ids = [f"guard_{segment}_{slug}" for slug in slugs]
        fork = expanded.nodes[f"guard_{segment}_fork"]
        join = expanded.nodes[f"guard_{segment}_join"]
        assert isinstance(fork, ForkNode)
        assert isinstance(join, JoinNode)
        assert fork.targets == checker_ids
        assert join.sources == checker_ids

    def test_gate_forward_edges_proceed(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        gate_id = f"guard_{segment}_gate"
        forward = [
            edge
            for edge in expanded.edges
            if edge.source == gate_id and edge.condition == VerdictType.PROCEED
        ]
        assert [edge.target for edge in forward] == ["publish"]

        halted = _toy()
        halted.edges = [Edge(source="draft", target="publish", condition=VerdictType.HALT)]
        with pytest.raises(ValueError, match="only unconditional or PROCEED"):
            expand_agent_guards(halted)

    def test_reloop_edge_targets_agent(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        reloops = [
            edge
            for edge in expanded.edges
            if edge.source == f"guard_{segment}_gate"
            and edge.condition == VerdictType.RELOOP
        ]
        assert len(reloops) == 1
        assert reloops[0].target == "draft"

    def test_explicit_fork_join_edge(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        assert Edge(
            source=f"guard_{segment}_fork", target=f"guard_{segment}_join"
        ) in expanded.edges

    def test_reset_clears_and_tokenises(self) -> None:
        agent_id = "Draft Builder"
        workflow = Workflow(
            name="quoted",
            nodes={agent_id: _guarded(agent_id)},
            edges=[],
            start_node=agent_id,
        )
        expanded = expand_agent_guards(workflow)
        segment = _agent_segment(agent_id)
        reset = expanded.nodes[f"guard_{segment}_reset"]
        assert isinstance(reset, FnNode)
        node = expanded.nodes[agent_id]
        assert isinstance(node, AgentNode)
        slugs = _unique_slugs(node.guards)
        payload = guard_rules_payload(slugs, node.guards)

        assert reset.command == _reset_command(segment, node)
        assert agent_id not in reset.command
        assert segment in reset.command
        assert f"'{payload}'" in reset.command
        assert reset.writes == {
            str(guard_dir(segment) / "token"),
            str(guard_dir(segment) / "rules.json"),
        }
        assert reset.reads == set()
        gate = expanded.nodes[f"guard_{segment}_gate"]
        assert isinstance(gate, GateNode)
        assert gate.evaluator_command == _check_command(segment, slugs)
        assert payload not in (gate.evaluator_command or "")

        project = Path("/tmp/guard-project")
        substituted = reset.command.replace("{project_path}", shlex.quote(str(project)))
        assert shlex.split(substituted)[1:] == [
            "-m",
            AGGREGATOR,
            "reset",
            str(project),
            segment,
            payload,
        ]

    def test_datanode_branch_agent_rejected(self) -> None:
        data = DataNode(id="items", source_path="items.jsonl", source_format="jsonl")
        branch = _guarded("branch")
        join = JoinNode(id="join", sources=["branch"])
        workflow = Workflow(
            name="branchy",
            nodes={"items": data, "branch": branch, "join": join},
            edges=[
                Edge(source="items", target="branch"),
                Edge(source="branch", target="join"),
            ],
            start_node="items",
            runtime_inputs=frozenset({"items.jsonl"}),
        )
        with pytest.raises(ValueError, match="DataNode") as exc_info:
            expand_agent_guards(workflow)
        message = str(exc_info.value)
        assert "'branch'" in message
        assert "'items'" in message
        assert not any(nid.startswith("guard_") for nid in workflow.nodes)

    def test_subgraph_fork_agent_rejected(self) -> None:
        fork = SubgraphForkNode(
            id="branches", subgraph_entry="branch", subgraph_exit="branch_done"
        )
        workflow = Workflow(
            name="subgraph-fork",
            nodes={
                "branches": fork,
                "branch": _guarded("branch"),
                "branch_done": _agent("branch_done"),
            },
            edges=[
                Edge(source="branches", target="branch"),
                Edge(source="branch", target="branch_done"),
            ],
            start_node="branches",
        )
        with pytest.raises(ValueError, match="SubgraphForkNode") as exc_info:
            expand_agent_guards(workflow)
        message = str(exc_info.value)
        assert "'branch'" in message
        assert "'branches'" in message
        assert not any(nid.startswith("guard_") for nid in workflow.nodes)

    def test_unguarded_branch_agent_does_not_block_expansion(self) -> None:
        data = DataNode(id="items", source_path="items.jsonl", source_format="jsonl")
        workflow = Workflow(
            name="mixed",
            nodes={
                "prep": _guarded("prep"),
                "items": data,
                "branch": _agent("branch"),
                "join": JoinNode(id="join", sources=["branch"]),
            },
            edges=[
                Edge(source="prep", target="items"),
                Edge(source="items", target="branch"),
                Edge(source="branch", target="join"),
            ],
            start_node="prep",
            runtime_inputs=frozenset({"items.jsonl"}),
        )
        expanded = expand_agent_guards(workflow)
        segment = _agent_segment("prep")
        assert f"guard_{segment}_gate" in expanded.nodes
        assert f"guard_{_agent_segment('branch')}_gate" not in expanded.nodes


def _assert_quoted_round_trip(project: Path, agent_id: str, segment: str) -> None:
    workflow = Workflow(
        name="quoting",
        nodes={agent_id: _guarded(agent_id, guards=[_guard()])},
        edges=[],
        start_node=agent_id,
    )
    expanded = expand_agent_guards(workflow)
    reset = expanded.nodes[f"guard_{segment}_reset"]
    gate = expanded.nodes[f"guard_{segment}_gate"]
    assert isinstance(reset, FnNode)
    assert isinstance(gate, GateNode)
    node = expanded.nodes[agent_id]
    assert isinstance(node, AgentNode)
    payload = guard_rules_payload(_unique_slugs(node.guards), node.guards)

    assert agent_id not in reset.command
    assert agent_id not in (gate.evaluator_command or "")
    assert segment in (gate.evaluator_command or "")

    substituted = reset.command.replace("{project_path}", shlex.quote(str(project)))
    argv = shlex.split(substituted)
    assert argv[1:] == [
        "-m",
        AGGREGATOR,
        "reset",
        str(project),
        segment,
        payload,
    ]

    proc = subprocess.run(
        substituted, shell=True, cwd=project, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert (project / GUARD_ARTIFACT_ROOT / segment / "rules.json").is_file()


class TestCommandQuoting:
    def test_spaced_agent_id_is_one_argv_token(self, tmp_path: Path) -> None:
        _assert_quoted_round_trip(tmp_path, "My Builder", "my_builder")

    def test_metacharacter_agent_id_executes_nothing(self, tmp_path: Path) -> None:
        agent_id = f"draft; touch {tmp_path / 'pwned'}"
        _assert_quoted_round_trip(tmp_path, agent_id, _agent_segment(agent_id))
        assert not (tmp_path / "pwned").exists()


class TestGraphValidation:
    @pytest.mark.parametrize("shape", ["one-successor", "two-successors", "unconditional", "terminal"])
    def test_validates_clean(self, shape: str) -> None:
        successors = {"one-successor": 1, "two-successors": 2, "unconditional": 1, "terminal": 0}[
            shape
        ]
        expanded = expand_agent_guards(_toy(successors=successors))
        assert validate_workflow(expanded) == []


class TestExpansionIdempotency:
    def test_second_call_is_noop(self) -> None:
        first = expand_agent_guards(_toy())
        second = expand_agent_guards(first)
        assert first.to_dict() == second.to_dict()

    def test_reloaded_graph_not_reexpanded(self) -> None:
        first = expand_agent_guards(_toy())
        reloaded = Workflow.from_dict(first.to_dict())
        again = expand_agent_guards(reloaded)
        assert reloaded.to_dict() == again.to_dict()

    def test_guards_survive_expansion(self) -> None:
        expanded = expand_agent_guards(_toy(max_iterations=2))
        node = expanded.nodes["draft"]
        assert isinstance(node, AgentNode)
        assert [guard.rule for guard in node.guards] == [RULE_A, RULE_B]
        assert node.guard_max_iterations == 2
        gate = expanded.nodes[f"guard_{_agent_segment('draft')}_gate"]
        assert isinstance(gate, GateNode)
        assert gate.max_iterations == node.guard_max_iterations == 2

    def test_stale_stage_rejected(self) -> None:
        old_rule = "one two three four five six seven eight"
        new_rule = "one two three four five six seven eight nine ten extra"
        assert guard_slug(old_rule) == guard_slug(new_rule)

        expanded = expand_agent_guards(_toy(guards=[_guard(old_rule)]))
        stale = expanded.model_copy(deep=True)
        node = stale.nodes["draft"]
        assert isinstance(node, AgentNode)
        stale.nodes["draft"] = with_guards(
            node, [Guard(rule=new_rule)], max_iterations=node.guard_max_iterations
        )
        with pytest.raises(ValueError, match="does not match"):
            expand_agent_guards(stale)

        hinted = expanded.model_copy(deep=True)
        node = hinted.nodes["draft"]
        assert isinstance(node, AgentNode)
        hinted.nodes["draft"] = with_guards(
            node,
            [Guard(rule=old_rule, audit_hint="also check the git log")],
            max_iterations=node.guard_max_iterations,
        )
        with pytest.raises(ValueError, match="does not match"):
            expand_agent_guards(hinted)

        rebudgeted_gate = expanded.model_copy(deep=True)
        gate = rebudgeted_gate.nodes[f"guard_{_agent_segment('draft')}_gate"]
        assert isinstance(gate, GateNode)
        rebudgeted_gate.nodes[gate.id] = gate.model_copy(update={"max_iterations": 9})
        with pytest.raises(ValueError, match="does not match"):
            expand_agent_guards(rebudgeted_gate)

        rebudgeted_node = expanded.model_copy(deep=True)
        node = rebudgeted_node.nodes["draft"]
        assert isinstance(node, AgentNode)
        rebudgeted_node.nodes["draft"] = with_guards(
            node, [Guard(rule=old_rule)], max_iterations=9
        )
        with pytest.raises(ValueError, match="does not match"):
            expand_agent_guards(rebudgeted_node)

    def test_join_sources_edits_rejected(self) -> None:
        expanded = expand_agent_guards(_toy())
        join_id = f"guard_{_agent_segment('draft')}_join"
        join = expanded.nodes[join_id]
        assert isinstance(join, JoinNode)
        assert len(join.sources) == 2
        tampered = expanded.model_copy(deep=True)
        tampered.nodes[join_id] = join.model_copy(update={"sources": list(join.sources)[:1]})
        with pytest.raises(ValueError, match="does not match"):
            expand_agent_guards(tampered)

    def test_partial_stage_rejected(self) -> None:
        expanded = expand_agent_guards(_toy())
        segment = _agent_segment("draft")
        missing_join = expanded.model_copy(deep=True)
        del missing_join.nodes[f"guard_{segment}_join"]
        with pytest.raises(ValueError, match="partially present"):
            expand_agent_guards(missing_join)

        node = expanded.nodes["draft"]
        assert isinstance(node, AgentNode)
        slug = _unique_slugs(node.guards)[0]
        missing_checker = expanded.model_copy(deep=True)
        del missing_checker.nodes[f"guard_{segment}_{slug}"]
        with pytest.raises(ValueError, match="partially present"):
            expand_agent_guards(missing_checker)


class TestExpansionPurity:
    def test_input_unchanged(self) -> None:
        workflow = _toy()
        before = workflow.to_dict()
        expand_agent_guards(workflow)
        assert workflow.to_dict() == before


class TestInjection:
    def test_scoped_and_ordered(self) -> None:
        rule_a = "alpha rule"
        rule_b = "beta rule"
        expanded = expand_agent_guards(_toy(guards=[_guard(rule_a), _guard(rule_b)]))
        draft_prompt = expanded.nodes["draft"].prompt_template
        assert GUARDRAILS_HEADER in draft_prompt
        assert draft_prompt.index(f"- {rule_a}") < draft_prompt.index(f"- {rule_b}")
        assert GUARDRAILS_HEADER not in expanded.nodes["publish"].prompt_template
        segment = _agent_segment("draft")
        default_checker = expanded.nodes[f"guard_{segment}_{guard_slug(rule_a)}"]
        assert f"How to establish it: {DEFAULT_AUDIT_HINT}" in default_checker.prompt_template
        hinted = expand_agent_guards(
            _toy(guards=[_guard(rule_a, audit_hint="diff the working tree")])
        )
        hinted_checker = hinted.nodes[f"guard_{segment}_{guard_slug(rule_a)}"]
        assert "How to establish it: diff the working tree" in hinted_checker.prompt_template
        for node_id, node in expanded.nodes.items():
            if node_id.startswith("guard_") and isinstance(node, AgentNode):
                assert GUARDRAILS_HEADER not in node.prompt_template

    def test_idempotent(self) -> None:
        once = expand_agent_guards(_toy())
        twice = expand_agent_guards(once)
        assert twice.nodes["draft"].prompt_template.count(GUARDRAILS_HEADER) == 1

    def test_no_template_markers(self) -> None:
        expanded = expand_agent_guards(_toy())
        for node in expanded.nodes.values():
            assert "{{" not in getattr(node, "prompt_template", "")
            assert "}}" not in getattr(node, "prompt_template", "")
            assert "<!--" not in getattr(node, "prompt_template", "")
            assert "-->" not in getattr(node, "prompt_template", "")
            assert "{{" not in (getattr(node, "command", "") or "")
            assert "}}" not in (getattr(node, "command", "") or "")
            assert "<!--" not in (getattr(node, "command", "") or "")
            assert "-->" not in (getattr(node, "command", "") or "")


class TestCompileHook:
    def test_compile_expands(self) -> None:
        package = _package()
        compiled = package.compile()
        segment = _agent_segment("draft")
        assert f"guard_{segment}_gate" in compiled.nodes
        assert not any(nid.startswith("guard_") for nid in package.graph.nodes)
        assert len(package.graph.nodes) == 2
        assert compiled.nodes["draft"].guards

    def test_injection_after_prompt_knob(self) -> None:
        workflow = _toy()
        workflow.knob_values["_prompt_draft"] = "MUTATED BY THE OUTER LOOP"
        compiled = _package(workflow).compile()
        prompt = compiled.nodes["draft"].prompt_template
        assert prompt.startswith("MUTATED BY THE OUTER LOOP")
        assert prompt.count(GUARDRAILS_HEADER) == 1
        assert RULE_A in prompt
        assert GUARDRAILS_HEADER not in compiled.nodes["publish"].prompt_template


class TestPromptRepair:
    def test_reinjection_after_prompt_mutation(self) -> None:
        expanded = expand_agent_guards(_toy())
        stage_nodes = set(expanded.nodes)
        edge_count = len(expanded.edges)

        mutated = expanded.model_copy(deep=True)
        draft = mutated.nodes["draft"]
        mutated.nodes["draft"] = draft.model_copy(update={"prompt_template": "WIPED PROMPT"})
        repaired = expand_agent_guards(mutated)

        repaired_prompt = repaired.nodes["draft"].prompt_template
        assert repaired_prompt.startswith("WIPED PROMPT")
        assert repaired_prompt.count(GUARDRAILS_HEADER) == 1
        assert set(repaired.nodes) == stage_nodes
        assert len(repaired.edges) == edge_count

    def test_unguarded_package_unchanged(self) -> None:
        workflow = Workflow(
            name="plain",
            nodes={"draft": _agent(), "publish": _agent("publish")},
            edges=[Edge(source="draft", target="publish")],
            start_node="draft",
        )
        compiled = _package(workflow).compile()
        assert len(compiled.nodes) == 2
        assert len(compiled.edges) == 1
        assert not any(nid.startswith("guard_") for nid in compiled.nodes)


class TestExpansionValidation:
    def test_fork_target_rejected(self) -> None:
        workflow = Workflow(
            name="fork-target",
            nodes={
                "fork": ForkNode(id="fork", targets=["draft"]),
                "draft": _guarded("draft"),
            },
            edges=[Edge(source="fork", target="draft")],
            start_node="fork",
        )
        with pytest.raises(ValueError) as exc_info:
            expand_agent_guards(workflow)
        assert "'draft'" in str(exc_info.value)
        assert "fork" in str(exc_info.value).lower()

    def test_non_blocking_rejected(self) -> None:
        workflow = Workflow(
            name="non-blocking",
            nodes={"draft": _guarded("draft", blocking=False)},
            edges=[],
            start_node="draft",
        )
        with pytest.raises(ValueError, match="blocking=False") as exc_info:
            expand_agent_guards(workflow)
        assert "'draft'" in str(exc_info.value)
        assert not any(nid.startswith("guard_") for nid in workflow.nodes)

    def test_payload_over_argv_budget_rejected(self) -> None:
        oversized = [
            _guard(f"r{index:02d}:" + "x" * (MAX_GUARD_TEXT_LEN - 4)) for index in range(17)
        ]
        payload_bytes = len(
            guard_rules_payload(_unique_slugs(oversized), oversized).encode("ascii")
        )
        assert payload_bytes > MAX_RULES_PAYLOAD_BYTES
        with pytest.raises(ValueError) as exc_info:
            expand_agent_guards(_toy(guards=oversized))
        message = str(exc_info.value)
        assert "'draft'" in message
        assert str(payload_bytes) in message
        assert str(MAX_RULES_PAYLOAD_BYTES) in message

        max_rules = [_guard("x" * MAX_GUARD_TEXT_LEN) for _ in range(15)]
        base = guard_rules_payload(_unique_slugs(max_rules), max_rules)
        last_slug = guard_slug("q" * 40)
        skeleton = base[:-1] + f',"{last_slug}":"'
        padding = MAX_RULES_PAYLOAD_BYTES - len(skeleton.encode("ascii")) - len('"}')
        assert 40 <= padding <= MAX_GUARD_TEXT_LEN
        at_cap = [*max_rules, _guard("q" * padding)]
        payload = guard_rules_payload(_unique_slugs(at_cap), at_cap)
        assert len(payload.encode("ascii")) == MAX_RULES_PAYLOAD_BYTES
        expanded = expand_agent_guards(_toy(guards=at_cap))
        assert f"guard_{_agent_segment('draft')}_gate" in expanded.nodes

    def test_segment_normalisation(self) -> None:
        assert _agent_segment("patch-builder") == "patch_builder"
        assert _agent_segment("My Builder") == "my_builder"
        assert _agent_segment("patch_builder") == "patch_builder"
        assert guard_dir("patch-builder") == Path(GUARD_ARTIFACT_ROOT) / "patch_builder"
        assert guard_dir("patch-builder") == guard_dir("patch_builder")
        with pytest.raises(ValueError):
            _agent_segment("")

    def test_duplicate_segment_rejected(self) -> None:
        workflow = Workflow(
            name="duplicate-segments",
            nodes={
                "patch-builder": _guarded("patch-builder"),
                "patch_builder": _guarded("patch_builder"),
            },
            edges=[],
            start_node="patch-builder",
        )
        with pytest.raises(ValueError) as exc_info:
            expand_agent_guards(workflow)
        message = str(exc_info.value)
        assert "'patch-builder'" in message
        assert "'patch_builder'" in message
        assert "share guard segment" in message

    def test_stage_insert_never_overwrites(self) -> None:
        workflow = Workflow(
            name="overwrite", nodes={"draft": _agent()}, edges=[], start_node="draft"
        )
        colliding: dict[str, NodeType] = {"draft": ForkNode(id="draft", targets=[])}
        with pytest.raises(ValueError, match="collides") as exc_info:
            _add_stage_nodes(workflow, colliding, "draft", {})
        message = str(exc_info.value)
        assert "'draft'" in message
        assert "existing node" in message
        assert isinstance(workflow.nodes["draft"], AgentNode)

    def test_two_segment_checker_id_collision_named(self) -> None:
        left_rule, right_rule = "b c", "c"
        left_id = f"guard_{_agent_segment('a')}_{guard_slug(left_rule)}"
        right_id = f"guard_{_agent_segment('a_b')}_{guard_slug(right_rule)}"
        assert left_id == right_id == "guard_a_b_c"
        workflow = Workflow(
            name="cross-segment",
            nodes={
                "a": _guarded("a", guards=[_guard(left_rule)]),
                "a_b": _guarded("a_b", guards=[_guard(right_rule)]),
            },
            edges=[],
            start_node="a",
        )
        with pytest.raises(ValueError) as exc_info:
            expand_agent_guards(workflow)
        message = str(exc_info.value)
        assert "guard_a_b_c" in message
        assert "'a'" in message and "'a_b'" in message
        assert "collides" in message
        assert "partially present" not in message
        assert "guard_a_b_c" not in workflow.nodes

    def test_two_agents_no_collision(self) -> None:
        shared = _guard("shared rule")
        workflow = Workflow(
            name="two-agents",
            nodes={
                "first": _guarded("first", guards=[shared]),
                "second": _guarded("second", guards=[shared]),
            },
            edges=[],
            start_node="first",
        )
        expanded = expand_agent_guards(workflow)
        first_ids = _stage_ids(expanded, "first")
        second_ids = _stage_ids(expanded, "second")
        assert first_ids and second_ids
        assert first_ids.isdisjoint(second_ids)
        assert guard_dir("first") != guard_dir("second")


class TestAggregatorCheck:
    def test_pass(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path)
        token = _token(stage)
        _write_verdict(stage, "alpha", token)
        _write_verdict(stage, "beta", token)
        proc = _run_check(tmp_path, ["alpha", "beta"])
        assert proc.returncode == 0
        assert _stdout_line(proc) == "pass: 2 guard(s) clean"
        assert not (stage / "violations.md").exists()

    def test_reloop(self, tmp_path: Path) -> None:
        rule = "no egress outside the sandbox"
        stage = _prepare_stage(tmp_path, {"alpha": rule, "beta": "beta rule"})
        token = _token(stage)
        _write_verdict(
            stage, "alpha", token, violated=True, evidence="curl in diff", fix="drop the curl call"
        )
        _write_verdict(stage, "beta", token)
        proc = _run_check(tmp_path, ["alpha", "beta"])
        assert proc.returncode == 0
        line1 = _stdout_line(proc)
        assert line1 == "reloop: alpha: drop the curl call"
        assert rule not in line1
        text = (stage / "violations.md").read_text()
        assert f"- rule: {rule}" in text
        assert "- evidence: curl in diff" in text
        assert "- fix: drop the curl call" in text

    def test_missing(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path)
        _write_verdict(stage, "alpha", _token(stage))
        proc = _run_check(tmp_path, ["alpha", "beta"])
        assert proc.returncode == 0
        line1 = _stdout_line(proc)
        assert line1.startswith("fail: guard verdicts missing or stale:")
        assert "beta: missing" in line1
        assert not (stage / "violations.md").exists()

    @pytest.mark.parametrize("remove_token_file", [False, True])
    def test_stale_token(self, tmp_path: Path, remove_token_file: bool) -> None:
        stage = _prepare_stage(tmp_path)
        token = _token(stage)
        _write_verdict(stage, "alpha", token, violated=True, fix="fix it")
        if remove_token_file:
            (stage / "token").unlink()
        else:
            _write_verdict(stage, "alpha", "0" * 32, violated=True, fix="fix it")
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert "alpha: stale" in _stdout_line(proc)
        assert not (stage / "violations.md").exists()

    def test_token_file_absent_is_stale(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path)
        token = _token(stage)
        (stage / "token").unlink()
        _write_verdict(stage, "alpha", token)
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        line1 = _stdout_line(proc)
        assert line1.startswith("fail: guard verdicts missing or stale:")
        assert "alpha: stale" in line1
        assert "alpha: missing" not in line1
        assert not (stage / "violations.md").exists()

    @pytest.mark.parametrize(
        "variant", ["truncated", "list", "string-violated", "wrong-guard", "non-object"]
    )
    def test_malformed_variants(self, tmp_path: Path, variant: str) -> None:
        stage = _prepare_stage(tmp_path)
        valid = {
            "guard": "alpha",
            "token": _token(stage),
            "violated": False,
            "evidence": "e",
            "fix": "",
        }
        raw = {
            "truncated": '{"guard": "alpha"',
            "list": '["alpha"]',
            "string-violated": json.dumps({**valid, "violated": "false"}),
            "wrong-guard": json.dumps({**valid, "guard": "beta"}),
            "non-object": '"not an object"',
        }[variant]
        (stage / "alpha.json").write_text(raw)
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        line1 = _stdout_line(proc)
        assert line1.startswith("fail: guard verdicts missing or stale:")
        assert "alpha: malformed" in line1

    def test_first_line_only(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path, {"alpha": "a", "beta": "b", "gamma": "c"})
        proc = _run_check(tmp_path, ["alpha", "beta", "gamma"])
        assert proc.returncode == 0
        lines = proc.stdout.splitlines()
        assert lines[0].startswith("fail: guard verdicts missing or stale:")
        for slug in ("alpha", "beta", "gamma"):
            assert f"{slug}: missing" in lines[0]
        assert "alpha: missing" in proc.stderr
        assert not (stage / "violations.md").exists()


class TestRuleChannel:
    def test_rules_json_written(self, tmp_path: Path) -> None:
        payload = {"alpha": "rule one", "beta": 'quotes " and\nnewline'}
        stage = _prepare_stage(tmp_path, payload)
        expected = (
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        )
        assert (stage / "rules.json").read_text() == expected
        guards = [Guard(rule=payload["alpha"]), Guard(rule=payload["beta"])]
        assert (stage / "rules.json").read_text() == guard_rules_payload(
            ["alpha", "beta"], guards
        ) + "\n"

    def test_rule_reaches_violations_md_from_reset(self, tmp_path: Path) -> None:
        rule = "no egress outside the sandbox"
        stage = _prepare_stage(tmp_path, {"alpha": rule})
        token = _token(stage)
        _write_verdict(stage, "alpha", token, violated=True, evidence="curl found", fix="remove it")
        assert "rule" not in json.loads((stage / "alpha.json").read_text())
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert f"- rule: {rule}" in (stage / "violations.md").read_text()

    def test_check_without_registry_fails_closed(self, tmp_path: Path) -> None:
        stage = _stage(tmp_path)
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert _stdout_line(proc).startswith("fail: guard rule registry missing:")
        assert not (stage / "violations.md").exists()

        _prepare_stage(tmp_path, {"alpha": "a"})
        proc = _run_check(tmp_path, ["alpha", "beta"])
        assert proc.returncode == 0
        assert _stdout_line(proc) == "fail: guard rule registry mismatch: beta"

        (stage / "rules.json").write_text("[1, 2]")
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert _stdout_line(proc).startswith("fail: guard rule registry malformed:")


class TestAggregatorReset:
    def test_reset(self, tmp_path: Path) -> None:
        stage = _stage(tmp_path)
        stage.mkdir(parents=True)
        (stage / "alpha.json").write_text("{}")
        (stage / "violations.md").write_text("stale violations")
        proc = _run_reset(tmp_path, {"alpha": "a", "beta": "b"})
        assert proc.returncode == 0
        assert not (stage / "alpha.json").exists()
        assert not (stage / "violations.md").exists()
        assert json.loads((stage / "rules.json").read_text()) == {"alpha": "a", "beta": "b"}
        assert _token(stage)
        assert "guard stage reset" in proc.stdout

        proc = _run_reset(tmp_path, {"gamma": "c"})
        assert proc.returncode == 0
        assert json.loads((stage / "rules.json").read_text()) == {"gamma": "c"}

    def test_reset_idempotent(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path)
        first_token = _token(stage)
        first_rules = (stage / "rules.json").read_bytes()
        proc = _run_reset(tmp_path, {"alpha": "alpha rule", "beta": "beta rule"})
        assert proc.returncode == 0
        assert (stage / "rules.json").read_bytes() == first_rules
        second_token = _token(stage)
        assert second_token
        assert second_token != first_token

    def test_reset_over_cap_exits_2(self, tmp_path: Path) -> None:
        oversized = {f"r{index:02d}": "x" * MAX_GUARD_TEXT_LEN for index in range(17)}
        canonical = json.dumps(
            oversized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        assert len(canonical.encode("ascii")) > MAX_RULES_PAYLOAD_BYTES
        stage = _stage(tmp_path)
        proc = _run_reset_raw(tmp_path, json.dumps(oversized))
        assert proc.returncode == 2
        assert "usage" in proc.stderr.lower()
        assert not (stage / "rules.json").exists()
        assert not (stage / "token").exists()

    def test_token_trailing_newline_matches(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path, {"alpha": "alpha rule"})
        token_file = (stage / "token").read_text()
        assert token_file.endswith("\n")
        bare = token_file.strip()
        _write_verdict(stage, "alpha", bare)
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert _stdout_line(proc) == "pass: 1 guard(s) clean"

        escaped = {
            "guard": "alpha",
            "token": f"{bare}\n",
            "violated": False,
            "evidence": "",
            "fix": "",
        }
        (stage / "alpha.json").write_text(json.dumps(escaped))
        proc = _run_check(tmp_path, ["alpha"])
        assert proc.returncode == 0
        assert _stdout_line(proc) == "pass: 1 guard(s) clean"


class TestFnGateParsing:
    def test_real_stdout(self, tmp_path: Path) -> None:
        stage = _prepare_stage(tmp_path, {"alpha": "alpha rule"})
        token = _token(stage)
        executor = _parsing_executor(tmp_path)

        _write_verdict(stage, "alpha", token)
        pass_verdict = executor._parse_fn_verdict(
            _run_check(tmp_path, ["alpha"]).stdout, "gate", max_iterations=2
        )
        assert pass_verdict.type == VerdictType.PROCEED

        _write_verdict(stage, "alpha", token, violated=True, fix="fix the thing")
        reloop_verdict = executor._parse_fn_verdict(
            _run_check(tmp_path, ["alpha"]).stdout, "gate", max_iterations=2
        )
        assert reloop_verdict.type == VerdictType.RELOOP
        assert reloop_verdict.target == "draft"
        assert reloop_verdict.feedback == "alpha: fix the thing"
        assert reloop_verdict.max_iterations == 2

        (stage / "alpha.json").unlink()
        fail_verdict = executor._parse_fn_verdict(
            _run_check(tmp_path, ["alpha"]).stdout, "gate", max_iterations=2
        )
        assert fail_verdict.type == VerdictType.HALT
        assert fail_verdict.reason is not None
        assert fail_verdict.reason.startswith("precheck failed: fail:")
