"""Tests for seed population diversity with designer-created variants."""

from __future__ import annotations

from factory.outer_loop.designer import DesignerAgent
from factory.outer_loop.engine import SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.models import EvalResult, SwarmConfig
from factory.outer_loop.similarity import NoveltyFilter, compute_features
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    VerdictType,
    Workflow,
)


def _make_config(**overrides: object) -> SwarmConfig:
    defaults: dict[str, object] = {
        "benchmark": "test_bench",
        "budget": 50,
        "population_size": 6,
        "tournament_size": 2,
        "mutation_rate": 0.3,
        "training_instances": ["t1", "t2"],
        "designer_count": 2,
    }
    defaults.update(overrides)
    return SwarmConfig(**defaults)  # type: ignore[arg-type]


def _make_base_workflow() -> Workflow:
    return Workflow(
        name="seed_base",
        nodes={
            "study": FnNode(
                id="study", command="factory study", writes={".factory/obs.md"},
            ),
            "researcher": AgentNode(
                id="researcher", role=AgentRole.RESEARCHER,
                reads={".factory/obs.md"}, writes={".factory/research.md"},
            ),
            "strategist": AgentNode(
                id="strategist", role=AgentRole.STRATEGIST,
                reads={".factory/research.md"}, writes={".factory/current.md"},
            ),
            "builder": AgentNode(
                id="builder", role=AgentRole.BUILDER,
                reads={".factory/current.md"}, writes={".factory/build.md"},
            ),
            "gate": GateNode(
                id="gate", evaluator_type="fn",
                reads={".factory/build.md"},
            ),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="strategist"),
            Edge(source="strategist", target="builder"),
            Edge(source="builder", target="gate"),
            Edge(source="gate", target="builder", condition=VerdictType.RELOOP),
        ],
        start_node="study",
    )


def _make_noop_evaluator(config: SwarmConfig) -> SwarmEvaluator:
    def noop_eval(wf: Workflow, project_dir: str, instances: list[str]) -> EvalResult:
        return EvalResult(
            score=0.5, benchmark_score=0.5, hygiene_score=0.5,
            cost_usd=0.01, complexity=float(len(wf.nodes)),
        )
    return SwarmEvaluator(config, evaluator_fn=noop_eval)


class TestSeedWithDesigner:
    def test_seed_includes_designer_variants(self) -> None:
        config = _make_config(population_size=6, designer_count=2)
        evaluator = _make_noop_evaluator(config)
        novelty = NoveltyFilter(min_edit_distance=1)
        engine = SwarmEngine(config, evaluator, novelty_filter=novelty)
        wf = _make_base_workflow()

        pop = engine.seed(wf)

        assert pop.size >= 3
        originals = [i for i in pop.individuals if i.parent_id is None]
        assert len(originals) >= 2

    def test_feature_vectors_differ(self) -> None:
        designer = DesignerAgent()
        minimal = designer.design_minimal("test")
        thorough = designer.design_thorough("test")

        min_features = compute_features(minimal)
        thor_features = compute_features(thorough)

        assert min_features != thor_features
        assert min_features[2] < thor_features[2]

    def test_designer_count_zero_skips_designs(self) -> None:
        config = _make_config(population_size=4, designer_count=0)
        evaluator = _make_noop_evaluator(config)
        engine = SwarmEngine(config, evaluator)
        wf = _make_base_workflow()

        pop = engine.seed(wf)

        originals = [i for i in pop.individuals if i.parent_id is None]
        assert len(originals) == 1

    def test_designer_count_3_includes_custom(self) -> None:
        config = _make_config(population_size=8, designer_count=3)
        evaluator = _make_noop_evaluator(config)
        novelty = NoveltyFilter(min_edit_distance=1)
        engine = SwarmEngine(config, evaluator, novelty_filter=novelty)
        wf = _make_base_workflow()

        pop = engine.seed(wf)

        originals = [i for i in pop.individuals if i.parent_id is None]
        assert len(originals) >= 3

    def test_minimal_has_fewer_nodes_than_thorough(self) -> None:
        designer = DesignerAgent()
        minimal = designer.design_minimal("test")
        thorough = designer.design_thorough("test")

        assert len(minimal.nodes) < len(thorough.nodes)

    def test_minimal_has_fewer_agents_than_thorough(self) -> None:
        designer = DesignerAgent()
        minimal = designer.design_minimal("test")
        thorough = designer.design_thorough("test")

        min_agents = sum(
            1 for n in minimal.nodes.values() if type(n).__name__ == "AgentNode"
        )
        thor_agents = sum(
            1 for n in thorough.nodes.values() if type(n).__name__ == "AgentNode"
        )
        assert min_agents < thor_agents
