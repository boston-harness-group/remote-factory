"""Tests for outer loop Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factory.outer_loop.models import (
    GenerationSummary,
    HyperparameterRecord,
    Individual,
    MutationRecord,
    MutationType,
    OuterLoopState,
    SwarmConfig,
)


class TestMutationType:
    def test_all_variants(self) -> None:
        assert len(MutationType) == 8
        assert MutationType.NODE_INSERT.value == "node_insert"
        assert MutationType.PARAM_MUTATE.value == "param_mutate"
        assert MutationType.PROMPT_MUTATE.value == "prompt_mutate"


class TestMutationRecord:
    def test_basic(self) -> None:
        rec = MutationRecord(
            operator=MutationType.NODE_INSERT,
            target_node="agent_1",
            rationale="test",
        )
        assert rec.operator == MutationType.NODE_INSERT
        assert rec.before == {}
        assert rec.after == {}

    def test_round_trip(self) -> None:
        rec = MutationRecord(
            operator=MutationType.EDGE_REDIRECT,
            target_node="gate_1",
            before={"target": "a"},
            after={"target": "b"},
            rationale="redirect",
        )
        dumped = rec.model_dump(mode="json")
        restored = MutationRecord.model_validate(dumped)
        assert restored == rec

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            MutationRecord(
                operator=MutationType.NODE_INSERT,
                target_node="x",
                rationale="test",
                unknown_field="bad",  # type: ignore[call-arg]
            )


class TestIndividual:
    def test_basic(self) -> None:
        ind = Individual(
            id="abc123",
            workflow_data={"name": "test"},
            score=0.85,
            features=(3, 2, 5, 1),
            generation=1,
        )
        assert ind.score == 0.85
        assert ind.features == (3, 2, 5, 1)
        assert ind.parent_id is None

    def test_round_trip(self) -> None:
        ind = Individual(
            id="xyz",
            workflow_data={"name": "w"},
            score=0.5,
            features=(1, 0, 2, 1),
            generation=0,
            parent_id="abc",
            mutation_record=MutationRecord(
                operator=MutationType.NODE_REMOVE,
                target_node="n1",
                rationale="r",
            ),
            cost_usd=1.5,
        )
        dumped = ind.model_dump(mode="json")
        restored = Individual.model_validate(dumped)
        assert restored.parent_id == "abc"
        assert restored.mutation_record is not None
        assert restored.mutation_record.operator == MutationType.NODE_REMOVE


class TestHyperparameterRecord:
    def test_basic(self) -> None:
        rec = HyperparameterRecord(
            generation=0,
            mutation_rate=0.3,
            population_size=4,
            tournament_size=3,
            designer_ratio=0.3,
            operator_weights={"node_insert": 0.2, "node_remove": 0.15},
            best_score=0.8,
            mean_score=0.6,
            diversity=0.4,
            novel_count=3,
        )
        assert rec.generation == 0
        assert rec.operator_weights["node_insert"] == 0.2

    def test_round_trip(self) -> None:
        rec = HyperparameterRecord(
            generation=5,
            mutation_rate=0.5,
            population_size=8,
            tournament_size=5,
            designer_ratio=0.4,
        )
        dumped = rec.model_dump(mode="json")
        restored = HyperparameterRecord.model_validate(dumped)
        assert restored == rec


class TestSwarmConfig:
    def test_defaults(self) -> None:
        cfg = SwarmConfig(benchmark="featurebench", budget=100)
        assert cfg.population_size == 4
        assert cfg.tournament_size == 3
        assert cfg.mutation_rate == 0.3
        assert cfg.designer_count == 2
        assert cfg.mutation_strategy == "weighted_random"
        assert cfg.target_project == ""

    def test_target_project(self) -> None:
        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=50,
            target_project="/tmp/featurebench-cancel-async",
        )
        assert cfg.target_project == "/tmp/featurebench-cancel-async"

    def test_target_project_round_trip(self) -> None:
        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=50,
            target_project="/tmp/test-project",
        )
        dumped = cfg.model_dump(mode="json")
        restored = SwarmConfig.model_validate(dumped)
        assert restored.target_project == "/tmp/test-project"

    def test_training_instances_accepted(self) -> None:
        cfg = SwarmConfig(
            benchmark="test",
            budget=50,
            training_instances=["p1", "p2", "p3"],
        )
        assert len(cfg.training_instances) == 3

    def test_no_holdout_instances_field(self) -> None:
        """holdout_instances was removed from SwarmConfig (PR A)."""
        cfg = SwarmConfig(benchmark="test", budget=50)
        assert not hasattr(cfg, "holdout_instances")


class TestOuterLoopState:
    def test_defaults(self) -> None:
        state = OuterLoopState()
        assert state.generation == 0
        assert state.convergence_reason is None
        assert state.hyperparameter_history == []

    def test_with_history(self) -> None:
        rec = HyperparameterRecord(
            generation=0,
            mutation_rate=0.3,
            population_size=4,
            tournament_size=3,
            designer_ratio=0.3,
        )
        state = OuterLoopState(
            generation=1,
            total_evaluations=8,
            best_score=0.85,
            budget_remaining=92,
            score_trajectory=[0.7, 0.85],
            hyperparameter_history=[rec],
        )
        dumped = state.model_dump(mode="json")
        restored = OuterLoopState.model_validate(dumped)
        assert len(restored.hyperparameter_history) == 1


class TestGenerationSummary:
    def test_basic(self) -> None:
        summary = GenerationSummary(
            generation=0,
            population_size=4,
            best_score=0.8,
            mean_score=0.6,
            diversity=0.4,
            novel_count=3,
            rejected_duplicates=1,
        )
        assert summary.hyperparameters is None
        assert summary.mutations_applied == []

    def test_with_mutations(self) -> None:
        rec = MutationRecord(
            operator=MutationType.PARALLELIZE,
            rationale="speed up",
        )
        summary = GenerationSummary(
            generation=1,
            population_size=4,
            best_score=0.9,
            mean_score=0.75,
            diversity=0.5,
            mutations_applied=[rec],
        )
        assert len(summary.mutations_applied) == 1
