"""Population management and MAP-Elites archive for evolutionary search."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.outer_loop.models import Individual
from factory.outer_loop.similarity import compute_features

if TYPE_CHECKING:
    from factory.workflow.primitives import Workflow

log = structlog.get_logger()


class Population:
    """Manages a collection of Individual candidates."""

    def __init__(self) -> None:
        self._individuals: dict[str, Individual] = {}

    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self) -> list[Individual]:
        return list(self._individuals.values())

    def add(self, individual: Individual) -> None:
        self._individuals[individual.id] = individual

    def remove(self, individual_id: str) -> Individual | None:
        return self._individuals.pop(individual_id, None)

    def get(self, individual_id: str) -> Individual | None:
        return self._individuals.get(individual_id)

    def best(self) -> Individual | None:
        if not self._individuals:
            return None
        scored = [
            i for i in self._individuals.values()
            if i.score is not None and not i.errored
        ]
        if not scored:
            return None
        return max(scored, key=lambda i: i.score)  # type: ignore[arg-type,return-value]

    def mean_score(self) -> float:
        if not self._individuals:
            return 0.0
        scored = [
            i for i in self._individuals.values()
            if i.score is not None and not i.errored
        ]
        if not scored:
            return 0.0
        return sum(i.score for i in scored) / len(scored)  # type: ignore[misc]

    @staticmethod
    def make_individual(
        workflow: Workflow,
        *,
        generation: int = 0,
        parent_id: str | None = None,
        mutation_record: object = None,
        score: float | None = None,
        cost_usd: float = 0.0,
    ) -> Individual:
        """Create an Individual from a Workflow, computing features automatically."""
        from factory.outer_loop.models import MutationRecord

        features = compute_features(workflow)
        return Individual(
            id=uuid.uuid4().hex[:12],
            workflow_data=workflow.to_dict(),
            score=score,
            features=features,
            generation=generation,
            parent_id=parent_id,
            mutation_record=mutation_record if isinstance(mutation_record, MutationRecord) else None,
            cost_usd=cost_usd,
        )

    def save(self, directory: Path) -> None:
        """Serialize the population to a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        data = [ind.model_dump(mode="json") for ind in self._individuals.values()]
        (directory / "population.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Population:
        """Deserialize a population from a directory."""
        pop = cls()
        path = directory / "population.json"
        if path.exists():
            data = json.loads(path.read_text())
            for item in data:
                pop.add(Individual.model_validate(item))
        return pop


# Structural axes from compute_features() 9-tuple:
# 0=depth, 1=fork_degree, 2=agent_count, 3=gate_count, 8=has_data_node
# Indices 4-7 are hash buckets (edge_sig, param, prompt, knob) — excluded.
STRUCTURAL_AXES: tuple[int, ...] = (0, 1, 2, 3, 8)


class MAPElitesArchive:
    """4D fixed-resolution grid archive for quality-diversity search.

    Axes: (depth, fork_degree, agent_count, gate_count).
    Each cell stores the best-scoring Individual for that feature combination.
    """

    def __init__(self) -> None:
        self._grid: dict[tuple[int, ...], Individual] = {}

    @property
    def size(self) -> int:
        return len(self._grid)

    def add(self, individual: Individual) -> bool:
        """Add an individual to the archive. Returns True if it was inserted or replaced.

        Rejects individuals with ``score is None`` (not yet evaluated)
        or ``errored=True`` (evaluation failed).
        """
        if individual.score is None or individual.errored:
            return False
        key = individual.features
        existing = self._grid.get(key)
        if existing is None or individual.score > (existing.score or 0.0):
            self._grid[key] = individual
            return True
        return False

    def best(self) -> Individual | None:
        if not self._grid:
            return None
        scored = [
            i for i in self._grid.values()
            if i.score is not None and not i.errored
        ]
        if not scored:
            return None
        return max(scored, key=lambda i: i.score)  # type: ignore[arg-type,return-value]

    def all_individuals(self) -> list[Individual]:
        return list(self._grid.values())

    def sample_parent(
        self,
        tournament_size: int = 3,
        rank_weighted: bool = False,
        auto_rank_weighted: bool = True,
        auto_rank_cell_threshold: int = 8,
    ) -> Individual | None:
        """Tournament selection: pick tournament_size individuals, return the best.

        When ``rank_weighted=True``, individuals are drawn with probability
        proportional to their rank (best=N, worst=1) instead of uniformly.
        This biases toward stronger parents while still allowing weaker
        individuals a small chance, preserving diversity.

        When ``auto_rank_weighted=True`` (default), rank-weighted selection
        activates automatically when the archive reaches
        ``auto_rank_cell_threshold`` occupied cells, mirroring
        ``on_plateau()``'s pattern of adapting strategy when the search
        state calls for it.
        """
        import random

        individuals = [i for i in self._grid.values() if i.score is not None]
        if not individuals:
            return None
        k = min(tournament_size, len(individuals))

        use_rank = rank_weighted
        if not use_rank and auto_rank_weighted and len(individuals) >= auto_rank_cell_threshold:
            use_rank = True

        if use_rank and len(individuals) >= 2:
            ranked = sorted(individuals, key=lambda i: i.score)  # type: ignore[arg-type,return-value]
            weights = [rank + 1.0 for rank in range(len(ranked))]
            tournament = random.choices(ranked, weights=weights, k=k)
        else:
            tournament = random.sample(individuals, k)
        return max(tournament, key=lambda i: i.score)  # type: ignore[arg-type,return-value]

    def pareto_front(self) -> list[Individual]:
        """Return the Pareto-optimal individuals (non-dominated on score + features).

        An individual is dominated if another has >= score and dominates on
        all feature axes (higher is better for diversity purposes).
        Individuals with ``score is None`` are excluded.
        """
        individuals = [i for i in self._grid.values() if i.score is not None]
        if len(individuals) <= 1:
            return list(individuals)

        front: list[Individual] = []
        for candidate in individuals:
            assert candidate.score is not None  # guaranteed by filter above
            dominated = False
            for other in individuals:
                if other is candidate:
                    continue
                assert other.score is not None  # guaranteed by filter above
                if other.score >= candidate.score and all(
                    o >= c for o, c in zip(other.features, candidate.features)
                ) and (
                    other.score > candidate.score
                    or any(o > c for o, c in zip(other.features, candidate.features))
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        return front

    def diversity_metric(self) -> float:
        """Fraction of occupied cells in the structural-axis subspace.

        Returns 0.0 for empty archive, approaches 1.0 as more cells are filled.
        Uses 5 structural axes from the 9-tuple returned by compute_features():
          depth (0), fork_degree (1), agent_count (2), gate_count (3), has_data_node (8).
        Hash-bucket axes (4-7: edge_sig, param, prompt, knob) are excluded because
        their high cardinality inflates the denominator without reflecting structural diversity.
        """
        if not self._grid:
            return 0.0
        sample_key = next(iter(self._grid))
        axes = [a for a in STRUCTURAL_AXES if a < len(sample_key)]
        unique_per_axis: list[set[int]] = [set() for _ in axes]
        for key in self._grid:
            for idx, a in enumerate(axes):
                unique_per_axis[idx].add(key[a])
        total_possible = 1
        for s in unique_per_axis:
            total_possible *= max(len(s), 1)
        structural_cells = {tuple(key[a] for a in axes) for key in self._grid}
        return len(structural_cells) / max(total_possible, 1)

    def save(self, directory: Path) -> None:
        """Serialize the archive to a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {}
        for key, ind in self._grid.items():
            str_key = ",".join(str(k) for k in key)
            data[str_key] = ind.model_dump(mode="json")
        (directory / "grid.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path) -> MAPElitesArchive:
        """Deserialize an archive from a directory."""
        archive = cls()
        path = directory / "grid.json"
        if path.exists():
            data = json.loads(path.read_text())
            for str_key, ind_data in data.items():
                ind = Individual.model_validate(ind_data)
                archive._grid[ind.features] = ind
        return archive
