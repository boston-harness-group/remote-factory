"""CompressInnerLoop — InnerLoop subclass with compression-specific tracking."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from factory.inner_loop import InnerLoop

log = structlog.get_logger()

_DEFAULT_FROZEN_NODES = frozenset({"gate_review", "gate_precheck"})


class CompressInnerLoop(InnerLoop):
    """InnerLoop specialized for model compression experiments.

    Adds technique tracking: each cycle records the compression technique used,
    enabling the outer loop to analyze which techniques work best.
    """

    def __init__(
        self,
        project_dir: Path,
        **kwargs,
    ) -> None:
        kwargs.setdefault("mode", "compress")
        kwargs.setdefault("frozen_nodes", _DEFAULT_FROZEN_NODES)
        super().__init__(project_dir, **kwargs)

    def technique_history(self) -> list[str]:
        """Return ordered list of techniques used across all cycles."""
        techniques: list[str] = []
        for record in self._history:
            technique = self._extract_technique(record)
            if technique:
                techniques.append(technique)
        return techniques

    def best_technique(self) -> str | None:
        """Return the technique that achieved the highest score."""
        best_score = -1.0
        best_technique: str | None = None
        for record in self._history:
            technique = self._extract_technique(record)
            if technique and record.score_end is not None and record.score_end > best_score:
                best_score = record.score_end
                best_technique = technique
        return best_technique

    def compression_trajectory(self) -> list[dict]:
        """Return per-cycle technique + score pairs."""
        trajectory: list[dict] = []
        for record in self._history:
            technique = self._extract_technique(record)
            trajectory.append({
                "cycle": record.cycle_number,
                "technique": technique,
                "score": record.score_end,
                "delta": record.score_delta,
            })
        return trajectory

    @staticmethod
    def _extract_technique(record) -> str | None:
        """Extract technique name from a CycleRecord's eval artifacts."""
        for exp in getattr(record, "experiments", []):
            for artifact_path in exp.eval_artifacts:
                try:
                    data = json.loads(Path(artifact_path).read_text())
                    if "technique" in data:
                        return str(data["technique"])
                except (json.JSONDecodeError, OSError):
                    continue
        return None
