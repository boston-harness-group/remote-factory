"""CompressEvaluator — parses compression result artifacts into structured EvalResults."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from factory.inner_loop import EvalResult

log = structlog.get_logger()


class CompressEvaluator:
    """Parses JSON artifacts from compression runs.

    Expected artifact schema:
        {compression_ratio, quality_retention, inference_latency, technique}

    Combined score: compression_ratio * w_compression + quality_retention * w_quality
                    - (1.0 - latency_penalty) * w_latency
    where latency_penalty = max(0.0, 1.0 - inference_latency).
    """

    def __init__(
        self,
        *,
        w_compression: float = 0.4,
        w_quality: float = 0.5,
        w_latency: float = 0.1,
    ) -> None:
        self.w_compression = w_compression
        self.w_quality = w_quality
        self.w_latency = w_latency

    def parse(self, artifact_path: Path) -> EvalResult:
        try:
            data = json.loads(Path(artifact_path).read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("compress_eval_parse_failed", path=str(artifact_path))
            return EvalResult(score=0.0, valid=False)

        compression_ratio = data.get("compression_ratio")
        quality_retention = data.get("quality_retention")
        if compression_ratio is None or quality_retention is None:
            log.warning("compress_eval_missing_fields", path=str(artifact_path))
            return EvalResult(score=0.0, valid=False)

        inference_latency = float(data.get("inference_latency", 0.0))
        latency_penalty = max(0.0, 1.0 - inference_latency)

        score = (
            float(compression_ratio) * self.w_compression
            + float(quality_retention) * self.w_quality
            - (1.0 - latency_penalty) * self.w_latency
        )

        metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

        return EvalResult(
            score=score,
            metrics=metrics,
            valid=True,
            artifacts=[str(artifact_path)],
        )

    def parse_many(self, artifact_paths: list[Path]) -> EvalResult:
        best = EvalResult(score=0.0, valid=False)
        for p in artifact_paths:
            result = self.parse(p)
            if result.score > best.score:
                best = result
        return best

    def get_info(self) -> dict:
        return {
            "benchmark": "compression",
            "weights": {
                "compression": self.w_compression,
                "quality": self.w_quality,
                "latency": self.w_latency,
            },
            "metrics": [
                "compression_ratio",
                "quality_retention",
                "inference_latency",
                "technique",
            ],
        }
