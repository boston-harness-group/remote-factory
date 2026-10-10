"""Canary test — real model outer-loop run via SwarmEngine.run().

Marked slow + skipif FACTORY_RUN_CANARY!=1. Uses claude-haiku-4-5-20251001.
No CLI pipeline — only SwarmEngine.run().

Run with:
    FACTORY_RUN_CANARY=1 pytest -m slow tests/test_outer_loop/test_canary.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Literal

import pytest

from tests.test_outer_loop.trust_helpers import assert_run_trustworthy

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("FACTORY_RUN_CANARY") != "1",
        reason="Set FACTORY_RUN_CANARY=1 to run canary tests",
    ),
]


# ── DocQualityTask ─────────────────────────────────────────────────


class DocQualityTask:
    """Simple doc quality task for canary testing.

    3 items: 2 train + 1 holdout. Agent writes documentation,
    verify scores on rubric-based heuristics (sections, terms, code, length).
    """

    RUBRICS: dict[str, dict[str, list[str]]] = {
        'readme-cli': {
            'required_sections': ['Overview', 'Installation', 'Usage', 'Errors'],
            'required_terms': ['command', 'flag', 'argument', 'usage', 'install'],
        },
        'tutorial-scraping': {
            'required_sections': ['Overview', 'Installation', 'Usage', 'Errors'],
            'required_terms': ['request', 'parse', 'selector', 'HTTP', 'scraping'],
        },
        'api-reference': {
            'required_sections': ['Overview', 'Installation', 'Usage', 'Errors'],
            'required_terms': ['endpoint', 'parameter', 'response', 'authentication', 'rate limit'],
        },
    }

    def __init__(self) -> None:
        from factory.task import TaskDefinition, InstancesConfig
        self._definition = TaskDefinition(
            name="doc-quality",
            instances_config=InstancesConfig(holdout_ids=["api-reference"]),
        )

    @property
    def definition(self) -> Any:
        return self._definition

    @property
    def name(self) -> str:
        return self._definition.name

    def instances(
        self, split: Literal["train", "val", "all"] = "all",
    ) -> Iterator[Any]:
        from factory.task import TaskInstance
        items = {
            "readme-cli": "train",
            "tutorial-scraping": "train",
            "api-reference": "val",
        }
        for iid, s in items.items():
            if split == "all" or s == split:
                yield TaskInstance(id=iid)

    def setup(self, instance: Any, workspace: Path) -> None:
        (workspace / ".factory").mkdir(parents=True, exist_ok=True)
        (workspace / ".factory" / "current_item.json").write_text(
            json.dumps({"id": instance.id, "topic": instance.id})
        )

    def prompt(self, instance: Any) -> str:
        return (
            f"Write comprehensive documentation about '{instance.id}'. "
            f"Create a file called document.md with clear, well-structured content."
        )

    def verify(self, instance: Any, workspace: Path) -> Any:
        import re as _re
        from factory.task import VerifyResult

        doc = workspace / "document.md"
        if not doc.exists():
            return VerifyResult(passed=False, score=0.0, details={"error": "no document.md"})

        content = doc.read_text()
        rubric = self.RUBRICS.get(instance.id, self.RUBRICS['readme-cli'])
        required_sections: list[str] = rubric['required_sections']
        required_terms: list[str] = rubric['required_terms']

        # 1. Section matching (case-insensitive heading search)
        matched_sections: list[str] = []
        missing_sections: list[str] = []
        content_lower = content.lower()
        for section in required_sections:
            # Match '# Section', '## Section', '### Section' etc.
            pattern = _re.compile(r'^#{1,6}\s+' + _re.escape(section.lower()), _re.MULTILINE)
            if pattern.search(content_lower):
                matched_sections.append(section)
            else:
                missing_sections.append(section)

        # 2. Term matching (case-insensitive)
        matched_terms: list[str] = []
        missing_terms: list[str] = []
        for term in required_terms:
            if term.lower() in content_lower:
                matched_terms.append(term)
            else:
                missing_terms.append(term)

        # 3. Code block detection (fenced triple backtick)
        has_code_block = '```' in content

        # 4. Length band
        word_count = len(content.split())
        if word_count < 150:
            length_score = 0.3
        elif word_count <= 200:
            length_score = 0.6
        elif word_count <= 800:
            length_score = 1.0
        elif word_count <= 2000:
            length_score = 0.8
        else:
            length_score = 0.5

        # 5. Composite score
        total_sections = len(required_sections)
        total_terms = len(required_terms)
        section_score = len(matched_sections) / total_sections if total_sections else 0.0
        term_score = len(matched_terms) / total_terms if total_terms else 0.0

        score = (
            0.35 * section_score
            + 0.30 * term_score
            + 0.15 * (1.0 if has_code_block else 0.0)
            + 0.20 * length_score
        )
        passed = score >= 0.7

        return VerifyResult(
            passed=passed,
            score=round(score, 4),
            details={
                "word_count": word_count,
                "matched_sections": matched_sections,
                "missing_sections": missing_sections,
                "matched_terms": matched_terms,
                "missing_terms": missing_terms,
                "has_code_block": has_code_block,
                "length_score": length_score,
                "section_score": round(section_score, 4),
                "term_score": round(term_score, 4),
            },
        )

    def get_evaluator(self) -> None:
        return None


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
    )


def _make_doc_workflow() -> Any:
    from factory.workflow.primitives import (
        AgentNode, AgentRole, DataNode, Edge, JoinNode, Workflow,
    )
    return Workflow(
        name="doc-quality-wf",
        nodes={
            "data": DataNode(id="data", parallelism=2),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                model="claude-haiku-4-5-20251001",
                prompt_template="Write a document about the given topic. Save it as document.md.",
                writes={"document.md"},
                reads=set(),
            ),
            "_join_data": JoinNode(id="_join_data", sources=["builder"]),
        },
        edges=[
            Edge(source="data", target="builder"),
            Edge(source="builder", target="_join_data"),
        ],
        start_node="data",
    )


@pytest.mark.timeout(3600)
class TestCanary:
    """Canary: one full outer-loop run via SwarmEngine.run() with real model.

    Validates: real project_dir used, evolution produces offspring,
    trust checks pass, reflection cites real item results.
    """

    @pytest.fixture()
    def project(self, tmp_path: Path) -> Iterator[Path]:
        project = tmp_path / 'canary-project'
        project.mkdir()
        _git(project, 'init')
        _git(project, 'config', 'user.email', 'canary@test')
        _git(project, 'config', 'user.name', 'canary')
        (project / '.gitignore').write_text('.factory/\n')
        factory_dir = project / '.factory'
        factory_dir.mkdir()
        (factory_dir / 'config.json').write_text(
            json.dumps({'inner_loop': {'aggregate': 'mean'}})
        )
        (project / 'README.md').write_text('# Canary\n')
        _git(project, 'add', '.')
        _git(project, 'commit', '-m', 'init')
        yield project
        # Copy run artifacts if FACTORY_CANARY_KEEP is set
        keep_dir = os.environ.get('FACTORY_CANARY_KEEP')
        if keep_dir:
            import shutil
            dst = Path(keep_dir)
            dst.mkdir(parents=True, exist_ok=True)
            ol_dir = project / '.factory' / 'outer_loop'
            if ol_dir.exists():
                shutil.copytree(ol_dir, dst / 'outer_loop', dirs_exist_ok=True)

    def _run_engine(self, project: Path) -> Any:
        """Run SwarmEngine with real model — evolves at least one offspring."""
        from factory.outer_loop.engine import SwarmEngine
        from factory.outer_loop.evaluator import SwarmEvaluator
        from factory.outer_loop.models import SwarmConfig

        model = os.environ.get("FACTORY_MODEL", "claude-haiku-4-5-20251001")

        task = DocQualityTask()
        wf = _make_doc_workflow()

        config = SwarmConfig(
            benchmark="doc-quality-canary",
            budget=4,
            population_size=2,
            training_instances=["readme-cli", "tutorial-scraping"],
            designer_count=0,
            mutation_rate=0.3,
        )
        config.set_task(task)

        evaluator = SwarmEvaluator(
            config, inner_loop_factory=True, project_dir=project,
        )
        engine = SwarmEngine(
            config, evaluator, project_dir=project, designer=None,
        )

        return engine.run(wf), engine, evaluator

    def test_canary_run(self, project: Path) -> None:
        """Full canary: evolution with real model, offspring, trust checks."""
        result, engine, evaluator = self._run_engine(project)

        # ── Basic score / convergence ──────────────────────────
        assert result.best_score > 0, (
            f"Best score should be positive, got {result.best_score}"
        )
        assert result.generations_completed >= 1
        assert result.convergence_reason in (
            "budget_exhausted", "target_score_reached", "plateau",
        )

        # ── Trust checks ──────────────────────────────────────
        diag = assert_run_trustworthy(result, project)
        assert diag["checks_failed"] == 0, f"Trust checks failed: {diag}"

        # ── Individuals from population ──────────────────────
        pop = engine._final_population
        seeds = [ind for ind in pop.individuals if ind.parent_id is None and ind.score is not None]
        offspring = [ind for ind in pop.individuals if ind.parent_id is not None and ind.score is not None]

        assert len(seeds) >= 1, 'No seeds evaluated'
        assert len(offspring) >= 2, f'Expected >=2 offspring, got {len(offspring)}'

        # Raw benchmark scores (not composite) from CycleRecord
        def _raw_score(ind_id: str) -> float | None:
            rec = evaluator.get_cycle_record(ind_id)
            return rec.score_end if rec and rec.score_end is not None else None

        seed_raw_scores = [_raw_score(s.id) for s in seeds if _raw_score(s.id) is not None]
        offspring_raw_scores = [_raw_score(o.id) for o in offspring if _raw_score(o.id) is not None]

        # Seed must score < 0.8 (rubric has headroom)
        for s in seeds:
            raw = _raw_score(s.id)
            assert raw is not None and raw < 0.8, f'Seed {s.id[:8]} raw score {raw} >= 0.8 — rubric too easy'

        best_seed_raw = max(seed_raw_scores)
        best_offspring_raw = max(offspring_raw_scores) if offspring_raw_scores else 0.0

        # Evolution MUST improve on seed — that's what the canary tests
        assert best_offspring_raw > best_seed_raw, (
            f'Offspring best raw ({best_offspring_raw:.4f}) did not beat seed best raw ({best_seed_raw:.4f}). '
            f'Seeds: {[(s.id[:8], _raw_score(s.id)) for s in seeds]}. '
            f'Offspring: {[(o.id[:8], _raw_score(o.id)) for o in offspring]}'
        )

        # ── Reflection cites real item results ────────────────
        # The reflector should have produced a reflection stored on the engine
        reflection = engine._last_reflection
        assert reflection is not None, "No reflection was produced"
        # Reflection report should contain patterns or suggestions
        assert len(reflection.failure_patterns) > 0 or len(reflection.mutation_suggestions) + len(reflection.typed_suggestions) > 0, (
            "Reflection has no failure patterns or suggestions"
        )

        # (e) Reflector's last reflection should reference rubric elements
        rubric_keywords = {
            'section', 'Overview', 'Installation', 'Usage', 'Errors',
            'term', 'code block', 'code_block', 'missing',
        }
        reflection_text = ""
        for fp in reflection.failure_patterns:
            reflection_text += f" {fp}"
        for sg in reflection.typed_suggestions:
            reflection_text += f" {sg.rationale}" if hasattr(sg, 'rationale') else f" {sg}"
        for sg_text in reflection.mutation_suggestions:
            reflection_text += f" {sg_text}"
        found_rubric_ref = any(kw in reflection_text for kw in rubric_keywords)
        assert found_rubric_ref, (
            f'Reflector did not reference any rubric element in patterns/suggestions. '
            f'Text: {reflection_text[:500]}'
        )

        # ── Diagnostic summary ────────────────────────────────
        summary_lines = ['=== CANARY DIAGNOSTIC SUMMARY ===']
        for ind in pop.individuals:
            rec = evaluator.get_cycle_record(ind.id)
            raw = rec.score_end if rec else None
            mutation_op = ind.mutation_record.operator.value if ind.mutation_record else 'seed'
            parent = ind.parent_id[:8] if ind.parent_id else 'none'
            summary_lines.append(
                f'\nIndividual {ind.id[:8]} (score={ind.score}, raw={raw}, '
                f'parent={parent}, mutation={mutation_op})'
            )

            # Get prompt from workflow
            wf_data = ind.workflow_data
            if isinstance(wf_data, dict):
                for nid, node in wf_data.get('nodes', {}).items():
                    pt = node.get('prompt_template', '')
                    if pt:
                        summary_lines.append(f'  Prompt ({nid}): {pt[:200]}...')

            # Per-item verify_details
            if rec and rec.instance_results:
                for item in rec.instance_results:
                    if isinstance(item, dict):
                        summary_lines.append(
                            f'  Item {item.get("item_id")}: score={item.get("score")}, '
                            f'words={item.get("verify_details", {}).get("word_count")}, '
                            f'missing_sections={item.get("verify_details", {}).get("missing_sections")}, '
                            f'missing_terms={item.get("verify_details", {}).get("missing_terms")}'
                        )

        # Reflection info
        if reflection:
            summary_lines.append(f'\nReflection failure_patterns: {reflection.failure_patterns}')
            summary_lines.append(f'Reflection prompt_improvements: {reflection.prompt_improvements}')

        summary_text = '\n'.join(summary_lines)
        print(summary_text)

        # Write to file
        diag_path = project / '.factory' / 'outer_loop' / 'canary-summary.txt'
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(summary_text)
