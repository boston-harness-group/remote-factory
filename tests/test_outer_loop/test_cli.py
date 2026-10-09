"""Tests for outer-loop CLI argument parsing and mode registration."""

from __future__ import annotations

from unittest.mock import patch


class TestDiskSpaceCheck:
    def test_sufficient_space_passes(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _check_disk_space

        assert _check_disk_space(Path(str(tmp_path)), population_size=4) is True

    def test_insufficient_space_fails(self, tmp_path: object) -> None:
        from collections import namedtuple
        from pathlib import Path

        from factory.cli.outer_loop import _check_disk_space

        DiskUsage = namedtuple("usage", ["total", "used", "free"])
        tiny = DiskUsage(total=100 * 1024**3, used=99 * 1024**3, free=1 * 1024**3)
        with patch("factory.cli.outer_loop.shutil.disk_usage", return_value=tiny):
            assert _check_disk_space(Path(str(tmp_path)), population_size=4) is False

    def test_required_space_formula(self) -> None:
        from collections import namedtuple
        from pathlib import Path

        from factory.cli.outer_loop import _check_disk_space

        DiskUsage = namedtuple("usage", ["total", "used", "free"])

        exactly_enough = DiskUsage(
            total=100 * 1024**3,
            used=80 * 1024**3,
            free=int(20.1 * 1024**3),
        )
        with patch("factory.cli.outer_loop.shutil.disk_usage", return_value=exactly_enough):
            assert _check_disk_space(Path("/tmp"), population_size=50) is True

        not_enough = DiskUsage(
            total=100 * 1024**3,
            used=81 * 1024**3,
            free=int(19.9 * 1024**3),
        )
        with patch("factory.cli.outer_loop.shutil.disk_usage", return_value=not_enough):
            assert _check_disk_space(Path("/tmp"), population_size=50) is False


class TestOuterLoopModeRegistration:
    def test_outer_loop_in_ceo_modes(self) -> None:
        from factory.cli._helpers import CEO_MODES

        assert "outer-loop" in CEO_MODES


class TestOuterLoopCLIParsing:
    def _parse_outer_loop(self, *args: str) -> object:
        from factory.cli._main import build_parser

        parser = build_parser()
        return parser.parse_args(["outer-loop", *args])

    def test_calibrate_subcommand(self) -> None:
        ns = self._parse_outer_loop("calibrate", "/tmp/project", "--seed-workflow", "m:f")
        assert ns.command == "outer-loop"
        assert ns.outer_loop_command == "calibrate"
        assert ns.project_path == "/tmp/project"

    def test_calibrate_with_options(self) -> None:
        ns = self._parse_outer_loop(
            "calibrate", "/tmp/project",
            "--benchmark", "featurebench",
            "--budget", "50",
            "--population-size", "4",
            "--seed-workflow", "m:f",
        )
        assert ns.benchmark == "featurebench"
        assert ns.budget == 50
        assert ns.population_size == 4

    def test_calibrate_with_seed_workflow(self) -> None:
        ns = self._parse_outer_loop(
            "calibrate", "/tmp/project",
            "--seed-workflow", "my.pipeline:build_pipeline",
        )
        assert ns.seed_workflow == "my.pipeline:build_pipeline"

    def test_calibrate_with_target_project(self) -> None:
        ns = self._parse_outer_loop(
            "calibrate", "/tmp/project",
            "--project-dir", "/tmp/featurebench-instance",
            "--seed-workflow", "m:f",
        )
        assert ns.project_dir == "/tmp/featurebench-instance"

    def test_calibrate_without_target_project(self) -> None:
        ns = self._parse_outer_loop("calibrate", "/tmp/project", "--seed-workflow", "m:f")
        assert ns.project_dir is None

    def test_evaluate_subcommand(self) -> None:
        ns = self._parse_outer_loop("evaluate", "/tmp/project", "--generation", "3")
        assert ns.outer_loop_command == "evaluate"
        assert ns.generation == 3

    def test_reflect_subcommand(self) -> None:
        ns = self._parse_outer_loop("reflect", "/tmp/project", "--generation", "2")
        assert ns.outer_loop_command == "reflect"
        assert ns.generation == 2

    def test_evolve_subcommand(self) -> None:
        ns = self._parse_outer_loop("evolve", "/tmp/project", "--generation", "1")
        assert ns.outer_loop_command == "evolve"
        assert ns.generation == 1

    def test_status_subcommand(self) -> None:
        ns = self._parse_outer_loop("status", "/tmp/project")
        assert ns.outer_loop_command == "status"

    def test_status_check_converge(self) -> None:
        ns = self._parse_outer_loop("status", "/tmp/project", "--check-converge")
        assert ns.check_converge is True

    def test_promote_subcommand(self) -> None:
        ns = self._parse_outer_loop("promote", "/tmp/project", "--mode-name", "evolve-gen5-abc")
        assert ns.outer_loop_command == "promote"
        assert ns.mode_name == "evolve-gen5-abc"

    def test_promote_with_permanent_name(self) -> None:
        ns = self._parse_outer_loop(
            "promote", "/tmp/project",
            "--mode-name", "evolve-gen5-abc",
            "--permanent-name", "my-evolved",
        )
        assert ns.permanent_name == "my-evolved"


class TestEvaluateTargetProjectFallback:
    def test_evaluate_uses_config_target_project(self, tmp_path: object) -> None:
        """_cmd_evaluate falls back to config.target_project when --project-dir not passed."""
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path)) / "factory-project"
        project.mkdir()

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=50,
            target_project="/tmp/featurebench-instance",
        )

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.load_checkpoint", return_value=None), \
             patch("factory.outer_loop.filesystem.save_checkpoint"):
            from factory.outer_loop.mode_registry import EphemeralModeRegistry

            with patch.object(EphemeralModeRegistry, "list_modes", return_value=[]):
                from factory.cli.outer_loop import _cmd_evaluate

                ns = argparse.Namespace(
                    project_path=str(project),
                    generation=0,
                    project_dir=None,
                )
                rc = _cmd_evaluate(ns)
                assert rc == 1  # no modes, but it should reach the "no modes" error


class TestInnerLoopFactoryReusesExistingModes:
    """Bug #16: _make_inner_loop_factory should reuse existing modes, not create eval copies."""

    def test_returns_existing_mode_by_structural_hash(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _make_inner_loop_factory
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        registry = EphemeralModeRegistry(project)

        wf = Workflow(
            name="test-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )
        registered_name = registry.register("abc12345", 0, wf)

        factory_fn = _make_inner_loop_factory(registry)
        result = factory_fn(wf)
        assert result == registered_name
        assert "eval" not in result

    def test_does_not_create_eval_copy_modes(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _make_inner_loop_factory
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        registry = EphemeralModeRegistry(project)

        wf = Workflow(
            name="test-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )
        registry.register("seed0001", 0, wf)

        factory_fn = _make_inner_loop_factory(registry)
        factory_fn(wf)
        factory_fn(wf)
        factory_fn(wf)

        modes = registry.list_modes()
        eval_modes = [m for m in modes if "eval" in m]
        assert eval_modes == [], f"Unexpected eval-copy modes: {eval_modes}"
        assert len(modes) == 1

    def test_caches_hash_lookups(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _make_inner_loop_factory
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        registry = EphemeralModeRegistry(project)

        wf = Workflow(
            name="test-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )
        registered_name = registry.register("abc12345", 0, wf)

        factory_fn = _make_inner_loop_factory(registry)
        r1 = factory_fn(wf)
        r2 = factory_fn(wf)
        assert r1 == r2 == registered_name

    def test_fallback_registers_new_mode_for_unknown_workflow(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _make_inner_loop_factory
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        registry = EphemeralModeRegistry(project)

        factory_fn = _make_inner_loop_factory(registry)

        wf = Workflow(
            name="new-wf",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    writes={".factory/reviews/builder-latest.md"},
                ),
            },
            edges=[],
            start_node="builder",
            terminal=True,
        )
        result = factory_fn(wf)
        assert result.startswith("evolve-gen0-")
        assert "eval" not in result


class TestReflectReadsCachedData:
    """Bug #17: _cmd_reflect should read cached data instead of re-evaluating."""

    def test_reflect_uses_saved_results_and_cycle_summary(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf1 = Workflow(
            name="mode-a",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        wf2 = Workflow(
            name="mode-b",
            nodes={"b": AgentNode(id="b", role=AgentRole.RESEARCHER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )

        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        name_a = registry.register("aaa", 0, wf1)
        name_b = registry.register("bbb", 0, wf2)

        gen_results = {
            name_a: {"score": 0.85, "cost_usd": 1.0},
            name_b: {"score": 0.72, "cost_usd": 0.5},
        }
        (results_dir / "gen0.json").write_text(json.dumps(gen_results))

        for name, score in [(name_a, 0.85), (name_b, 0.72)]:
            runs_dir = project / ".factory" / "outer_loop" / "runs" / name
            runs_dir.mkdir(parents=True)
            summary = {"mode": name, "score": score, "cost_usd": 0.5, "kept": 2, "reverted": 1}
            (runs_dir / "cycle_summary.json").write_text(json.dumps(summary))

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        from unittest.mock import MagicMock as _MagicMock
        mock_llm = _MagicMock()
        mock_llm.stdout = '{"failure_patterns": [], "success_patterns": [], "mutation_suggestions": []}'
        mock_llm.returncode = 0

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("subprocess.run", return_value=mock_llm):
            from factory.cli.outer_loop import _cmd_reflect

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_reflect(ns)
            assert rc == 0

    def test_evaluate_persists_cycle_summary(self, tmp_path: object) -> None:
        """_cmd_evaluate should write cycle_summary.json for each evaluated mode."""
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import EvalResult, SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        mode_name = registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)
        mock_result = EvalResult(
            score=0.75, benchmark_score=0.8, cost_usd=2.0,
            details={"kept": 2, "reverted": 1},
        )

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_result

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.load_checkpoint", return_value=None), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator", return_value=mock_evaluator):
            from factory.cli.outer_loop import _cmd_evaluate

            ns = argparse.Namespace(
                project_path=str(project), generation=0, project_dir=None,
            )
            rc = _cmd_evaluate(ns)
            assert rc == 0

        summary_path = (
            project / ".factory" / "outer_loop" / "runs" / mode_name / "cycle_summary.json"
        )
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["score"] == 0.75
        assert data["kept"] == 2


class TestReflectPassesKnobValues:
    def test_reflect_passes_knob_values_by_id(self, tmp_path: object) -> None:
        """_cmd_reflect builds knob_values_by_id from registry workflows and passes to reflect."""
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf1 = Workflow(
            name="mode-a",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"temperature": 0.7, "strategy": "greedy"},
        )
        wf2 = Workflow(
            name="mode-b",
            nodes={"b": AgentNode(id="b", role=AgentRole.RESEARCHER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"temperature": 0.9},
        )

        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        name_a = registry.register("aaa", 0, wf1)
        name_b = registry.register("bbb", 0, wf2)

        gen_results = {
            name_a: {"score": 0.85, "cost_usd": 1.0},
            name_b: {"score": 0.72, "cost_usd": 0.5},
        }
        (results_dir / "gen0.json").write_text(json.dumps(gen_results))

        for name, score in [(name_a, 0.85), (name_b, 0.72)]:
            runs_dir = project / ".factory" / "outer_loop" / "runs" / name
            runs_dir.mkdir(parents=True)
            summary = {"mode": name, "score": score, "cost_usd": 0.5, "kept": 2, "reverted": 1}
            (runs_dir / "cycle_summary.json").write_text(json.dumps(summary))

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop.reflector import OuterLoopReflector, ReflectionReport

        def spy_reflect(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return ReflectionReport()

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(OuterLoopReflector, "reflect", side_effect=spy_reflect):
            from factory.cli.outer_loop import _cmd_reflect

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_reflect(ns)
            assert rc == 0

        assert len(captured_kwargs) == 1
        kvbi = captured_kwargs[0].get("knob_values_by_id")
        assert kvbi is not None
        assert len(kvbi) == 2
        assert kvbi[name_a] == {"temperature": 0.7, "strategy": "greedy"}
        assert kvbi[name_b] == {"temperature": 0.9}


class TestEvolveHandlesMalformedReflection:
    def test_evolve_handles_malformed_reflection_json(self, tmp_path: object) -> None:
        """_cmd_evolve gracefully handles malformed reflection JSON (returns 0, report=None)."""
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        reflections_dir = project / ".factory" / "outer_loop" / "reflections"
        reflections_dir.mkdir(parents=True)

        (reflections_dir / "gen0.json").write_text("not json {{{")

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop import mutations as _mutations_mod

        original_apply = _mutations_mod.apply_random_mutation

        def spy_apply(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_apply(*args, **kwargs)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=spy_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_evolve(ns)
            assert rc == 0

        assert len(captured_kwargs) > 0
        assert captured_kwargs[0].get("reflection_report") is None


class TestEvolveLoadsReflectionReport:
    def test_evolve_loads_persisted_reflection_json(self, tmp_path: object) -> None:
        """_cmd_evolve should load reflections/gen0.json and pass it to apply_random_mutation."""
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        reflections_dir = project / ".factory" / "outer_loop" / "reflections"
        reflections_dir.mkdir(parents=True)

        report_data = {
            "generation": 0,
            "failure_patterns": ["Agent builder failed"],
            "success_patterns": ["Winner used researcher"],
            "mutation_suggestions": ["NODE_INSERT: Add researcher"],
            "prompt_improvements": [],
            "structural_recommendations": [],
            "top_k_ids": ["w1"],
            "bottom_k_ids": ["l1"],
        }
        (reflections_dir / "gen0.json").write_text(json.dumps(report_data))

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop import mutations as _mutations_mod

        original_apply = _mutations_mod.apply_random_mutation

        def spy_apply(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_apply(*args, **kwargs)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=spy_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            _cmd_evolve(ns)

        assert len(captured_kwargs) > 0
        assert captured_kwargs[0].get("reflection_report") is not None

    def test_evolve_works_without_reflection_file(self, tmp_path: object) -> None:
        """_cmd_evolve should still work when no reflection JSON exists."""
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_evolve(ns)
            assert rc == 0


class TestResolveSeedWorkflow:
    """Cover _resolve_seed_workflow dynamic import + Package/Workflow handling."""

    def test_resolve_returns_workflow_directly(self) -> None:
        from unittest.mock import MagicMock, patch

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="direct-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"style": "fast"},
        )

        fake_mod = MagicMock()
        fake_mod.build_pipeline = MagicMock(return_value=wf)

        with patch("importlib.import_module", return_value=fake_mod):
            from factory.cli.outer_loop import _resolve_seed_workflow
            result = _resolve_seed_workflow("my.mod:build_pipeline")

        assert result.name == "direct-wf"
        assert result.knob_values == {"style": "fast"}

    def test_resolve_compiles_package(self) -> None:
        from unittest.mock import MagicMock, patch

        from factory.workflow.package import OptKnob, Package
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        inner_wf = Workflow(
            name="pkg-inner",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        pkg = Package(
            name="test-pkg",
            graph=inner_wf,
            entry_node="b",
            exit_node="b",
            knobs=[OptKnob(name="retries", kind="threshold", node_id="b", default=3, bounds=[1, 3, 5])],
        )

        fake_mod = MagicMock()
        fake_mod.build = MagicMock(return_value=pkg)

        with patch("importlib.import_module", return_value=fake_mod):
            from factory.cli.outer_loop import _resolve_seed_workflow
            result = _resolve_seed_workflow("my.mod:build")

        assert result.knob_values["retries"] == 3
        assert result.knob_bounds["retries"] == [1, 3, 5]

    def test_resolve_invalid_format_raises(self) -> None:
        import pytest

        from factory.cli.outer_loop import _resolve_seed_workflow

        with pytest.raises(ValueError, match="Expected 'module.path:callable'"):
            _resolve_seed_workflow("no_colon_here")

    def test_resolve_missing_module_raises(self) -> None:
        import pytest

        from factory.cli.outer_loop import _resolve_seed_workflow

        with pytest.raises(ImportError, match="Could not import module"):
            _resolve_seed_workflow("nonexistent.module.xyz:build")

    def test_resolve_missing_attr_raises(self) -> None:
        import pytest
        from unittest.mock import MagicMock, patch

        fake_mod = MagicMock(spec=[])

        with patch("importlib.import_module", return_value=fake_mod):
            from factory.cli.outer_loop import _resolve_seed_workflow

            with pytest.raises(ImportError, match="has no attribute"):
                _resolve_seed_workflow("my.mod:nonexistent_fn")

    def test_resolve_not_callable_raises(self) -> None:
        import pytest
        from unittest.mock import MagicMock, patch

        fake_mod = MagicMock()
        fake_mod.not_a_fn = "just a string"

        with patch("importlib.import_module", return_value=fake_mod):
            from factory.cli.outer_loop import _resolve_seed_workflow

            with pytest.raises(TypeError, match="not callable"):
                _resolve_seed_workflow("my.mod:not_a_fn")

    def test_resolve_wrong_return_type_raises(self) -> None:
        import pytest
        from unittest.mock import MagicMock, patch

        fake_mod = MagicMock()
        fake_mod.build = MagicMock(return_value={"not": "a workflow"})

        with patch("importlib.import_module", return_value=fake_mod):
            from factory.cli.outer_loop import _resolve_seed_workflow

            with pytest.raises(TypeError, match="expected Package or Workflow"):
                _resolve_seed_workflow("my.mod:build")


class TestCalibrateSeedWorkflowModule:
    """Cover --seed-workflow flag in _cmd_calibrate."""

    def test_seed_workflow_module_package_knobs_propagate(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path)) / "cal-swm"
        project.mkdir()

        compiled_wf = Workflow(
            name="compiled-seed",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"depth": 3, "style": "focused"},
            knob_bounds={"depth": [1, 3, 5], "style": ["focused", "broad"]},
        )

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            population_size=2,
            seed_workflow_module="my.pipeline:build",
        )

        ns = argparse.Namespace(project_path=str(project))

        mock_engine = MagicMock()
        mock_pop = MagicMock()
        mock_engine.seed.return_value = mock_pop
        captured_base_wf = []

        def spy_seed(wf, cfg_arg):
            captured_base_wf.append(wf)
            return mock_pop

        mock_engine.seed = spy_seed

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.init_filesystem", return_value=project / ".factory" / "outer_loop"), \
             patch("factory.outer_loop.engine.SwarmEngine", return_value=mock_engine), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator"), \
             patch("factory.outer_loop.similarity.NoveltyFilter"), \
             patch.object(mock_pop, "save"), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.cli.outer_loop._resolve_seed_workflow", return_value=compiled_wf):
            from factory.cli.outer_loop import _cmd_calibrate
            result = _cmd_calibrate(ns)

        assert result == 0
        assert len(captured_base_wf) == 1
        assert captured_base_wf[0].knob_values == {"depth": 3, "style": "focused"}
        assert captured_base_wf[0].knob_bounds == {"depth": [1, 3, 5], "style": ["focused", "broad"]}

    def test_seed_workflow_module_returns_workflow_directly(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path)) / "cal-swm-wf"
        project.mkdir()

        direct_wf = Workflow(
            name="direct-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"retries": 2},
        )

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            population_size=2,
            seed_workflow_module="my.pipeline:build_wf",
        )

        ns = argparse.Namespace(project_path=str(project))

        mock_engine = MagicMock()
        mock_pop = MagicMock()
        mock_engine.seed.return_value = mock_pop
        captured_base_wf = []

        def spy_seed(wf, cfg_arg):
            captured_base_wf.append(wf)
            return mock_pop

        mock_engine.seed = spy_seed

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.init_filesystem", return_value=project / ".factory" / "outer_loop"), \
             patch("factory.outer_loop.engine.SwarmEngine", return_value=mock_engine), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator"), \
             patch("factory.outer_loop.similarity.NoveltyFilter"), \
             patch.object(mock_pop, "save"), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.cli.outer_loop._resolve_seed_workflow", return_value=direct_wf):
            from factory.cli.outer_loop import _cmd_calibrate
            result = _cmd_calibrate(ns)

        assert result == 0
        assert len(captured_base_wf) == 1
        assert captured_base_wf[0].knob_values == {"retries": 2}

    def test_seed_workflow_module_error_returns_exit_1(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path)) / "cal-swm-err"
        project.mkdir()

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            population_size=2,
            seed_workflow_module="bad.module:fn",
        )

        ns = argparse.Namespace(project_path=str(project))

        mock_engine = MagicMock()
        mock_pop = MagicMock()
        mock_engine.seed.return_value = mock_pop

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.init_filesystem", return_value=project / ".factory" / "outer_loop"), \
             patch("factory.outer_loop.engine.SwarmEngine", return_value=mock_engine), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator"), \
             patch("factory.outer_loop.similarity.NoveltyFilter"), \
             patch.object(mock_pop, "save"), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.cli.outer_loop._resolve_seed_workflow", side_effect=ImportError("no such module")):
            from factory.cli.outer_loop import _cmd_calibrate
            result = _cmd_calibrate(ns)

        assert result == 1

    def test_seed_workflow_module_takes_precedence_over_registry(self, tmp_path: object) -> None:
        """--seed-workflow should win over config.seed_workflow (registry name)."""
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path)) / "cal-swm-prec"
        project.mkdir()

        module_wf = Workflow(
            name="from-module",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"from": "module"},
        )

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            population_size=2,
            seed_workflow="my-registry-wf",
            seed_workflow_module="my.pipeline:build",
        )

        ns = argparse.Namespace(project_path=str(project))

        mock_engine = MagicMock()
        mock_pop = MagicMock()
        mock_engine.seed.return_value = mock_pop
        captured_base_wf = []

        def spy_seed(wf, cfg_arg):
            captured_base_wf.append(wf)
            return mock_pop

        mock_engine.seed = spy_seed

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.init_filesystem", return_value=project / ".factory" / "outer_loop"), \
             patch("factory.outer_loop.engine.SwarmEngine", return_value=mock_engine), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator"), \
             patch("factory.outer_loop.similarity.NoveltyFilter"), \
             patch.object(mock_pop, "save"), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.cli.outer_loop._resolve_seed_workflow", return_value=module_wf) as mock_resolve:
            from factory.cli.outer_loop import _cmd_calibrate
            result = _cmd_calibrate(ns)

        assert result == 0
        assert len(captured_base_wf) == 1
        assert captured_base_wf[0].name == "from-module"
        mock_resolve.assert_called_once_with("my.pipeline:build")

    def test_seed_workflow_module_with_task_module_independent(self, tmp_path: object) -> None:
        """--seed-workflow and --task-module can be specified independently."""
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path)) / "cal-swm-both"
        project.mkdir()

        module_wf = Workflow(
            name="seed-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
            knob_values={"topology": "parallel"},
        )

        cfg = SwarmConfig(
            benchmark="featurebench",
            budget=10,
            population_size=2,
            task_module="my.task:MyTask",
            seed_workflow_module="my.pipeline:build",
        )

        ns = argparse.Namespace(project_path=str(project))

        mock_engine = MagicMock()
        mock_pop = MagicMock()
        mock_engine.seed.return_value = mock_pop
        captured_base_wf = []

        def spy_seed(wf, cfg_arg):
            captured_base_wf.append(wf)
            return mock_pop

        mock_engine.seed = spy_seed

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.init_filesystem", return_value=project / ".factory" / "outer_loop"), \
             patch("factory.outer_loop.engine.SwarmEngine", return_value=mock_engine), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator"), \
             patch("factory.outer_loop.similarity.NoveltyFilter"), \
             patch.object(mock_pop, "save"), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.cli.outer_loop._resolve_seed_workflow", return_value=module_wf):
            from factory.cli.outer_loop import _cmd_calibrate
            result = _cmd_calibrate(ns)

        assert result == 0
        assert len(captured_base_wf) == 1
        assert captured_base_wf[0].knob_values == {"topology": "parallel"}


class TestEvolveTypedSuggestionsDeserialization:
    """Cover typed_suggestions deserialization in _cmd_evolve."""

    def test_evolve_deserializes_typed_suggestions(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        reflections_dir = project / ".factory" / "outer_loop" / "reflections"
        reflections_dir.mkdir(parents=True)

        report_data = {
            "generation": 0,
            "failure_patterns": ["Agent builder failed"],
            "success_patterns": ["Winner used researcher"],
            "mutation_suggestions": ["NODE_INSERT: Add researcher"],
            "prompt_improvements": [],
            "structural_recommendations": [],
            "top_k_ids": ["w1"],
            "bottom_k_ids": ["l1"],
            "typed_suggestions": [
                {
                    "operator": "knob_mutate",
                    "target": "style",
                    "rationale": "focused works best",
                    "value": "focused",
                },
                {
                    "operator": "node_insert",
                    "target": "researcher",
                    "rationale": "present in winners",
                    "value": None,
                },
            ],
        }
        (reflections_dir / "gen0.json").write_text(json.dumps(report_data))

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop import mutations as _mutations_mod

        original_apply = _mutations_mod.apply_random_mutation

        def spy_apply(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_apply(*args, **kwargs)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=spy_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            _cmd_evolve(ns)

        assert len(captured_kwargs) > 0
        report = captured_kwargs[0].get("reflection_report")
        assert report is not None
        assert len(report.typed_suggestions) == 2
        assert report.typed_suggestions[0].operator == "knob_mutate"
        assert report.typed_suggestions[0].value == "focused"
        assert report.typed_suggestions[1].operator == "node_insert"


class TestReflectNotEnoughCandidates:
    """Cover lines 455-456: reflect returns 1 when < 2 candidates."""

    def test_reflect_returns_error_with_one_candidate(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="only-one",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        name = registry.register("solo", 0, wf)

        gen_results = {name: {"score": 0.9, "cost_usd": 1.0}}
        (results_dir / "gen0.json").write_text(json.dumps(gen_results))
        runs_dir = project / ".factory" / "outer_loop" / "runs" / name
        runs_dir.mkdir(parents=True)
        (runs_dir / "cycle_summary.json").write_text(
            json.dumps({"mode": name, "score": 0.9, "cost_usd": 1.0, "kept": 1, "reverted": 0})
        )

        cfg = SwarmConfig(benchmark="featurebench", budget=50)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg):
            from factory.cli.outer_loop import _cmd_reflect

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_reflect(ns)
            assert rc == 1


class TestEvolveNoModes:
    """Cover lines 495-496: evolve returns 1 when no ephemeral modes exist."""

    def test_evolve_returns_error_with_no_modes(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_evolve(ns)
            assert rc == 1


class TestEvolveNonDictTypedSuggestion:
    """Cover partial branch 512->511: non-dict items in typed_suggestions are skipped."""

    def test_evolve_skips_non_dict_typed_suggestion(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        reflections_dir = project / ".factory" / "outer_loop" / "reflections"
        reflections_dir.mkdir(parents=True)

        report_data = {
            "failure_patterns": ["f1"],
            "success_patterns": ["s1"],
            "mutation_suggestions": [],
            "prompt_improvements": [],
            "structural_recommendations": [],
            "top_k_ids": ["w1"],
            "bottom_k_ids": ["l1"],
            "typed_suggestions": [
                "not-a-dict",
                {"operator": "knob_mutate", "target": "t", "rationale": "r", "value": "v"},
            ],
        }
        (reflections_dir / "gen0.json").write_text(json.dumps(report_data))

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop import mutations as _mutations_mod

        original_apply = _mutations_mod.apply_random_mutation

        def spy_apply(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_apply(*args, **kwargs)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=spy_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            _cmd_evolve(ns)

        assert len(captured_kwargs) > 0
        report = captured_kwargs[0].get("reflection_report")
        assert report is not None
        assert len(report.typed_suggestions) == 1
        assert report.typed_suggestions[0].operator == "knob_mutate"


class TestEvolveNonListTypedSuggestions:
    """Cover partial branch 510->521: non-list typed_suggestions is skipped."""

    def test_evolve_skips_non_list_typed_suggestions(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)
        reflections_dir = project / ".factory" / "outer_loop" / "reflections"
        reflections_dir.mkdir(parents=True)

        report_data = {
            "failure_patterns": ["f1"],
            "success_patterns": ["s1"],
            "mutation_suggestions": [],
            "prompt_improvements": [],
            "structural_recommendations": [],
            "top_k_ids": ["w1"],
            "bottom_k_ids": ["l1"],
            "typed_suggestions": "not-a-list",
        }
        (reflections_dir / "gen0.json").write_text(json.dumps(report_data))

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="test-wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )
        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("test01", 0, wf)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        captured_kwargs: list[dict] = []
        from factory.outer_loop import mutations as _mutations_mod

        original_apply = _mutations_mod.apply_random_mutation

        def spy_apply(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_apply(*args, **kwargs)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=spy_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            _cmd_evolve(ns)

        assert len(captured_kwargs) > 0
        report = captured_kwargs[0].get("reflection_report")
        assert report is not None
        assert len(report.typed_suggestions) == 0


class TestEvolveRegistryLoadNone:
    """Cover line 532: continue when registry.load returns None in evolve loop."""

    def test_evolve_skips_none_workflow(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.models import SwarmConfig

        project = Path(str(tmp_path))
        modes_dir = project / ".factory" / "outer_loop" / "modes"
        modes_dir.mkdir(parents=True)

        cfg = SwarmConfig(benchmark="featurebench", budget=50)

        mock_registry = MagicMock()
        mock_registry.list_modes.return_value = ["mode-a", "mode-b"]
        mock_registry.load.return_value = None
        mock_registry.prune_stale_modes.return_value = None

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.mode_registry.EphemeralModeRegistry",
                   return_value=mock_registry):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            rc = _cmd_evolve(ns)
            assert rc == 0


class TestOuterLoopWorkflowGraph:
    def test_workflow_validates(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow

        wf = workflow()
        issues = wf.validate_graph()
        assert issues == [], f"Workflow validation issues: {issues}"

    def test_workflow_name(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow

        wf = workflow()
        assert wf.name == "outer-loop"

    def test_workflow_start_node(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow

        wf = workflow()
        assert wf.start_node == "seed"

    def test_workflow_has_expected_nodes(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow

        wf = workflow()
        expected = {"seed", "evaluate", "reflect", "evolve", "gate_converge", "promote"}
        assert set(wf.nodes.keys()) == expected

    def test_workflow_generation_loop(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow
        from factory.workflow.primitives import VerdictType

        wf = workflow()
        loop_edge = [
            e for e in wf.edges
            if e.source == "gate_converge"
            and e.target == "evaluate"
            and e.condition == VerdictType.RELOOP
        ]
        assert len(loop_edge) == 1

    def test_workflow_is_terminal(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow

        wf = workflow()
        assert wf.terminal is True

    def test_workflow_serialization_round_trip(self) -> None:
        from factory.workflow.contributed.outer_loop.workflow import workflow
        from factory.workflow.primitives import Workflow

        wf = workflow()
        data = wf.to_dict()
        restored = Workflow.from_dict(data)

        assert restored.name == wf.name
        assert set(restored.nodes.keys()) == set(wf.nodes.keys())
        assert len(restored.edges) == len(wf.edges)


class TestEvaluatePropagatesScoresToPopulation:
    """_cmd_evaluate must write scores back to population/population.json.

    Without this the population keeps score=0.0 for every individual and the
    evolve step has no fitness signal for parent selection.
    """

    @staticmethod
    def _builder_workflow(name: str) -> object:
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        return Workflow(
            name=name,
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        )

    def test_population_scores_updated_from_results(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.outer_loop.models import EvalResult, SwarmConfig
        from factory.outer_loop.population import Population

        project = Path(str(tmp_path))
        (project / ".factory" / "outer_loop" / "modes").mkdir(parents=True)
        pop_dir = project / ".factory" / "outer_loop" / "population"

        registry = EphemeralModeRegistry(project)
        population = Population()
        scores_by_mode: dict[str, float] = {}
        expected: dict[str, float] = {}
        for i, score in enumerate([0.25, 0.75]):
            wf = self._builder_workflow(f"seed-{i}")
            ind = Population.make_individual(wf, generation=0)
            population.add(ind)
            mode_name = registry.register(ind.id, 0, wf)
            scores_by_mode[mode_name] = score
            expected[ind.id] = score
        population.save(pop_dir)

        assert all(i.score is None for i in Population.load(pop_dir).individuals)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = lambda wf, *a, **k: EvalResult(
            score=scores_by_mode[wf.name],
            benchmark_score=scores_by_mode[wf.name],
            cost_usd=scores_by_mode[wf.name] * 10,
        )

        cfg = SwarmConfig(benchmark="featurebench", budget=50)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.load_checkpoint", return_value=None), \
             patch("factory.outer_loop.filesystem.save_checkpoint"), \
             patch("factory.outer_loop.evaluator.SwarmEvaluator", return_value=mock_evaluator):
            from factory.cli.outer_loop import _cmd_evaluate

            ns = argparse.Namespace(project_path=str(project), generation=0, project_dir=None)
            assert _cmd_evaluate(ns) == 0

        reloaded = Population.load(pop_dir)
        assert reloaded.size == 2
        for ind in reloaded.individuals:
            assert ind.score == expected[ind.id]
            assert ind.cost_usd == expected[ind.id] * 10

        results = json.loads(
            (project / ".factory" / "outer_loop" / "results" / "gen0.json").read_text()
        )
        by_id = {i.id: i.score for i in reloaded.individuals}
        for mode_name, entry in results.items():
            suffix = mode_name[len("evolve-gen0-"):]
            match = next(iid for iid in by_id if iid.startswith(suffix))
            assert by_id[match] == entry["score"]

    def test_missing_population_file_is_tolerated(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _propagate_scores_to_population

        project = Path(str(tmp_path))
        updated = _propagate_scores_to_population(
            project, 0, {"evolve-gen0-abc": {"score": 0.5, "cost_usd": 1.0}}
        )
        assert updated == 0
        assert not (project / ".factory" / "outer_loop" / "population").exists()

    def test_unmatched_mode_is_skipped(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _propagate_scores_to_population
        from factory.outer_loop.population import Population

        project = Path(str(tmp_path))
        pop_dir = project / ".factory" / "outer_loop" / "population"
        population = Population()
        ind = Population.make_individual(self._builder_workflow("seed"), generation=0)
        population.add(ind)
        population.save(pop_dir)

        updated = _propagate_scores_to_population(
            project, 0, {"evolve-gen0-zzzzzzzz": {"score": 0.9, "cost_usd": 1.0}}
        )
        assert updated == 0
        assert Population.load(pop_dir).get(ind.id).score is None


class TestLoadModeScores:
    def test_missing_results_file_returns_empty(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _load_mode_scores

        assert _load_mode_scores(Path(str(tmp_path)), 0) == {}

    def test_scores_parsed_from_results(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        from factory.cli.outer_loop import _load_mode_scores

        project = Path(str(tmp_path))
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "gen0.json").write_text(json.dumps({
            "evolve-gen0-aaa": {"score": 8.11, "cost_usd": 1.0},
            "evolve-gen0-bbb": {"score": 4.81, "cost_usd": 2.0},
        }))

        assert _load_mode_scores(project, 0) == {
            "evolve-gen0-aaa": 8.11,
            "evolve-gen0-bbb": 4.81,
        }

    def test_malformed_results_returns_empty(self, tmp_path: object) -> None:
        from pathlib import Path

        from factory.cli.outer_loop import _load_mode_scores

        project = Path(str(tmp_path))
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "gen0.json").write_text("{not json")

        assert _load_mode_scores(project, 0) == {}


class TestSelectParent:
    def test_empty_modes_returns_none(self) -> None:
        from factory.cli.outer_loop import _select_parent

        assert _select_parent([], {}) is None

    def test_single_mode_always_selected(self) -> None:
        from factory.cli.outer_loop import _select_parent

        assert _select_parent(["only"], {"only": 0.0}) == "only"

    def test_worst_parent_never_wins_a_tournament(self) -> None:
        from factory.cli.outer_loop import _select_parent

        modes = ["a", "b", "c", "d"]
        scores = {"a": 8.11, "b": 6.0, "c": 5.5, "d": 4.81}
        counts = dict.fromkeys(modes, 0)
        for _ in range(400):
            counts[_select_parent(modes, scores)] += 1

        assert counts["d"] == 0
        assert counts["a"] > counts["b"] > counts["c"]

    def test_unscored_modes_degrade_to_uniform(self) -> None:
        from factory.cli.outer_loop import _select_parent

        modes = ["a", "b", "c", "d"]
        counts = dict.fromkeys(modes, 0)
        for _ in range(400):
            counts[_select_parent(modes, {})] += 1

        assert all(c > 0 for c in counts.values())


class TestEvolveSelectionPressure:
    """_cmd_evolve must bias offspring toward higher-scoring parents."""

    def test_offspring_biased_toward_high_scoring_parents(self, tmp_path: object) -> None:
        import argparse
        import json
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop import mutations as _mutations_mod
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.outer_loop.models import MutationRecord, MutationType, SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        (project / ".factory" / "outer_loop" / "modes").mkdir(parents=True)
        results_dir = project / ".factory" / "outer_loop" / "results"
        results_dir.mkdir(parents=True)

        registry = EphemeralModeRegistry(project)
        labelled_scores = {"best0000": 8.11, "mid00000": 6.0, "low00000": 5.5, "worst000": 4.81}
        results: dict[str, dict[str, float]] = {}
        for ind_id, score in labelled_scores.items():
            wf = Workflow(
                name=f"wf-{ind_id}",
                nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
                edges=[], start_node="b", terminal=True,
            )
            mode_name = registry.register(ind_id, 0, wf)
            results[mode_name] = {"score": score, "cost_usd": 1.0}
        (results_dir / "gen0.json").write_text(json.dumps(results))

        parents: list[str] = []

        def fake_apply(wf, *args, **kwargs):
            # Stubbed: the real operators can call out to an LLM (PROMPT_MUTATE),
            # which would make a 200-trial statistical test take minutes.
            parents.append(wf.name)
            return wf, MutationRecord(operator=MutationType.PARAM_MUTATE)

        cfg = SwarmConfig(benchmark="featurebench", budget=50, population_size=200)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch("factory.outer_loop.filesystem.load_checkpoint", return_value=None), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=fake_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            assert _cmd_evolve(ns) == 0

        assert len(parents) == 200
        counts = {ind_id: 0 for ind_id in labelled_scores}
        for name in parents:
            counts[name[len("evolve-gen0-"):]] += 1

        assert counts["worst000"] == 0
        assert counts["best0000"] > counts["mid00000"]
        assert counts["best0000"] > counts["low00000"]

    def test_only_current_generation_modes_are_parents(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop import mutations as _mutations_mod
        from factory.outer_loop.mode_registry import EphemeralModeRegistry
        from factory.outer_loop.models import MutationRecord, MutationType, SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        (project / ".factory" / "outer_loop" / "modes").mkdir(parents=True)

        registry = EphemeralModeRegistry(project)
        for gen, ind_id in ((0, "genzero0"), (1, "genone000")):
            wf = Workflow(
                name=f"wf-{ind_id}",
                nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
                edges=[], start_node="b", terminal=True,
            )
            registry.register(ind_id, gen, wf)

        parents: list[str] = []

        def fake_apply(wf, *args, **kwargs):
            parents.append(wf.name)
            return wf, MutationRecord(operator=MutationType.PARAM_MUTATE)

        cfg = SwarmConfig(benchmark="featurebench", budget=50, population_size=10)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=fake_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            assert _cmd_evolve(ns) == 0

        assert parents
        assert all(p.startswith("evolve-gen0-") for p in parents)


class TestEvolveAddsOffspringToPopulation:
    """_cmd_evolve must mirror each new offspring mode into population.json.

    Without this the population never grows past generation 0, so
    _propagate_scores_to_population finds no individual to attach gen 1+
    scores to and the population silently goes stale.
    """

    @staticmethod
    def _seed(project, *, parent_ids, population_size):
        """Register gen 0 modes and a matching population; return the registry."""
        from factory.outer_loop.models import SwarmConfig
        from factory.outer_loop.population import Population
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        (project / ".factory" / "outer_loop" / "modes").mkdir(parents=True)

        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        population = Population()
        for ind_id in parent_ids:
            wf = Workflow(
                name=f"wf-{ind_id}",
                nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
                edges=[], start_node="b", terminal=True,
            )
            registry.register(ind_id, 0, wf)
            population.add(
                Population.make_individual(wf, generation=0).model_copy(update={"id": ind_id})
            )
        population.save(project / ".factory" / "outer_loop" / "population")

        cfg = SwarmConfig(
            benchmark="featurebench", budget=50, population_size=population_size
        )
        return registry, cfg

    @staticmethod
    def _load_population(project):
        from factory.outer_loop.population import Population

        return Population.load(project / ".factory" / "outer_loop" / "population")

    def test_offspring_individuals_are_persisted(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop import mutations as _mutations_mod
        from factory.outer_loop.models import MutationRecord, MutationType

        project = Path(str(tmp_path))
        _, cfg = self._seed(project, parent_ids=["p0"], population_size=2)

        def fake_apply(wf, *args, **kwargs):
            return wf, MutationRecord(operator=MutationType.PARAM_MUTATE)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=fake_apply):
            from factory.cli.outer_loop import _cmd_evolve

            assert _cmd_evolve(argparse.Namespace(project_path=str(project), generation=0)) == 0

        population = self._load_population(project)
        offspring = [i for i in population.individuals if i.generation == 1]
        assert len(offspring) == 2
        assert {i.id for i in offspring} == {"gen1_0", "gen1_1"}
        for ind in offspring:
            assert ind.score == 0.0
            assert ind.parent_id == "p0"
            assert ind.mutation_record is not None
            assert ind.mutation_record.operator == MutationType.PARAM_MUTATE
            assert ind.workflow_data["nodes"]

    def test_offspring_ids_match_mode_names_so_scores_propagate(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop import mutations as _mutations_mod
        from factory.outer_loop.models import MutationRecord, MutationType

        project = Path(str(tmp_path))
        _, cfg = self._seed(project, parent_ids=["p0"], population_size=2)

        def fake_apply(wf, *args, **kwargs):
            return wf, MutationRecord(operator=MutationType.NODE_INSERT)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=fake_apply):
            from factory.cli.outer_loop import _cmd_evolve

            assert _cmd_evolve(argparse.Namespace(project_path=str(project), generation=0)) == 0

        from factory.cli.outer_loop import _propagate_scores_to_population

        results = {
            "evolve-gen1-gen1_0": {"score": 0.75, "cost_usd": 1.5},
            "evolve-gen1-gen1_1": {"score": 0.25, "cost_usd": 0.5},
        }
        assert _propagate_scores_to_population(project, 1, results) == 2

        population = self._load_population(project)
        assert population.get("gen1_0").score == 0.75
        assert population.get("gen1_0").cost_usd == 1.5
        assert population.get("gen1_1").score == 0.25
        assert population.get("p0").score is None

    def test_population_grows_each_generation(self, tmp_path: object) -> None:
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop import mutations as _mutations_mod
        from factory.outer_loop.models import MutationRecord, MutationType

        project = Path(str(tmp_path))
        _, cfg = self._seed(project, parent_ids=["p0", "p1"], population_size=3)
        assert self._load_population(project).size == 2

        def fake_apply(wf, *args, **kwargs):
            return wf, MutationRecord(operator=MutationType.PARAM_MUTATE)

        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg), \
             patch.object(_mutations_mod, "apply_random_mutation", side_effect=fake_apply):
            from factory.cli.outer_loop import _cmd_evolve

            ns = argparse.Namespace(project_path=str(project), generation=0)
            assert _cmd_evolve(ns) == 0
            after_gen1 = self._load_population(project)
            assert after_gen1.size == 2 + 3
            assert {i.id for i in after_gen1.individuals} >= {"p0", "p1"}

            ns2 = argparse.Namespace(project_path=str(project), generation=1)
            assert _cmd_evolve(ns2) == 0

        after_gen2 = self._load_population(project)
        assert after_gen2.size == 2 + 3 + 3
        assert len([i for i in after_gen2.individuals if i.generation == 2]) == 3

    def test_evolve_without_population_file_is_a_noop(self, tmp_path: object) -> None:
        """A project that never seeded a population still evolves modes fine."""
        import argparse
        from pathlib import Path
        from unittest.mock import patch

        from factory.outer_loop.models import SwarmConfig
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        project = Path(str(tmp_path))
        (project / ".factory" / "outer_loop" / "modes").mkdir(parents=True)

        from factory.outer_loop.mode_registry import EphemeralModeRegistry

        registry = EphemeralModeRegistry(project)
        registry.register("p0", 0, Workflow(
            name="wf",
            nodes={"b": AgentNode(id="b", role=AgentRole.BUILDER, writes=set())},
            edges=[], start_node="b", terminal=True,
        ))

        cfg = SwarmConfig(benchmark="featurebench", budget=50, population_size=2)
        with patch("factory.outer_loop.filesystem.load_config", return_value=cfg):
            from factory.cli.outer_loop import _cmd_evolve

            assert _cmd_evolve(argparse.Namespace(project_path=str(project), generation=0)) == 0

        assert not (project / ".factory" / "outer_loop" / "population" / "population.json").exists()
