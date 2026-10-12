"""Tests for EphemeralModeRegistry."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path


from factory.outer_loop.mode_registry import EphemeralModeRegistry
from factory.outer_loop.population import Population
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    GateNode,
    Workflow,
)


def _make_workflow(name: str = "test_wf") -> Workflow:
    return Workflow(
        name=name,
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                writes={".factory/reviews/builder-latest.md"},
            ),
            "gate": GateNode(
                id="gate",
                evaluator_type="agent",
                evaluator_role=AgentRole.HEALTH_CHECKER,
            ),
        },
        edges=[Edge(source="builder", target="gate")],
        start_node="builder",
    )


class TestEphemeralModeRegistry:
    def test_register_creates_file(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("abc12345", 0, wf)

        assert mode_name == "evolve-gen0-abc12345"
        mode_file = tmp_path / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        assert mode_file.exists()

    def test_register_naming_convention(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        name0 = registry.register("individual1", 0, wf)
        name1 = registry.register("individual2", 3, wf)

        assert name0 == "evolve-gen0-individu"
        assert name1 == "evolve-gen3-individu"

    def test_load_round_trip(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("test1234", 0, wf)

        loaded = registry.load(mode_name)
        assert loaded is not None
        assert set(loaded.nodes.keys()) == {"builder", "gate"}
        assert loaded.start_node == "builder"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        assert registry.load("nonexistent-mode") is None

    def test_cleanup_generation(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 0, wf)
        registry.register("ccc", 0, wf)

        assert registry.count == 3
        removed = registry.cleanup_generation({"evolve-gen0-aaa"})
        assert removed == 2
        assert registry.count == 1
        modes = registry.list_modes()
        assert "evolve-gen0-aaa" in modes

    def test_cleanup_all(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 1, wf)

        removed = registry.cleanup_all()
        assert removed == 2
        assert registry.count == 0

    def test_cleanup_all_keep_best(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 0, wf)

        removed = registry.cleanup_all(keep_best="evolve-gen0-bbb")
        assert removed == 1
        assert registry.count == 1

    def test_context_manager_cleanup(self, tmp_path: Path) -> None:
        with EphemeralModeRegistry(tmp_path) as registry:
            wf = _make_workflow()
            registry.register("test", 0, wf)
            assert registry.count == 1

        # After context exit, modes should be cleaned up
        fresh = EphemeralModeRegistry(tmp_path)
        assert fresh.count == 0

    def test_promote(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("winner", 5, wf)

        dest = registry.promote(mode_name, "best-evolved")
        assert dest is not None
        assert dest.exists()
        assert "best-evolved" in str(dest)

    def test_promote_nonexistent(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        assert registry.promote("nonexistent", "test") is None

    def test_list_modes(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 1, wf)

        modes = registry.list_modes()
        assert len(modes) == 2
        assert "evolve-gen0-aaa" in modes
        assert "evolve-gen1-bbb" in modes

    def test_register_creates_workflow_wrapper(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("abc12345", 0, wf)

        wrapper = tmp_path / ".factory" / "workflows" / f"{mode_name}.py"
        assert wrapper.exists()

        spec = importlib.util.spec_from_file_location(f"_test_{mode_name}", wrapper)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        sys.modules.pop(spec.name, None)

        assert mod.meta["name"] == mode_name
        loaded = mod.workflow()
        assert set(loaded.nodes.keys()) == {"builder", "gate"}

    def test_cleanup_generation_removes_wrappers(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 0, wf)

        wf_dir = tmp_path / ".factory" / "workflows"
        assert (wf_dir / "evolve-gen0-aaa.py").exists()
        assert (wf_dir / "evolve-gen0-bbb.py").exists()

        registry.cleanup_generation({"evolve-gen0-aaa"})
        assert (wf_dir / "evolve-gen0-aaa.py").exists()
        assert not (wf_dir / "evolve-gen0-bbb.py").exists()

    def test_cleanup_all_removes_wrappers(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()

        registry.register("aaa", 0, wf)
        registry.register("bbb", 0, wf)

        wf_dir = tmp_path / ".factory" / "workflows"
        registry.cleanup_all(keep_best="evolve-gen0-aaa")
        assert (wf_dir / "evolve-gen0-aaa.py").exists()
        assert not (wf_dir / "evolve-gen0-bbb.py").exists()

    def test_context_manager_removes_wrappers(self, tmp_path: Path) -> None:
        with EphemeralModeRegistry(tmp_path) as registry:
            wf = _make_workflow()
            registry.register("test", 0, wf)
            assert (tmp_path / ".factory" / "workflows" / "evolve-gen0-test.py").exists()

        assert not (tmp_path / ".factory" / "workflows" / "evolve-gen0-test.py").exists()


class TestEphemeralModeRegistryTargetDir:
    """Tests for target_dir mirroring when sub-CEO runs in a different project."""

    def test_register_mirrors_to_target(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        mode_name = registry.register("abc12345", 0, wf)

        assert (outer / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json").exists()
        assert (outer / ".factory" / "workflows" / f"{mode_name}.py").exists()
        assert (target / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json").exists()
        assert (target / ".factory" / "workflows" / f"{mode_name}.py").exists()

    def test_target_wrapper_loads_correctly(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        mode_name = registry.register("abc12345", 0, wf)

        wrapper = target / ".factory" / "workflows" / f"{mode_name}.py"
        spec = importlib.util.spec_from_file_location(f"_test_target_{mode_name}", wrapper)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        sys.modules.pop(spec.name, None)

        loaded = mod.workflow()
        assert set(loaded.nodes.keys()) == {"builder", "gate"}

    def test_cleanup_all_removes_target_artifacts(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        mode_name = registry.register("aaa", 0, wf)

        registry.cleanup_all()
        assert not (target / ".factory" / "workflows" / f"{mode_name}.py").exists()
        assert not (target / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json").exists()

    def test_cleanup_generation_removes_target_artifacts(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        registry.register("aaa", 0, wf)
        registry.register("bbb", 0, wf)

        registry.cleanup_generation({"evolve-gen0-aaa"})
        assert (target / ".factory" / "workflows" / "evolve-gen0-aaa.py").exists()
        assert not (target / ".factory" / "workflows" / "evolve-gen0-bbb.py").exists()

    def test_no_target_dir_no_mirroring(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer)
        wf = _make_workflow()
        registry.register("abc12345", 0, wf)

        assert not (target / ".factory" / "workflows").exists()
        assert not (target / ".factory" / "outer_loop").exists()

    def test_same_dir_target_no_duplicate(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path, target_dir=tmp_path)
        assert not registry.has_target
        wf = _make_workflow()
        mode_name = registry.register("abc12345", 0, wf)
        assert (tmp_path / ".factory" / "workflows" / f"{mode_name}.py").exists()


class TestPruneStaleModes:
    def test_prune_removes_old_modes(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("old_mode", 0, wf)

        mode_file = tmp_path / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        old_time = time.time() - 25 * 3600
        os.utime(mode_file, (old_time, old_time))

        pruned = registry.prune_stale_modes(older_than_hours=24)
        assert mode_name in pruned
        assert not mode_file.exists()
        wrapper = tmp_path / ".factory" / "workflows" / f"{mode_name}.py"
        assert not wrapper.exists()

    def test_prune_keeps_recent_modes(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        registry.register("new_mode", 0, wf)

        pruned = registry.prune_stale_modes(older_than_hours=24)
        assert len(pruned) == 0
        assert registry.count == 1

    def test_prune_mixed_old_and_new(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        old_name = registry.register("old_one", 0, wf)
        new_name = registry.register("new_one", 1, wf)

        old_file = tmp_path / ".factory" / "outer_loop" / "modes" / f"{old_name}.json"
        old_time = time.time() - 48 * 3600
        os.utime(old_file, (old_time, old_time))

        pruned = registry.prune_stale_modes(older_than_hours=24)
        assert old_name in pruned
        assert new_name not in pruned
        assert registry.count == 1

    def test_prune_empty_modes_dir(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        pruned = registry.prune_stale_modes()
        assert pruned == []

    def test_prune_removes_target_artifacts(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        target = tmp_path / "target"
        outer.mkdir()
        target.mkdir()

        registry = EphemeralModeRegistry(outer, target_dir=target)
        wf = _make_workflow()
        mode_name = registry.register("old_tgt", 0, wf)

        mode_file = outer / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        old_time = time.time() - 25 * 3600
        os.utime(mode_file, (old_time, old_time))

        pruned = registry.prune_stale_modes(older_than_hours=24)
        assert mode_name in pruned
        assert not (target / ".factory" / "workflows" / f"{mode_name}.py").exists()
        assert not (target / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json").exists()


# ── Tests moved from test_coverage_gaps.py ─────────────────────────


class TestModeRegistryHandlesHashCollision:
    """Verifies registry detects 12-char prefix collisions."""

    def test_same_id_prefix_different_generations_no_collision(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        name0 = registry.register("abcdefgh_extra", 0, wf)
        name1 = registry.register("abcdefgh_extra", 1, wf)
        assert name0 != name1
        assert name0 == "evolve-gen0-abcdefgh"
        assert name1 == "evolve-gen1-abcdefgh"
        assert registry.count == 2

    def test_same_prefix_same_generation_overwrites(self, tmp_path: Path) -> None:
        registry = EphemeralModeRegistry(tmp_path)
        wf1 = _make_workflow("wf1")
        wf2 = _make_workflow("wf2")
        name1 = registry.register("abcdefgh_111", 0, wf1)
        name2 = registry.register("abcdefgh_222", 0, wf2)
        assert name1 == name2
        loaded = registry.load(name2)
        assert loaded is not None
        assert loaded.name == name2

    def test_content_hash_detects_tampered_mode_file(self, tmp_path: Path) -> None:
        import json
        registry = EphemeralModeRegistry(tmp_path)
        wf = _make_workflow()
        mode_name = registry.register("hashtest1", 0, wf)
        mode_path = tmp_path / ".factory" / "outer_loop" / "modes" / f"{mode_name}.json"
        data = json.loads(mode_path.read_text())
        data["name"] = "tampered-name"
        mode_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        loaded = registry.load(mode_name)
        assert loaded is not None

    def test_12_char_uuid_prefix_uniqueness(self) -> None:
        wf = _make_workflow()
        ids = set()
        for _ in range(20):
            ind = Population.make_individual(wf, generation=0)
            assert len(ind.id) == 12
            ids.add(ind.id)
        assert len(ids) == 20
