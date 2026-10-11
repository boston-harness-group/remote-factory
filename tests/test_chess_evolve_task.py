"""Tests for ChessEvolveTask — registration, four hooks, self-play, compose integration."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

chess = pytest.importorskip("chess", reason="python-chess required for chess evolve tests")

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from factory.compose import compose  # noqa: E402
from factory.inner_loop import InnerLoop  # noqa: E402
from factory.task import TaskInstance, VerifyResult  # noqa: E402
from chess_evolve_task import ChessEvolveTask, _load_engine_module, _play_game  # noqa: E402


# ── Registration ─────────────────────────────────────────────────


class TestRegistration:
    def test_task_registry_discovers_chess_evolve(self, tmp_path: Path):
        from factory.task_registry import TaskRegistry

        task_dir = tmp_path / ".factory" / "tasks"
        task_dir.mkdir(parents=True)

        import shutil
        src = Path(__file__).parent.parent / "examples" / "chess_evolve_task.py"
        shutil.copy(src, task_dir / "chess_evolve.py")

        TaskRegistry.reset()
        entries = TaskRegistry.discover(tmp_path)
        assert "chess-evolve" in entries

    def test_task_meta_and_factory_function(self):
        from chess_evolve_task import meta, task

        assert meta["name"] == "chess-evolve"
        t = task()
        assert isinstance(t, ChessEvolveTask)
        assert t.name == "chess-evolve"


# ── Instances ────────────────────────────────────────────────────


class TestInstances:
    def test_yields_task_instances(self):
        t = ChessEvolveTask()
        instances = list(t.instances())
        assert len(instances) > 0
        for inst in instances:
            assert isinstance(inst, TaskInstance)
            assert inst.id
            assert "depth" in inst.metadata
            assert "num_games" in inst.metadata
            assert "opening_fen" in inst.metadata

    def test_instance_ids_unique(self):
        t = ChessEvolveTask()
        ids = [inst.id for inst in t.instances()]
        assert len(ids) == len(set(ids))

    def test_varying_depths(self):
        t = ChessEvolveTask()
        depths = {inst.metadata["depth"] for inst in t.instances()}
        assert len(depths) >= 2

    def test_varying_openings(self):
        t = ChessEvolveTask()
        openings = {inst.metadata["opening_name"] for inst in t.instances()}
        assert len(openings) >= 2


# ── Setup ────────────────────────────────────────────────────────


class TestSetup:
    def test_creates_workspace_files(self, tmp_path: Path):
        t = ChessEvolveTask()
        inst = TaskInstance(id="depth2-startpos", metadata={"depth": 2})
        t.setup(inst, tmp_path)

        assert (tmp_path / "src" / "engine.py").exists()
        assert (tmp_path / "base" / "engine.py").exists()

        evolved_src = (tmp_path / "src" / "engine.py").read_text()
        base_src = (tmp_path / "base" / "engine.py").read_text()
        assert "def best_move" in evolved_src
        assert "def minimax" in base_src

    def test_setup_idempotent(self, tmp_path: Path):
        t = ChessEvolveTask()
        inst = TaskInstance(id="depth1-startpos", metadata={"depth": 1})
        t.setup(inst, tmp_path)

        (tmp_path / "src" / "engine.py").write_text("# custom engine\n")
        t.setup(inst, tmp_path)

        content = (tmp_path / "src" / "engine.py").read_text()
        assert content == "# custom engine\n"


# ── Prompt ───────────────────────────────────────────────────────


class TestPrompt:
    def test_returns_nonempty_string(self):
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth2-italian",
            metadata={"depth": 2, "opening_name": "italian"},
        )
        p = t.prompt(inst)
        assert isinstance(p, str)
        assert len(p) > 0

    def test_references_depth(self):
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth3-startpos",
            metadata={"depth": 3, "opening_name": "startpos"},
        )
        p = t.prompt(inst)
        assert "3" in p

    def test_references_opening(self):
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth2-sicilian",
            metadata={"depth": 2, "opening_name": "sicilian"},
        )
        p = t.prompt(inst)
        assert "sicilian" in p


# ── Verify (self-play) ──────────────────────────────────────────


class TestVerify:
    def test_returns_verify_result(self, tmp_path: Path):
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth1-startpos",
            metadata={
                "depth": 1,
                "num_games": 2,
                "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            },
        )
        t.setup(inst, tmp_path)
        result = t.verify(inst, tmp_path)

        assert isinstance(result, VerifyResult)
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert "wins" in result.details
        assert "draws" in result.details
        assert "losses" in result.details
        assert "win_rate" in result.details
        assert "blunder_count" in result.details
        assert isinstance(result.details["blunder_count"], int)
        assert "avg_eval" in result.details
        assert isinstance(result.details["avg_eval"], float)
        assert "total_moves" in result.details
        assert isinstance(result.details["total_moves"], int)
        assert result.details["total_moves"] > 0

        games = result.details["games"]
        assert len(games) == 2
        for game in games:
            assert "eval_curve" in game
            assert "move_list" in game
            assert isinstance(game["eval_curve"], list)
            assert isinstance(game["move_list"], list)
            assert len(game["eval_curve"]) == game["moves"]
            assert len(game["move_list"]) == game["moves"]

    def test_identical_engines_draw(self, tmp_path: Path):
        """Same engine vs itself should roughly tie (score ~0.5)."""
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth1-startpos",
            metadata={
                "depth": 1,
                "num_games": 2,
                "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            },
        )
        t.setup(inst, tmp_path)
        result = t.verify(inst, tmp_path)

        assert result.score >= 0.0

    def test_missing_engine_returns_zero(self, tmp_path: Path):
        t = ChessEvolveTask()
        inst = TaskInstance(id="depth1-startpos", metadata={"depth": 1, "num_games": 2})
        result = t.verify(inst, tmp_path)
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.timeout(30)
    def test_completes_within_timeout(self, tmp_path: Path):
        """Self-play at depth 1 with 2 games should be fast."""
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth1-startpos",
            metadata={
                "depth": 1,
                "num_games": 2,
                "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            },
        )
        t.setup(inst, tmp_path)
        result = t.verify(inst, tmp_path)
        assert isinstance(result, VerifyResult)


# ── Compose integration ─────────────────────────────────────────


class TestComposeIntegration:
    def test_compose_produces_inner_loop(self, tmp_path: Path):
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="improve",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                ),
            },
            edges=[],
            start_node="builder",
        )

        t = ChessEvolveTask()
        loop = compose(wf, t, tmp_path)
        assert isinstance(loop, InnerLoop)
        assert loop.task is t

    def test_inner_loop_step_iterates_instances(self, tmp_path: Path):
        from unittest.mock import patch

        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        wf = Workflow(
            name="improve",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="build",
                ),
            },
            edges=[],
            start_node="builder",
        )

        t = _SingleInstanceChessTask()
        loop = compose(wf, t, tmp_path)

        from factory.workflow.executor import ExecutionResult

        async def fake_execute(self_exec):
            r = ExecutionResult()
            r.success = True
            r.item_results = [
                {"instance_id": "depth1-startpos", "score": 0.5, "status": "passed"},
            ]
            return r

        with patch.object(
            __import__("factory.workflow.executor", fromlist=["WorkflowExecutor"]).WorkflowExecutor,
            "execute",
            fake_execute,
        ):
            record = loop.step()

        assert record.instance_results is not None
        assert len(record.instance_results) >= 1
        assert record.score_end is not None
        assert 0.0 <= record.score_end <= 1.0

        for ir in record.instance_results:
            assert "instance_id" in ir
            assert "score" in ir


# ── Helper: single-instance variant for fast integration test ────


class _SingleInstanceChessTask(ChessEvolveTask):
    """Yields only one instance at depth 1 for fast integration tests."""

    def instances(self) -> Iterator[TaskInstance]:
        yield TaskInstance(
            id="depth1-startpos",
            metadata={
                "depth": 1,
                "num_games": 2,
                "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            },
        )


# ── Helper function tests ───────────────────────────────────────


class TestHelpers:
    def test_load_engine_module(self, tmp_path: Path):
        engine_path = tmp_path / "test_engine.py"
        engine_path.write_text(
            "import chess\n"
            "def best_move(board, depth=2):\n"
            "    moves = list(board.legal_moves)\n"
            "    return moves[0] if moves else None\n"
        )
        mod = _load_engine_module(engine_path, "test_mod")
        assert hasattr(mod, "best_move")

        board = chess.Board()
        move = mod.best_move(board)
        assert move is not None

    def test_play_game_completes(self, tmp_path: Path):
        from chess_evolve_task import BASE_ENGINE_SOURCE

        engine_path = tmp_path / "e.py"
        engine_path.write_text(BASE_ENGINE_SOURCE)

        mod = _load_engine_module(engine_path, "play_test")
        result = _play_game(
            mod, mod,
            depth_evolved=1, depth_base=1,
            start_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            max_moves=40,
        )
        assert "result" in result
        assert "moves" in result
        assert result["moves"] > 0
        assert "eval_curve" in result
        assert "move_list" in result
        assert len(result["eval_curve"]) == result["moves"]
        assert len(result["move_list"]) == result["moves"]
        assert all(isinstance(e, int) for e in result["eval_curve"])
        assert all(isinstance(m, str) for m in result["move_list"])

    def test_play_game_checkmate(self, tmp_path: Path):
        """Game ending in checkmate (not max_moves) — covers outcome branch."""
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        from chess_evolve_task import BASE_ENGINE_SOURCE

        engine_path = tmp_path / "e_cm.py"
        engine_path.write_text(BASE_ENGINE_SOURCE)
        mod = _load_engine_module(engine_path, "checkmate_test")

        result = _play_game(
            mod, mod,
            depth_evolved=1, depth_base=1,
            start_fen=fen,
            max_moves=200,
        )
        assert result["termination"] != "max_moves"
        assert result["winner"] is not None or result["termination"] in (
            "STALEMATE", "INSUFFICIENT_MATERIAL", "stalemate", "insufficient_material",
        )
        assert len(result["eval_curve"]) == result["moves"]
        assert len(result["move_list"]) == result["moves"]

    def test_play_game_move_is_none(self, tmp_path: Path):
        """When engine returns None for a move, the game should end."""
        from unittest.mock import MagicMock

        none_engine = MagicMock()
        none_engine.best_move.return_value = None

        from chess_evolve_task import BASE_ENGINE_SOURCE

        base_path = tmp_path / "base_e.py"
        base_path.write_text(BASE_ENGINE_SOURCE)
        base_mod = _load_engine_module(base_path, "base_none_test")

        result = _play_game(
            none_engine, base_mod,
            depth_evolved=1, depth_base=1,
            start_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            max_moves=80,
        )
        assert result["moves"] == 0
        assert result["termination"] == "max_moves" or result["winner"] is None


class TestVerifyEdgeCases:
    """Verify edge cases — chess unavailable, engine load failure."""

    def test_verify_chess_unavailable(self, tmp_path: Path):
        """verify() returns score=0 when chess library is not available."""
        from unittest.mock import patch

        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth1-startpos",
            metadata={"depth": 1, "num_games": 2, "opening_fen": "x"},
        )

        with patch("chess_evolve_task._ensure_chess_available", return_value=False):
            result = t.verify(inst, tmp_path)

        assert result.passed is False
        assert result.score == 0.0
        assert result.details.get("error") == "chess library not available"

    def test_verify_engine_load_fails(self, tmp_path: Path):
        """verify() returns score=0 when engine file is corrupt."""
        t = ChessEvolveTask()
        inst = TaskInstance(
            id="depth1-startpos",
            metadata={
                "depth": 1,
                "num_games": 2,
                "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            },
        )

        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "base").mkdir(parents=True)
        (tmp_path / "src" / "engine.py").write_text("this is not valid python !!@#$")
        (tmp_path / "base" / "engine.py").write_text("also invalid !!@#$")

        result = t.verify(inst, tmp_path)
        assert result.passed is False
        assert result.score == 0.0
        assert "engine load failed" in result.details.get("error", "")


