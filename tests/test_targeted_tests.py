"""Tests for graph-based targeted test selection."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from factory.graph import find_dependent_tests


def _make_graph(
    tmp_path: Path,
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> Path:
    gpath = tmp_path / "graph.json"
    gpath.write_text(json.dumps({
        "nodes": nodes or [],
        "edges": edges or [],
    }))
    return gpath


def _simple_graph(tmp_path: Path) -> Path:
    """A -> B -> test_B (via imports edges), plus unrelated tests to stay under fan-out."""
    nodes = [
        {"id": "mod_a", "source_file": "src/a.py"},
        {"id": "mod_b", "source_file": "src/b.py"},
        {"id": "mod_c", "source_file": "src/c.py"},
        {"id": "test_b", "source_file": "tests/test_b.py"},
        {"id": "test_a", "source_file": "tests/test_a.py"},
        {"id": "test_c1", "source_file": "tests/test_c1.py"},
        {"id": "test_c2", "source_file": "tests/test_c2.py"},
        {"id": "test_c3", "source_file": "tests/test_c3.py"},
        {"id": "test_c4", "source_file": "tests/test_c4.py"},
        {"id": "test_c5", "source_file": "tests/test_c5.py"},
        {"id": "test_c6", "source_file": "tests/test_c6.py"},
        {"id": "test_c7", "source_file": "tests/test_c7.py"},
        {"id": "test_c8", "source_file": "tests/test_c8.py"},
    ]
    edges = [
        {"source": "mod_b", "target": "mod_a", "relation": "imports"},
        {"source": "test_b", "target": "mod_b", "relation": "imports"},
        {"source": "test_a", "target": "mod_a", "relation": "imports"},
        {"source": "test_c1", "target": "mod_c", "relation": "imports"},
        {"source": "test_c2", "target": "mod_c", "relation": "imports"},
        {"source": "test_c3", "target": "mod_c", "relation": "imports"},
        {"source": "test_c4", "target": "mod_c", "relation": "imports"},
        {"source": "test_c5", "target": "mod_c", "relation": "imports"},
        {"source": "test_c6", "target": "mod_c", "relation": "imports"},
        {"source": "test_c7", "target": "mod_c", "relation": "imports"},
        {"source": "test_c8", "target": "mod_c", "relation": "imports"},
    ]
    return _make_graph(tmp_path, nodes, edges)


class TestFindDependentTests:
    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_basic_reverse_bfs(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is not None
        assert "tests/test_a.py" in result
        assert "tests/test_b.py" in result

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_direct_import_only(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        result = find_dependent_tests(tmp_path, ["src/b.py"])
        assert result is not None
        assert "tests/test_b.py" in result
        assert "tests/test_a.py" not in result

    @patch("factory.graph.is_graph_stale", return_value=True)
    def test_returns_none_when_stale(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["src/a.py"]) is None

    @patch("factory.graph.is_graph_stale", return_value=None)
    def test_returns_none_when_staleness_unknown(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["src/a.py"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_conftest(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["tests/conftest.py"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_init(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["src/__init__.py"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_pyproject(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["pyproject.toml"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_pytest_ini(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["pytest.ini"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_workflow_file(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, [".github/workflows/ci.yml"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_non_python_no_node(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["README.md"]) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_none_on_unknown_py_file(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        assert find_dependent_tests(tmp_path, ["src/unknown.py"]) is None

    def test_returns_none_on_empty_changed_files(self) -> None:
        assert find_dependent_tests(Path("/tmp"), []) is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_fan_out_returns_none(self, _stale: MagicMock, tmp_path: Path) -> None:
        """When >80% of test files are dependent, return None."""
        nodes = [
            {"id": "mod_a", "source_file": "src/a.py"},
        ]
        edges = []
        for i in range(10):
            nid = f"test_{i}"
            nodes.append({"id": nid, "source_file": f"tests/test_{i}.py"})
            edges.append({"source": nid, "target": "mod_a", "relation": "imports"})
        _make_graph(tmp_path, nodes, edges)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is None

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_fan_out_under_threshold(self, _stale: MagicMock, tmp_path: Path) -> None:
        """When <80% of test files are dependent, return the set."""
        nodes = [
            {"id": "mod_a", "source_file": "src/a.py"},
            {"id": "mod_b", "source_file": "src/b.py"},
        ]
        edges = []
        for i in range(3):
            nid = f"test_a_{i}"
            nodes.append({"id": nid, "source_file": f"tests/test_a_{i}.py"})
            edges.append({"source": nid, "target": "mod_a", "relation": "imports"})
        for i in range(10):
            nid = f"test_b_{i}"
            nodes.append({"id": nid, "source_file": f"tests/test_b_{i}.py"})
        _make_graph(tmp_path, nodes, edges)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is not None
        assert len(result) == 3

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_transitive_multi_hop(self, _stale: MagicMock, tmp_path: Path) -> None:
        """C imports B imports A; changing A reaches test_C."""
        nodes = [
            {"id": "mod_a", "source_file": "src/a.py"},
            {"id": "mod_b", "source_file": "src/b.py"},
            {"id": "mod_c", "source_file": "src/c.py"},
            {"id": "test_c", "source_file": "tests/test_c.py"},
        ]
        # Add unrelated tests to stay under fan-out threshold
        for i in range(8):
            nodes.append({"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"})
        edges = [
            {"source": "mod_b", "target": "mod_a", "relation": "imports"},
            {"source": "mod_c", "target": "mod_b", "relation": "imports_from"},
            {"source": "test_c", "target": "mod_c", "relation": "imports"},
        ]
        _make_graph(tmp_path, nodes, edges)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is not None
        assert "tests/test_c.py" in result

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_changed_test_file_included(self, _stale: MagicMock, tmp_path: Path) -> None:
        _simple_graph(tmp_path)
        result = find_dependent_tests(tmp_path, ["tests/test_a.py"])
        assert result is not None
        assert "tests/test_a.py" in result

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_returns_empty_set_when_no_tests_depend(
        self, _stale: MagicMock, tmp_path: Path,
    ) -> None:
        nodes = [
            {"id": "mod_a", "source_file": "src/a.py"},
            {"id": "mod_b", "source_file": "src/b.py"},
            {"id": "test_x", "source_file": "tests/test_x.py"},
        ]
        edges = [
            {"source": "test_x", "target": "mod_b", "relation": "imports"},
        ]
        _make_graph(tmp_path, nodes, edges)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is not None
        assert len(result) == 0

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_non_import_edges_ignored(self, _stale: MagicMock, tmp_path: Path) -> None:
        nodes = [
            {"id": "mod_a", "source_file": "src/a.py"},
            {"id": "test_a", "source_file": "tests/test_a.py"},
            # Unrelated tests to stay under fan-out threshold
            *[{"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"} for i in range(10)],
        ]
        edges = [
            {"source": "test_a", "target": "mod_a", "relation": "calls"},
        ]
        _make_graph(tmp_path, nodes, edges)
        result = find_dependent_tests(tmp_path, ["src/a.py"])
        assert result is not None
        assert len(result) == 0


class TestGrepInlineImportScan:
    """Tests for the grep-based inline import scan that catches inline imports."""

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_grep_finds_test_with_inline_import_not_found_by_bfs(
        self, _stale: MagicMock, tmp_path: Path,
    ) -> None:
        """A test file with an inline import (not in graph edges) is found by grep."""
        nodes = [
            {"id": "mod_runner", "source_file": "factory/runners/claude.py"},
            {"id": "test_runner", "source_file": "tests/test_claude.py"},
            *[{"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"} for i in range(10)],
        ]
        edges: list[dict] = []
        _make_graph(tmp_path, nodes, edges)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_claude.py").write_text(
            "def test_it():\n    from factory.runners.claude import run\n    run()\n"
        )
        for i in range(10):
            (tests_dir / f"test_u{i}.py").write_text("def test_pass(): pass\n")
        result = find_dependent_tests(tmp_path, ["factory/runners/claude.py"])
        assert result is not None
        assert "tests/test_claude.py" in result

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_grep_does_not_duplicate_bfs_results(
        self, _stale: MagicMock, tmp_path: Path,
    ) -> None:
        """A test already found by BFS is not duplicated by grep."""
        nodes = [
            {"id": "mod_a", "source_file": "factory/a.py"},
            {"id": "test_a", "source_file": "tests/test_a.py"},
            *[{"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"} for i in range(10)],
        ]
        edges = [
            {"source": "test_a", "target": "mod_a", "relation": "imports"},
        ]
        _make_graph(tmp_path, nodes, edges)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").write_text("import factory.a\n")
        for i in range(10):
            (tests_dir / f"test_u{i}.py").write_text("def test_pass(): pass\n")
        result = find_dependent_tests(tmp_path, ["factory/a.py"])
        assert result is not None
        assert "tests/test_a.py" in result
        assert len([t for t in result if t == "tests/test_a.py"]) == 1

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_grep_failure_silently_skipped(
        self, _stale: MagicMock, tmp_path: Path,
    ) -> None:
        """If grep fails (e.g. subprocess error), the function still returns a set, not None."""
        nodes = [
            {"id": "mod_a", "source_file": "factory/a.py"},
            *[{"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"} for i in range(10)],
        ]
        edges: list[dict] = []
        _make_graph(tmp_path, nodes, edges)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        for i in range(10):
            (tests_dir / f"test_u{i}.py").write_text("def test_pass(): pass\n")
        with patch("factory.graph.subprocess.run", side_effect=OSError("grep not found")):
            result = find_dependent_tests(tmp_path, ["factory/a.py"])
        assert result is not None
        assert isinstance(result, set)

    @patch("factory.graph.is_graph_stale", return_value=False)
    def test_grep_skips_non_python_changed_files(
        self, _stale: MagicMock, tmp_path: Path,
    ) -> None:
        """Non-.py changed files are not processed by the grep scan."""
        nodes = [
            {"id": "mod_readme", "source_file": "factory/README.md"},
            {"id": "mod_a", "source_file": "factory/a.py"},
            *[{"id": f"test_u{i}", "source_file": f"tests/test_u{i}.py"} for i in range(10)],
        ]
        edges: list[dict] = []
        _make_graph(tmp_path, nodes, edges)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        for i in range(10):
            (tests_dir / f"test_u{i}.py").write_text("def test_pass(): pass\n")
        result = find_dependent_tests(tmp_path, ["factory/a.py", "factory/README.md"])
        assert result is not None


class TestPythonEvaluatorTestPaths:
    @patch("factory.eval.languages.python.PythonEvaluator._detect_cov_target", return_value="src")
    def test_test_paths_appended_to_cmd(self, _cov: MagicMock) -> None:
        from factory.eval.languages.python import PythonEvaluator

        ev = PythonEvaluator()
        with patch("factory.eval.languages.python._run_cmd") as mock_run:
            mock_run.return_value = (0, "1 passed", "TOTAL 100 10 90%")
            ev.run_tests_with_coverage(
                Path("/fake"),
                timeout=60,
                test_paths=["tests/test_a.py", "tests/test_b.py"],
            )
            cmd = mock_run.call_args[0][0]
            assert "tests/test_a.py" in cmd
            assert "tests/test_b.py" in cmd
            assert cmd.index("tests/test_a.py") > cmd.index("-q")

    @patch("factory.eval.languages.python.PythonEvaluator._detect_cov_target", return_value="src")
    def test_no_test_paths_unchanged(self, _cov: MagicMock) -> None:
        from factory.eval.languages.python import PythonEvaluator

        ev = PythonEvaluator()
        with patch("factory.eval.languages.python._run_cmd") as mock_run:
            mock_run.return_value = (0, "1 passed", "TOTAL 100 10 90%")
            ev.run_tests_with_coverage(Path("/fake"), timeout=60)
            cmd = mock_run.call_args[0][0]
            assert not any(c.startswith("tests/") for c in cmd)

    @patch("factory.eval.languages.python.PythonEvaluator._detect_cov_target", return_value="src")
    def test_run_tests_forwards_test_paths(self, _cov: MagicMock) -> None:
        from factory.eval.languages.python import PythonEvaluator

        ev = PythonEvaluator()
        with patch("factory.eval.languages.python._run_cmd") as mock_run:
            mock_run.return_value = (0, "1 passed", "TOTAL 100 10 90%")
            ev.run_tests(Path("/fake"), timeout=60, test_paths=["tests/test_x.py"])
            cmd = mock_run.call_args[0][0]
            assert "tests/test_x.py" in cmd


# ── CLI integration: _compute_targeted_test_paths ──────────────────


def _make_merge_base_result(returncode: int = 0, stdout: str = "abc123\n") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["git", "merge-base", "HEAD", "main"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _make_diff_result(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["git", "diff", "--name-only", "abc123..HEAD"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


class TestComputeTargetedTestPaths:
    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.graph.find_dependent_tests", return_value={"tests/test_a.py", "tests/test_b.py"})
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_success_path(
        self, mock_run: MagicMock, mock_find: MagicMock, _branch: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(stdout="abc123\n"),
            _make_diff_result(stdout="src/foo.py\nsrc/bar.py\n"),
        ]
        result = _compute_targeted_test_paths(tmp_path)
        assert result == ["tests/test_a.py", "tests/test_b.py"]
        mock_find.assert_called_once_with(tmp_path, ["src/foo.py", "src/bar.py"])

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_merge_base_fails(self, mock_run: MagicMock, _branch: MagicMock, tmp_path: Path) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.return_value = _make_merge_base_result(returncode=1)
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_diff_fails(self, mock_run: MagicMock, _branch: MagicMock, tmp_path: Path) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(),
            _make_diff_result(returncode=1),
        ]
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_merge_base_timeout(self, mock_run: MagicMock, _branch: MagicMock, tmp_path: Path) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git"], timeout=10)
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_diff_timeout(self, mock_run: MagicMock, _branch: MagicMock, tmp_path: Path) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(),
            subprocess.TimeoutExpired(cmd=["git"], timeout=10),
        ]
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_empty_changed_files(self, mock_run: MagicMock, _branch: MagicMock, tmp_path: Path) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(),
            _make_diff_result(stdout="\n"),
        ]
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.graph.find_dependent_tests", return_value=None)
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_find_dependent_tests_returns_none(
        self, mock_run: MagicMock, mock_find: MagicMock, _branch: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(),
            _make_diff_result(stdout="src/foo.py\n"),
        ]
        assert _compute_targeted_test_paths(tmp_path) is None

    @patch("factory.cli.eval_cmds._read_target_branch", return_value="main")
    @patch("factory.graph.find_dependent_tests", return_value={"tests/test_x.py"})
    @patch("factory.cli.eval_cmds.subprocess.run")
    def test_result_is_sorted(
        self, mock_run: MagicMock, _find: MagicMock, _branch: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import _compute_targeted_test_paths

        mock_run.side_effect = [
            _make_merge_base_result(),
            _make_diff_result(stdout="src/z.py\n"),
        ]
        result = _compute_targeted_test_paths(tmp_path)
        assert result == sorted(result)


# ── Hygiene: _collect_test_and_coverage inspect.signature guard ────


class TestCollectTestAndCoverageSignatureGuard:
    def test_passes_test_paths_when_evaluator_supports_it(self, tmp_path: Path) -> None:
        from factory.eval.hygiene import _collect_test_and_coverage
        from factory.eval.languages.base import EvalFragment

        mock_evaluator = MagicMock()
        captured_kwargs: dict = {}

        def fake_run_tests_with_coverage(
            project_path: Path, timeout: int = 300, test_paths: list[str] | None = None,
        ) -> tuple:
            captured_kwargs["test_paths"] = test_paths
            return (
                EvalFragment(passed=10, failed=0, score=1.0, details="ok"),
                EvalFragment(passed=10, failed=0, score=0.9, details="ok", coverage_pct=90.0),
            )

        mock_evaluator.run_tests_with_coverage = fake_run_tests_with_coverage

        with patch("factory.eval.hygiene._find_sub_projects", return_value=[tmp_path]), \
             patch("factory.eval.hygiene.detect_languages", return_value=[mock_evaluator]):
            test_result, cov_result = _collect_test_and_coverage(
                tmp_path, test_paths=["tests/test_a.py"],
            )
        assert test_result["score"] == 1.0
        assert cov_result["score"] == 0.9
        assert captured_kwargs["test_paths"] == ["tests/test_a.py"]

    def test_omits_test_paths_when_evaluator_lacks_param(self, tmp_path: Path) -> None:
        from factory.eval.hygiene import _collect_test_and_coverage
        from factory.eval.languages.base import EvalFragment

        mock_evaluator = MagicMock()
        captured_kwargs: dict = {}

        def fake_run_tests_with_coverage(project_path: Path, timeout: int = 300) -> tuple:
            captured_kwargs["timeout"] = timeout
            return (
                EvalFragment(passed=8, failed=2, score=0.8, details="ok"),
                EvalFragment(passed=7, failed=3, score=0.7, details="ok", coverage_pct=70.0),
            )

        mock_evaluator.run_tests_with_coverage = fake_run_tests_with_coverage

        with patch("factory.eval.hygiene._find_sub_projects", return_value=[tmp_path]), \
             patch("factory.eval.hygiene.detect_languages", return_value=[mock_evaluator]):
            test_result, cov_result = _collect_test_and_coverage(
                tmp_path, test_paths=["tests/test_a.py"],
            )
        assert test_result["score"] == 0.8
        assert "test_paths" not in captured_kwargs

    def test_no_test_paths_skips_signature_check(self, tmp_path: Path) -> None:
        from factory.eval.hygiene import _collect_test_and_coverage
        from factory.eval.languages.base import EvalFragment

        mock_evaluator = MagicMock()

        def fake_run_tests_with_coverage(project_path: Path, timeout: int = 300) -> tuple:
            return (
                EvalFragment(passed=10, failed=0, score=1.0, details="ok"),
                None,
            )

        mock_evaluator.run_tests_with_coverage = fake_run_tests_with_coverage

        with patch("factory.eval.hygiene._find_sub_projects", return_value=[tmp_path]), \
             patch("factory.eval.hygiene.detect_languages", return_value=[mock_evaluator]):
            test_result, cov_result = _collect_test_and_coverage(tmp_path)
        assert test_result["score"] == 1.0
        assert cov_result["score"] == 0.5  # neutral — no coverage fragment


# ── cmd_eval --targeted flag wiring ────────────────────────────────


class TestCmdEvalTargetedFlag:
    @patch("factory.cli.eval_cmds._emit_cli_event")
    @patch("factory.cli.eval_cmds._run")
    @patch("factory.cli.eval_cmds._compute_targeted_test_paths", return_value=["tests/test_a.py"])
    def test_targeted_flag_passes_test_paths_to_run_eval(
        self, mock_compute: MagicMock, mock_run: MagicMock, _event: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import cmd_eval

        mock_config = MagicMock()
        mock_config.eval_command = "pytest"
        mock_config.project_eval = None
        mock_config.eval_weights = None
        mock_config.eval_threshold = 0.7
        mock_config.test_timeout = 300

        mock_score = MagicMock()
        mock_score.total = 0.9
        mock_score.passed = True
        mock_score.results = []
        mock_score.model_dump.return_value = {"total": 0.9}

        mock_run.side_effect = [mock_config, mock_score]

        args = MagicMock()
        args.path = str(tmp_path)
        args.targeted = True
        args.skip_project_eval = False

        result = cmd_eval(args)
        assert result == 0
        mock_compute.assert_called_once_with(tmp_path)

    @patch("factory.cli.eval_cmds._emit_cli_event")
    @patch("factory.cli.eval_cmds._run")
    @patch("factory.cli.eval_cmds._compute_targeted_test_paths", return_value=None)
    def test_targeted_fallback_to_full(
        self, mock_compute: MagicMock, mock_run: MagicMock, _event: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import cmd_eval

        mock_config = MagicMock()
        mock_config.eval_command = "pytest"
        mock_config.project_eval = None
        mock_config.eval_weights = None
        mock_config.eval_threshold = 0.7
        mock_config.test_timeout = 300

        mock_score = MagicMock()
        mock_score.total = 0.85
        mock_score.passed = True
        mock_score.results = []
        mock_score.model_dump.return_value = {"total": 0.85}

        mock_run.side_effect = [mock_config, mock_score]

        args = MagicMock()
        args.path = str(tmp_path)
        args.targeted = True
        args.skip_project_eval = False

        result = cmd_eval(args)
        assert result == 0
        mock_compute.assert_called_once()

    @patch("factory.cli.eval_cmds._emit_cli_event")
    @patch("factory.cli.eval_cmds._run")
    def test_no_targeted_flag_skips_compute(
        self, mock_run: MagicMock, _event: MagicMock, tmp_path: Path,
    ) -> None:
        from factory.cli.eval_cmds import cmd_eval

        mock_config = MagicMock()
        mock_config.eval_command = "pytest"
        mock_config.project_eval = None
        mock_config.eval_weights = None
        mock_config.eval_threshold = 0.7
        mock_config.test_timeout = 300

        mock_score = MagicMock()
        mock_score.total = 0.9
        mock_score.passed = True
        mock_score.results = []
        mock_score.model_dump.return_value = {"total": 0.9}

        mock_run.side_effect = [mock_config, mock_score]

        args = MagicMock()
        args.path = str(tmp_path)
        args.targeted = False
        args.skip_project_eval = False

        with patch("factory.cli.eval_cmds._compute_targeted_test_paths") as mock_compute:
            result = cmd_eval(args)
            mock_compute.assert_not_called()
        assert result == 0
