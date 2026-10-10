"""Python language evaluator."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from factory.eval.languages.base import EvalFragment, _run_cmd


def _resolve_python(project_path: Path) -> str:
    from factory.worktree import is_factory_venv
    if is_factory_venv(project_path):
        return str(project_path / ".venv" / "bin" / "python")
    return sys.executable


class PythonEvaluator:
    @property
    def name(self) -> str:
        return "python"

    def _detect_cov_target(self, project_path: Path) -> str:
        src_dirs = [
            c.name for c in sorted(project_path.iterdir())
            if c.is_dir() and (c / "__init__.py").exists()
        ]
        return src_dirs[0] if src_dirs else "."

    def detect(self, project_path: Path) -> bool:
        return (
            (project_path / "pyproject.toml").exists()
            or (project_path / "setup.py").exists()
        )

    def run_tests_with_coverage(
        self,
        project_path: Path,
        timeout: int = 300,
        test_paths: list[str] | None = None,
    ) -> tuple[EvalFragment | None, EvalFragment | None]:
        cov_target = self._detect_cov_target(project_path)
        cmd = [
            _resolve_python(project_path), "-m", "pytest",
            f"--cov={cov_target}", "--cov-report=term",
            "-v", "--tb=no", "-q",
        ]
        if test_paths:
            cmd.extend(test_paths)
        rc, stdout, stderr = _run_cmd(
            cmd,
            project_path,
            timeout=timeout,
        )
        output = stdout + stderr

        # Parse test results
        test_frag: EvalFragment | None = None
        p_match = re.search(r"(\d+)\s+passed", output)
        f_match = re.search(r"(\d+)\s+failed", output)
        p = int(p_match.group(1)) if p_match else 0
        f = int(f_match.group(1)) if f_match else 0
        if p + f > 0:
            total = p + f
            test_frag = EvalFragment(
                passed=p,
                failed=f,
                score=p / total,
                details=f"{project_path.name}: {p} passed, {f} failed",
            )

        # Parse coverage only if tests were collected
        cov_frag: EvalFragment | None = None
        total_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if total_match and test_frag is not None:
            pct = int(total_match.group(1))
            cov_frag = EvalFragment(
                passed=0,
                failed=0,
                score=pct / 100.0,
                coverage_pct=pct,
                details=f"{project_path.name}: {pct}%",
            )

        return test_frag, cov_frag

    def run_tests(
        self,
        project_path: Path,
        timeout: int = 300,
        test_paths: list[str] | None = None,
    ) -> EvalFragment | None:
        """Prefer run_tests_with_coverage() to avoid a redundant pytest invocation."""
        return self.run_tests_with_coverage(project_path, timeout=timeout, test_paths=test_paths)[0]

    def run_lint(self, project_path: Path) -> EvalFragment | None:
        rc, stdout, stderr = _run_cmd(
            [_resolve_python(project_path), "-m", "ruff", "check", "."], project_path
        )
        output = stdout + stderr
        if rc == 0:
            return EvalFragment(passed=1, failed=0, score=1.0, details=f"{project_path.name}: clean")
        err_match = re.search(r"Found\s+(\d+)\s+error", output)
        count = int(err_match.group(1)) if err_match else 1
        return EvalFragment(passed=0, failed=count, score=0.0, details=f"{project_path.name}: {count} errors")

    def run_type_check(self, project_path: Path) -> EvalFragment | None:
        src_dirs = []
        for child in sorted(project_path.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                src_dirs.append(child.name)
        target = src_dirs[0] if src_dirs else "."
        rc, stdout, stderr = _run_cmd(
            [_resolve_python(project_path), "-m", "mypy", target], project_path
        )
        output = stdout + stderr
        if rc == 0:
            return EvalFragment(passed=1, failed=0, score=1.0, details=f"{project_path.name}: clean")
        err_match = re.search(r"Found\s+(\d+)\s+error", output)
        count = int(err_match.group(1)) if err_match else 1
        return EvalFragment(
            passed=0, failed=count, score=0.0,
            details=f"{project_path.name}: {count} errors",
        )

    def run_coverage(self, project_path: Path, timeout: int = 300) -> EvalFragment | None:
        """Prefer run_tests_with_coverage() to avoid a redundant pytest invocation."""
        return self.run_tests_with_coverage(project_path, timeout=timeout)[1]


def register_evaluator() -> PythonEvaluator:
    return PythonEvaluator()
