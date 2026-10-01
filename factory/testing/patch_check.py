"""AST-based detection of over-mocking in test files.

Analyses ``@patch`` decorator targets and ``monkeypatch.setattr``/``setenv``/
``delenv`` call targets.  Classifies each target as:

* **external** — patches an external I/O boundary (good)
* **allowlisted** — patches a known-legitimate internal target (ok)
* **internal** — patches the project's own modules (over-mocking warning)

Exit codes:
    0  pass — only external/allowlisted patches found
    1  advisory warning — internal patches detected (not a hard failure)
    2  error — file could not be parsed or analysed
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# Modules that are legitimate to patch even though they may appear as
# internal imports.  These are standard-library singletons, time sources,
# or environment accessors that tests routinely need to control.
ALLOWLISTED_MODULES: frozenset[str] = frozenset(
    {
        "datetime",
        "random",
        "time",
        "os.environ",
        "os.cpu_count",
        "uuid",
        "tempfile",
    }
)


@dataclass
class PatchTarget:
    """A single patch target extracted from a test file."""

    target: str
    line: int
    kind: str  # "decorator" | "monkeypatch_setattr" | "monkeypatch_setenv" | "monkeypatch_delenv"
    classification: str = ""  # "external" | "internal" | "allowlisted"


@dataclass
class CheckResult:
    """Result of analysing a single test file."""

    path: str
    targets: list[PatchTarget] = field(default_factory=list)
    error: str | None = None

    @property
    def exit_code(self) -> int:
        if self.error:
            return 2
        if any(t.classification == "internal" for t in self.targets):
            return 1
        return 0


def _classify(target: str, package_names: frozenset[str]) -> str:
    """Classify a dotted target string."""
    # Check allowlist first — these are always ok
    for allowed in ALLOWLISTED_MODULES:
        if target == allowed or target.startswith(allowed + "."):
            return "allowlisted"

    # Internal = starts with any of the project's package names
    for name in package_names:
        if target.startswith(name + ".") or target == name:
            return "internal"

    return "external"


class PatchVisitor(ast.NodeVisitor):
    """Walks an AST and extracts patch targets from decorators and calls."""

    def __init__(self, package_names: frozenset[str]) -> None:
        self.package_names = package_names
        self.targets: list[PatchTarget] = []

    # ── decorator-based patches ──────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_decorators(node)
        self.generic_visit(node)

    def _check_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            target_str = self._extract_patch_target(decorator)
            if target_str is not None:
                self.targets.append(
                    PatchTarget(
                        target=target_str,
                        line=decorator.lineno,
                        kind="decorator",
                        classification=_classify(target_str, self.package_names),
                    )
                )

    def _extract_patch_target(self, node: ast.expr) -> str | None:
        """Extract the target string from a @patch(...) or @patch.object(...) decorator."""
        if not isinstance(node, ast.Call):
            return None

        func = node.func

        # @patch("some.module.thing")
        if isinstance(func, ast.Attribute) and func.attr == "patch":
            # Could be mock.patch or unittest.mock.patch
            if node.args:
                return self._const_str(node.args[0])
            return None

        # @patch("target") where patch is imported directly
        if isinstance(func, ast.Name) and func.id == "patch":
            if node.args:
                return self._const_str(node.args[0])
            return None

        # @patch.object(module, "attr") — target is module.attr
        if isinstance(func, ast.Attribute) and func.attr == "object":
            if isinstance(func.value, ast.Attribute) and func.value.attr == "patch":
                return None  # patch.object is harder to resolve statically — skip
            if isinstance(func.value, ast.Name) and func.value.id == "patch":
                return None  # same — skip patch.object

        return None

    @staticmethod
    def _const_str(node: ast.expr) -> str | None:
        """Extract a string constant from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    # ── monkeypatch call detection ───────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "setattr" and self._is_monkeypatch(node.func.value):
                target_str = self._monkeypatch_setattr_target(node)
                if target_str is not None:
                    self.targets.append(
                        PatchTarget(
                            target=target_str,
                            line=node.lineno,
                            kind="monkeypatch_setattr",
                            classification=_classify(target_str, self.package_names),
                        )
                    )
            elif attr in ("setenv", "delenv") and self._is_monkeypatch(node.func.value):
                # setenv/delenv are always os.environ manipulation — allowlisted
                env_key = ""
                if node.args:
                    key = self._const_str(node.args[0])
                    env_key = f"os.environ[{key}]" if key else "os.environ"
                self.targets.append(
                    PatchTarget(
                        target=env_key or "os.environ",
                        line=node.lineno,
                        kind=f"monkeypatch_{attr}",
                        classification="allowlisted",
                    )
                )
        self.generic_visit(node)

    @staticmethod
    def _is_monkeypatch(node: ast.expr) -> bool:
        """Check if a node likely refers to a monkeypatch fixture."""
        if isinstance(node, ast.Name):
            return node.id in ("monkeypatch", "mp")
        return False

    def _monkeypatch_setattr_target(self, node: ast.Call) -> str | None:
        """Extract the target from monkeypatch.setattr(target, name, value)
        or monkeypatch.setattr("dotted.path", value)."""
        if not node.args:
            return None

        first = node.args[0]

        # monkeypatch.setattr("dotted.path", value) — string form
        s = self._const_str(first)
        if s is not None:
            return s

        # monkeypatch.setattr(module, "attr", value) — object form
        if len(node.args) >= 2:
            attr_name = self._const_str(node.args[1])
            module_name = self._resolve_name(first)
            if module_name and attr_name:
                return f"{module_name}.{attr_name}"

        return None

    @staticmethod
    def _resolve_name(node: ast.expr) -> str | None:
        """Resolve a simple Name or Attribute chain to a dotted string."""
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None


_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "node_modules",
        ".git",
        "__pycache__",
        ".factory",
        ".tox",
        "build",
        "dist",
        ".eggs",
    }
)


def _detect_package_names(start_dir: Path | None = None) -> frozenset[str]:
    """Detect importable package names by finding directories with __init__.py.

    Also includes the pyproject.toml project name (dash→underscore) as a fallback.
    Returns a frozenset of all discovered package names.
    """
    search_dir = start_dir or Path.cwd()

    # Walk up to find pyproject.toml (project root)
    project_root: Path | None = None
    current = search_dir.resolve()
    for _ in range(20):  # safety limit
        candidate = current / "pyproject.toml"
        if candidate.is_file():
            project_root = current
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    names: set[str] = set()

    if project_root is not None:
        # Scan project root for directories containing __init__.py
        for child in project_root.iterdir():
            if not child.is_dir():
                continue
            dir_name = child.name
            # Skip common non-package directories and egg-info dirs
            if dir_name in _EXCLUDED_DIRS or dir_name.endswith(".egg-info"):
                continue
            if (child / "__init__.py").is_file():
                names.add(dir_name)

        # Also include pyproject.toml name as fallback
        toml_name = _parse_project_name(project_root / "pyproject.toml")
        if toml_name:
            names.add(toml_name)

    return frozenset(names)


def _parse_project_name(toml_path: Path) -> str:
    """Parse project name from pyproject.toml, converting dashes to underscores."""
    import tomllib

    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        name = data.get("project", {}).get("name", "")
        # PEP 503: normalize — dashes become underscores for import names
        return name.replace("-", "_")
    except Exception:
        return ""


def check_file(
    path: str | Path,
    project_name: str | frozenset[str] | None = None,
) -> CheckResult:
    """Analyse a test file for over-mocking.

    Parameters
    ----------
    path:
        Path to the Python test file.
    project_name:
        Importable package name(s).  Accepts a single string, a frozenset of
        strings, or ``None`` (auto-detected from the project root).

    Returns
    -------
    CheckResult with classified targets and an exit code.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return CheckResult(path=str(file_path), error=f"File not found: {file_path}")

    # Normalise to frozenset[str]
    package_names: frozenset[str]
    if project_name is None:
        package_names = _detect_package_names(file_path.parent)
    elif isinstance(project_name, str):
        package_names = frozenset({project_name})
    else:
        package_names = project_name

    if not package_names:
        return CheckResult(
            path=str(file_path),
            error="Could not detect package names from project root",
        )

    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return CheckResult(path=str(file_path), error=f"Cannot read file: {exc}")

    if not source.strip():
        # Empty file is fine — no patches to check
        return CheckResult(path=str(file_path))

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return CheckResult(path=str(file_path), error=f"Syntax error: {exc}")

    visitor = PatchVisitor(package_names)
    visitor.visit(tree)

    return CheckResult(path=str(file_path), targets=visitor.targets)


def _format_result(result: CheckResult) -> str:
    """Format a CheckResult for human-readable output."""
    lines: list[str] = []
    lines.append(f"File: {result.path}")

    if result.error:
        lines.append(f"ERROR: {result.error}")
        return "\n".join(lines)

    if not result.targets:
        lines.append("No patch targets found.")
        return "\n".join(lines)

    # Group by classification
    for classification in ("internal", "external", "allowlisted"):
        group = [t for t in result.targets if t.classification == classification]
        if not group:
            continue
        label = {
            "internal": "⚠ INTERNAL (over-mocking advisory)",
            "external": "✓ External I/O (good)",
            "allowlisted": "✓ Allowlisted (ok)",
        }[classification]
        lines.append(f"\n{label}:")
        for t in group:
            lines.append(f"  line {t.line}: {t.target} ({t.kind})")

    exit_code = result.exit_code
    status = {0: "PASS", 1: "ADVISORY WARNING", 2: "ERROR"}[exit_code]
    lines.append(f"\nResult: {status} (exit code {exit_code})")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for patch-target checking."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="factory check-patch-targets",
        description="Check test files for over-mocking (internal patch targets)",
    )
    parser.add_argument("test_file", help="Path to the test file to analyse")
    parser.add_argument(
        "--project-name",
        default=None,
        help="Project package name (auto-detected from project root if omitted)",
    )
    args = parser.parse_args(argv)

    result = check_file(args.test_file, project_name=args.project_name)
    print(_format_result(result))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
