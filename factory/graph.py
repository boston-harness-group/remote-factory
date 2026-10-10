"""Graphify integration — extract, update, and query code knowledge graphs."""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import structlog

log = structlog.get_logger()

GRAPH_FILE = "graph.json"
GRAPHIFY_OUT_DIR = ".factory/graphify-out"


def _graph_path(project_path: Path) -> Path:
    return project_path / GRAPH_FILE


def is_graphify_installed() -> bool:
    """Check whether the graphify CLI is available on PATH."""
    return shutil.which("graphify") is not None


def is_graph_available(project_path: Path) -> bool:
    """Check whether a graph.json exists for the given project."""
    return _graph_path(project_path).is_file()


def graph_stats(project_path: Path) -> dict[str, int] | None:
    """Return node/edge counts from graph.json, or None if unavailable."""
    gpath = _graph_path(project_path)
    if not gpath.is_file():
        return None
    try:
        data = json.loads(gpath.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        edges = data.get("edges", data.get("links", []))
        return {"nodes": len(nodes), "edges": len(edges)}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("graph.stats.failed", error=str(exc))
        return None


def is_graph_stale(project_path: Path) -> bool | None:
    """Compare graph.json mtime against latest git commit timestamp.

    Returns True if stale, False if fresh, None if comparison not possible.
    """
    gpath = _graph_path(project_path)
    if not gpath.is_file():
        return None

    try:
        graph_mtime = gpath.stat().st_mtime
    except OSError:
        return None

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        latest_commit_ts = float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None

    return graph_mtime < latest_commit_ts


def _run_graphify(project_path: Path, extra_args: list[str] | None = None) -> Path | None:
    """Run graphify extract and copy graph.json to the project root.

    Graphify writes to .factory/graphify-out/ (cache, reports, etc.).
    The graph.json is then copied to the project root for easy access.
    Returns path to root graph.json on success, None on failure.
    """
    if not is_graphify_installed():
        log.warning("graph.extract.skipped", reason="graphify not installed")
        return None

    factory_dir = project_path / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "graphify",
        "extract",
        str(project_path),
        "--code-only",
        "--out",
        str(factory_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.error("graph.extract.failed", error=str(exc))
        return None

    if result.returncode != 0:
        log.error(
            "graph.extract.failed",
            returncode=result.returncode,
            stderr=result.stderr[:500],
        )
        return None

    graphify_out = project_path / GRAPHIFY_OUT_DIR / GRAPH_FILE
    if not graphify_out.is_file():
        log.error("graph.extract.no_output", expected=str(graphify_out))
        return None

    gpath = _graph_path(project_path)
    shutil.copy2(graphify_out, gpath)

    stats = graph_stats(project_path)
    log.info("graph.extract.complete", output=str(gpath), **(stats or {}))
    return gpath


def extract_graph(project_path: Path) -> Path | None:
    """Run graphify extract on the project directory.

    Returns path to root graph.json on success, None on failure.
    """
    return _run_graphify(project_path)


def update_graph(project_path: Path) -> Path | None:
    """Run graphify extract with --update for incremental refresh.

    Returns path to root graph.json on success, None on failure.
    """
    return _run_graphify(project_path, extra_args=["--update"])


_FULL_SUITE_TRIGGERS = [
    "**/conftest.py",
    "**/__init__.py",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    ".github/workflows/**",
]

_IMPORT_EDGE_TYPES = frozenset({"imports", "imports_from"})

_FAN_OUT_THRESHOLD = 0.80


def find_dependent_tests(
    project_path: Path,
    changed_files: list[str],
) -> set[str] | None:
    """Reverse-import BFS over graph.json to find tests affected by changed files.

    Returns None when selection cannot be trusted — callers treat None as
    "fall back to full suite".
    """
    if not changed_files:
        return None

    staleness = is_graph_stale(project_path)
    if staleness is not False:
        log.info("targeted.skip", reason="graph_stale_or_unavailable", staleness=staleness)
        return None

    for cf in changed_files:
        for pattern in _FULL_SUITE_TRIGGERS:
            if fnmatch.fnmatch(cf, pattern):
                log.info("targeted.skip", reason="full_suite_trigger", file=cf, pattern=pattern)
                return None

    gpath = _graph_path(project_path)
    try:
        data = json.loads(gpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("targeted.skip", reason="graph_read_error", error=str(exc))
        return None

    nodes = data.get("nodes", [])
    edges = data.get("edges", data.get("links", []))

    file_to_node_ids: dict[str, list[str]] = {}
    for node in nodes:
        sf = node.get("source_file", "")
        nid = node.get("id", "")
        if sf and nid:
            file_to_node_ids.setdefault(sf, []).append(nid)

    all_test_files = {sf for sf in file_to_node_ids if sf.startswith("tests/") and sf.endswith(".py")}

    G: nx.DiGraph[str] = nx.DiGraph()
    for node in nodes:
        nid = node.get("id", "")
        if nid:
            G.add_node(nid, source_file=node.get("source_file", ""))
    for edge in edges:
        if edge.get("relation") in _IMPORT_EDGE_TYPES:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src and tgt:
                G.add_edge(src, tgt)

    seed_node_ids: set[str] = set()
    for cf in changed_files:
        if not cf.endswith(".py"):
            if cf not in file_to_node_ids:
                log.info("targeted.skip", reason="non_python_no_node", file=cf)
                return None
        nids = file_to_node_ids.get(cf, [])
        if not nids:
            log.info("targeted.skip", reason="file_not_in_graph", file=cf)
            return None
        seed_node_ids.update(nids)

    # Reverse BFS: find all nodes that transitively import the changed nodes
    reverse_G = G.reverse(copy=False)
    reached: set[str] = set()
    for seed in seed_node_ids:
        if seed in reverse_G:
            reached.update(nx.descendants(reverse_G, seed))
    reached.update(seed_node_ids)

    dependent_test_files: set[str] = set()
    for nid in reached:
        sf = G.nodes[nid].get("source_file", "") if nid in G else ""
        if sf in all_test_files:
            dependent_test_files.add(sf)

    # Also include any changed files that are themselves test files
    for cf in changed_files:
        if cf in all_test_files:
            dependent_test_files.add(cf)

    # Grep-based inline import scan: catch tests that import changed modules
    # inside function bodies, which graphify doesn't capture in its graph
    tests_dir = project_path / "tests"
    if tests_dir.is_dir():
        for cf in changed_files:
            if not cf.endswith(".py"):
                continue
            parts = Path(cf).parts
            if len(parts) < 2:
                continue
            module_path = cf.replace("/", ".").removesuffix(".py")
            grep_pattern = module_path.replace(".", r"\.")
            try:
                grep_result = subprocess.run(
                    ["grep", "-rl", grep_pattern, str(tests_dir), "--include=*.py"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if grep_result.returncode == 0 and grep_result.stdout.strip():
                    for match in grep_result.stdout.strip().splitlines():
                        try:
                            rel = str(Path(match).relative_to(project_path))
                        except ValueError:
                            continue
                        if rel in all_test_files and rel not in dependent_test_files:
                            dependent_test_files.add(rel)
                            log.info(
                                "targeted.grep_inline_import",
                                changed_file=cf,
                                test_file=rel,
                            )
            except (subprocess.TimeoutExpired, OSError):
                pass

    if not dependent_test_files:
        return dependent_test_files

    if all_test_files and len(dependent_test_files) / len(all_test_files) > _FAN_OUT_THRESHOLD:
        log.info(
            "targeted.skip",
            reason="fan_out_exceeded",
            dependent=len(dependent_test_files),
            total=len(all_test_files),
        )
        return None

    log.info(
        "targeted.selected",
        dependent_tests=len(dependent_test_files),
        total_tests=len(all_test_files),
    )
    return dependent_test_files
