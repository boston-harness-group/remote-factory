"""Aggregator / reset CLI for generated guard stages (INT-002, DEC-008).

Invoked by the two commands :mod:`factory.workflow.guardrails` generates::

    python -m factory.workflow.guardrails_aggregate reset project_path agent_segment json_payload
    python -m factory.workflow.guardrails_aggregate check project_path agent_segment slug [slug ...]

``reset`` clears stale verdicts, persists the slug→rule registry
(``<guard_dir>/rules.json``) from its argv payload and writes a fresh iteration
token. ``check`` reconciles the per-guard verdict JSONs against that token and
emits the fn-gate protocol on **stdout line 1**:

    pass: <N> guard(s) clean
    reloop: <slug>: <fix>[ | <slug>: <fix>]
    fail: <reason>

Exit-code contract (R-008): **exit 0 for every verdict case**, including
``fail:`` — the executor turns a non-zero gate exit into a halt with a different,
unlabelled reason and would make the reloop branch unreachable. Non-zero (2) is
reserved for usage / argument-validation errors, which fail closed at the FnNode
or gate. Diagnostics go to stderr so nothing can precede the verdict on line 1.

The module is standalone: stdlib plus the :mod:`factory.workflow.guardrails`
constants only (no executor, no package, no network), so it runs from a bare
subprocess whose ``cwd`` is the project directory.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from factory.workflow.guardrails import (
    GUARD_SLUG_RE,
    MAX_RULES_PAYLOAD_BYTES,
    guard_dir,
)

_USAGE = (
    "usage: python -m factory.workflow.guardrails_aggregate reset "
    "project_path agent_segment json_payload\n"
    "       python -m factory.workflow.guardrails_aggregate check "
    "project_path agent_segment slug [slug ...]"
)

_VERDICT_KINDS = frozenset({"missing", "malformed", "stale"})


def _usage_error(detail: str) -> SystemExit:
    """Build the ``SystemExit(2)`` for an argument/validation error (fail-closed)."""
    print(_USAGE, file=sys.stderr)
    print(f"error: {detail}", file=sys.stderr)
    return SystemExit(2)


def _stage_dir(project_path: str, segment: str) -> Path:
    if not GUARD_SLUG_RE.fullmatch(segment):
        raise _usage_error(
            f"agent_segment must match {GUARD_SLUG_RE.pattern}, got {segment!r}"
        )
    return Path(project_path) / guard_dir(segment)


def _canonical_json(payload: dict[str, str]) -> str:
    """The one canonical serialization of the rule registry (§4.6/§6.5)."""
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a sibling temp file + ``os.replace`` (crash-atomic).

    The stage has a single serial writer (``reset`` runs before the checker fork), so
    this is not a concurrency guard: it keeps a crash mid-write from leaving a torn
    ``rules.json``/``token`` for the next run to classify.
    """
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _one_line(text: str) -> str:
    """Collapse every run of whitespace to one space (the reloop fix is single-line)."""
    return " ".join(text.split())


def _emit(line: str) -> int:
    """Print the verdict on stdout line 1 and return the always-zero exit code."""
    print(line)
    return 0


def _reset(project_path: str, segment: str, raw_payload: str) -> int:
    stage = _stage_dir(project_path, segment)
    try:
        parsed: Any = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise _usage_error(f"json_payload is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise _usage_error("json_payload must be a non-empty JSON object of str -> str")
    payload: dict[str, str] = {}
    for slug, rule in parsed.items():
        if not GUARD_SLUG_RE.fullmatch(slug):
            raise _usage_error(
                f"json_payload slug {slug!r} must match {GUARD_SLUG_RE.pattern}"
            )
        if not isinstance(rule, str):
            raise _usage_error(f"json_payload value for {slug!r} must be a string")
        payload[slug] = rule
    canonical = _canonical_json(payload)
    payload_bytes = len(canonical.encode("ascii"))
    if payload_bytes > MAX_RULES_PAYLOAD_BYTES:
        raise _usage_error(
            f"json_payload is {payload_bytes} bytes; max {MAX_RULES_PAYLOAD_BYTES}"
        )
    stage.mkdir(parents=True, exist_ok=True)
    _write_atomic(stage / "rules.json", canonical + "\n")
    for slug in payload:
        (stage / f"{slug}.json").unlink(missing_ok=True)
    (stage / "violations.md").unlink(missing_ok=True)
    _write_atomic(stage / "token", uuid.uuid4().hex + "\n")
    print(f"guard stage reset: {stage} ({len(payload)} rule(s))")
    return 0


def _classify_verdict(path: Path, slug: str, token: str) -> tuple[str, str, str]:
    """Classify one verdict file per §4.6 (``missing``/``malformed``/``stale``/…)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "missing", "", ""
    except OSError:
        return "malformed", "", ""
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        return "malformed", "", ""
    if not isinstance(verdict, dict):
        return "malformed", "", ""
    verdict_token = verdict.get("token")
    candidate = verdict_token.strip() if isinstance(verdict_token, str) else ""
    if not candidate or not token or candidate != token:
        return "stale", "", ""
    violated = verdict.get("violated")
    if not isinstance(violated, bool):
        return "malformed", "", ""
    guard_name = verdict.get("guard")
    if guard_name is not None and guard_name != slug:
        return "malformed", "", ""
    evidence = verdict.get("evidence")
    fix = verdict.get("fix")
    evidence = evidence if isinstance(evidence, str) else ""
    fix = fix if isinstance(fix, str) else ""
    if violated:
        return "violation", evidence, fix
    return "pass", evidence, fix


def _write_violations(
    stage: Path,
    segment: str,
    token: str,
    registry: dict[str, Any],
    violations: list[tuple[str, str, str]],
) -> None:
    lines = [f"# Guard violations — {segment} (iteration token {token[:8]})"]
    for slug, evidence, fix in violations:
        lines.append("")
        lines.append(f"## {slug}")
        lines.append(f"- rule: {registry[slug].strip()}")
        lines.append(f"- evidence: {evidence if evidence else '(none given)'}")
        lines.append(f"- fix: {fix if fix else '(none given)'}")
    lines.append("")
    _write_atomic(stage / "violations.md", "\n".join(lines))


def _check(project_path: str, segment: str, slugs: list[str]) -> int:
    stage = _stage_dir(project_path, segment)
    registry_path = stage / "rules.json"
    try:
        registry_raw = registry_path.read_text(encoding="utf-8")
    except OSError:
        return _emit(f"fail: guard rule registry missing: {registry_path}")
    try:
        registry: Any = json.loads(registry_raw)
    except json.JSONDecodeError:
        return _emit(f"fail: guard rule registry malformed: {registry_path}")
    if not isinstance(registry, dict) or not all(
        isinstance(value, str) for value in registry.values()
    ):
        return _emit(f"fail: guard rule registry malformed: {registry_path}")
    for slug in slugs:
        if slug not in registry:
            return _emit(f"fail: guard rule registry mismatch: {slug}")
    try:
        token = (stage / "token").read_text(encoding="utf-8").strip()
    except OSError:
        token = ""

    problems: list[tuple[str, str]] = []
    violations: list[tuple[str, str, str]] = []
    for slug in slugs:
        kind, evidence, fix = _classify_verdict(stage / f"{slug}.json", slug, token)
        if kind in _VERDICT_KINDS:
            problems.append((slug, kind))
        elif kind == "violation":
            violations.append((slug, evidence, fix))
    if problems:
        problems.sort()
        detail = "; ".join(f"{slug}: {kind}" for slug, kind in problems)
        print(detail, file=sys.stderr)
        return _emit(f"fail: guard verdicts missing or stale: {detail}")
    if violations:
        violations.sort(key=lambda item: item[0])
        _write_violations(stage, segment, token, registry, violations)
        line = " | ".join(
            f"{slug}: {_one_line(fix) or '(none given)'}"
            for slug, _evidence, fix in violations
        )
        return _emit(f"reloop: {line}")
    return _emit(f"pass: {len(slugs)} guard(s) clean")


def main(argv: list[str]) -> int:
    """Dispatch one aggregator verb; returns the process exit code."""
    if not argv:
        raise _usage_error("missing verb (expected reset or check)")
    verb, args = argv[0], argv[1:]
    if verb == "reset":
        if len(args) != 3:
            raise _usage_error(
                "reset expects 3 arguments (project_path agent_segment json_payload), "
                f"got {len(args)}"
            )
        project_path, segment, raw_payload = args
        if not project_path:
            raise _usage_error("project_path must be non-empty")
        return _reset(project_path, segment, raw_payload)
    if verb == "check":
        if len(args) < 3:
            raise _usage_error(
                "check expects at least 3 arguments "
                f"(project_path agent_segment slug [slug ...]), got {len(args)}"
            )
        project_path, segment = args[0], args[1]
        if not project_path:
            raise _usage_error("project_path must be non-empty")
        return _check(project_path, segment, args[2:])
    raise _usage_error(f"unknown verb {verb!r} (expected reset or check)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
