"""Canonical identity for agent invocations.

Multiple subsystems need to answer the same question — *is this agent
invocation the same as that one?* Node-level caching (the Ledger proposal),
record-and-replay integration testing, and artifact provenance all need a
stable key over an ``AgentRunRequest``. This module is that key, and nothing
else: no storage, no cassettes, no cache machinery (see issue #1485).

The naive approach — hashing the raw request — fails because prompts embed
volatility: absolute paths that differ per machine and per checkout, run IDs,
timestamps. Correctness lives in the normalization applied before hashing,
and normalization is a blunt instrument in both directions:

- Too aggressive (strips semantic content): a cache silently replays stale
  outputs and hides prompt regressions.
- Too timid (keeps volatility): a cache never hits; replay cassettes must be
  re-recorded on every innocuous change.

The rules below are deliberately conservative — only true volatility is
removed. **Any change to them is a cache-invalidation event for every
consumer** and should be treated with the same care as a prompt-template
change.

## Normalization table

What is applied to prompt and task text, in order:

| # | Rule | Input example | Output | Why |
|---|------|---------------|--------|-----|
| 1 | Project path → ``{project}`` | ``/Users/x/code/app/src/main.py`` | ``{project}/src/main.py`` | Checkout location is machine state, not semantics. Both the literal and resolved forms are replaced. |
| 2 | Home directory → ``{home}`` | ``/Users/x/.factory/agents/prompts/r.md`` | ``{home}/.factory/...`` | Usernames differ across machines. |
| 3 | Temp directory → ``{tmp}`` | ``/var/folders/zy/T/pytest-123/x.md`` | ``{tmp}/pytest-123/x.md`` | OS temp roots differ per machine, per run. |
| 4 | UUIDs → ``{uuid}`` | ``a3f1...`` (dashed or 32-hex) | ``{uuid}`` | Session IDs, trace IDs, and other opaque run-scoped identifiers. |
| 5 | ISO-8601 timestamps → ``{timestamp}`` | ``2026-09-09T14:33:21+00:00`` | ``{timestamp}`` | Wall-clock time is not semantics. Date-only strings are kept (a date in a prompt is usually content). |
| 6 | Epoch-milliseconds → ``{timestamp}`` | ``1757428401123`` | ``{timestamp}`` | 13-digit epoch-millis appear in run-scoped IDs; 13-digit integers are vanishingly rare as prompt content. Plain 10-digit values are kept (they can be legitimate content). |
| 7 | Whitespace runs → single space | ``"a\\n\\n  b"`` | ``"a b"`` | Re-templating that only changes indentation must not invalidate. |

What is deliberately **kept**: everything else, including numbers (except
13-digit epoch-millis), file contents, instructions, constraints, paths that
are not under the project/home/temp roots, and the request's ``role``,
``model``, and ``task``.

## Identity composition

``invocation_identity`` digests, in a fixed field order:

1. ``role``
2. ``model`` (empty string when unset)
3. normalized ``prompt``
4. normalized ``task``
5. ``attempt`` — see below
6. sorted ``input_artifact_hashes``

**Attempt number is load-bearing.** A node rejected and retried (e.g. via a
RELOOP verdict) runs with byte-identical inputs. If the identity excluded the
attempt, a cache would replay the just-rejected output, the gate would reject
it again, and the workflow would loop forever without consuming budget. Same
inputs plus a different attempt must be a cache miss.

The identity is project-agnostic by design: two identical requests against
different projects produce the same digest. Consumers that need per-project
separation should namespace keys themselves (e.g. one cache directory per
project).

**Consumer contract for volatile roots.** Normalization only templates paths
under roots it knows about: the given ``project_path``, the user's home, and
the system temp root. A consumer whose prompts embed a per-run workspace
(a tmp scratch dir, a worktree) should pass that root as ``project_path`` —
then the run-scoped root is templated while its semantic subpaths survive
(``{project}/inputs/x.json``).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from factory.models import AgentRunRequest

# RFC 4122 UUIDs: dashed 8-4-4-4-12, or bare 32 hex chars.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    r"|\b[0-9a-fA-F]{32}\b"
)

# ISO-8601 datetimes with a time component (date-only is content, not noise).
_ISO_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b"
)

# Epoch milliseconds: 13 digits, standalone.
_EPOCH_MS_RE = re.compile(r"(?<![\d.])\d{13}(?![\d.])")

_WHITESPACE_RE = re.compile(r"\s+")


def _path_replacements(project_path: Path | None) -> list[tuple[str, str]]:
    """Literal → placeholder pairs for known-volatile roots, longest first."""
    pairs: list[tuple[str, str]] = []
    if project_path is not None:
        candidates = [project_path]
        try:
            resolved = project_path.resolve()
        except OSError:
            resolved = None
        if resolved is not None and str(resolved) != str(project_path):
            candidates.append(resolved)
        for candidate in candidates:
            pairs.append((str(candidate), "{project}"))

    home = Path.home()
    pairs.append((str(home), "{home}"))
    try:
        home_resolved = home.resolve()
    except OSError:
        home_resolved = None
    if home_resolved is not None and str(home_resolved) != str(home):
        pairs.append((str(home_resolved), "{home}"))

    temp = Path(tempfile.gettempdir())
    pairs.append((str(temp), "{tmp}"))
    try:
        temp_resolved = temp.resolve()
    except OSError:
        temp_resolved = None
    if temp_resolved is not None and str(temp_resolved) != str(temp):
        pairs.append((str(temp_resolved), "{tmp}"))

    # Longest literal first so /home/u/proj is replaced before /home/u.
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def normalize_prompt(prompt: str, *, project_path: Path | None = None) -> str:
    """Canonicalize prompt/task text per the normalization table.

    Deterministic, environment-independent in output, and conservative: only
    paths under known-volatile roots, UUIDs, timestamps, epoch-millis, and
    whitespace runs are touched.
    """
    text = prompt
    for literal, placeholder in _path_replacements(project_path):
        text = text.replace(literal, placeholder)
    text = _UUID_RE.sub("{uuid}", text)
    text = _ISO_DATETIME_RE.sub("{timestamp}", text)
    text = _EPOCH_MS_RE.sub("{timestamp}", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def invocation_identity(
    request: AgentRunRequest,
    *,
    attempt: int = 0,
    input_artifact_hashes: list[str] | None = None,
) -> str:
    """Stable sha256 hex digest over the identity of an agent invocation.

    Covers (role, model, normalized prompt, normalized task, attempt, sorted
    input artifact hashes). The digest is project- and path-independent;
    consumers needing project separation should namespace it themselves.
    """
    project_path = request.project_path
    payload_parts = [
        f"role={request.role}",
        f"model={request.model or ''}",
        f"prompt={normalize_prompt(request.prompt, project_path=project_path)}",
        f"task={normalize_prompt(request.task, project_path=project_path)}",
        f"attempt={attempt}",
        "artifacts="
        + ",".join(sorted(input_artifact_hashes or [])),
    ]
    return hashlib.sha256("\n".join(payload_parts).encode()).hexdigest()
