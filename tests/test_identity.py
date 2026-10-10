"""Tests for the canonical agent-invocation identity (factory/identity.py).

Covers the six properties from issue #1485 plus the RELOOP/attempt
regression: a retry with byte-identical inputs must NOT hit the identity of
the rejected first attempt.
"""

from __future__ import annotations

from pathlib import Path

from factory.identity import invocation_identity, normalize_prompt
from factory.models import AgentRunRequest

HOME = str(Path.home())


def _request(
    prompt: str = "Build the auth feature described in {project}/docs/spec.md",
    task: str = "ship the auth feature",
    role: str = "builder",
    model: str | None = "opus",
    project: str = "/proj/checkout",
) -> AgentRunRequest:
    prompt = prompt.replace("{project}", project)
    return AgentRunRequest(
        prompt=prompt,
        task=task,
        cwd=Path("/tmp/work"),
        role=role,
        model=model,
        project_path=Path(project),
    )


class TestNormalizePrompt:
    def test_project_path_becomes_placeholder(self):
        out = normalize_prompt(
            f"Read {HOME}/code/app/src/main.py", project_path=Path(f"{HOME}/code/app")
        )
        assert out == "Read {project}/src/main.py"

    def test_project_path_resolved_form_replaced(self):
        # macOS: /var/folders is a symlink to /private/var/folders
        raw = normalize_prompt("/tmp/foo/../bar/x.md", project_path=Path("/tmp/foo/../bar"))
        assert raw == "{project}/x.md"

    def test_home_paths_become_placeholder_without_project(self):
        out = normalize_prompt(f"See {HOME}/.factory/agents/prompts/r.md")
        assert out == "See {home}/.factory/agents/prompts/r.md"

    def test_unknown_absolute_paths_kept(self):
        out = normalize_prompt("Edit /etc/hosts carefully")
        assert out == "Edit /etc/hosts carefully"

    def test_dashed_uuid_replaced(self):
        out = normalize_prompt("session 550e8400-e29b-41d4-a716-446655440000 done")
        assert out == "session {uuid} done"

    def test_bare_hex32_replaced(self):
        out = normalize_prompt("trace 550e8400e29b41d4a716446655440000 done")
        assert out == "trace {uuid} done"

    def test_iso_datetime_replaced(self):
        for stamp in (
            "2026-09-09T14:33:21+00:00",
            "2026-09-09T14:33:21Z",
            "2026-09-09 14:33:21",
            "2026-09-09T14:33",
            "2026-09-09T14:33:21.123456",
        ):
            assert normalize_prompt(f"at {stamp} then") == "at {timestamp} then", stamp

    def test_date_only_kept(self):
        out = normalize_prompt("release scheduled for 2026-09-09")
        assert out == "release scheduled for 2026-09-09"

    def test_epoch_millis_replaced(self):
        out = normalize_prompt("run ts 1757428401123 ok")
        assert out == "run ts {timestamp} ok"

    def test_small_numbers_kept(self):
        out = normalize_prompt("retry 3 times over 10 iterations at depth 42")
        assert out == "retry 3 times over 10 iterations at depth 42"

    def test_whitespace_collapsed(self):
        out = normalize_prompt("line one\n\n   line two\t\ttabbed")
        assert out == "line one line two tabbed"

    def test_semantic_content_preserved(self):
        out = normalize_prompt("Use bcrypt, not sha1, for password hashing")
        assert out == "Use bcrypt, not sha1, for password hashing"


class TestInvocationIdentity:
    def test_path_independence(self):
        a = _request(project="/Users/alice/code/app")
        b = _request(project="/tmp/pytest-42/bob/app")
        assert invocation_identity(a) == invocation_identity(b)

    def test_path_independence_across_home_roots(self, monkeypatch):
        # The same logical prompt written under bob's home on one machine and
        # ours on another must normalize identically. Each digest is computed
        # while its machine's home is active.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/bob")))
        b = AgentRunRequest(
            prompt="Read /home/bob/docs/a.md and act",
            task="t",
            cwd=Path("/tmp/y"),
            role="builder",
        )
        digest_b = invocation_identity(b)

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(HOME)))
        a = AgentRunRequest(
            prompt=f"Read {HOME}/docs/a.md and act",
            task="t",
            cwd=Path("/tmp/x"),
            role="builder",
        )
        digest_a = invocation_identity(a)

        assert digest_a == digest_b

    def test_cwd_not_part_of_identity(self):
        a = _request()
        b = a.model_copy(update={"cwd": Path("/somewhere/else")})
        assert invocation_identity(a) == invocation_identity(b)

    def test_session_fields_not_part_of_identity(self):
        a = _request()
        b = a.model_copy(
            update={"session_id": "s2", "session_name": "tmux-b", "resume_session_id": "s0"}
        )
        assert invocation_identity(a) == invocation_identity(b)

    def test_semantic_prompt_change_changes_digest(self):
        a = _request()
        b = a.model_copy(update={"prompt": a.prompt + " Also add integration tests"})
        assert invocation_identity(a) != invocation_identity(b)

    def test_semantic_task_change_changes_digest(self):
        a = _request()
        b = a.model_copy(update={"task": "ship it, but keep the old auth module"})
        assert invocation_identity(a) != invocation_identity(b)

    def test_attempt_sensitivity(self):
        a = _request()
        assert invocation_identity(a) != invocation_identity(a, attempt=1)
        assert invocation_identity(a, attempt=1) != invocation_identity(a, attempt=2)

    def test_model_sensitivity(self):
        a = _request()
        b = a.model_copy(update={"model": "sonnet"})
        assert invocation_identity(a) != invocation_identity(b)

    def test_role_sensitivity(self):
        a = _request()
        b = a.model_copy(update={"role": "code_reviewer"})
        assert invocation_identity(a) != invocation_identity(b)

    def test_unset_model_vs_empty(self):
        a = _request(model=None)
        b = _request(model="")
        assert invocation_identity(a) == invocation_identity(b)

    def test_artifact_hashes_order_insensitive(self):
        a = _request()
        left = invocation_identity(a, input_artifact_hashes=["h1", "h2", "h3"])
        right = invocation_identity(a, input_artifact_hashes=["h3", "h1", "h2"])
        assert left == right

    def test_artifact_hashes_content_sensitive(self):
        a = _request()
        left = invocation_identity(a, input_artifact_hashes=["h1"])
        right = invocation_identity(a, input_artifact_hashes=["h2"])
        assert left != right

    def test_artifact_hashes_vs_none(self):
        a = _request()
        assert invocation_identity(a) != invocation_identity(a, input_artifact_hashes=["h1"])

    def test_determinism(self):
        a = _request()
        assert invocation_identity(a) == invocation_identity(a)
        assert len(invocation_identity(a)) == 64

    def test_digest_is_hex(self):
        import re

        assert re.fullmatch(r"[0-9a-f]{64}", invocation_identity(_request()))


class TestReloopRegression:
    """The correctness trap from issue #1485: a RELOOP retry runs with
    byte-identical inputs; if the identity excluded the attempt number, an
    identity-keyed cache would replay the just-rejected output forever."""

    def test_rejected_retry_must_be_cache_miss(self):
        first = _request()
        rejected_output = "the output the gate rejected"

        cache: dict[str, str] = {}
        cache[invocation_identity(first)] = rejected_output

        # Retry after RELOOP: same request object, attempt incremented.
        retry_key = invocation_identity(first, attempt=1)
        assert retry_key not in cache, "retry must not hit the rejected attempt's entry"

        cache[retry_key] = "fixed output"
        again = invocation_identity(first, attempt=2)
        assert again not in cache

    def test_prompt_volatility_does_not_break_hits(self):
        """The flip side: innocuous volatility (workspace paths, timestamps,
        session IDs) must not turn a genuine repeat into a miss. The consumer
        contract is to pass the per-run workspace as project_path so the
        run-scoped root is templated while its semantic subpaths survive."""
        def make(workspace: str, ts: str, session: str) -> AgentRunRequest:
            return AgentRunRequest(
                prompt=(
                    "Build the auth feature described in /proj/checkout/docs/spec.md. "
                    f"Workspace: {workspace}/inputs. Started {ts}. Session {session}."
                ),
                task="ship the auth feature",
                cwd=Path(workspace),
                role="builder",
                model="opus",
                project_path=Path(workspace),
            )

        run_a = make(
            "/tmp/pytest-111/work",
            "2026-09-09T09:00:00Z",
            "550e8400-e29b-41d4-a716-446655440000",
        )
        run_b = make(
            "/tmp/pytest-222/work",
            "2026-09-09T11:30:45Z",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        )
        assert invocation_identity(run_a) == invocation_identity(run_b)


class TestProjectAgnosticism:
    def test_identical_requests_same_digest_across_projects(self):
        a = _request(project="/proj/alpha")
        b = _request(project="/proj/beta")
        assert invocation_identity(a) == invocation_identity(b)
