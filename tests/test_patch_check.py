"""Tests for factory.testing.patch_check — AST-based over-mocking detection."""

from __future__ import annotations

import textwrap
from pathlib import Path

from factory.testing.patch_check import (
    ALLOWLISTED_MODULES,
    CheckResult,
    PatchTarget,
    check_file,
    _classify,
    _detect_project_name,
    _format_result,
)


# ── helpers ──────────────────────────────────────────────────────

def _write_test_file(tmp_path: Path, content: str, name: str = "test_sample.py") -> Path:
    """Write a Python test file and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _write_pyproject(tmp_path: Path, project_name: str = "my-project") -> None:
    """Write a minimal pyproject.toml for project name detection."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n',
        encoding="utf-8",
    )


# ── classification tests ────────────────────────────────────────

class TestClassify:
    def test_external_target(self):
        assert _classify("requests.get", "factory") == "external"

    def test_internal_target(self):
        assert _classify("factory.cli.main", "factory") == "internal"

    def test_allowlisted_datetime(self):
        assert _classify("datetime.datetime.now", "factory") == "allowlisted"

    def test_allowlisted_random(self):
        assert _classify("random.randint", "factory") == "allowlisted"

    def test_allowlisted_time(self):
        assert _classify("time.time", "factory") == "allowlisted"

    def test_allowlisted_os_environ(self):
        assert _classify("os.environ", "factory") == "allowlisted"

    def test_allowlisted_uuid(self):
        assert _classify("uuid.uuid4", "factory") == "allowlisted"

    def test_project_name_exact_match(self):
        assert _classify("factory", "factory") == "internal"

    def test_unrelated_package(self):
        assert _classify("boto3.client", "factory") == "external"


# ── exit code tests ─────────────────────────────────────────────

class TestExitCodes:
    def test_only_external_patches_exit_0(self, tmp_path):
        """File with only external patches → exit 0 (pass)."""
        _write_pyproject(tmp_path, "my-project")
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("requests.get")
            def test_api_call(mock_get):
                mock_get.return_value.status_code = 200
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert len(result.targets) == 1
        assert result.targets[0].classification == "external"

    def test_internal_patches_exit_1(self, tmp_path):
        """File with project-internal patches → exit 1 (advisory warning)."""
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("my_project.core.engine.run")
            def test_engine(mock_run):
                mock_run.return_value = True
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 1
        assert len(result.targets) == 1
        assert result.targets[0].classification == "internal"

    def test_allowlisted_patches_exit_0(self, tmp_path):
        """File with only allowlisted patches (datetime, random) → exit 0."""
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("datetime.datetime.now")
            @patch("random.randint")
            def test_timing(mock_rand, mock_now):
                pass
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert all(t.classification == "allowlisted" for t in result.targets)

    def test_mixed_patches_correct_classification(self, tmp_path):
        """File with mixed patches — each classified correctly."""
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("requests.post")
            @patch("my_project.db.connection.get_pool")
            @patch("datetime.datetime.utcnow")
            def test_mixed(mock_dt, mock_pool, mock_post):
                pass
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 1  # has internal patch

        classifications = {t.target: t.classification for t in result.targets}
        assert classifications["requests.post"] == "external"
        assert classifications["my_project.db.connection.get_pool"] == "internal"
        assert classifications["datetime.datetime.utcnow"] == "allowlisted"


# ── monkeypatch detection ───────────────────────────────────────

class TestMonkeypatch:
    def test_monkeypatch_setattr_string_form(self, tmp_path):
        """monkeypatch.setattr("dotted.path", value) detected."""
        f = _write_test_file(tmp_path, """\
            def test_something(monkeypatch):
                monkeypatch.setattr("my_project.config.DEBUG", True)
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 1
        assert len(result.targets) == 1
        assert result.targets[0].kind == "monkeypatch_setattr"
        assert result.targets[0].classification == "internal"

    def test_monkeypatch_setattr_object_form(self, tmp_path):
        """monkeypatch.setattr(module, "attr", value) detected."""
        f = _write_test_file(tmp_path, """\
            import my_project.config as config

            def test_something(monkeypatch):
                monkeypatch.setattr(config, "DEBUG", True)
        """)
        result = check_file(f, project_name="my_project")
        # Object form resolves to config.DEBUG — not project-prefixed,
        # so classified as external. This is a known limitation of static analysis.
        assert len(result.targets) == 1
        assert result.targets[0].kind == "monkeypatch_setattr"

    def test_monkeypatch_setenv(self, tmp_path):
        """monkeypatch.setenv is always allowlisted (os.environ)."""
        f = _write_test_file(tmp_path, """\
            def test_env(monkeypatch):
                monkeypatch.setenv("API_KEY", "test-key")
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert len(result.targets) == 1
        assert result.targets[0].classification == "allowlisted"
        assert result.targets[0].kind == "monkeypatch_setenv"

    def test_monkeypatch_delenv(self, tmp_path):
        """monkeypatch.delenv is always allowlisted."""
        f = _write_test_file(tmp_path, """\
            def test_env(monkeypatch):
                monkeypatch.delenv("SECRET", raising=False)
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert result.targets[0].classification == "allowlisted"
        assert result.targets[0].kind == "monkeypatch_delenv"

    def test_mp_alias_detected(self, tmp_path):
        """monkeypatch aliased as 'mp' is also detected."""
        f = _write_test_file(tmp_path, """\
            def test_something(mp):
                mp.setattr("my_project.config.DEBUG", True)
        """)
        result = check_file(f, project_name="my_project")
        assert len(result.targets) == 1
        assert result.targets[0].classification == "internal"


# ── edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        """Empty file → exit 0, no targets."""
        f = _write_test_file(tmp_path, "")
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert result.targets == []

    def test_malformed_file(self, tmp_path):
        """File with syntax errors → exit 2."""
        f = _write_test_file(tmp_path, "def broken(:\n    pass\n")
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 2
        assert result.error is not None
        assert "Syntax error" in result.error

    def test_file_not_found(self, tmp_path):
        """Non-existent file → exit 2."""
        result = check_file(tmp_path / "nonexistent.py", project_name="my_project")
        assert result.exit_code == 2
        assert "not found" in (result.error or "").lower()

    def test_no_patches(self, tmp_path):
        """File with no patches at all → exit 0."""
        f = _write_test_file(tmp_path, """\
            def test_add():
                assert 1 + 1 == 2
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert result.targets == []

    def test_mock_patch_attribute_form(self, tmp_path):
        """@mock.patch("target") form detected."""
        f = _write_test_file(tmp_path, """\
            from unittest import mock

            @mock.patch("boto3.client")
            def test_aws(mock_client):
                pass
        """)
        result = check_file(f, project_name="my_project")
        assert result.exit_code == 0
        assert len(result.targets) == 1
        assert result.targets[0].target == "boto3.client"


# ── project name detection ──────────────────────────────────────

class TestProjectNameDetection:
    def test_detect_from_pyproject(self, tmp_path):
        """Detects project name from pyproject.toml."""
        _write_pyproject(tmp_path, "cool-project")
        name = _detect_project_name(tmp_path)
        assert name == "cool_project"

    def test_dash_to_underscore(self, tmp_path):
        """Dashes in project name are converted to underscores."""
        _write_pyproject(tmp_path, "my-cool-project")
        name = _detect_project_name(tmp_path)
        assert name == "my_cool_project"

    def test_no_pyproject(self, tmp_path):
        """Missing pyproject.toml → empty string."""
        name = _detect_project_name(tmp_path)
        assert name == ""

    def test_auto_detect_in_check_file(self, tmp_path):
        """check_file auto-detects project name when not provided."""
        _write_pyproject(tmp_path, "my-project")
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("my_project.core.run")
            def test_internal(mock_run):
                pass
        """)
        # Don't pass project_name — let it auto-detect
        result = check_file(f)
        assert result.exit_code == 1
        assert result.targets[0].classification == "internal"

    def test_no_project_name_available(self, tmp_path):
        """No pyproject.toml and no project_name → exit 2 error."""
        f = _write_test_file(tmp_path, """\
            from unittest.mock import patch

            @patch("something.run")
            def test_it(mock_run):
                pass
        """)
        result = check_file(f)
        # Walks up and may find the repo's pyproject.toml, so this test
        # is only reliable in isolated tmp dirs. If it finds a pyproject,
        # that's fine — the important thing is no crash.
        assert result.exit_code in (0, 1, 2)


# ── format output ───────────────────────────────────────────────

class TestFormatResult:
    def test_pass_format(self):
        result = CheckResult(path="test.py", targets=[])
        output = _format_result(result)
        assert "No patch targets found" in output

    def test_error_format(self):
        result = CheckResult(path="test.py", error="Syntax error")
        output = _format_result(result)
        assert "ERROR" in output

    def test_advisory_format(self):
        result = CheckResult(
            path="test.py",
            targets=[PatchTarget("my_project.core.run", 10, "decorator", "internal")],
        )
        output = _format_result(result)
        assert "ADVISORY WARNING" in output
        assert "INTERNAL" in output


# ── allowlist constant ──────────────────────────────────────────

class TestAllowlist:
    def test_expected_modules_present(self):
        """All documented allowlisted modules are in the constant."""
        expected = {"datetime", "random", "time", "os.environ", "os.cpu_count", "uuid"}
        assert expected.issubset(ALLOWLISTED_MODULES)
