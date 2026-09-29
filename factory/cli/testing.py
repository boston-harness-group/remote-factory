"""CLI handlers for testing-related subcommands."""

from __future__ import annotations

import argparse


def cmd_check_patch_targets(args: argparse.Namespace) -> int:
    """Run the patch-target over-mocking check on a test file."""
    from factory.testing.patch_check import check_file, _format_result

    project_name: str | None = getattr(args, "project_name", None)
    result = check_file(args.test_file, project_name=project_name)
    print(_format_result(result))
    return result.exit_code
