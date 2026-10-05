"""`factory contained` — run any factory command inside a podman container or a cluster pod.

The runtime is a place to run the factory, not a mode of the factory: everything after `--` is
handed inward verbatim, except for path rewriting.

This module is only the front door: register the parser, then hand one interpreted command to
whoever owns it. The things it hands to are peers, and none of them knows about the others —
`contained_args.py` reads the command line, `contained_local.py` runs one podman container,
`contained_k8s.py` runs one cluster pod, `contained_openshell.py` runs one OpenShell sandbox.
"""

from __future__ import annotations

import argparse
import sys

from factory.cli.contained_args import (
    HELP_EPILOG,
    HELP_SUBCOMMAND,
    interpret,
    target_given,
)
from factory.cli.contained_local import run_local
from factory.contained.lifecycle import dispatch_lifecycle
from factory.contained.prereq import local_checks, render_checks
from factory.contained.setup import run_setup


# Set by `build_contained_parser`, read by `cmd_contained`. `interpret` needs the parser itself (to
# call `.error()` on) and the namespace has no room for it: `set_defaults` would put every key into
# `--help` output and into every namespace repr, which is noise in exactly the place a user is
# trying to read.
_PARSER: argparse.ArgumentParser | None = None


def build_contained_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `contained` subcommand.

    The payload after `--` is `argparse.REMAINDER`: it is handed to the factory inside the runtime
    verbatim. Validating it here would mean the host has to know every subcommand the contained
    factory supports, which it cannot — and a passthrough that second-guesses its payload breaks
    every time the CLI grows.
    """
    global _PARSER
    p = sub.add_parser(
        "contained",
        help="Run any factory command in a container (local), a pod (k8s), or a sandbox (openshell)",
        usage="factory contained [runtime flags] -- <factory command>\n"
              "       factory contained {ls|attach|rm|sync|setup|verify|bundle|help} [name]",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # `--name` and `--namespace` share a prefix. Without this, argparse's default prefix
        # matching lets `--name` silently resolve to `--namespace` (or any future flag that happens
        # to share a prefix with another), which is exactly the kind of flag-aliasing this parser
        # has to name loudly rather than let happen quietly.
        allow_abbrev=False,
    )
    # One REMAINDER for everything positional, split afterwards by `interpret`. A declarative split
    # is not expressible: an optional positional carrying `choices` would try to match the first
    # word of the payload and reject it as an invalid choice.
    # Every flag is SUPPRESSed from argparse's own listing and described in the epilog instead:
    # a flat list hides which target each flag belongs to, and printing both lists each flag twice.
    p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.add_argument("--target", choices=["local", "k8s", "openshell"], default="local",
                   help=argparse.SUPPRESS)
    p.add_argument("--division", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--name", default=None, help=argparse.SUPPRESS)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", dest="extra_env",
                   help=argparse.SUPPRESS)
    p.add_argument("--forward", action="append", default=[], metavar="VAR", help=argparse.SUPPRESS)
    p.add_argument("--mount", action="append", default=[], metavar="PATH", help=argparse.SUPPRESS)
    p.add_argument("--namespace", default=None, help=argparse.SUPPRESS)
    p.add_argument("--storage-class", default=None, dest="storage_class", help=argparse.SUPPRESS)
    p.add_argument("--context", default=None, help=argparse.SUPPRESS)
    p.add_argument("--image", default=None, help=argparse.SUPPRESS)
    # Openshell-scoped: the policy is a full replacement (never merged with the default), and
    # the gateway names which registered gateway this machine talks to.
    p.add_argument("--policy", default=None, help=argparse.SUPPRESS)
    p.add_argument("--gateway", default=None, help=argparse.SUPPRESS)
    # `rm` prompts before deleting an active runtime and the cluster upload prompts on a secret-scan
    # finding; `--yes` skips both, for automation.
    p.add_argument("--yes", action="store_true", default=False, help=argparse.SUPPRESS)
    _PARSER = p
    return p


def _verify(args: argparse.Namespace) -> int:
    if args.target == "openshell":
        from factory.contained.prereq import format_check, summary_line
        from factory.contained.openshell_prereq import verify_openshell

        # Streamed for the same reason the cluster checks stream: several are gateway round
        # trips, and silence until the last one lands is indistinguishable from a hang.
        checks = verify_openshell(
            gateway=args.gateway,
            on_check=lambda c: print(format_check(c), flush=True),
        )
        print()
        print(summary_line(
            checks,
            ready_command="factory contained --target openshell -- ceo <path> --headless",
        ))
        return 0 if all(c.ok for c in checks) else 1
    if args.target == "k8s":
        from factory.contained.k8s_setup import verify_k8s
        from factory.contained.prereq import format_check, summary_line

        # Streamed for the same reason `setup` streams: the cluster checks take minutes between
        # them, and silence until the last one lands is indistinguishable from a hang.
        checks = verify_k8s(
            namespace=args.namespace, division=args.division,
            on_check=lambda c: print(format_check(c), flush=True),
        )
        print()
        print(summary_line(checks, ready_command="factory contained --target k8s -- ceo <path>"))
        return 0 if all(c.ok for c in checks) else 1
    checks = local_checks()
    print(render_checks(checks))
    return 0 if all(c.ok for c in checks) else 1


def cmd_contained(args: argparse.Namespace) -> int:
    """Run the factory inside a container (local) or a pod (k8s).

    Ctrl-C is caught here rather than allowed to unwind. Backing out of a wizard partway through is
    an ordinary thing to do — the flow is a sequence of questions and someone will always change
    their mind at question three — and answering that with a stack trace reads as a crash the user
    caused. The two exit paths that need their own message (a container that may still be running,
    a namespace left half-prepared) handle it closer in and never reach this.
    """
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130                       # what a shell expects from a process killed by SIGINT


def _dispatch(args: argparse.Namespace) -> int:
    assert _PARSER is not None, "build_contained_parser must run before cmd_contained"
    interpret(_PARSER, args)

    if getattr(args, "context", None):
        # Pinned once, here, for every cluster command this invocation issues. `factory/contained/
        # k8s.py:cli()` is the single place it is applied, so nothing downstream has to remember.
        from factory.contained.k8s import set_active_context

        set_active_context(args.context)

    if args.subcommand == HELP_SUBCOMMAND:
        _PARSER.print_help()
        return 0
    if args.subcommand == "verify":
        return _verify(args)
    if args.subcommand == "setup":
        return run_setup(
            args.target if target_given(args) else None,
            interactive=sys.stdin.isatty(),
            namespace=args.namespace,
            division=args.division,
            assume_yes=args.yes,
            gateway=getattr(args, "gateway", None),
        )
    if args.subcommand == "bundle":
        from factory.contained.bundle import render_bundle
        from factory.podman import resolve_image

        print(render_bundle(namespace=args.namespace, storage_class=args.storage_class,
                            division=args.division, image=args.image or resolve_image()))
        return 0
    if args.subcommand:
        return dispatch_lifecycle(args)

    if args.target == "k8s":
        from factory.cli.contained_k8s import run_k8s

        return run_k8s(args)
    if args.target == "openshell":
        from factory.cli.contained_openshell import run_openshell

        return run_openshell(args)

    return run_local(args)
