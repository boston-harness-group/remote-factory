"""How `factory contained`'s command line is read — separate from what it then does.

`contained` has two positional shapes sharing one parser: a lifecycle subcommand (`ls`, `rm`, …)
and a verbatim payload after `--`. argparse cannot express that split declaratively — an optional
positional carrying `choices` would try to match the first word of the payload and reject it as an
invalid choice — so a single `REMAINDER` swallows everything and `interpret` divides it afterwards.

Everything in this module is about *reading* the command line: which shape it is, which flags are
in scope for the chosen target, the help text that says so, and the two readers that look inside the
verbatim payload — the project directory a run works on, and `--env`. Nothing here provisions
anything, which is why both runtimes can share it without either one importing the other.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog

from factory.contained.errors import ContainedError

log = structlog.get_logger()

LIFECYCLE_SUBCOMMANDS = ("ls", "attach", "rm", "sync", "setup", "verify", "bundle")

# `help` is not a lifecycle subcommand — it provisions nothing and acts on no runtime — but it is
# what people type, and without it the word falls through to the passthrough path and fails with
# "no existing directory found in ['help']", a message about materializing workspaces for what is a
# request to read the manual.
HELP_SUBCOMMAND = "help"

# Lifecycle subcommands that act on one named runtime, so a name is not optional for them.
_NAMED_SUBCOMMANDS = ("attach", "rm", "sync")

# Flags whose meaning exists only for one runtime. Using one against the other is a mistake worth
# naming: silently ignoring it makes a user believe a namespace or a mount took effect.
_LOCAL_ONLY = ("mount",)
_K8S_ONLY = ("namespace", "storage_class", "context")
_OPENSHELL_ONLY = ("policy", "gateway")

# Interactive payloads are refused for the openshell target — but per command, because the
# commands differ in what makes them interactive and in how to opt out:
#
#   ceo      interactive by default (a `claude` subprocess with inherited stdio); opts out
#            via --headless (pipe mode), --bg (background session) or --auto-approve
#            (headless design mode — its own validation requires the flag with --headless).
#   run      headless by default (the CEO is invoked in pipe mode); only --tmux-persist
#            makes it interactive. NOTE: `run` accepts no --headless — suggesting one would
#            pass this check and then die inside the sandbox with an argparse error.
#   agent    same shape as `run`: headless by default, interactive only with --tmux-persist.
#   resume   inherently interactive (Claude --resume with no headless form) — refused
#            outright, with no flag suggested because none exists.
#   tmux     a tmux session is a PTY by definition, and a sandbox cannot allocate one
#            (OpenShell #749) — refused outright.
#
# Everything else (study, diff, backlog, ...) is non-interactive and passes untouched.
_CEO_HEADLESS_FLAGS = ("--headless", "--bg", "--auto-approve")
_TERMINAL_COMMANDS = frozenset({"resume", "tmux"})
_PERSIST_COMMANDS = frozenset({"run", "agent"})

# Flags are described here rather than in argparse's own listing: which target a flag belongs to is
# the thing a user most needs to know, and a flat alphabetical list hides it.
HELP_EPILOG = """\
Run any factory command against a pinned toolchain and a copy of your project, so your
working tree is untouched. Everything after `--` is passed through unchanged.

  factory contained -- ceo ~/code/my-project

Targets:
  local   a podman container on this machine (the default). Fastest to start.
  k8s     a pod on a Kubernetes/OpenShift cluster. For long, unattended runs.
  openshell  (experimental) a policy-governed sandbox. For runs whose input you
          do not trust. The run is unattended: a sandbox has no terminal, so
          ceo needs --headless (or --bg/--auto-approve); resume and tmux are
          refused; run/agent only need attention with --tmux-persist.

Subcommands:
  setup                  Install what is missing, then check it
  verify                 Check prerequisites; report the fix for each failure
  ls                     List the runtimes this tool created
  attach NAME            Watch a running run (Ctrl-b d detaches; the run continues)
  sync NAME              Show how to get the run's work back
  rm NAME                Delete a runtime
  bundle                 Print the cluster prerequisites as YAML (k8s)
  help                   Print this text (same as --help)

All targets:
  --target local|k8s|openshell  Which runtime                       (default: local)
  --division             Let the agent build container images
  --name NAME            Name this run                              (default: derived)
  --env KEY=VALUE        Extra environment for the run, repeatable
  --forward VAR          Pass a variable from your shell inward, repeatable
  --image REF            Use a different runtime image
  --yes                  Skip confirmation prompts

Local only:
  --mount PATH           Also mount this host path, repeatable

K8s only:
  --namespace NS         Namespace                    (default: your current context)
  --context NAME         Which kubeconfig context to use    (default: your current one)
  --storage-class SC     Storage class for the workspace volume

Openshell only:
  --policy PATH          Replace the default sandbox policy (full replacement, never merged)
  --gateway NAME         Which registered OpenShell gateway to use  (default: the active one)

Environment:
  FACTORY_CONTAINED_IMAGE          Runtime image to use
  FACTORY_CONTAINED_HOME           Where workspace copies live (default ~/.factory-contained)
  FACTORY_CONTAINED_DRY_RUN=1      Print what would run; provision nothing

`contained` gives a run a reproducible environment and keeps it off your working tree.
The local and k8s targets are not security sandboxes: they do not restrict what the
agent's code can do, and they do not replace reviewing the result. The openshell target
is different by design — kernel-enforced filesystem and network allowlists, credentials
held by the gateway — but reviewing the result is still yours to do. `--division`
additionally opens an unauthenticated build endpoint on this machine for the length of
the run (local), and is not supported on openshell.

Full guide: https://akashgit.github.io/remote-factory/contained/
"""


def interpret(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Split the positional remainder and check flag scoping. Call once, before anything else.

    argparse offers no post-parse hook, so this is invoked explicitly — by `cmd_contained`, and by
    the tests, which must exercise the same interpretation the CLI performs.

    Sets `args.subcommand` and `args.factory_args` always; `args.name` only when a lifecycle
    positional supplies one. `--name` is parsed onto `args.name` before this runs, and the
    verbatim-payload branches must leave it alone — otherwise a run like
    `contained --name foo -- study /p` would have its explicit name overwritten with None here.
    """
    _split_positional(parser, args)

    # `bundle` only ever emits cluster YAML, so it implies the cluster target. Without this the
    # namespace flag it needs is rejected as out-of-scope for the default target, and the command
    # the generated manifest tells you to run cannot be run.
    if args.subcommand == "bundle":
        args.target = "k8s"

    _reject_out_of_scope_flags(parser, args)
    _reject_interactive_payload(parser, args)

    if args.subcommand in _NAMED_SUBCOMMANDS and not args.name:
        parser.error(f"`factory contained {args.subcommand}` needs a runtime name. Try `ls`.")
    if not args.subcommand and not args.factory_args:
        parser.error(
            "`factory contained` expects a factory command after `--`, for example:\n"
            "  factory contained -- ceo ~/code/my-project\n"
            "  factory contained --division -- study ~/code/my-project"
        )


def _split_positional(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Decide which of the four positional shapes was typed, and set `subcommand`/`factory_args`."""
    rest = list(args.rest)
    if rest and rest[0] == "--":          # argparse leaves the separator inside a REMAINDER
        args.subcommand, args.factory_args = None, rest[1:]
    elif rest and rest[0] == HELP_SUBCOMMAND:
        # Handled here rather than by argparse so that `help` behaves like `--help` without the
        # payload separator: everything after it is discarded, because there is no per-subcommand
        # help to select and silently ignoring `help ls` would imply there is.
        args.subcommand, args.factory_args = HELP_SUBCOMMAND, []
    elif rest and rest[0] in LIFECYCLE_SUBCOMMANDS:
        args.subcommand, args.factory_args = rest[0], []
        _read_lifecycle_tail(parser, args, rest[1:])
    else:
        args.subcommand, args.factory_args = None, rest
        _reject_subcommand_typo(parser, rest)


def _read_lifecycle_tail(
    parser: argparse.ArgumentParser, args: argparse.Namespace, tail: list[str]
) -> None:
    """What may follow a lifecycle subcommand: a runtime name, and `--yes`. Nothing else."""
    # `--yes` is the one trailing flag accepted here, because `rm <name> --yes` is the order
    # people type it. It is documented as the exception; every other flag in this position is
    # rejected below rather than silently dropped.
    if "--yes" in tail:
        args.yes = True
        tail = [token for token in tail if token != "--yes"]
    # Everything else that looks like a flag here is a mistake worth naming, not swallowing.
    # The REMAINDER split means `--target k8s` typed *after* the subcommand never reaches
    # `args.target` — it lands here as a plain string instead, so a silent absorption would
    # leave `args.target` at its default ("local") while the user believes they asked for k8s,
    # and would hand a lifecycle command a name like "--target" to resolve.
    flag_like = [token for token in tail if token.startswith("-")]
    if flag_like:
        parser.error(
            f"unrecognized flag {flag_like[0]!r} after `factory contained "
            f"{args.subcommand}`. Runtime flags (--target, --namespace, --name, ...) go before "
            f"the subcommand, for example:\n"
            f"  factory contained --target k8s {args.subcommand}"
        )
    # Only the positional overrides `--name` here, and only when one was actually given —
    # `ls` takes no name.
    if tail:
        args.name = tail[0]


def _reject_out_of_scope_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """A flag that belongs to the other target is named, never quietly ignored."""
    for dest in _LOCAL_ONLY:
        if getattr(args, dest) and args.target != "local":
            parser.error(f"--{dest.replace('_', '-')} only applies to --target local")
    for dest in _K8S_ONLY:
        if getattr(args, dest) and args.target != "k8s":
            parser.error(f"--{dest.replace('_', '-')} only applies to --target k8s")
    for dest in _OPENSHELL_ONLY:
        if getattr(args, dest) and args.target != "openshell":
            parser.error(f"--{dest.replace('_', '-')} only applies to --target openshell")
    if args.division and args.target == "openshell":
        # The build plane reaches outward through an unauthenticated endpoint (local) or the
        # OpenShift Build API (k8s); an OpenShell sandbox has neither, and egress wide enough to
        # build images is exactly what its policy denies. Refused here rather than at launch so
        # nobody reads three steps of provisioning before finding out.
        parser.error("--division is not supported by --target openshell")


def _reject_interactive_payload(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """An openshell run has no terminal, so an interactive payload cannot work there.

    The payload after `--` is verbatim by contract, and this is the one narrow exception — the
    same precedent as refusing `--division`: a command that cannot possibly succeed in the
    target is named at parse time rather than three provisioning steps in, after the workspace
    was copied and the sandbox created. Only the *first word* is looked at, per command,
    because the commands differ in what makes them interactive and in which flag opts out —
    a refusal that suggests a flag the command does not accept would pass this check and then
    die inside the sandbox.
    """
    if args.target != "openshell" or args.subcommand or not args.factory_args:
        return
    command = args.factory_args[0]

    if command in _TERMINAL_COMMANDS:
        # A PTY is the command's whole point (tmux) or its only form (resume): no flag can
        # make it work without a terminal, so the message says so rather than suggesting a
        # flag the command does not accept — advice that would pass this check and then die
        # inside the sandbox.
        parser.error(
            f"`factory {command}` needs a terminal, and an OpenShell sandbox has none — the "
            f"run is a detached process writing .factory/run.log.\n"
            f"  There is no headless form of `{command}`. Start a fresh headless run instead:\n"
            f"  factory contained --target openshell -- ceo <path> --headless"
        )
        return

    if command == "ceo" and not any(flag in args.factory_args for flag in _CEO_HEADLESS_FLAGS):
        parser.error(
            "`factory ceo` runs interactively by default, and an OpenShell sandbox has no "
            "terminal for it — the run is a detached process writing .factory/run.log.\n"
            "  Re-run with --headless:  factory contained --target openshell -- ceo "
            "<path> --headless\n"
            "  (--bg and --auto-approve also opt out; design mode needs --auto-approve, "
            "which --headless requires there)"
        )
        return

    if command in _PERSIST_COMMANDS and "--tmux-persist" in args.factory_args:
        # `run` and `agent` are headless by default — only --tmux-persist makes them
        # interactive, and neither accepts a --headless to be told otherwise.
        parser.error(
            f"`factory {command} --tmux-persist` runs interactively in a tmux window, and an "
            f"OpenShell sandbox has no terminal for it — the run is a detached process "
            f"writing .factory/run.log.\n"
            f"  Drop --tmux-persist:  factory contained --target openshell -- {command} "
            f"<path> runs headless by default"
        )


def _reject_subcommand_typo(parser: argparse.ArgumentParser, rest: list[str]) -> None:
    """Catch `lst` for `ls` before it is treated as a factory command.

    Without this the token falls through to the passthrough path and fails much later with "no
    existing directory found in ['lst']" — a message about materializing workspaces, for what is
    simply a typo.
    """
    if not rest:
        return
    first = rest[0]
    if first.startswith("-") or Path(first).expanduser().exists():
        return
    close = [
        c for c in (*LIFECYCLE_SUBCOMMANDS, HELP_SUBCOMMAND) if _within_one_edit(first, c)
    ]
    if close:
        parser.error(
            f"unknown subcommand {first!r} — did you mean {close[0]!r}?\n"
            f"  factory contained {close[0]}"
        )


def _within_one_edit(a: str, b: str) -> bool:
    """A cheap edit-distance-1 check: one substitution, insertion, or deletion."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    for index in range(len(longer)):
        if shorter == longer[:index] + longer[index + 1:]:
            return True
    return False


def target_given(args: argparse.Namespace) -> bool:
    """Whether the user actually typed `--target`, not just landed on its default.

    `--target` defaults to `"local"` (never `None`), so the parsed value alone cannot tell "the user
    asked for local" from "the user didn't say" — and only the second case should trigger
    `run_setup`'s interactive question. Recognizes both the space form (`--target local`) and the
    equals form; an explicit `--target=local` must not be mistaken for "didn't say".
    """
    return any(token == "--target" or token.startswith("--target=") for token in sys.argv)


def validate_env_args(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    """Check `--env` and `--forward` before anything is created.

    Both cost nothing to validate and everything to validate late: by the time the plan is built the
    workspace copy already exists and a container probe has run, so a typo would be reported after
    real work — or masked by an unrelated failure in between.
    """
    extra = parse_extra_env(args.extra_env)
    forwarded: dict[str, str] = {}
    for name in args.forward:
        value = os.environ.get(name)
        if value is None:
            raise ContainedError(f"--forward {name}: not set in this environment")
        forwarded[name] = value
    return extra, forwarded


def parse_extra_env(pairs: list[str]) -> dict[str, str]:
    """Parse repeated `--env KEY=VALUE` into a mapping, rejecting anything malformed."""
    parsed: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ContainedError(
                f"--env {pair!r} is not KEY=VALUE. Each --env takes one variable, and the value may "
                "be empty but the '=' may not be omitted."
            )
        parsed[key.strip()] = value
    return parsed


def resolve_project(factory_args: list[str]) -> Path:
    """The first existing directory named in the payload — the project a run works on.

    Everything after `--` is opaque to the host: it is not parsed as `factory ceo`'s own
    flags, so the one thing that can safely be assumed is that a contained run always starts from a
    project already on this machine, somewhere in that payload.
    """
    for token in factory_args:
        candidate = Path(token).expanduser()
        if candidate.is_dir():
            resolved = candidate.resolve()
            # The rule is generic — the first existing directory anywhere in the payload — so a
            # free-text value that coincidentally names one is picked silently otherwise. Logging it
            # is what keeps that visible.
            log.debug("contained_project_resolved", argument=token, project=str(resolved))
            return resolved
    raise ContainedError(
        f"no existing directory found in {factory_args!r}. `factory contained` materializes a "
        "workspace from a project already on this machine, for example:\n"
        "  factory contained -- ceo ~/code/my-project"
    )
