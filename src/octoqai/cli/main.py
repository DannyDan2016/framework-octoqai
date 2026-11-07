"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

import httpx

from octoqai import __version__
from octoqai.agents import AdapterKwargs, adapter_for
from octoqai.db import SqlDatabase, connection_secrets
from octoqai.errors import ConfigError
from octoqai.experiment import Experiment, load_experiment
from octoqai.hooks import HookEnvironment, HookExecutor, HookKwargs
from octoqai.reports import write_json
from octoqai.running import RunResult, hook_error, run_experiment
from octoqai.security import Redactor
from octoqai.verdict import EXIT_CODES, Verdict, decide

EXIT_ERROR = 3


@dataclass(frozen=True)
class _Seams:
    transport: httpx.BaseTransport | None
    adapter: AdapterKwargs
    hooks: HookKwargs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="octoqai", description="Evaluate AI agents with statistically sound experiments."
    )
    parser.add_argument("--version", action="version", version=f"octoqai {__version__}")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="run an experiment and write the reports")
    run.add_argument("experiment", type=Path, help="path to the experiment YAML file")
    run.add_argument(
        "--out", type=Path, default=Path("octoqai-out"), help="output directory for reports"
    )
    run.add_argument("--env", help="environment to run in, e.g. dev or qa")
    run.add_argument(
        "--tags", type=_tag_list, default=[], help="comma separated tags; runs cases with any"
    )
    return parser


def _tag_list(text: str) -> list[str]:
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def _print_summary(verdict: Verdict) -> None:
    for summary in verdict.variants:
        lo, hi = summary.ci95
        print(
            f"{summary.name:<20} {summary.status:<12} "
            f"pass_rate={summary.pass_rate:.2f} [{lo:.2f}, {hi:.2f}] "
            f"leaks={summary.leaks} forbidden={summary.forbidden_crud}"
        )
    for comparison in verdict.comparisons:
        lo, hi = comparison.ci95
        print(
            f"{comparison.variant} vs {comparison.vs}: diff={comparison.diff:+.2f} "
            f"[{lo:+.2f}, {hi:+.2f}]{' significant' if comparison.significant else ''}"
        )
    for reason in verdict.reasons:
        print(f"  - {reason}")
    print(f"VERDICT: {verdict.status} (recommended: {verdict.recommended})")


def _never_ran(run: RunResult) -> bool:
    return all(
        (t.agent_error or t.setup_error) and t.leak_count == 0 and t.forbidden_count == 0
        for t in run.trials
    )


def _never_ran_message(run: RunResult) -> str:
    if all(t.setup_error for t in run.trials):
        first_setup = next(t.setup_error for t in run.trials)
        return f"Error: before_trial hook failed in every trial ({first_setup})."
    first = next(turn.reply.error for t in run.trials for turn in t.turns if turn.reply.error)
    return f"Error: agent unreachable in every trial ({first})."


def _warn_failed_cleanup(run: RunResult, redactor: Redactor) -> bool:
    failed = [r for r in run.hooks if r.stage == "after_run" and not r.ok]
    for result in failed:
        print(redactor.redact(f"Warning: hook {hook_error(result)}"), file=sys.stderr)
    return bool(failed)


def _exit_code(verdict: Verdict, cleanup_failed: bool) -> int:
    if not cleanup_failed:
        return EXIT_CODES[verdict.status]
    return 1 if verdict.status == "no-go" else EXIT_ERROR


def _with_url_secrets(exp: Experiment) -> Experiment:
    if exp.database is None:
        return exp
    known = list(exp.redacted_values)
    known += [v for v in connection_secrets(exp.database.url) if v not in known]
    return exp.model_copy(update={"redacted_values": known})


def _run(args: argparse.Namespace, seams: _Seams) -> int:
    exp = _with_url_secrets(load_experiment(args.experiment, env=args.env))
    redactor = Redactor.for_report(exp.security, exp.redacted_values)
    try:
        return _execute(args, exp, redactor, seams)
    except ConfigError as exc:
        print(redactor.redact(f"Config error: {exc}"), file=sys.stderr)
    except Exception as exc:
        print(redactor.redact(f"Error: {type(exc).__name__}: {exc}"), file=sys.stderr)
    return EXIT_ERROR


def _open_and_run(exp: Experiment, seams: _Seams) -> RunResult:
    with ExitStack() as stack:
        db = None
        if exp.database is not None:
            db = stack.enter_context(closing(SqlDatabase(exp.database.url)))
        environment = HookEnvironment(db, exp.base_dir, exp.hook_env)
        hooks = stack.enter_context(closing(HookExecutor(exp.hooks, environment, **seams.hooks)))
        adapter = stack.enter_context(
            closing(adapter_for(exp.agent, seams.transport, **seams.adapter))
        )
        return run_experiment(exp, adapter, db=db, hooks=hooks)


def _execute(args: argparse.Namespace, exp: Experiment, redactor: Redactor, seams: _Seams) -> int:
    if args.tags:
        exp = exp.select(args.tags)
    out: Path = args.out
    run = _open_and_run(exp, seams)
    cleanup_failed = _warn_failed_cleanup(run, redactor)
    if run.setup_error:
        print(redactor.redact(f"Error: {run.setup_error}. No trial was run."), file=sys.stderr)
        return EXIT_ERROR
    verdict = decide(run)
    report = out / "report.json"
    write_json(run, verdict, report)
    if _never_ran(run):
        message = redactor.redact(_never_ran_message(run))
        print(f"{message} Report: {report}", file=sys.stderr)
        return EXIT_ERROR
    _print_summary(verdict)
    print(f"Report: {report}")
    return _exit_code(verdict, cleanup_failed)


def main(
    argv: list[str],
    transport: httpx.BaseTransport | None = None,
    *,
    adapter_kwargs: AdapterKwargs | None = None,
    hook_kwargs: HookKwargs | None = None,
) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Command line arguments without the program name.
        transport: Optional httpx transport, used to test without a network.
        adapter_kwargs: Optional ``clock``, ``sleep``, ``jitter`` and ``watchdog`` for the
            adapter, used to test without real waiting.
        hook_kwargs: Optional ``transport``, ``run_command`` and ``clock`` for the hook
            executor, used to test without a network or real processes.

    Returns:
        0 for go, 1 for no-go, 2 for insufficient evidence and 3 for errors.
    """
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_ERROR
    if args.command != "run":
        parser.print_usage(sys.stderr)
        return EXIT_ERROR
    try:
        return _run(
            args, _Seams(transport, adapter_kwargs or AdapterKwargs(), hook_kwargs or HookKwargs())
        )
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return EXIT_ERROR


def entrypoint() -> None:
    """Console script entry point."""
    sys.exit(main(sys.argv[1:]))
