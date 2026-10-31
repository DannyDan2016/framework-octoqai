"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from octoqai import __version__
from octoqai.agents import HttpAgentAdapter
from octoqai.errors import ConfigError
from octoqai.experiment import load_experiment
from octoqai.reports import write_json
from octoqai.running import run_experiment
from octoqai.verdict import EXIT_CODES, Verdict, decide

EXIT_ERROR = 3


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
    return parser


def _print_summary(verdict: Verdict) -> None:
    for summary in verdict.variants:
        lo, hi = summary.ci95
        print(
            f"{summary.name:<20} {summary.status:<12} "
            f"pass_rate={summary.pass_rate:.2f} [{lo:.2f}, {hi:.2f}]"
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


def _run(
    experiment: Path, out: Path, env: str | None, transport: httpx.BaseTransport | None
) -> int:
    exp = load_experiment(experiment, env=env)
    adapter = HttpAgentAdapter(exp.agent, transport=transport)
    try:
        run = run_experiment(exp, adapter)
    finally:
        adapter.close()
    verdict = decide(run)
    report = out / "report.json"
    write_json(run, verdict, report)
    if all(trial.agent_error for trial in run.trials):
        first = next(turn.reply.error for turn in run.trials[0].turns if turn.reply.error)
        print(
            f"Error: agent unreachable in every trial ({first}). Report: {report}", file=sys.stderr
        )
        return EXIT_ERROR
    _print_summary(verdict)
    print(f"Report: {report}")
    return EXIT_CODES[verdict.status]


def main(argv: list[str], transport: httpx.BaseTransport | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Command line arguments without the program name.
        transport: Optional httpx transport, used to test without a network.

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
        return _run(args.experiment, args.out, args.env, transport)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return EXIT_ERROR


def entrypoint() -> None:
    """Console script entry point."""
    sys.exit(main(sys.argv[1:]))
