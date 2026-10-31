"""Machine-readable JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from octoqai import __version__
from octoqai.running import RunResult, TrialResult, TurnResult
from octoqai.security import Redactor
from octoqai.verdict import Verdict

SCHEMA_VERSION = "1"


def _turn(turn: TurnResult) -> dict[str, Any]:
    reply = turn.reply
    return {
        "user": turn.user,
        "answer": reply.answer,
        "tool_calls": reply.tool_calls,
        "tokens": reply.tokens,
        "latency_ms": reply.latency_ms,
        "ttft_ms": reply.ttft_ms,
        "fragments": len(reply.fragments),
        "retries": reply.retries,
        "error": reply.error,
        "checks": [asdict(check) for check in turn.checks],
        "leaks": [asdict(leak) for leak in turn.leaks],
    }


def _trial(trial: TrialResult) -> dict[str, Any]:
    return {
        "variant": trial.variant,
        "case_id": trial.case_id,
        "trial": trial.trial,
        "outcome": trial.outcome,
        "hard_violation": trial.hard_violation,
        "tokens": trial.tokens,
        "latency_ms": trial.latency_ms,
        "leaks": trial.leak_count,
        "ttft_ms": trial.ttft_ms,
        "turns": [_turn(turn) for turn in trial.turns],
    }


def build_report(run: RunResult, verdict: Verdict) -> dict[str, Any]:
    """Build the report document without agent connection details.

    Every string passes through the redactor: canaries, secrets, personal data and values
    from ``${VAR}`` expansions are replaced by placeholders. Raw stream fragments are not
    written, only their count.

    Args:
        run: The collected trial results.
        verdict: The decision for the run.

    Returns:
        A JSON-serializable report.
    """
    exp = run.experiment
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "octoqai", "version": __version__},
        "experiment": {
            "name": exp.name,
            "trials": exp.trials,
            "thresholds": exp.thresholds.model_dump(),
            "variants": [v.model_dump() for v in exp.variants],
        },
        "environment": exp.environment,
        "tags": exp.selected_tags,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "verdict": {
            "status": verdict.status,
            "recommended": verdict.recommended,
            "reasons": verdict.reasons,
        },
        "variants": [asdict(summary) for summary in verdict.variants],
        "comparisons": [asdict(comparison) for comparison in verdict.comparisons],
        "trials": [_trial(trial) for trial in run.trials],
    }
    redactor = Redactor.for_report(exp.security, exp.redacted_values)
    redacted: dict[str, Any] = redactor.redact_tree(report)
    return redacted


def write_json(run: RunResult, verdict: Verdict, path: Path) -> None:
    """Write the JSON report as UTF-8, creating parent directories.

    Args:
        run: The collected trial results.
        verdict: The decision for the run.
        path: Destination file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_report(run, verdict), ensure_ascii=False, indent=2), encoding="utf-8"
    )
