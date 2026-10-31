"""Machine-readable JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from octoqai import __version__
from octoqai.running import RunResult, TrialResult
from octoqai.verdict import Verdict

SCHEMA_VERSION = "1"
REDACTED = "[redacted]"
MIN_REDACTED_LENGTH = 4


def _trial(trial: TrialResult) -> dict[str, Any]:
    return {
        "variant": trial.variant,
        "case_id": trial.case_id,
        "trial": trial.trial,
        "outcome": trial.outcome,
        "hard_violation": trial.hard_violation,
        "tokens": trial.tokens,
        "latency_ms": trial.latency_ms,
        "turns": [
            {
                "user": turn.user,
                "answer": turn.reply.answer,
                "tool_calls": turn.reply.tool_calls,
                "tokens": turn.reply.tokens,
                "latency_ms": turn.reply.latency_ms,
                "error": turn.reply.error,
                "checks": [asdict(check) for check in turn.checks],
            }
            for turn in trial.turns
        ],
    }


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
    return value


def build_report(run: RunResult, verdict: Verdict) -> dict[str, Any]:
    """Build the report document without agent connection details.

    Every value that came from a ``${VAR}`` expansion is replaced by ``[redacted]``.

    Args:
        run: The collected trial results.
        verdict: The decision for the run.

    Returns:
        A JSON-serializable report.
    """
    exp = run.experiment
    secrets = sorted(
        (v for v in exp.redacted_values if len(v) >= MIN_REDACTED_LENGTH), key=len, reverse=True
    )
    report = {
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
    redacted: dict[str, Any] = _redact(report, secrets)
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
