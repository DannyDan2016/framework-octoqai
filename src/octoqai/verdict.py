"""Go/no-go decision with hard gates and A/B recommendation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from octoqai.runner import RunResult, TrialResult
from octoqai.stats import paired_bootstrap_diff, pass_at_k, pass_hat_k, wilson_interval

Status = Literal["go", "no-go", "insufficient"]

EXIT_CODES: dict[str, int] = {"go": 0, "no-go": 1, "insufficient": 2}

MIN_CASES_FOR_COMPARISON = 5


@dataclass
class VariantSummary:
    """Aggregated metrics and status of one variant."""

    name: str
    baseline: bool
    n_cases: int
    n_evaluable: int
    n_not_evaluable: int
    pass_rate: float
    ci95: tuple[float, float]
    pass_at_k: float
    pass_hat_k: float
    hard_violations: int
    mean_tokens: float | None
    mean_latency_ms: float
    status: Status
    reasons: list[str] = field(default_factory=list)


@dataclass
class Comparison:
    """Paired difference of a candidate against the baseline."""

    variant: str
    vs: str
    diff: float
    ci95: tuple[float, float]
    significant: bool


@dataclass
class Verdict:
    """Final decision for the run."""

    status: Status
    recommended: str
    variants: list[VariantSummary]
    comparisons: list[Comparison]
    reasons: list[str]


def _by_case(trials: list[TrialResult]) -> dict[str, list[TrialResult]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for trial in trials:
        grouped[trial.case_id].append(trial)
    return grouped


def _summarize(name: str, baseline: bool, trials: list[TrialResult], threshold: float):
    evaluable = [t for t in trials if t.outcome != "not_evaluable"]
    passes = sum(t.outcome == "pass" for t in evaluable)
    n_not_evaluable = len(trials) - len(evaluable)
    n_cases = len(_by_case(evaluable))
    pass_rate = passes / len(evaluable) if evaluable else 0.0
    lo, hi = wilson_interval(pass_rate * n_cases, n_cases)
    cases = _by_case(trials)
    case_outcomes = [[t.outcome == "pass" for t in group] for group in cases.values()]
    hard = sum(t.hard_violation for t in trials)
    tokens = [t.tokens for t in trials if t.tokens is not None]
    reasons: list[str] = []
    if hard:
        status: Status = "no-go"
        reasons.append(f"{hard} hard gate violation(s)")
    elif hi < threshold:
        status = "no-go"
        reasons.append(f"pass rate upper bound {hi:.2f} is below threshold {threshold:.2f}")
    elif lo >= threshold and n_not_evaluable == 0:
        status = "go"
        reasons.append(f"pass rate lower bound {lo:.2f} meets threshold {threshold:.2f}")
    else:
        status = "insufficient"
        if n_not_evaluable:
            reasons.append(f"{n_not_evaluable} trial(s) could not be evaluated")
        if lo < threshold:
            reasons.append(
                f"interval [{lo:.2f}, {hi:.2f}] contains threshold {threshold:.2f}: "
                f"more cases or trials are needed"
            )
    return VariantSummary(
        name=name,
        baseline=baseline,
        n_cases=n_cases,
        n_evaluable=len(evaluable),
        n_not_evaluable=n_not_evaluable,
        pass_rate=pass_rate,
        ci95=(lo, hi),
        pass_at_k=sum(map(pass_at_k, case_outcomes)) / len(case_outcomes),
        pass_hat_k=sum(map(pass_hat_k, case_outcomes)) / len(case_outcomes),
        hard_violations=hard,
        mean_tokens=sum(tokens) / len(tokens) if tokens else None,
        mean_latency_ms=sum(t.latency_ms for t in trials) / len(trials),
        status=status,
        reasons=reasons,
    )


def _case_rates(trials: list[TrialResult], case_ids: list[str]) -> list[float]:
    cases = _by_case(trials)
    return [
        sum(t.outcome == "pass" for t in cases[cid]) / len(cases[cid]) if cases[cid] else 0.0
        for cid in case_ids
    ]


def decide(run: RunResult) -> Verdict:
    """Decide go, no-go or insufficient evidence for a run.

    Args:
        run: The collected trial results.

    Returns:
        The verdict, with per-variant summaries and comparisons against the baseline.
    """
    exp = run.experiment
    threshold = exp.thresholds.pass_rate
    baseline = exp.baseline
    by_variant: dict[str, list[TrialResult]] = defaultdict(list)
    for trial in run.trials:
        by_variant[trial.variant].append(trial)
    summaries = {
        v.name: _summarize(v.name, v.name == baseline.name, by_variant[v.name], threshold)
        for v in exp.variants
    }
    case_ids = [c.id for c in exp.cases]
    base_rates = _case_rates(by_variant[baseline.name], case_ids)
    comparisons = []
    for variant in exp.variants:
        if variant.name == baseline.name:
            continue
        diff, lo, hi = paired_bootstrap_diff(
            _case_rates(by_variant[variant.name], case_ids), base_rates
        )
        significant = lo > 0 and len(case_ids) >= MIN_CASES_FOR_COMPARISON
        comparisons.append(Comparison(variant.name, baseline.name, diff, (lo, hi), significant))
    winners = [c for c in comparisons if c.significant and summaries[c.variant].status != "no-go"]
    recommended = max(winners, key=lambda c: c.diff).variant if winners else baseline.name
    chosen = summaries[recommended]
    reasons = [f"{recommended}: {reason}" for reason in chosen.reasons]
    if comparisons and len(case_ids) < MIN_CASES_FOR_COMPARISON:
        reasons.append(
            f"at least {MIN_CASES_FOR_COMPARISON} cases are needed to compare variants, "
            f"got {len(case_ids)}"
        )
    elif comparisons and not winners:
        reasons.append("no candidate is significantly better than the baseline")
    return Verdict(chosen.status, recommended, list(summaries.values()), comparisons, reasons)
