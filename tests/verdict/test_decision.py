import pytest

from octoqai.agents import AgentReply
from octoqai.expectations import CheckResult
from octoqai.experiment import Experiment
from octoqai.running import RunResult, TrialResult, TurnResult
from octoqai.security import Leak
from octoqai.verdict import EXIT_CODES, decide
from support.builders import ExperimentBuilder, turn

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def all_pass(n_cases, k):
    return [["pass"] * k for _ in range(n_cases)]


def rate(n_cases, k, fraction):
    passing = int(n_cases * fraction)
    return [["pass" if i < passing else "fail"] * k for i in range(n_cases)]


def _trial(variant, case_id, index, outcome, hard=False):
    passed = {"pass": True, "fail": False, "not_evaluable": None}[outcome]
    check = CheckResult("contains", False if hard else passed, hard, "")
    reply = AgentReply("ok", [], 7, 20.0, None)
    return TrialResult(variant, case_id, index, [TurnResult("u", reply, [check])])


def make_run(plan, threshold=0.8, baseline=None, hard_in=None, not_eval_in=None):
    names = list(plan)
    n_cases = len(next(iter(plan.values())))
    builder = ExperimentBuilder().agent(url="http://a", request={}, response={"answer": "$.a"})
    for name in names:
        builder.variant(name, baseline=name == baseline)
    for i in range(n_cases):
        builder.case(f"c{i}", turn("u", {"contains": "."}))
    builder.trials(len(next(iter(plan.values()))[0])).threshold(threshold)
    exp = Experiment.model_validate(builder.build())
    trials = []
    for name, cases in plan.items():
        for ci, outcomes in enumerate(cases):
            for ti, outcome in enumerate(outcomes):
                if not_eval_in == (name, ci):
                    outcome = "not_evaluable"
                hard = hard_in == (name, ci) and ti == 0
                trials.append(_trial(name, f"c{ci}", ti, outcome, hard))
    return RunResult(exp, trials, "2026-10-30T00:00:00+00:00", "2026-10-30T00:01:00+00:00")


@pytest.mark.smoke
def test_clear_pass_is_go():
    v = decide(make_run({"base": all_pass(30, 5)}))
    assert v.status == "go" and v.recommended == "base"
    assert v.variants[0].ci95[0] >= 0.8 and v.variants[0].pass_hat_k == 1.0


@pytest.mark.smoke
def test_hard_violation_is_no_go_even_if_rate_high():
    v = decide(make_run({"base": all_pass(30, 5)}, hard_in=("base", 0)))
    assert v.status == "no-go" and v.variants[0].hard_violations == 1


def test_clear_failure_is_no_go():
    assert decide(make_run({"base": rate(30, 5, 0.3)})).status == "no-go"


def test_small_sample_is_insufficient():
    assert decide(make_run({"base": all_pass(3, 1)})).status == "insufficient"


def test_not_evaluable_caps_at_insufficient():
    v = decide(make_run({"base": all_pass(30, 5)}, not_eval_in=("base", 0)))
    assert v.status == "insufficient" and v.variants[0].n_not_evaluable == 5


def test_significantly_better_candidate_recommended():
    v = decide(make_run({"base": rate(30, 5, 0.5), "cand": all_pass(30, 5)}, baseline="base"))
    assert v.recommended == "cand" and v.comparisons[0].significant
    assert v.comparisons[0].diff > 0 and v.status == "go"


def test_no_significant_winner_keeps_baseline():
    v = decide(make_run({"base": all_pass(30, 5), "cand": all_pass(30, 5)}, baseline="base"))
    assert v.recommended == "base" and not v.comparisons[0].significant


def test_better_candidate_with_hard_violation_is_not_recommended():
    v = decide(
        make_run(
            {"base": rate(30, 5, 0.5), "cand": all_pass(30, 5)},
            baseline="base",
            hard_in=("cand", 0),
        )
    )
    assert v.recommended == "base"


def test_metrics_aggregated():
    s = decide(make_run({"base": all_pass(4, 2)})).variants[0]
    assert s.mean_tokens == 7 and s.mean_latency_ms == 20.0 and s.n_evaluable == 8


def test_exit_codes():
    assert EXIT_CODES == {"go": 0, "no-go": 1, "insufficient": 2}


def test_one_case_many_trials_is_insufficient():
    v = decide(make_run({"base": all_pass(1, 30)}))
    assert v.status == "insufficient" and v.variants[0].n_cases == 1


def test_comparison_with_few_cases_is_never_significant():
    v = decide(make_run({"base": rate(2, 5, 0.0), "cand": all_pass(2, 5)}, baseline="base"))
    assert not v.comparisons[0].significant and v.recommended == "base"


def _leaky_trial(variant, case_id, index):
    leak = Leak("canary", "canary-1", "answer", "T…")
    check = CheckResult("leak", False, True, "canary 'canary-1' in answer: T…")
    reply = AgentReply("ok", [], 7, 20.0, None)
    return TrialResult(variant, case_id, index, [TurnResult("u", reply, [check], [leak])])


@pytest.mark.security
def test_leak_forces_no_go_with_reason():
    run = make_run({"base": all_pass(30, 5)})
    run.trials[0] = _leaky_trial("base", "c0", 0)
    verdict = decide(run)
    summary = verdict.variants[0]
    assert verdict.status == "no-go" and summary.leaks == 1
    assert summary.reasons == ["1 confidential data leak(s)"]
    assert verdict.reasons[0] == "base: 1 confidential data leak(s)"


@pytest.mark.security
def test_leak_with_other_hard_violation_reports_both_reasons_once():
    run = make_run({"base": all_pass(30, 5)}, hard_in=("base", 1))
    run.trials[0] = _leaky_trial("base", "c0", 0)
    summary = decide(run).variants[0]
    assert summary.reasons == ["1 confidential data leak(s)", "1 hard gate violation(s)"]


def test_mean_ttft_ignores_trials_without_streaming():
    run = make_run({"base": all_pass(5, 1)})
    assert decide(run).variants[0].mean_ttft_ms is None
    run.trials[0].turns[0].reply.ttft_ms = 40.0
    run.trials[1].turns[0].reply.ttft_ms = 60.0
    assert decide(run).variants[0].mean_ttft_ms == 50.0


def _forbidden_trial(variant, case_id, index, leak=False):
    checks = [CheckResult("forbidden_crud", False, True, "forbidden change 'x' after turn 1")]
    leaks = [Leak("canary", "canary-1", "answer", "T…")] if leak else []
    if leak:
        checks.append(CheckResult("leak", False, True, "canary 'canary-1' in answer: T…"))
    reply = AgentReply("ok", [], 7, 20.0, None)
    return TrialResult(variant, case_id, index, [TurnResult("u", reply, checks, leaks)])


@pytest.mark.security
def test_forbidden_change_forces_no_go_with_its_own_reason():
    run = make_run({"base": all_pass(30, 5)})
    run.trials[0] = _forbidden_trial("base", "c0", 0)
    summary = decide(run).variants[0]
    assert (summary.status, summary.forbidden_crud, summary.hard_violations) == ("no-go", 1, 1)
    assert summary.reasons == ["1 forbidden database change(s)"]


@pytest.mark.security
def test_leak_and_forbidden_change_are_listed_separately():
    run = make_run({"base": all_pass(30, 5)})
    run.trials[0] = _forbidden_trial("base", "c0", 0, leak=True)
    assert decide(run).variants[0].reasons == [
        "1 confidential data leak(s)",
        "1 forbidden database change(s)",
    ]


def test_trial_skipped_by_a_failed_reset_blocks_go():
    run = make_run({"base": all_pass(30, 5)})
    run.trials[0] = TrialResult(
        "base", "c0", 0, [], setup_error="before_trial[0] (command) exited with code 1"
    )
    summary = decide(run).variants[0]
    assert summary.status == "insufficient" and summary.n_not_evaluable == 1
    assert summary.mean_latency_ms == 20.0


def test_a_variant_that_never_ran_is_insufficient_and_empty():
    run = make_run({"base": all_pass(30, 5), "cand": all_pass(30, 5)}, baseline="base")
    run.trials = [t for t in run.trials if t.variant == "base"]
    summary = next(s for s in decide(run).variants if s.name == "cand")
    assert summary.status == "insufficient" and summary.n_evaluable == 0
    assert summary.mean_latency_ms == 0.0 and summary.mean_tokens is None


@pytest.mark.parametrize("skipped", ["no-trials", "failed-reset"])
def test_a_candidate_where_no_trial_ran_is_named_in_the_reasons(skipped):
    run = make_run({"base": all_pass(30, 5), "cand": all_pass(30, 5)}, baseline="base")
    if skipped == "no-trials":
        run.trials = [t for t in run.trials if t.variant == "base"]
    else:
        run.trials = [
            TrialResult(t.variant, t.case_id, t.trial, [], setup_error="before_trial[0] failed")
            if t.variant == "cand"
            else t
            for t in run.trials
        ]
    verdict = decide(run)
    assert "cand: no trial ran" in verdict.reasons
    assert verdict.recommended == "base"
    assert next(s for s in verdict.variants if s.name == "cand").status != "go"


def test_a_candidate_that_ran_is_not_reported_as_unrun():
    run = make_run({"base": all_pass(30, 5), "cand": all_pass(30, 5)}, baseline="base")
    assert not any("no trial ran" in reason for reason in decide(run).reasons)
