import pytest

from octoqai.agents import AgentReply
from octoqai.expectations import CheckResult
from octoqai.experiment import Experiment
from octoqai.running import RunResult, TrialResult, TurnResult
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
