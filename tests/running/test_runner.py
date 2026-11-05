import pytest

from octoqai.experiment import Experiment
from octoqai.running import run_experiment
from support.agents import RESPONSE_MAP, FakeSseAgent
from support.builders import ExperimentBuilder, turn
from support.hooks import RecordingHooks

pytestmark = [pytest.mark.unit, pytest.mark.regression]

REQUEST = {
    "conversation_id": "{{ conversation_id }}",
    "message": "{{ message }}",
    "history": "{{ history }}",
    "temperature": "{{ variant.temperature }}",
}


def experiment(*cases, two_variants=False, trials=1):
    builder = ExperimentBuilder().agent(request=REQUEST, response=RESPONSE_MAP).trials(trials)
    builder.variant("base", baseline=True, temperature=0)
    if two_variants:
        builder.variant("cand", temperature=1)
    for case in cases:
        builder.case(*case)
    return Experiment.model_validate(builder.build())


def multiturn():
    return (
        "conciliar",
        turn("Concilia octubre", {"tool_called": ["buscar"]}),
        turn("¿Y la diferencia?", {"contains": "ok"}),
    )


def two_cases():
    return [
        ("c1", turn("hola", {"contains": "ok"})),
        ("c2", turn("adiós", {"contains": "ok"})),
    ]


def hard_case(pattern):
    return ("leak", turn("x", {"not_contains": {"pattern": pattern, "hard": True}}))


@pytest.fixture
def agent(fake_agent):
    return fake_agent.answers("ok", tools=["buscar"], tokens=5)


def run(exp, fake):
    return run_experiment(exp, fake.adapter(exp.agent))


@pytest.mark.smoke
def test_runs_variants_x_cases_x_trials(agent):
    result = run(experiment(*two_cases(), two_variants=True, trials=3), agent)
    assert len(result.trials) == 2 * 2 * 3
    assert {t.variant for t in result.trials} == {"base", "cand"}


def test_variant_params_sent_per_request(agent):
    run(experiment(*two_cases(), two_variants=True), agent)
    assert sorted({b["temperature"] for b in agent.requests}) == [0, 1]


def test_conversation_id_stable_within_trial_new_across_trials(agent):
    run(experiment(multiturn(), trials=2), agent)
    ids = [b["conversation_id"] for b in agent.requests]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_history_grows_across_turns_and_resets_per_trial(agent):
    run(experiment(multiturn(), trials=2), agent)
    assert [len(b["history"]) for b in agent.requests] == [0, 2, 0, 2]
    assert agent.requests[1]["history"][0] == {"role": "user", "content": "Concilia octubre"}


def test_trial_passes_when_all_turns_pass(agent):
    trial = run(experiment(multiturn()), agent).trials[0]
    assert trial.outcome == "pass" and trial.tokens == 10 and not trial.hard_violation


def test_error_stops_trial_and_fails_it(fake_agent):
    trial = run(experiment(multiturn()), fake_agent.fails(500, "down")).trials[0]
    assert len(trial.turns) == 1 and trial.outcome == "fail"


def test_not_evaluable_when_no_failures_but_unknown_checks(fake_agent):
    trial = run(experiment(multiturn()), fake_agent.answers("ok")).trials[0]
    assert trial.outcome == "not_evaluable"


def test_hard_violation_flagged(agent):
    trial = run(experiment(hard_case("ok")), agent).trials[0]
    assert trial.hard_violation and trial.outcome == "fail"


def test_timestamps_are_iso_utc(agent):
    result = run(experiment(*two_cases()), agent)
    assert result.started_at.endswith("+00:00") and result.finished_at >= result.started_at


def test_agent_error_is_not_a_hard_violation(fake_agent):
    trial = run(experiment(hard_case("z")), fake_agent.fails(500)).trials[0]
    assert trial.outcome == "fail" and not trial.hard_violation and trial.agent_error


def test_request_template_sees_data(fake_agent):
    builder = ExperimentBuilder().agent(
        request={"message": "{{ message }}", "account": "{{ data.cuenta }}", "env": "{{ env }}"}
    )
    exp = Experiment.model_validate(builder.build())
    exp = exp.model_copy(update={"data_values": {"cuenta": "ES-0002"}, "environment": "qa"})
    run(exp, fake_agent)
    assert fake_agent.requests[0] == {"message": "Hola", "account": "ES-0002", "env": "qa"}


def streaming_experiment():
    builder = ExperimentBuilder().streaming().security(canaries=[{"value": "TEN-B-0042"}])
    builder.case("c1", turn("Concilia octubre", {"contains": "."}))
    return Experiment.model_validate(builder.build())


def run_sse(fake):
    exp = streaming_experiment()
    return run_experiment(exp, fake.adapter(exp.agent)).trials[0]


@pytest.mark.security
def test_leak_makes_trial_fail_with_hard_violation():
    trial = run_sse(FakeSseAgent().leaks("TEN-B-0042"))
    assert trial.outcome == "fail" and trial.hard_violation and trial.leak_count == 1
    assert trial.turns[0].checks[-1].kind == "leak"
    assert trial.turns[0].leaks[0].location == "fragment 2"


@pytest.mark.security
def test_leak_in_partial_stream_counts_despite_agent_error():
    trial = run_sse(FakeSseAgent().cuts_after("dato TEN-B-0042"))
    assert trial.agent_error and trial.hard_violation and trial.leak_count == 1


@pytest.mark.security
def test_canary_written_by_the_user_is_not_a_leak():
    builder = ExperimentBuilder().streaming().security(canaries=[{"value": "TEN-B-0042"}])
    exp = Experiment.model_validate(
        builder.case("c1", turn("Dame TEN-B-0042", {"contains": "."})).build()
    )
    trial = run_experiment(exp, FakeSseAgent().leaks("TEN-B-0042").adapter(exp.agent)).trials[0]
    assert trial.leak_count == 0 and not trial.hard_violation


def test_ttft_reported_per_trial():
    trial = run_sse(FakeSseAgent().streams("ok.", first_s=0.2))
    assert trial.ttft_ms == pytest.approx(200) and trial.leak_count == 0


def test_json_trials_have_no_ttft(agent):
    assert run(experiment(*two_cases()), agent).trials[0].ttft_ms is None


def lifecycle_experiment(trials=1, n_cases=2):
    builder = ExperimentBuilder().trials(trials)
    for i in range(n_cases):
        builder.case(f"c{i}", turn("Hola", {"contains": "."}))
    return Experiment.model_validate(builder.build())


@pytest.mark.smoke
def test_hooks_wrap_every_trial_in_series(fake_agent):
    events = []
    hooks = RecordingHooks(events)
    exp = lifecycle_experiment()
    agent = fake_agent.answers("ok.").effect(lambda _: events.append("send"))
    run_experiment(exp, agent.adapter(exp.agent), hooks=hooks)
    assert events == [
        "before_run",
        "before_trial",
        "send",
        "after_trial",
        "before_trial",
        "send",
        "after_trial",
        "after_run",
    ]


def test_failed_before_run_skips_trials_and_still_runs_after_run(fake_agent):
    hooks = RecordingHooks().fails("before_run")
    exp = lifecycle_experiment()
    run = run_experiment(exp, fake_agent.adapter(exp.agent), hooks=hooks)
    assert run.trials == [] and fake_agent.requests == []
    assert run.setup_error == "before_run[0] (command) exited with code 1"
    assert hooks.events == ["before_run", "after_run"]
    assert [h.stage for h in run.hooks] == ["before_run", "after_run"]


def test_failed_reset_skips_the_trial_without_calling_the_agent(fake_agent):
    hooks = RecordingHooks().fails("before_trial", times=1)
    exp = lifecycle_experiment(trials=2, n_cases=1)
    adapter = fake_agent.answers("ok.").adapter(exp.agent)
    first, second = run_experiment(exp, adapter, hooks=hooks).trials
    assert (first.outcome, first.turns, first.tokens) == ("not_evaluable", [], None)
    assert first.setup_error == "before_trial[0] (command) exited with code 1"
    assert second.outcome == "pass" and len(fake_agent.requests) == 1
    assert hooks.events.count("after_trial") == 2


def test_failed_after_trial_is_recorded_without_changing_the_outcome(fake_agent):
    hooks = RecordingHooks().fails("after_trial")
    exp = lifecycle_experiment(n_cases=1)
    adapter = fake_agent.answers("ok.").adapter(exp.agent)
    [trial] = run_experiment(exp, adapter, hooks=hooks).trials
    assert trial.outcome == "pass"
    assert [(h.stage, h.ok) for h in trial.hooks] == [
        ("before_trial", True),
        ("after_trial", False),
    ]


def _boom(_request):
    raise RuntimeError("boom")


def test_after_run_runs_even_if_a_trial_raises(fake_agent):
    hooks = RecordingHooks()
    exp = lifecycle_experiment(n_cases=1)
    fake_agent.effect(_boom)
    with pytest.raises(RuntimeError):
        run_experiment(exp, fake_agent.adapter(exp.agent), hooks=hooks)
    assert hooks.events[-1] == "after_run"


def test_after_trial_runs_even_if_the_conversation_raises(fake_agent):
    hooks = RecordingHooks()
    exp = lifecycle_experiment(n_cases=1)
    fake_agent.effect(_boom)
    with pytest.raises(RuntimeError):
        run_experiment(exp, fake_agent.adapter(exp.agent), hooks=hooks)
    assert hooks.events[-2:] == ["after_trial", "after_run"]
