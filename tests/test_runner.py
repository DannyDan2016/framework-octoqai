import json

import httpx
import pytest

from octoqai.agents import HttpAgentAdapter
from octoqai.experiment import Experiment
from octoqai.running import run_experiment


def experiment(variants=None, cases=None, trials=1):
    return Experiment.model_validate(
        {
            "name": "t",
            "agent": {
                "url": "http://agent.local/chat",
                "request": {
                    "conversation_id": "{{ conversation_id }}",
                    "message": "{{ message }}",
                    "history": "{{ history }}",
                    "temperature": "{{ variant.temperature }}",
                },
                "response": {"answer": "$.answer", "tool_calls": "$.tools", "tokens": "$.tokens"},
            },
            "variants": variants
            or [
                {"name": "base", "baseline": True, "params": {"temperature": 0}},
                {"name": "cand", "params": {"temperature": 1}},
            ],
            "trials": trials,
            "thresholds": {"pass_rate": 0.8},
            "cases": cases
            or [
                {"id": "c1", "turns": [{"user": "hola", "expect": [{"contains": "ok"}]}]},
                {"id": "c2", "turns": [{"user": "adiós", "expect": [{"contains": "ok"}]}]},
            ],
        }
    )


MULTITURN = [
    {
        "id": "conciliar",
        "turns": [
            {"user": "Concilia octubre", "expect": [{"tool_called": ["buscar"]}]},
            {"user": "¿Y la diferencia?", "expect": [{"contains": "ok"}]},
        ],
    }
]


class Recorder:
    def __init__(self, status=200):
        self.bodies = []
        self.status = status

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(json.loads(json.dumps(body)))
        if self.status != 200:
            return httpx.Response(self.status, text="down")
        return httpx.Response(200, json={"answer": "ok", "tools": ["buscar"], "tokens": 5})


@pytest.fixture
def recorder():
    return Recorder()


def adapter_for(exp, handler):
    return HttpAgentAdapter(exp.agent, transport=httpx.MockTransport(handler))


def test_runs_variants_x_cases_x_trials(recorder):
    exp = experiment(trials=3)
    run = run_experiment(exp, adapter_for(exp, recorder))
    assert len(run.trials) == 2 * 2 * 3
    assert {t.variant for t in run.trials} == {"base", "cand"}


def test_variant_params_sent_per_request(recorder):
    exp = experiment()
    run_experiment(exp, adapter_for(exp, recorder))
    assert sorted({b["temperature"] for b in recorder.bodies}) == [0, 1]


def test_conversation_id_stable_within_trial_new_across_trials(recorder):
    exp = experiment(
        variants=[{"name": "base", "params": {"temperature": 0}}], cases=MULTITURN, trials=2
    )
    run_experiment(exp, adapter_for(exp, recorder))
    ids = [b["conversation_id"] for b in recorder.bodies]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_history_grows_across_turns_and_resets_per_trial(recorder):
    exp = experiment(
        variants=[{"name": "base", "params": {"temperature": 0}}], cases=MULTITURN, trials=2
    )
    run_experiment(exp, adapter_for(exp, recorder))
    assert [len(b["history"]) for b in recorder.bodies] == [0, 2, 0, 2]
    assert recorder.bodies[1]["history"][0] == {"role": "user", "content": "Concilia octubre"}


def test_trial_passes_when_all_turns_pass(recorder):
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}], cases=MULTITURN)
    trial = run_experiment(exp, adapter_for(exp, recorder)).trials[0]
    assert trial.outcome == "pass" and trial.tokens == 10 and not trial.hard_violation


def test_error_stops_trial_and_fails_it():
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}], cases=MULTITURN)
    trial = run_experiment(exp, adapter_for(exp, Recorder(status=500))).trials[0]
    assert len(trial.turns) == 1 and trial.outcome == "fail"


def test_not_evaluable_when_no_failures_but_unknown_checks():
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}], cases=MULTITURN)

    def no_tools(request):
        return httpx.Response(200, json={"answer": "ok"})

    trial = run_experiment(exp, adapter_for(exp, no_tools)).trials[0]
    assert trial.outcome == "not_evaluable"


def test_hard_violation_flagged():
    cases = [
        {
            "id": "leak",
            "turns": [{"user": "x", "expect": [{"not_contains": {"pattern": "ok", "hard": True}}]}],
        }
    ]
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}], cases=cases)
    trial = run_experiment(exp, adapter_for(exp, Recorder())).trials[0]
    assert trial.hard_violation and trial.outcome == "fail"


def test_timestamps_are_iso_utc(recorder):
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}])
    run = run_experiment(exp, adapter_for(exp, recorder))
    assert run.started_at.endswith("+00:00") and run.finished_at >= run.started_at


def test_agent_error_is_not_a_hard_violation():
    cases = [
        {
            "id": "leak",
            "turns": [{"user": "x", "expect": [{"not_contains": {"pattern": "z", "hard": True}}]}],
        }
    ]
    exp = experiment(variants=[{"name": "base", "params": {"temperature": 0}}], cases=cases)
    trial = run_experiment(exp, adapter_for(exp, Recorder(status=500))).trials[0]
    assert trial.outcome == "fail" and not trial.hard_violation and trial.agent_error
