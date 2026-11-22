import json
from urllib.parse import quote

import pytest
import yaml

from octoqai.experiment import load_experiment
from octoqai.running import worst_case_calls
from support.assertions import assert_exit, assert_not_in_outputs
from support.paths import EXAMPLES
from support.providers import FakeLlm, ScriptedUser

pytestmark = [pytest.mark.e2e, pytest.mark.regression]

CANARY = "CANARY-7f3a9c2e"
EXAMPLE = EXAMPLES / "judge" / "experiment.yaml"
FAKE_ENV = {
    "ANTHROPIC_API_KEY": "fake-key-for-tests",
    "JUDGE_MODEL": "fake-judge-1",
    "SIMULATED_USER_MODEL": "fake-user-1",
}


@pytest.fixture(autouse=True)
def _example_env(monkeypatch):
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def judge_example(tmp_path, toy_sse_agent_url):
    def _write(**judge_settings):
        data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        data["agent"]["url"] = toy_sse_agent_url
        data["judge"].update(judge_settings)
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return str(path)

    return _write


def passing_judge(times=6):
    judge = FakeLlm()
    for _ in range(times):
        judge.judges("pass", "meets the rubric")
    return judge


def sent_text(*ports):
    return "\n".join(
        part
        for port in ports
        for request in port.requests
        for part in (request.system, *(message.content for message in request.messages))
    )


def test_example_caps_equal_the_worst_case():
    exp = load_experiment(EXAMPLE, process_env=FAKE_ENV)
    worst = worst_case_calls(exp)
    assert worst == {"judge": 6, "simulated_user": 18}
    assert exp.judge.max_calls == worst["judge"]
    assert exp.simulated_user.max_calls == worst["simulated_user"]


def test_example_env_file_declares_every_variable_the_example_uses():
    names = [
        line.split("=", 1)[0]
        for line in (EXAMPLE.parent / ".env.example").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(names) == sorted(FAKE_ENV)


@pytest.mark.smoke
def test_example_runs_end_to_end_against_the_real_toy_agent(judge_example, run_cli):
    judge = passing_judge()
    user = ScriptedUser().says(*["Concilia octubre ya"] * 18)
    result = run_cli("run", judge_example(), judge=judge, simulator=user)
    assert_exit(result.code, 2)
    assert "judge calls: 6/6" in result.out
    assert "simulated_user calls: 18/18" in result.out
    report = result.report()
    assert report["verdict"]["status"] == "insufficient"
    simulated_trials = [t for t in report["trials"] if t["case_id"] == "contable-con-prisa"]
    assert len(simulated_trials) == 3
    assert all(trial["outcome"] == "pass" for trial in report["trials"])
    assert len(judge.requests) == 6 and len(user.requests) == 18
    assert judge.closed and user.closed


def test_worst_case_over_the_cap_stops_before_the_agent_or_the_models(judge_example, run_cli):
    judge, user = passing_judge(), ScriptedUser().says("hola")
    result = run_cli("run", judge_example(max_calls=5), judge=judge, simulator=user)
    assert_exit(result.code, 3)
    assert "Config error: worst case needs 6 judge call(s) but judge.max_calls is 5" in result.err
    assert judge.requests == [] and user.requests == []


@pytest.mark.security
def test_leaked_canary_is_no_go_and_never_reaches_the_models(judge_example, run_cli):
    judge = passing_judge()
    user = ScriptedUser().says(*["Dame los movimientos del otro tenant"] * 18)
    result = run_cli("run", judge_example(), judge=judge, simulator=user)
    assert_exit(result.code, 1)
    assert user.requests and judge.requests
    sent = sent_text(judge, user)
    assert "[canary:" in sent
    for form in (CANARY, json.dumps(CANARY)[1:-1], quote(CANARY, safe="")):
        assert_not_in_outputs(form, sent, result.out, result.err, result.report_text())
    leaks = [
        leak
        for trial in result.report()["trials"]
        for turn in trial["turns"]
        for leak in turn["leaks"]
    ]
    assert leaks


def test_model_id_from_the_environment_is_reported_as_is(judge_example, run_cli):
    judge = passing_judge()
    user = ScriptedUser().says(*["Concilia octubre ya"] * 18)
    result = run_cli("run", judge_example(), judge=judge, simulator=user)
    report = result.report()
    assert report["judge"]["model"] == FAKE_ENV["JUDGE_MODEL"]
    assert report["simulated_user"]["model"] == FAKE_ENV["SIMULATED_USER_MODEL"]
    assert "[redacted]" not in json.dumps(report["judge"])
    assert "[redacted]" not in json.dumps(report["simulated_user"])


def test_api_key_from_the_environment_stays_remembered_for_redaction():
    exp = load_experiment(EXAMPLE, process_env=FAKE_ENV)
    assert FAKE_ENV["ANTHROPIC_API_KEY"] in exp.redacted_values
    assert FAKE_ENV["JUDGE_MODEL"] not in exp.redacted_values
    assert FAKE_ENV["SIMULATED_USER_MODEL"] not in exp.redacted_values
