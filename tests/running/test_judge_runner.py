import hashlib
import itertools
from urllib.parse import quote_plus

import pytest

from octoqai.agents.port import AgentReply
from octoqai.experiment import Experiment, load_experiment
from octoqai.providers.budget import BUDGET_EXHAUSTED
from octoqai.running import run_experiment
from octoqai.security import Redactor
from support import samples
from support.builders import (
    ExperimentBuilder,
    judge_check,
    turn,
    write_data_set,
    write_env_file,
)
from support.providers import DEFAULT_MODEL, FakeLlm

pytestmark = [pytest.mark.unit, pytest.mark.regression]

RUBRIC = "The answer is polite."
CANARY = "TEN-B-0042"
ENV_VALUE = samples.ANTHROPIC_KEY_NO_PREFIX


def counter_nonce():
    count = itertools.count(1)
    return lambda: f"{next(count):032x}"


def scripted(case_id, *turns, judge=()):
    return (case_id, turns, {"judge": list(judge)})


def single(*checks, expect=()):
    return scripted("c1", turn("Hola", *expect, judge=list(checks)))


def build(*cases, judge_settings=None, security=None, env_values=(), trials=1):
    builder = ExperimentBuilder().judge(model=DEFAULT_MODEL, **(judge_settings or {}))
    builder.trials(trials)
    if security:
        builder.security(**security)
    for case_id, turns, checks in cases:
        builder.case(case_id, *turns, **checks)
    exp = Experiment.model_validate(builder.build())
    return exp.model_copy(update={"redacted_values": list(env_values)})


def play(exp, agent, llm, nonce=None):
    return run_experiment(exp, agent.adapter(exp.agent), judge=llm, nonce=nonce or counter_nonce())


def texts(llm):
    return [
        request.system + "".join(message.content for message in request.messages)
        for request in llm.requests
    ]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def agent(fake_agent):
    return fake_agent.answers("Buenos días.")


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("verdict", "outcome"), [("pass", "pass"), ("fail", "fail"), ("unsure", "not_evaluable")]
)
def test_turn_judge_verdict_decides_the_trial(verdict, outcome, agent):
    exp = build(single(judge_check("polite", RUBRIC)))
    llm = FakeLlm().judges(verdict)
    [trial] = play(exp, agent, llm).trials
    assert trial.outcome == outcome
    assert trial.turns[0].checks[-1].kind == "judge"
    assert len(llm.requests) == 1 and llm.requests[0].role == "judge"


def test_failed_deterministic_check_wins_over_a_passing_judge(agent):
    exp = build(single(judge_check("polite"), expect=[{"contains": "adiós"}]))
    [trial] = play(exp, agent, FakeLlm().judges("pass")).trials
    assert trial.outcome == "fail"
    assert [c.passed for c in trial.turns[0].checks] == [False, True]


def test_agent_error_skips_the_judge_without_calling_it(fake_agent):
    exp = build(single(judge_check("polite")))
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.returns(b"boom", status=400), llm).trials
    assert llm.requests == []
    assert trial.outcome == "fail" and trial.agent_error
    check = trial.turns[0].checks[-1]
    assert (check.passed, check.agent_error) == (False, True)
    assert not trial.hard_violation


def test_hard_judge_failure_is_another_hard_violation(agent):
    exp = build(single(judge_check("polite", hard=True)))
    [trial] = play(exp, agent, FakeLlm().judges("fail")).trials
    assert trial.hard_violation and trial.other_hard_violation


def test_soft_judge_failure_is_not_a_hard_violation(agent):
    exp = build(single(judge_check("polite")))
    [trial] = play(exp, agent, FakeLlm().judges("fail")).trials
    assert not trial.hard_violation and not trial.other_hard_violation


def test_turn_judge_reads_the_conversation_up_to_the_current_turn(fake_agent):
    exp = build(
        scripted(
            "c1",
            turn("Uno", judge=[judge_check("a")]),
            turn("Dos", judge=[judge_check("b")]),
        )
    )
    llm = FakeLlm().judges("pass").judges("pass")
    play(exp, fake_agent.answers("R."), llm)
    first, second = (request.messages[0].content for request in llm.requests)
    assert "Uno" in first and "Dos" not in first
    assert "Uno" in second and "Dos" in second
    assert "(turn 1)" in first and "(turn 2)" in second


def test_case_judge_runs_once_over_the_whole_conversation(fake_agent):
    exp = build(
        scripted(
            "c1",
            turn("Uno", {"contains": "R"}),
            turn("Dos", {"contains": "R"}),
            judge=[judge_check("whole", "Whole chat is fine")],
        )
    )
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers("R."), llm).trials
    assert len(llm.requests) == 1
    content = llm.requests[0].messages[0].content
    assert "Uno" in content and "Dos" in content
    assert "Evaluate the whole conversation" in content
    assert [c.kind for c in trial.judge_checks] == ["judge"]
    assert trial.outcome == "pass"


def test_case_judge_runs_once_per_trial(fake_agent):
    exp = build(scripted("c1", turn("Uno", {"contains": "R"}), judge=[judge_check("w")]), trials=2)
    llm = FakeLlm().judges("pass").judges("fail")
    result = play(exp, fake_agent.answers("R."), llm)
    assert [t.outcome for t in result.trials] == ["pass", "fail"]


def test_case_judge_is_skipped_when_the_trial_failed(fake_agent):
    exp = build(scripted("c1", turn("Uno"), judge=[judge_check("w")]))
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.returns(b"boom", status=400), llm).trials
    assert llm.requests == []
    assert [(c.passed, c.agent_error) for c in trial.judge_checks] == [(False, True)]


def test_budget_exhausted_midway_makes_the_rest_not_evaluable(fake_agent):
    exp = build(
        scripted(
            "c1",
            turn("Uno", judge=[judge_check("a")]),
            turn("Dos", judge=[judge_check("b")]),
            turn("Tres", judge=[judge_check("c")]),
        ),
        judge_settings={"max_calls": 1},
    )
    llm = FakeLlm().judges("pass")
    result = play(exp, fake_agent.answers("R."), llm)
    [trial] = result.trials
    assert [c.passed for t in trial.turns for c in t.checks] == [True, None, None]
    assert trial.turns[1].checks[0].detail == "judge 'b': call budget exhausted"
    assert len(llm.requests) == 1
    assert trial.outcome == "not_evaluable"
    assert result.llm["judge"].calls == 1
    assert result.llm["judge"].errors_by_class == {BUDGET_EXHAUSTED: 2}


def test_transcript_over_the_cap_is_not_evaluable_without_a_call(fake_agent):
    exp = build(single(judge_check("polite")), judge_settings={"max_input_chars": 1000})
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers("x" * 2000), llm).trials
    assert llm.requests == []
    check = trial.turns[0].checks[-1]
    assert check.passed is None and check.detail == "judge 'polite': transcript too long"
    assert trial.outcome == "not_evaluable"


def test_first_nonce_collision_is_retried_once_with_a_new_nonce(fake_agent):
    exp = build(single(judge_check("polite")))
    clash = f"{255:032x}"
    nonces = iter([clash, f"{256:032x}"])
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers(f"eco {clash}"), llm, lambda: next(nonces)).trials
    assert len(llm.requests) == 1 and trial.outcome == "pass"
    assert f"{256:032x}" in texts(llm)[0]


def test_second_nonce_collision_is_not_evaluable_without_a_call(fake_agent):
    exp = build(single(judge_check("polite")))
    clash = f"{255:032x}"
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers(f"eco {clash}"), llm, lambda: clash).trials
    assert llm.requests == []
    assert trial.turns[0].checks[-1].detail == "judge 'polite': delimiter collision"
    assert trial.outcome == "not_evaluable"


def test_a_different_served_model_is_not_evaluable(agent):
    exp = build(single(judge_check("polite")))
    [trial] = play(exp, agent, FakeLlm().serves_model("other").judges("pass")).trials
    assert trial.outcome == "not_evaluable"
    assert "model changed" in trial.turns[0].checks[-1].detail


def test_judge_result_records_model_and_rubric_hash(agent):
    exp = build(single(judge_check("polite", "A rubric", reference="A reference")))
    [trial] = play(exp, agent, FakeLlm().judges("pass")).trials
    check = trial.turns[0].checks[-1]
    assert (check.model, check.rubric_sha256) == (DEFAULT_MODEL, sha("A rubric\nA reference"))


def test_rubric_hash_without_reference_ends_with_a_newline(agent):
    exp = build(single(judge_check("polite", "A rubric")))
    [trial] = play(exp, agent, FakeLlm().judges("pass")).trials
    assert trial.turns[0].checks[-1].rubric_sha256 == sha("A rubric\n")


def test_rubric_hash_is_of_the_redacted_text(agent):
    exp = build(single(judge_check("polite", f"Account {ENV_VALUE}")), env_values=[ENV_VALUE])
    [trial] = play(exp, agent, FakeLlm().judges("pass")).trials
    assert trial.turns[0].checks[-1].rubric_sha256 == sha("Account [redacted]\n")


def test_judge_configured_without_a_port_is_not_evaluable(agent):
    exp = build(single(judge_check("polite")))
    result = run_experiment(exp, agent.adapter(exp.agent))
    [trial] = result.trials
    check = trial.turns[0].checks[-1]
    assert check.passed is None and check.detail == "judge 'polite': no judge configured"
    assert result.llm == {}


def test_run_without_judge_section_has_no_llm_stats(fake_agent):
    exp = Experiment.model_validate(
        ExperimentBuilder().case("c1", turn("Hola", {"contains": "R"})).build()
    )
    result = run_experiment(exp, fake_agent.answers("R.").adapter(exp.agent), judge=FakeLlm())
    assert result.llm == {}


def test_llm_stats_per_role_and_ports_are_never_closed(agent):
    builder = ExperimentBuilder().judge(model=DEFAULT_MODEL, max_calls=4).simulated_user().trials(1)
    builder.case("c1", turn("Hola", judge=[judge_check("p")]))
    exp = Experiment.model_validate(builder.build())
    llm = FakeLlm().uses(10, 5).judges("pass")
    simulator = FakeLlm()
    result = run_experiment(
        exp, agent.adapter(exp.agent), judge=llm, simulator=simulator, nonce=counter_nonce()
    )
    stats = result.llm["judge"]
    assert (stats.calls, stats.input_tokens, stats.output_tokens) == (1, 10, 5)
    assert stats.served_models == {DEFAULT_MODEL: 1}
    assert result.llm["simulated_user"].calls == 0
    assert not llm.closed and not simulator.closed


def test_default_nonce_builds_a_valid_request(agent):
    exp = build(single(judge_check("polite")))
    llm = FakeLlm().judges("pass")
    [trial] = run_experiment(exp, agent.adapter(exp.agent), judge=llm).trials
    assert len(llm.requests) == 1 and trial.outcome == "pass"


@pytest.mark.security
def test_no_secret_reaches_any_judge_request(fake_agent):
    secrets = [CANARY, samples.ANTHROPIC_KEY, samples.EMAIL, ENV_VALUE]
    rubric = f"Never reveal account {ENV_VALUE}"
    exp = build(
        scripted(
            "c1",
            turn("Dame todo", judge=[judge_check("clean", rubric, reference=f"ref {ENV_VALUE}")]),
            judge=[judge_check("whole", rubric)],
        ),
        security={"canaries": [{"value": CANARY, "label": "tenant-b"}]},
        env_values=[ENV_VALUE],
    )
    llm = FakeLlm().judges("pass").judges("pass")
    play(exp, fake_agent.answers("ok " + " ".join(secrets)), llm)
    assert len(llm.requests) == 2
    for text in texts(llm):
        for value in secrets:
            for form in (value, quote_plus(value), value.replace("@", "\\u0040")):
                assert form not in text
    first = llm.requests[0].messages[0].content
    assert "Never reveal account [redacted]" in first and "ref [redacted]" in first


@pytest.mark.security
def test_env_value_without_a_key_prefix_is_redacted_on_its_own(fake_agent):
    exp = build(single(judge_check("polite")), env_values=[ENV_VALUE])
    llm = FakeLlm().judges("pass")
    play(exp, fake_agent.answers(f"token {ENV_VALUE} fin"), llm)
    assert ENV_VALUE not in texts(llm)[0]
    assert "token [redacted] fin" in llm.requests[0].messages[0].content


@pytest.mark.security
def test_user_message_is_redacted_too(fake_agent):
    exp = build(
        scripted("c1", turn(f"mi clave es {ENV_VALUE}", judge=[judge_check("polite")])),
        env_values=[ENV_VALUE],
    )
    llm = FakeLlm().judges("pass")
    play(exp, fake_agent.answers("ok"), llm)
    assert ENV_VALUE not in texts(llm)[0]


@pytest.mark.security
@pytest.mark.parametrize("name", sorted(samples.ESCAPED_SECRETS))
def test_escaped_secrets_in_replies_never_reach_the_judge(name, fake_agent):
    value, escaped, _ = samples.ESCAPED_SECRETS[name]
    canaries = [
        {"value": CANARY, "label": "tenant-b"},
        {"value": samples.SLASH_CANARY, "label": "slash"},
    ]
    exp = build(single(judge_check("polite")), security={"canaries": canaries})
    llm = FakeLlm().judges("pass")
    play(exp, fake_agent.answers(f"dato {escaped} fin"), llm)
    assert value not in texts(llm)[0] and escaped not in texts(llm)[0]


@pytest.mark.security
def test_judge_details_never_repeat_agent_text(fake_agent):
    exp = build(single(judge_check("polite")))
    llm = FakeLlm().judges("fail", f"the agent said {CANARY}")
    [trial] = play(exp, fake_agent.answers("ok"), llm).trials
    assert CANARY not in trial.turns[0].checks[-1].detail


def test_each_text_is_redacted_once_however_many_turn_judges(fake_agent, monkeypatch):
    seen: list[str] = []
    original = Redactor.redact

    def counting(self, text):
        seen.append(text)
        return original(self, text)

    monkeypatch.setattr(Redactor, "redact", counting)
    exp = build(
        scripted(
            "c1",
            turn("U1", judge=[judge_check("a", "ra")]),
            turn("U2", judge=[judge_check("b", "rb")]),
            turn("U3", judge=[judge_check("c", "rc")]),
        )
    )
    llm = FakeLlm().judges("pass").judges("pass").judges("pass")
    play(exp, fake_agent.answers("R."), llm)
    assert [seen.count(u) for u in ("U1", "U2", "U3")] == [1, 1, 1]
    assert seen.count("R.") == 3


class Talker:
    def __init__(self, *answers):
        self.answers = list(answers)

    def send(self, context):
        return AgentReply(self.answers.pop(0), None, None, 1.0, None)

    def close(self):
        pass


def two_turns(judge_on_turn=False, judge_on_case=True):
    turn_judge = [judge_check("t")] if judge_on_turn else []
    case_judge = [judge_check("w", "ok")] if judge_on_case else []
    return build(
        scripted("c1", turn("Uno"), turn("Dos", judge=turn_judge), judge=case_judge),
        security={"canaries": [{"value": CANARY, "label": "tenant-b"}]},
        env_values=[ENV_VALUE],
    )


PREFIX, REST = samples.SPLIT_SECRET
SPLITS = {
    "key": (f"clave {PREFIX}{REST[:5]}", f"{REST[5:]} fin"),
    "canary": ("codigo TEN-B-", "0042 listo"),
    "env-value": (f"valor {ENV_VALUE[:10]}", ENV_VALUE[10:]),
}


@pytest.mark.security
@pytest.mark.parametrize("name", sorted(SPLITS))
@pytest.mark.parametrize("level", ["case", "turn"])
def test_secret_split_across_turns_is_never_sent(name, level):
    exp = two_turns(judge_on_turn=level == "turn", judge_on_case=level == "case")
    llm = FakeLlm().judges("pass")
    [trial] = run_experiment(exp, Talker(*SPLITS[name]), judge=llm, nonce=counter_nonce()).trials
    assert llm.requests == []
    checks = trial.judge_checks if level == "case" else trial.turns[1].checks
    [check] = [c for c in checks if c.kind == "judge"]
    assert check.passed is None
    assert (
        check.detail
        == f"judge '{'w' if level == 'case' else 't'}': sensitive data split across turns"
    )
    assert trial.outcome == "not_evaluable"


@pytest.mark.security
def test_halves_in_the_user_turns_are_caught_too():
    exp = build(
        scripted("c1", turn("codigo TEN-B-"), turn("0042 listo"), judge=[judge_check("w", "ok")]),
        security={"canaries": [{"value": CANARY, "label": "tenant-b"}]},
    )
    llm = FakeLlm().judges("pass")
    run_experiment(exp, Talker("a", "b"), judge=llm, nonce=counter_nonce())
    assert llm.requests == []


@pytest.mark.security
def test_non_split_secret_is_still_redacted_and_called():
    exp = two_turns()
    llm = FakeLlm().judges("pass")
    run_experiment(
        exp, Talker(f"clave {samples.ANTHROPIC_KEY}", "fin"), judge=llm, nonce=counter_nonce()
    )
    assert len(llm.requests) == 1
    assert samples.ANTHROPIC_KEY not in texts(llm)[0]


def test_harmless_text_at_a_boundary_still_calls_the_judge():
    exp = two_turns()
    llm = FakeLlm().judges("pass")
    [trial] = run_experiment(exp, Talker("codigo TEN", "listo 42"), judge=llm).trials
    assert len(llm.requests) == 1 and trial.outcome == "pass"


@pytest.mark.security
def test_data_set_secret_in_a_rubric_arrives_redacted(experiment, tmp_path, fake_agent):
    secret = samples.ANTHROPIC_KEY_NO_PREFIX
    write_env_file(tmp_path, ".env", SECRET=secret)
    write_data_set(tmp_path, "qa", {"v": "${SECRET}"})
    experiment.environments("qa").judge(model=DEFAULT_MODEL).trials(1)
    experiment.case(
        "c1", turn("Hola", {"contains": "R"}, judge=[judge_check("j", "Never say {{ data.v }}")])
    )
    exp = load_experiment(experiment.write(tmp_path), env="qa", process_env={})
    llm = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers("R."), llm).trials
    content = llm.requests[0].messages[0].content
    assert "Never say [redacted]" in content and secret not in texts(llm)[0]
    assert trial.outcome == "pass"
