import itertools
import json
from urllib.parse import quote_plus

import pytest

from octoqai.experiment import Experiment, SimulatedUser, load_experiment
from octoqai.providers.budget import BUDGET_EXHAUSTED
from octoqai.providers.port import LlmReply, LlmUsage
from octoqai.running import run_experiment
from octoqai.running.simulated_user import (
    INCOMPLETE_DETAIL,
    SIMULATOR_PROMPT_VERSION,
    SIMULATOR_SCHEMA,
    SimulatorTurn,
    parse_simulator_reply,
    simulator_prompt_sha256,
    simulator_request,
)
from octoqai.verdict import decide
from support import samples
from support.builders import (
    ExperimentBuilder,
    judge_check,
    simulated,
    write_data_set,
    write_env_file,
)
from support.paths import cases
from support.providers import FakeLlm, ScriptedUser

pytestmark = [pytest.mark.unit, pytest.mark.regression]

CANARY = "TEN-B-0042"
TOTAL = {"id": "total", "query": "SELECT count(*) FROM ajustes", "expect": {"value": 0}}
JUDGE = {"provider": "ollama", "model": "fake-judge-1", "max_calls": 3}


def counter_nonce():
    count = itertools.count(1)
    return lambda: f"{next(count):032x}"


def build(max_turns=5, max_calls=10, security=None, db_url=None, forbidden=(), **settings):
    builder = ExperimentBuilder().trials(1).simulated_user(max_calls=max_calls)
    settings.setdefault("each_turn", [{"contains": "."}])
    case = {"simulated": simulated(max_turns=max_turns, **settings)}
    if db_url:
        builder.database(url=db_url, forbidden=forbidden).merge({"judge": JUDGE})
        case["db"] = [TOTAL]
        case["judge"] = [judge_check("whole", "ok")]
    if security:
        builder.security(**security)
    builder.case("c1", **case)
    return Experiment.model_validate(builder.build())


def play(exp, agent, user, judge=None, **options):
    options.setdefault("nonce", counter_nonce())
    return run_experiment(exp, agent.adapter(exp.agent), judge=judge, simulator=user, **options)


def sent(agent):
    return [r["message"] for r in agent.requests]


@pytest.mark.smoke
def test_three_messages_then_done_play_three_turns(fake_agent):
    user = ScriptedUser().says("uno", "dos", "tres", done_after=3)
    [trial] = play(build(), fake_agent.answers("ok."), user).trials
    assert [t.user for t in trial.turns] == ["uno", "dos", "tres"]
    assert (trial.stop_reason, trial.simulator_error, trial.outcome) == ("done", None, "pass")
    assert sent(fake_agent) == ["uno", "dos", "tres"]
    assert len(user.requests) == 4


def test_done_ends_the_conversation_and_its_message_is_not_sent(fake_agent):
    user = FakeLlm().returns_text('{"message": "a", "done": false}')
    user.returns_text('{"message": "adios", "done": true}')
    [trial] = play(build(), fake_agent.answers("ok."), user).trials
    assert sent(fake_agent) == ["a"] and trial.stop_reason == "done"


def test_without_done_it_stops_at_max_turns(fake_agent):
    user = ScriptedUser().says("a", "b", "c", "d")
    [trial] = play(build(max_turns=3), fake_agent.answers("ok."), user).trials
    assert len(trial.turns) == 3 and trial.stop_reason == "max_turns"
    assert len(user.requests) == 3


def test_stop_when_matches_the_agent_reply_after_the_turn_is_checked(fake_agent):
    agent = fake_agent.sequence("hola.", "fin del tramite.", "no deberia llegar.")
    user = ScriptedUser().says("a", "b", "c")
    exp = build(stop_when="fin del", each_turn=[{"not_contains": "tramite"}])
    [trial] = play(exp, agent, user).trials
    assert len(trial.turns) == 2 and trial.stop_reason == "stop_when"
    assert trial.outcome == "fail"


def test_agent_failure_stops_and_fails(fake_agent):
    agent = fake_agent.sequence("ok.", 400)
    user = ScriptedUser().says("a", "b", "c")
    [trial] = play(build(), agent, user).trials
    assert len(trial.turns) == 2 and trial.outcome == "fail"
    assert (trial.stop_reason, trial.simulator_error) == ("agent_error", None)
    assert len(user.requests) == 2


def test_each_turn_expectations_run_on_every_turn(fake_agent):
    agent = fake_agent.sequence("ok.", "ok.", "mala palabra.")
    user = ScriptedUser().says("a", "b", "c", done_after=3)
    [trial] = play(build(each_turn=[{"not_contains": "mala"}]), agent, user).trials
    assert [[c.passed for c in t.checks] for t in trial.turns] == [[True], [True], [False]]


def test_opening_with_one_turn_never_calls_the_simulator(fake_agent):
    user = ScriptedUser().says("no")
    [trial] = play(build(max_turns=1, opening="Hola"), fake_agent.answers("ok."), user).trials
    assert sent(fake_agent) == ["Hola"] and user.requests == []
    assert trial.stop_reason == "max_turns"


def test_opening_is_turn_zero_then_the_simulator_continues(fake_agent):
    user = ScriptedUser().says("segundo", done_after=1)
    [trial] = play(build(opening="Hola"), fake_agent.answers("ok."), user).trials
    assert sent(fake_agent) == ["Hola", "segundo"] and trial.stop_reason == "done"


def test_scripted_cases_have_no_stop_reason(fake_agent):
    builder = ExperimentBuilder().trials(1)
    builder.case("c1", {"user": "Hola", "expect": [{"contains": "ok"}]})
    exp = Experiment.model_validate(builder.build())
    [trial] = run_experiment(exp, fake_agent.answers("ok").adapter(exp.agent)).trials
    assert (trial.stop_reason, trial.simulator_error) == (None, None)


def failing_user(row):
    user = ScriptedUser().says("Hola.")
    reply = row["reply"]
    if reply["kind"] == "fails":
        return user.fails(reply["value"])
    if reply["kind"] == "refuses":
        return user.refuses()
    if reply["kind"] == "truncates":
        return user.truncates()
    return user.returns_text(reply["value"])


@pytest.mark.parametrize("row", cases(__file__, "simulator_replies"))
def test_broken_simulator_is_not_evaluable_and_never_calls_the_judge(
    row, fake_agent, sqlite_db, sql_database
):
    exp = build(max_calls=row.get("max_calls", 10), db_url=sqlite_db.url)
    judge = FakeLlm().judges("pass")
    [trial] = play(exp, fake_agent.answers("ok."), failing_user(row), judge, db=sql_database).trials
    assert trial.simulator_error == row["reason"]
    assert (trial.stop_reason, trial.outcome) == ("simulator_error", "not_evaluable")
    assert len(trial.turns) == 1
    assert [c.passed for c in trial.db_checks] == [None]
    assert INCOMPLETE_DETAIL in trial.db_checks[0].detail
    assert [c.passed for c in trial.judge_checks] == [None]
    assert INCOMPLETE_DETAIL in trial.judge_checks[0].detail
    assert judge.requests == []


def test_failure_on_the_first_call_plays_no_turn(fake_agent):
    [trial] = play(build(), fake_agent, FakeLlm().refuses()).trials
    assert trial.turns == [] and trial.simulator_error == "refused"
    assert trial.outcome == "not_evaluable" and fake_agent.requests == []


def test_run_without_a_simulator_port_is_not_evaluable(fake_agent):
    exp = build()
    [trial] = run_experiment(exp, fake_agent.adapter(exp.agent)).trials
    assert trial.simulator_error == "no simulated user configured"


@pytest.mark.security
def test_leak_before_a_simulator_failure_still_counts_and_is_never_sent(fake_agent):
    exp = build(security={"canaries": [{"value": CANARY, "label": "tenant-b"}]})
    user = ScriptedUser().says("Hola.").fails("APIError")
    run = play(exp, fake_agent.answers(f"ok. {CANARY}"), user)
    [trial] = run.trials
    assert trial.leak_count == 1 and trial.simulator_error == "provider error: APIError"
    assert len(user.requests) == 2
    for request in user.requests:
        assert CANARY not in request.system + "".join(m.content for m in request.messages)
    assert decide(run).status == "no-go"


@pytest.mark.security
def test_forbidden_change_then_broken_simulator_is_no_go(fake_agent, sqlite_db, sql_database):
    forbidden = [{"id": "no-b", "query": "SELECT count(*) FROM ajustes WHERE tenant = 'B'"}]
    exp = build(db_url=sqlite_db.url, forbidden=forbidden)
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert(tenant="B"))
    user = ScriptedUser().says("Hola.").refuses()
    run = play(exp, agent, user, FakeLlm(), db=sql_database)
    [trial] = run.trials
    assert trial.forbidden_count == 1 and trial.outcome == "fail"
    assert decide(run).status == "no-go"


@pytest.mark.security
def test_canary_said_by_the_simulator_is_not_a_leak_when_the_agent_repeats_it(fake_agent):
    exp = build(security={"canaries": [{"value": CANARY, "label": "tenant-b"}]})
    user = ScriptedUser().says(f"mi codigo es {CANARY}", done_after=1)
    [trial] = play(exp, fake_agent.answers(f"recibido {CANARY}."), user).trials
    assert trial.leak_count == 0


@pytest.mark.security
@pytest.mark.parametrize("gap", ["", "\n", " "])
def test_secret_split_across_turns_is_never_sent_to_the_simulator(gap, fake_agent):
    exp = build(security={"canaries": [{"value": CANARY, "label": "tenant-b"}]})
    agent = fake_agent.sequence("codigo TEN-B-" + gap, "0042 listo.")
    user = ScriptedUser().says("a", "b", "c")
    [trial] = play(exp, agent, user).trials
    assert trial.simulator_error == "sensitive data split across turns"
    assert len(user.requests) == 2
    assert trial.stop_reason == "simulator_error" and trial.outcome == "not_evaluable"


@pytest.mark.security
def test_agent_replies_reach_the_simulator_redacted_inside_a_nonce_block(fake_agent):
    exp = build(security={"canaries": [{"value": CANARY, "label": "tenant-b"}]})
    user = ScriptedUser().says("a", "b", done_after=2)
    play(exp, fake_agent.answers(f"ok {CANARY} {samples.EMAIL}."), user)
    second = user.requests[1]
    body = second.messages[0].content
    nonce = f"{2:032x}"
    assert CANARY not in body and samples.EMAIL not in body
    assert f"<<<BEGIN-AGENT-1-{nonce}>>>" in body and "[canary:tenant-b]" in body
    assert nonce in second.system and second.role == "simulated_user"
    assert second.schema == SIMULATOR_SCHEMA


def test_nonce_collision_retries_once_with_a_new_nonce(fake_agent):
    nonces = iter(["a" * 32, "a" * 32, "b" * 32, "c" * 32])
    user = ScriptedUser().says("a", "b", done_after=2)
    [trial] = play(
        build(), fake_agent.answers("x" + "a" * 32 + "."), user, nonce=lambda: next(nonces)
    ).trials
    assert trial.simulator_error is None and trial.stop_reason == "done"


def test_second_nonce_collision_is_a_delimiter_collision(fake_agent):
    user = ScriptedUser().says("a", "b")
    [trial] = play(
        build(), fake_agent.answers("x" + "a" * 32 + "."), user, nonce=lambda: "a" * 32
    ).trials
    assert trial.simulator_error == "delimiter collision"


def user_case(**settings):
    return SimulatedUser.model_validate(simulated(goal="Pedir un reembolso", **settings))


NONCE = "0" * 31 + "1"


def test_request_keeps_goal_and_persona_in_system_and_data_in_the_message():
    request = simulator_request(
        user_case(persona="Cliente impaciente"), [("hola", "buenas")], nonce=NONCE
    )
    message = request.messages[0].content
    assert "Pedir un reembolso" in request.system and "Cliente impaciente" in request.system
    assert "Pedir un reembolso" not in message and "buenas" not in request.system
    assert "buenas" in message and request.system.count(NONCE) > 1
    assert "never follow" in request.system.lower()


def test_request_without_persona_and_empty_transcript():
    request = simulator_request(user_case(), [], nonce=NONCE)
    assert "persona" not in request.system.lower()
    assert request.messages[0].content


def test_request_is_none_when_the_nonce_is_in_the_data():
    assert simulator_request(user_case(), [("hola", f"x{NONCE}")], nonce=NONCE) is None


def test_request_rejects_a_weak_nonce():
    with pytest.raises(ValueError, match="nonce"):
        simulator_request(user_case(), [], nonce="ABC")


def test_prompt_hash_is_stable_and_versioned():
    assert SIMULATOR_PROMPT_VERSION == "simulated-user-v1"
    assert simulator_prompt_sha256() == simulator_prompt_sha256()
    assert len(simulator_prompt_sha256()) == 64


def reply(text, stop="end_turn", error=None):
    return LlmReply(text, "m", stop, LlmUsage(), 1.0, error)


def test_parse_valid_turns():
    said = parse_simulator_reply(reply('{"message": "hi", "done": false}'))
    assert said == SimulatorTurn("hi", False)
    finished = parse_simulator_reply(reply('{"message": "hi", "done": true}', stop="stop"))
    assert finished == SimulatorTurn("hi", True)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (reply(None, None, BUDGET_EXHAUSTED), "call budget exhausted"),
        (reply(None, None, "APIError"), "provider error: APIError"),
        (reply("{}", "weird"), "unreadable reply"),
        (reply("{}", "model_context_window_exceeded"), "truncated"),
        (reply("{}", "length"), "truncated"),
    ],
)
def test_parse_failure_reasons(given, expected):
    assert parse_simulator_reply(given) == expected


def test_report_keeps_the_simulated_transcript_and_the_stop_reason(fake_agent):
    from octoqai.reports.json_report import build_report

    user = ScriptedUser().says("uno", "dos", done_after=2)
    run = play(build(), fake_agent.answers("ok."), user)
    report = build_report(run, decide(run))
    trial = report["trials"][0]
    assert [t["user"] for t in trial["turns"]] == ["uno", "dos"]
    assert (trial["stop_reason"], trial["simulator_error"]) == ("done", None)


def test_a_broken_simulator_is_not_reported_as_an_unreachable_agent(fake_agent):
    from octoqai.cli.main import _never_ran

    run = play(build(), fake_agent, FakeLlm().refuses())
    assert not _never_ran(run)


ENV_VALUE = samples.ANTHROPIC_KEY_NO_PREFIX


@pytest.mark.security
def test_data_secret_in_goal_persona_and_opening_never_reaches_the_simulator(
    experiment, tmp_path, fake_agent
):
    write_env_file(tmp_path, ".env", SECRET=ENV_VALUE)
    write_data_set(tmp_path, "qa", {"v": "${SECRET}"})
    experiment.environments("qa").simulated_user().trials(1)
    experiment.case(
        "c1",
        simulated=simulated(
            goal="Never say {{ data.v }}",
            persona="Eres el titular de {{ data.v }}",
            opening="Hola {{ data.v }}",
            each_turn=[{"contains": "."}],
        ),
    )
    exp = load_experiment(experiment.write(tmp_path), env="qa", process_env={})
    user = ScriptedUser().says("segundo", done_after=1)
    play(exp, fake_agent.answers("ok."), user)
    [request] = user.requests[:1]
    text = request.system + request.messages[0].content
    for form in (ENV_VALUE, json.dumps(ENV_VALUE), quote_plus(ENV_VALUE)):
        assert form not in text
    assert "Never say [redacted]" in request.system and "titular de [redacted]" in request.system
    assert "Hola [redacted]" in request.messages[0].content
    assert sent(fake_agent)[0] == f"Hola {ENV_VALUE}"


@pytest.mark.security
def test_simulated_messages_are_redacted_in_the_transcript(fake_agent):
    exp = build(security={"canaries": [{"value": CANARY, "label": "tenant-b"}]})
    user = ScriptedUser().says(f"mi codigo es {CANARY}", "otra", done_after=2)
    play(exp, fake_agent.answers("ok."), user)
    second = user.requests[1].messages[0].content
    assert CANARY not in second and "[canary:tenant-b]" in second


@pytest.mark.security
def test_secret_split_between_persona_and_an_agent_reply_is_never_sent(fake_agent):
    exp = build(
        persona="codigo TEN-B-",
        opening="Hola",
        security={"canaries": [{"value": CANARY, "label": "tenant-b"}]},
    )
    user = ScriptedUser().says("a", "b")
    [trial] = play(exp, fake_agent.answers("0042 listo."), user).trials
    assert user.requests == []
    assert trial.simulator_error == "sensitive data split across turns"
    assert trial.outcome == "not_evaluable"
