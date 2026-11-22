import json
from pathlib import Path

import pytest

from octoqai.expectations import JUDGE_PROMPT_VERSION, judge_prompt_sha256
from octoqai.experiment import Experiment
from octoqai.reports import build_report, write_json
from octoqai.running import run_experiment
from octoqai.running.simulated_user import (
    SIMULATOR_PROMPT_VERSION,
    simulator_prompt_sha256,
)
from octoqai.verdict import decide
from support import samples
from support.agents import FakeJsonAgent, FakeSseAgent
from support.assertions import assert_not_in_outputs
from support.builders import ExperimentBuilder, judge_check, simulated, turn
from support.hooks import RecordingHooks
from support.providers import DEFAULT_MODEL, FakeLlm, ScriptedUser

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def _run(answer, headers=None):
    builder = ExperimentBuilder().agent(headers=headers or {}).variant("base")
    builder.merge({"name": "conciliación demo"}).trials(2).threshold(0.5)
    exp = Experiment.model_validate(
        builder.case("c1", turn("¿Concilia?", {"contains": "ñ"})).build()
    )
    run = run_experiment(exp, FakeJsonAgent().answers(answer).adapter(exp.agent))
    return run, decide(run)


@pytest.mark.security
def test_report_never_contains_header_secret(tmp_path):
    run, verdict = _run("ñ ok", headers={"Authorization": "Bearer s3cret"})
    write_json(run, verdict, tmp_path / "r.json")
    assert_not_in_outputs("s3cret", (tmp_path / "r.json").read_text(encoding="utf-8"))


def test_report_preserves_spanish_text(tmp_path):
    write_json(*_run("conciliación ñ"), tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert "conciliación ñ" in text and "¿Concilia?" in text


@pytest.mark.smoke
def test_report_structure(tmp_path):
    write_json(*_run("ñ"), tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert data["tool"]["name"] == "octoqai"
    assert data["verdict"]["status"] in {"go", "no-go", "insufficient"}
    assert data["variants"][0]["ci95"][0] <= data["variants"][0]["ci95"][1]
    assert len(data["trials"]) == 2
    turn_data = data["trials"][0]["turns"][0]
    assert {"user", "answer", "tool_calls", "tokens", "latency_ms", "error", "checks"} <= set(
        turn_data
    )
    assert "headers" not in data["experiment"]
    assert data["environment"] is None and data["tags"] == []


def test_report_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "out" / "r.json"
    write_json(*_run("ñ"), target)
    assert target.is_file()


@pytest.mark.security
def test_report_redacts_canary_everywhere(tmp_path):
    canary = "TEN-B-0042"
    builder = (
        ExperimentBuilder().variant("base", note=canary).security(canaries=[{"value": canary}])
    )
    exp = Experiment.model_validate(
        builder.case("c1", turn(f"¿Conoces {canary}?", {"contains": "."})).build()
    )
    run = run_experiment(exp, FakeJsonAgent().answers(f"sí, {canary}.").adapter(exp.agent))
    write_json(run, decide(run), tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert canary.casefold() not in text.casefold() and "[canary:canary-1]" in text


@pytest.mark.security
def test_report_counts_fragments_and_lists_masked_leaks():
    builder = ExperimentBuilder().streaming().security(canaries=[{"value": "TEN-B-0042"}])
    exp = Experiment.model_validate(
        builder.variant("base").trials(1).case("c1", turn("Hola", {"contains": "."})).build()
    )
    run = run_experiment(exp, FakeSseAgent().leaks("TEN-B-0042").adapter(exp.agent))
    report = build_report(run, decide(run))
    turn_data = report["trials"][0]["turns"][0]
    assert turn_data["fragments"] == 3 and turn_data["retries"] == []
    assert turn_data["ttft_ms"] is not None
    assert turn_data["leaks"] == [
        {"kind": "canary", "label": "canary-1", "location": "fragment 2", "masked": "T…"}
    ]
    assert report["trials"][0]["leaks"] == 1 and report["variants"][0]["leaks"] == 1
    assert "TEN-B-0042" not in str(report)


def _secure_run(answer, security, response=None):
    builder = ExperimentBuilder().variant("base").trials(1).security(**security)
    if response:
        builder.agent(response=response)
    exp = Experiment.model_validate(builder.case("c1", turn("Hola", {"contains": "."})).build())
    run = run_experiment(exp, FakeJsonAgent().answers(answer).adapter(exp.agent))
    return run, decide(run)


@pytest.mark.security
def test_report_masks_pii_and_secrets_even_when_detectors_are_disabled(tmp_path):
    key = samples.SECRET_SAMPLES["llm api key"]
    run, verdict = _secure_run(f"ñ {samples.EMAIL} {key}.", {"detectors": []})
    write_json(run, verdict, tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert samples.EMAIL not in text and key not in text
    assert "[pii:email]" in text and "[secret:llm api key]" in text


@pytest.mark.security
def test_report_masks_allowed_values_that_are_not_leaks(tmp_path):
    key = samples.SECRET_SAMPLES["llm api key"]
    answer = f"ñ {samples.EMAIL} {samples.DATABASE_URL} {key}."
    run, verdict = _secure_run(answer, {"allow": [".*"]})
    write_json(run, verdict, tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert samples.EMAIL not in text and "hunter2" not in text and key not in text
    assert json.loads(text)["trials"][0]["leaks"] == 0


@pytest.mark.security
def test_report_masks_canary_in_error_and_check_messages(tmp_path):
    canary = "TEN_B_0042"
    security = {"canaries": [{"value": canary}]}
    run, verdict = _secure_run("x", security, response={"answer": f"$.{canary}"})
    write_json(run, verdict, tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    turn_data = data["trials"][0]["turns"][0]
    assert canary not in str(data)
    assert "[canary:canary-1]" in turn_data["error"]
    assert "[canary:canary-1]" in turn_data["checks"][0]["detail"]


@pytest.mark.security
def test_report_masks_a_canary_hidden_by_json_escapes(tmp_path):
    canary = samples.NON_ASCII_CANARY
    run, verdict = _secure_run(json.dumps({"dbg": canary}), {"canaries": [{"value": canary}]})
    write_json(run, verdict, tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert "contrase" not in text and "[canary:canary-1]" in data["trials"][0]["turns"][0]["answer"]
    assert data["trials"][0]["leaks"] == 1


def test_report_lists_hooks_database_checks_and_counters(sqlite_db, sql_database, fake_agent):
    check = {"id": "total", "query": "SELECT count(*) FROM ajustes", "expect": {"value": 0}}
    builder = ExperimentBuilder().database(url=sqlite_db.url).trials(1)
    exp = Experiment.model_validate(
        builder.case("c1", turn("Hola", {"contains": "."}), db=[check]).build()
    )
    run = run_experiment(
        exp, fake_agent.answers("ok.").adapter(exp.agent), db=sql_database, hooks=RecordingHooks()
    )
    report = build_report(run, decide(run))
    trial = report["trials"][0]
    assert report["database"] is True
    assert [h["stage"] for h in report["hooks"]] == ["before_run", "after_run"]
    assert [h["stage"] for h in trial["hooks"]] == ["before_trial", "after_trial"]
    assert trial["db_checks"][0]["detail"] == "db 'total': value matches"
    assert trial["forbidden_crud"] == 0 and trial["setup_error"] is None
    assert report["variants"][0]["forbidden_crud"] == 0
    assert sqlite_db.url not in str(report)


GOLDEN = Path(__file__).parent / "golden_without_judge.json"
VOLATILE = {"started_at", "finished_at", "latency_ms", "ttft_ms", "mean_latency_ms", "mean_ttft_ms"}
ADDITIVE_TOP = {"judge", "simulated_user"}
ADDITIVE_TRIAL = {"stop_reason", "simulator_error"}
CASE_LEVEL_JUDGE = "judge_checks"
ADDITIVE_VARIANT = {"judge_not_evaluable", "simulator_errors"}


def _normalize(node):
    if isinstance(node, dict):
        normalized = {k: (None if k in VOLATILE else _normalize(v)) for k, v in node.items()}
        if isinstance(node.get("tool"), dict):
            normalized["tool"] = {**node["tool"], "version": None}
        return normalized
    if isinstance(node, list):
        return [_normalize(v) for v in node]
    return node


def _reference_report():
    builder = ExperimentBuilder().agent().variant("base").trials(2).threshold(0.5)
    exp = Experiment.model_validate(
        builder.case("c1", turn("Hola", {"contains": "ñ"}))
        .case("c2", turn("Adiós", {"contains": "ok"}))
        .build()
    )
    run = run_experiment(exp, FakeJsonAgent().answers("ñ ok").adapter(exp.agent))
    return _normalize(json.loads(json.dumps(build_report(run, decide(run)))))


def test_report_without_judge_matches_golden_except_additive_keys():
    report = _reference_report()
    assert {k: report[k] for k in ADDITIVE_TOP} == {"judge": None, "simulated_user": None}
    for trial in report["trials"]:
        assert {k: trial[k] for k in ADDITIVE_TRIAL} == {
            "stop_reason": None,
            "simulator_error": None,
        }
    for variant in report["variants"]:
        assert {k: variant[k] for k in ADDITIVE_VARIANT} == {
            "judge_not_evaluable": 0,
            "simulator_errors": 0,
        }
    assert all(trial[CASE_LEVEL_JUDGE] == [] for trial in report["trials"])
    stripped = {k: v for k, v in report.items() if k not in ADDITIVE_TOP}
    stripped["trials"] = [
        {k: v for k, v in t.items() if k != CASE_LEVEL_JUDGE} for t in report["trials"]
    ]
    stripped["variants"] = [
        {k: v for k, v in e.items() if k not in ADDITIVE_VARIANT} for e in report["variants"]
    ]
    golden = _normalize(json.loads(GOLDEN.read_text(encoding="utf-8")))
    assert stripped == golden


def _judged_run(llm, judge_settings=None, security=None, simulated_llm=None, **checks):
    builder = ExperimentBuilder().variant("base").trials(1)
    builder.judge(model=DEFAULT_MODEL, **(judge_settings or {}))
    if security:
        builder.security(**security)
    exp = Experiment.model_validate(
        builder.case("c1", turn("Hola", judge=[judge_check("polite")])).build()
    )
    run = run_experiment(exp, FakeJsonAgent().answers("ok").adapter(exp.agent), judge=llm)
    return run, decide(run)


def test_report_lists_judge_block_with_stats_and_results():
    llm = FakeLlm().judges("pass", "fine").uses(11, 5)
    run, verdict = _judged_run(llm, {"effort": "medium", "max_tokens": 512, "max_calls": 4})
    report = build_report(run, verdict)
    assert report["simulated_user"] is None
    assert report["judge"] == {
        "provider": "ollama",
        "model": DEFAULT_MODEL,
        "served_models": {DEFAULT_MODEL: 1},
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": judge_prompt_sha256(),
        "effort": "medium",
        "max_tokens": 512,
        "max_calls": 4,
        "calls": 1,
        "errors_by_class": {},
        "input_tokens": 11,
        "output_tokens": 5,
    }
    check = report["trials"][0]["turns"][0]["checks"][0]
    assert check["verdict"] == "pass" and check["rationale"] == "fine"
    assert check["check_id"] == "polite" and check["rubric_sha256"]


def test_report_judge_block_for_a_role_that_never_ran_is_zeroed():
    run, verdict = _judged_run(FakeLlm().judges("pass"))
    run.llm.clear()
    block = build_report(run, verdict)["judge"]
    assert block["calls"] == 0 and block["served_models"] == {} and block["errors_by_class"] == {}
    assert block["input_tokens"] is None and block["output_tokens"] is None


def test_report_lists_simulated_user_block():
    builder = ExperimentBuilder().variant("base").trials(1).simulated_user(max_calls=5)
    exp = Experiment.model_validate(
        builder.case("c1", simulated=simulated(max_turns=2, each_turn=[{"contains": "."}])).build()
    )
    run = run_experiment(
        exp,
        FakeJsonAgent().answers("ok.").adapter(exp.agent),
        simulator=ScriptedUser().says("hola", "otra"),
    )
    report = build_report(run, decide(run))
    block = report["simulated_user"]
    assert block["prompt_version"] == SIMULATOR_PROMPT_VERSION
    assert block["prompt_sha256"] == simulator_prompt_sha256()
    assert block["max_calls"] == 5 and report["judge"] is None


@pytest.mark.security
def test_report_judge_block_never_holds_key_or_timeout(tmp_path):
    settings = {
        "provider": "anthropic",
        "api_key": samples.ANTHROPIC_KEY,
        "allow_remote": True,
        "timeout_s": 77.5,
    }
    run, verdict = _judged_run(FakeLlm().judges("pass"), settings)
    write_json(run, verdict, tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    block = json.loads(text)["judge"]
    assert samples.ANTHROPIC_KEY not in text and "77.5" not in text
    assert not {"api_key", "timeout_s", "timeout", "max_retries", "allow_remote"} & set(block)
    assert "base_url" not in block


@pytest.mark.security
def test_report_redacts_a_canary_in_the_judge_rationale(tmp_path):
    canary = "TEN-B-0042"
    llm = FakeLlm().judges("fail", f"the agent revealed {canary}")
    run, verdict = _judged_run(llm, security={"canaries": [{"value": canary}]})
    write_json(run, verdict, tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    check = json.loads(text)["trials"][0]["turns"][0]["checks"][0]
    assert canary not in text and "[canary:canary-1]" in check["rationale"]
    assert check["verdict"] == "fail"


@pytest.mark.security
def test_report_keeps_only_closed_values_for_verdict_and_stop_reason():
    run, verdict = _judged_run(FakeLlm().judges("pass"))
    run.trials[0].turns[0].checks[0].verdict = samples.ANTHROPIC_KEY
    run.trials[0].stop_reason = "done"
    report = build_report(run, verdict)
    assert report["trials"][0]["turns"][0]["checks"][0]["verdict"] != samples.ANTHROPIC_KEY
    assert samples.ANTHROPIC_KEY not in str(report)
    assert report["trials"][0]["stop_reason"] == "done"


def test_report_includes_ollama_base_url():
    run, verdict = _judged_run(FakeLlm().judges("pass"), {"base_url": "http://127.0.0.1:11434"})
    assert build_report(run, verdict)["judge"]["base_url"] == "http://127.0.0.1:11434"


@pytest.mark.security
def test_report_redacts_a_secret_in_the_ollama_base_url_userinfo():
    run, verdict = _judged_run(
        FakeLlm().judges("pass"), {"base_url": samples.LOOPBACK_URL_WITH_USERINFO}
    )
    report = build_report(run, verdict)
    assert samples.DB_PASSWORD not in json.dumps(report)
    assert "base_url" in report["judge"]


def _redacting_pass(case_judge):
    builder = ExperimentBuilder().variant("base").trials(1)
    builder.judge(model=DEFAULT_MODEL)
    if case_judge:
        builder.case("c1", turn("Hola", {"contains": "pass"}), judge=[judge_check("polite")])
    else:
        builder.case("c1", turn("Hola", judge=[judge_check("polite")]))
    exp = Experiment.model_validate(builder.build())
    exp = exp.model_copy(update={"redacted_values": ["pass"]})
    agent = FakeJsonAgent().answers("pass").adapter(exp.agent)
    run = run_experiment(exp, agent, judge=FakeLlm().judges("pass"))
    return build_report(run, decide(run))


@pytest.mark.security
def test_case_level_judge_verdict_is_restored_but_the_agent_answer_stays_redacted():
    trial = _redacting_pass(case_judge=True)["trials"][0]
    assert [check["verdict"] for check in trial["judge_checks"]] == ["pass"]
    assert trial["turns"][0]["answer"] != "pass"


@pytest.mark.security
def test_turn_judge_verdict_is_restored_but_the_agent_answer_stays_redacted():
    trial = _redacting_pass(case_judge=False)["trials"][0]
    assert [check["verdict"] for check in trial["turns"][0]["checks"]] == ["pass"]
    assert trial["turns"][0]["answer"] != "pass"
