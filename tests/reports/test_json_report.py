import json

import pytest

from octoqai.experiment import Experiment
from octoqai.reports import build_report, write_json
from octoqai.running import run_experiment
from octoqai.verdict import decide
from support import samples
from support.agents import FakeJsonAgent, FakeSseAgent
from support.assertions import assert_not_in_outputs
from support.builders import ExperimentBuilder, turn
from support.hooks import RecordingHooks

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
