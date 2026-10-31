import json

import pytest

from octoqai.experiment import Experiment
from octoqai.reports import write_json
from octoqai.running import run_experiment
from octoqai.verdict import decide
from support.agents import FakeJsonAgent
from support.assertions import assert_not_in_outputs
from support.builders import ExperimentBuilder, turn

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


def test_report_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "out" / "r.json"
    write_json(*_run("ñ"), target)
    assert target.is_file()
