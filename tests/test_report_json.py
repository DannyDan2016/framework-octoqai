import json

import httpx

from octoqai.adapter import HttpAgentAdapter
from octoqai.config import Experiment
from octoqai.report_json import write_json
from octoqai.runner import run_experiment
from octoqai.verdict import decide


def _run(answer, headers=None):
    exp = Experiment.model_validate(
        {
            "name": "conciliación demo",
            "agent": {
                "url": "http://agent.local/chat",
                "headers": headers or {},
                "request": {"message": "{{ message }}"},
                "response": {"answer": "$.answer"},
            },
            "variants": [{"name": "base"}],
            "trials": 2,
            "thresholds": {"pass_rate": 0.5},
            "cases": [
                {"id": "c1", "turns": [{"user": "¿Concilia?", "expect": [{"contains": "ñ"}]}]}
            ],
        }
    )
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"answer": answer}))
    run = run_experiment(exp, HttpAgentAdapter(exp.agent, transport=transport))
    return run, decide(run)


def test_report_never_contains_header_secret(tmp_path):
    run, verdict = _run("ñ ok", headers={"Authorization": "Bearer s3cret"})
    write_json(run, verdict, tmp_path / "r.json")
    assert "s3cret" not in (tmp_path / "r.json").read_text(encoding="utf-8")


def test_report_preserves_spanish_text(tmp_path):
    write_json(*_run("conciliación ñ"), tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert "conciliación ñ" in text and "¿Concilia?" in text


def test_report_structure(tmp_path):
    write_json(*_run("ñ"), tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert data["tool"]["name"] == "octoqai"
    assert data["verdict"]["status"] in {"go", "no-go", "insufficient"}
    assert data["variants"][0]["ci95"][0] <= data["variants"][0]["ci95"][1]
    assert len(data["trials"]) == 2
    turn = data["trials"][0]["turns"][0]
    assert {"user", "answer", "tool_calls", "tokens", "latency_ms", "error", "checks"} <= set(turn)
    assert "headers" not in data["experiment"]


def test_report_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "out" / "r.json"
    write_json(*_run("ñ"), target)
    assert target.is_file()
