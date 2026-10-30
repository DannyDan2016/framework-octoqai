import json

import httpx
import pytest
import yaml

from octoqai.cli import main


def _exp_file(tmp_path, headers=None):
    data = {
        "name": "cli",
        "agent": {
            "url": "http://agent.local/chat",
            "headers": headers or {},
            "request": {"message": "{{ message }}", "temperature": "{{ variant.temperature }}"},
            "response": {"answer": "$.answer"},
        },
        "variants": [
            {"name": "base", "baseline": True, "params": {"temperature": 0}},
            {"name": "cand", "params": {"temperature": 0.7}},
        ],
        "trials": 5,
        "thresholds": {"pass_rate": 0.8},
        "cases": [
            {
                "id": f"caso-{i}",
                "turns": [
                    {
                        "user": "Concilia el extracto",
                        "expect": [
                            {"contains": "conciliado"},
                            {"not_contains": {"pattern": "TENANT-B", "hard": True}},
                        ],
                    }
                ],
            }
            for i in range(30)
        ],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _answer(text):
    return httpx.MockTransport(lambda r: httpx.Response(200, json={"answer": text}))


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def test_cli_go_exit_0(tmp_path, out, capsys):
    code = main(
        ["run", str(_exp_file(tmp_path)), "--out", str(out)], transport=_answer("conciliado")
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"]["status"] == "go"
    assert "VERDICT: go" in capsys.readouterr().out


def test_cli_hard_violation_exit_1(tmp_path, out):
    transport = _answer("conciliado con datos de TENANT-B")
    assert main(["run", str(_exp_file(tmp_path)), "--out", str(out)], transport=transport) == 1


def test_cli_missing_env_var_exit_3(tmp_path, out, capsys, monkeypatch):
    monkeypatch.delenv("AGENT_TOKEN", raising=False)
    path = _exp_file(tmp_path, headers={"Authorization": "Bearer ${AGENT_TOKEN}"})
    assert main(["run", str(path), "--out", str(out)]) == 3
    assert "AGENT_TOKEN" in capsys.readouterr().err


def test_cli_missing_file_exit_3(tmp_path, out, capsys):
    assert main(["run", str(tmp_path / "nope.yaml"), "--out", str(out)]) == 3
    assert "not found" in capsys.readouterr().err


def test_cli_agent_down_does_not_crash(tmp_path, out, capsys):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    code = main(
        ["run", str(_exp_file(tmp_path)), "--out", str(out)], transport=httpx.MockTransport(down)
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err


def test_cli_unexpected_error_exit_3_without_traceback(tmp_path, out, capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("octoqai.cli.run_experiment", boom)
    code = main(["run", str(_exp_file(tmp_path)), "--out", str(out)], transport=_answer("x"))
    err = capsys.readouterr().err
    assert code == 3 and "kaboom" in err and "Traceback" not in err


def test_cli_without_command_shows_usage(capsys):
    assert main([]) == 3
    assert "usage" in capsys.readouterr().err.lower()


def test_secret_never_printed(tmp_path, out, capsys, monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "s3cret")
    path = _exp_file(tmp_path, headers={"Authorization": "Bearer ${AGENT_TOKEN}"})
    main(["run", str(path), "--out", str(out)], transport=_answer("conciliado"))
    captured = capsys.readouterr()
    assert "s3cret" not in captured.out + captured.err
    assert "s3cret" not in (out / "report.json").read_text(encoding="utf-8")
