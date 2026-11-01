import importlib
import json

import pytest

from octoqai.agents import AgentReply
from octoqai.experiment import SecurityConfig
from octoqai.security import Detector, scan_reply
from support.agents import FakeSseAgent
from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import turn, write_data_set, write_env_file
from support.paths import EXAMPLES
from support.samples import ESCAPED_KEY_ANSWER, HOSTILE_JSON

pytestmark = [pytest.mark.integration, pytest.mark.regression]


@pytest.fixture
def thirty_cases(experiment, tmp_path):
    def _write(headers=None):
        experiment.agent(
            headers=headers or {},
            request={"message": "{{ message }}", "temperature": "{{ variant.temperature }}"},
        )
        experiment.variant("base", baseline=True, temperature=0)
        experiment.variant("cand", temperature=0.7).trials(5)
        for i in range(30):
            experiment.case(
                f"caso-{i}",
                turn(
                    "Concilia el extracto",
                    {"contains": "conciliado"},
                    {"not_contains": {"pattern": "TENANT-B", "hard": True}},
                ),
            )
        return str(experiment.write(tmp_path))

    return _write


@pytest.mark.smoke
def test_cli_go_exit_0(thirty_cases, run_cli, fake_agent):
    result = run_cli("run", thirty_cases(), agent=fake_agent.answers("conciliado"))
    assert_exit(result.code, 0)
    assert result.report()["verdict"]["status"] == "go"
    assert "VERDICT: go" in result.out


def test_cli_hard_violation_exit_1(thirty_cases, run_cli, fake_agent):
    fake_agent.answers("conciliado con datos de TENANT-B")
    assert_exit(run_cli("run", thirty_cases(), agent=fake_agent).code, 1)


def test_cli_missing_env_var_exit_3(thirty_cases, run_cli, monkeypatch):
    monkeypatch.delenv("AGENT_TOKEN", raising=False)
    result = run_cli("run", thirty_cases(headers={"Authorization": "Bearer ${AGENT_TOKEN}"}))
    assert_exit(result.code, 3)
    assert "AGENT_TOKEN" in result.err


def test_cli_missing_file_exit_3(tmp_path, run_cli):
    result = run_cli("run", str(tmp_path / "nope.yaml"))
    assert_exit(result.code, 3)
    assert "not found" in result.err


def test_cli_agent_down_does_not_crash(thirty_cases, run_cli, fake_agent):
    result = run_cli("run", thirty_cases(), agent=fake_agent.refuses())
    assert_exit(result.code, 3)
    assert "unreachable" in result.err and "Traceback" not in result.err
    assert (result.out_dir / "report.json").is_file()


def test_cli_unexpected_error_exit_3_without_traceback(
    thirty_cases, run_cli, fake_agent, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(importlib.import_module("octoqai.cli.main"), "run_experiment", boom)
    result = run_cli("run", thirty_cases(), agent=fake_agent.answers("x"))
    assert_exit(result.code, 3)
    assert "kaboom" in result.err and "Traceback" not in result.err


def test_cli_without_command_shows_usage(run_cli):
    result = run_cli()
    assert_exit(result.code, 3)
    assert "usage" in result.err.lower()


@pytest.mark.security
def test_secret_never_printed(thirty_cases, run_cli, fake_agent, monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "s3cret")
    path = thirty_cases(headers={"Authorization": "Bearer ${AGENT_TOKEN}"})
    result = run_cli("run", path, agent=fake_agent.answers("conciliado"))
    assert_not_in_outputs("s3cret", result.out, result.err, result.report_text())


@pytest.mark.security
def test_run_with_env_uses_env_file_and_reports_environment(
    experiment, tmp_path, run_cli, fake_agent, monkeypatch
):
    monkeypatch.delenv("AGENT_URL", raising=False)
    write_env_file(tmp_path, ".env.qa", AGENT_URL="http://qa.agent.local/chat")
    path = experiment.agent(url="${AGENT_URL}").write(tmp_path)
    result = run_cli("run", str(path), "--env", "qa", agent=fake_agent.answers("ok."))
    assert result.code != 3, result.err
    assert str(fake_agent.urls[0]) == "http://qa.agent.local/chat"
    assert result.report()["environment"] == "qa"
    assert_not_in_outputs("qa.agent.local", result.out, result.err, result.report_text())


@pytest.mark.parametrize("env", ["dev", "qa"])
def test_multienv_example_is_valid_for_every_env(env, run_cli, fake_agent, monkeypatch):
    monkeypatch.setenv("AGENT_URL", "http://127.0.0.1:8000/chat")
    monkeypatch.setenv("AGENT_TOKEN", "example-token")
    example = EXAMPLES / "multienv" / "experiment.yaml"
    result = run_cli("run", str(example), "--env", env, agent=fake_agent.answers("ok"))
    assert result.code != 3, result.err
    assert result.report()["environment"] == env


@pytest.mark.parametrize("extra", [(), ("--env", "prod")], ids=["missing", "undeclared"])
def test_wrong_env_is_config_error_exit_3(extra, run_cli, monkeypatch):
    monkeypatch.setenv("AGENT_URL", "http://127.0.0.1:8000/chat")
    monkeypatch.setenv("AGENT_TOKEN", "example-token")
    result = run_cli("run", str(EXAMPLES / "multienv" / "experiment.yaml"), *extra)
    assert_exit(result.code, 3)
    assert "dev, qa" in result.err


@pytest.mark.smoke
def test_cli_tags_filter_runs_only_smoke_cases(experiment, tmp_path, run_cli, fake_agent):
    experiment.trials(2)
    experiment.case("c1", turn("a", {"contains": "."}), tags=["smoke"])
    experiment.case("c2", turn("b", {"contains": "."}), tags=["regression"])
    experiment.case("c3", turn("c", {"contains": "."}), tags=["smoke", "pagos"])
    path = str(experiment.write(tmp_path))
    result = run_cli("run", path, "--tags", "smoke, pagos", agent=fake_agent.answers("ok."))
    assert result.code != 3, result.err
    report = result.report()
    assert report["tags"] == ["smoke", "pagos"]
    assert {t["case_id"] for t in report["trials"]} == {"c1", "c3"}
    assert len(report["trials"]) == 2 * 2


def test_cli_tags_without_matches_exit_3(experiment, tmp_path, run_cli):
    path = str(experiment.case("c1", turn("a", {"contains": "."}), tags=["smoke"]).write(tmp_path))
    result = run_cli("run", path, "--tags", "nightly")
    assert_exit(result.code, 3)
    assert "no case matches tags" in result.err


def test_multienv_example_smoke_suite_renders_qa_data(run_cli, fake_agent, monkeypatch):
    monkeypatch.setenv("AGENT_URL", "http://127.0.0.1:8000/chat")
    monkeypatch.setenv("AGENT_TOKEN", "example-token")
    example = str(EXAMPLES / "multienv" / "experiment.yaml")
    result = run_cli("run", example, "--env", "qa", "--tags", "smoke", agent=fake_agent)
    assert result.code != 3, result.err
    assert {t["case_id"] for t in result.report()["trials"]} == {"conciliar-extracto"}
    assert "ES-QA-0001" in fake_agent.requests[0]["message"]


@pytest.mark.security
def test_values_from_env_vars_in_data_are_masked_in_report(
    experiment, tmp_path, run_cli, fake_agent
):
    write_data_set(tmp_path, "qa", {"cuenta": "${QA_ACCOUNT}"})
    write_env_file(tmp_path, ".env.qa", QA_ACCOUNT="ES-REAL-998877")
    experiment.environments("qa").case(
        "c1", turn("Concilia la cuenta {{ data.cuenta }}", {"contains": "."})
    )
    path = str(experiment.write(tmp_path))
    result = run_cli("run", path, "--env", "qa", agent=fake_agent.answers("ok ES-REAL-998877."))
    assert result.code != 3, result.err
    assert "ES-REAL-998877" in fake_agent.requests[0]["message"]
    assert_not_in_outputs("ES-REAL-998877", result.out, result.err, result.report_text())
    assert "[redacted]" in result.report_text()


@pytest.mark.smoke
def test_cli_runs_a_streaming_agent(experiment, tmp_path, run_cli):
    experiment.streaming().case("c1", turn("Concilia", {"contains": "conciliado"}))
    path = str(experiment.trials(1).write(tmp_path))
    result = run_cli("run", path, agent=FakeSseAgent().streams("Extracto ", "conciliado."))
    assert result.code != 3, result.err
    assert result.report()["trials"][0]["turns"][0]["answer"] == "Extracto conciliado."


@pytest.fixture
def streaming(experiment, tmp_path):
    def _write(canary="TEN-B-0042"):
        experiment.streaming().security(canaries=[{"value": canary}]).trials(2)
        experiment.case("c1", turn("Concilia octubre", {"contains": "."}))
        return str(experiment.write(tmp_path))

    return _write


@pytest.mark.security
def test_streaming_leak_exits_no_go_and_never_prints_canary(streaming, run_cli):
    result = run_cli("run", streaming(), agent=FakeSseAgent().leaks("TEN-B-0042"))
    assert_exit(result.code, 1)
    assert_not_in_outputs("TEN-B-0042", result.out, result.err, result.report_text())
    assert "leaks=2" in result.out


@pytest.mark.security
def test_env_canary_never_written(experiment, tmp_path, run_cli, monkeypatch):
    monkeypatch.setenv("CANARY_KEY", "tenant-b-secret-01")
    experiment.streaming().security(canaries=[{"value": "${CANARY_KEY}"}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path, agent=FakeSseAgent().leaks("tenant-b-secret-01"))
    assert_exit(result.code, 1)
    assert_not_in_outputs("tenant-b-secret-01", result.out, result.err, result.report_text())


@pytest.mark.security
def test_leak_before_a_stream_cut_is_no_go_not_unreachable(streaming, run_cli):
    result = run_cli("run", streaming(), agent=FakeSseAgent().cuts_after("dato TEN-B-0042"))
    assert_exit(result.code, 1)
    assert "unreachable" not in result.err


def test_retries_are_written_to_the_report(experiment, tmp_path, run_cli, fake_agent):
    path = str(experiment.agent(retry={"retries": 1, "backoff_s": 0}).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.answers("ok.").refuses_first(1))
    assert result.code != 3, result.err
    turns = [t["turns"][0] for t in result.report()["trials"]]
    assert turns[0]["retries"] == ["cannot connect to agent: ConnectError"]
    assert all(t["retries"] == [] for t in turns[1:])


def test_streaming_example_runs_against_a_fake_agent(run_cli):
    example = str(EXAMPLES / "streaming" / "experiment.yaml")
    result = run_cli("run", example, agent=FakeSseAgent().streams("Extracto conciliado."))
    assert result.code != 3, result.err
    assert all(v["leaks"] == 0 for v in result.report()["variants"])


@pytest.mark.security
def test_unreachable_error_never_prints_canary(experiment, tmp_path, run_cli, fake_agent):
    canary = "TEN_B_0042"
    experiment.agent(response={"answer": f"$.{canary}"}).security(canaries=[{"value": canary}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.answers("x"))
    assert_exit(result.code, 3)
    assert "unreachable" in result.err
    assert_not_in_outputs(canary, result.out, result.err, result.report_text())


@pytest.mark.security
def test_unexpected_error_never_prints_canary(
    experiment, tmp_path, run_cli, fake_agent, monkeypatch
):
    canary = "TEN-B-0042"

    def boom(*args, **kwargs):
        raise RuntimeError(f"kaboom {canary}")

    monkeypatch.setattr(importlib.import_module("octoqai.cli.main"), "run_experiment", boom)
    experiment.security(canaries=[{"value": canary}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent)
    assert_exit(result.code, 3)
    assert "kaboom [canary:canary-1]" in result.err
    assert_not_in_outputs(canary, result.out, result.err)


@pytest.mark.security
def test_config_error_after_loading_never_prints_canary(experiment, tmp_path, run_cli):
    canary = "TEN_B_0042"
    experiment.agent(response={"answer": f"$.{canary}["}).security(canaries=[{"value": canary}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path)
    assert_exit(result.code, 3)
    assert "invalid JSONPath" in result.err
    assert_not_in_outputs(canary, result.out, result.err)


@pytest.mark.security
def test_mismatched_charset_leak_is_found_and_never_written(
    experiment, tmp_path, run_cli, fake_agent
):
    canary = "TEN-B-0042"
    experiment.security(canaries=[{"value": canary}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    agent = fake_agent.fails(500, f"dato {canary}", charset="utf-16", encoding="ascii")
    result = run_cli("run", path, agent=agent)
    assert any(v["leaks"] for v in result.report()["variants"])
    assert_not_in_outputs(canary, result.out, result.err, result.report_text())


@pytest.mark.security
@pytest.mark.parametrize("body", HOSTILE_JSON.values(), ids=HOSTILE_JSON.keys())
def test_hostile_json_with_a_leak_still_writes_the_report_and_is_no_go(
    experiment, tmp_path, run_cli, fake_agent, body
):
    experiment.security(canaries=[{"value": "TEN-B-0042"}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    agent = fake_agent.returns(body.encode("ascii"), content_type="application/json")
    result = run_cli("run", path, agent=agent)
    assert_exit(result.code, 1)
    assert "Traceback" not in result.err
    assert all(v["leaks"] for v in result.report()["variants"])
    assert_not_in_outputs("TEN-B-0042", result.out, result.err, result.report_text())


@pytest.mark.security
def test_value_hidden_in_an_escaped_json_key_never_reaches_the_report(
    experiment, tmp_path, run_cli, fake_agent
):
    canary = "TEN-B-0042"
    experiment.security(canaries=[{"value": canary}])
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.answers(ESCAPED_KEY_ANSWER))
    assert_exit(result.code, 1)
    text = result.report_text()
    detector = Detector(SecurityConfig.model_validate({"canaries": [{"value": canary}]}))
    assert scan_reply(detector, AgentReply(text, None, None, 1.0, None), []) == []
    assert_not_in_outputs(canary, result.out, result.err, text, json.dumps(json.loads(text)))
