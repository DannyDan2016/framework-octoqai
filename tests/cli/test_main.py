import importlib

import pytest

from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import turn, write_data_set, write_env_file
from support.paths import EXAMPLES

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
