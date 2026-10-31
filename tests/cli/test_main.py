import importlib

import pytest

from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import turn

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
