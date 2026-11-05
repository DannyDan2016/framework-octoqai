import importlib

import pytest

from support import samples
from support.agents import FakeJsonAgent, FakeSseAgent
from support.assertions import (
    assert_canary_hidden,
    assert_exit,
    assert_not_in_outputs,
    assert_value_hidden,
)
from support.builders import turn, write_data_set, write_env_file
from support.hooks import FakeCommands
from support.paths import EXAMPLES
from support.samples import ESCAPED_KEY_ANSWERS, ESCAPED_LEAKS, ESCAPED_WRAPPERS, HOSTILE_JSON

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
@pytest.mark.parametrize(
    ("streaming", "case"),
    [(False, "alone"), (False, "credential-value"), (True, "credential-key")],
    ids=["json-alone", "json-credential-value", "sse-credential-key"],
)
def test_value_hidden_in_an_escaped_json_key_never_reaches_the_report(
    experiment, tmp_path, run_cli, streaming, case
):
    canary, answer = "TEN-B-0042", ESCAPED_KEY_ANSWERS[case]
    fake_agent = FakeJsonAgent()
    experiment.security(canaries=[{"value": canary}])
    if streaming:
        experiment.streaming()
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    agent = FakeSseAgent().streams(answer) if streaming else fake_agent.answers(answer)
    result = run_cli("run", path, agent=agent)
    assert_exit(result.code, 1)
    text = result.report_text()
    answers = [t["turns"][0]["answer"] for t in result.report()["trials"]]
    assert_canary_hidden(canary, result.out, result.err, text, *answers)


UNESCAPE = importlib.import_module("octoqai.security._json_text").unescape
REPORTED_LEAKS = {
    f"{wrapper}-{secret}": ESCAPED_LEAKS[f"{wrapper}-{secret}"]
    for wrapper in ESCAPED_WRAPPERS
    for secret in ("canary-dash", "email", "credential-url")
}


@pytest.mark.security
@pytest.mark.parametrize("leak", REPORTED_LEAKS.values(), ids=REPORTED_LEAKS.keys())
def test_escaped_value_in_an_answer_that_is_not_json_is_no_go_and_never_reported(
    experiment, tmp_path, run_cli, fake_agent, leak
):
    value, answer, kind = leak
    experiment.security(canaries=[{"value": "TEN-B-0042"}]).trials(1)
    path = str(experiment.case("c1", turn("Hola", {"contains": "."})).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.answers(answer))
    assert_exit(result.code, 1)
    trials = result.report()["trials"]
    assert all(kind in [leak["kind"] for leak in t["turns"][0]["leaks"]] for t in trials)
    text = result.report_text()
    answers = [t["turns"][0]["answer"] for t in trials]
    unescaped = [UNESCAPE(form) for form in (text, *answers)]
    assert_value_hidden(value, result.out, result.err, text, *answers, *unescaped)


@pytest.fixture
def db_experiment(experiment, tmp_path, sqlite_db, monkeypatch):
    monkeypatch.setenv("DB_URL", sqlite_db.url)

    def _write(forbidden=(), **hooks):
        experiment.database(forbidden=forbidden).hooks(
            before_trial=[{"sql": ["DELETE FROM ajustes"]}], **hooks
        ).trials(2)
        experiment.case(
            "c1",
            turn(
                "Crea un ajuste",
                db=[
                    {
                        "id": "creado",
                        "query": "SELECT count(*) FROM ajustes",
                        "expect": {"delta": 1},
                    }
                ],
            ),
        )
        return str(experiment.write(tmp_path))

    return _write


@pytest.mark.smoke
def test_database_checks_pass_end_to_end(db_experiment, run_cli, fake_agent, sqlite_db):
    result = run_cli(
        "run", db_experiment(), agent=fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert())
    )
    assert result.code != 3, result.err
    assert all(t["outcome"] == "pass" for t in result.report()["trials"])
    assert "forbidden=0" in result.out


@pytest.mark.security
def test_forbidden_change_exits_no_go(db_experiment, run_cli, fake_agent, sqlite_db):
    forbidden = [{"id": "no-tocar-b", "query": "SELECT count(*) FROM ajustes WHERE tenant = 'B'"}]
    path = db_experiment(forbidden=forbidden)
    result = run_cli(
        "run", path, agent=fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert(tenant="B"))
    )
    assert_exit(result.code, 1)
    assert "forbidden=2" in result.out
    assert result.report()["variants"][0]["reasons"][0] == "2 forbidden database change(s)"


@pytest.mark.security
def test_forbidden_change_before_an_agent_error_is_no_go_not_unreachable(
    db_experiment, run_cli, fake_agent, sqlite_db
):
    forbidden = [{"id": "no-tocar-b", "query": "SELECT count(*) FROM ajustes WHERE tenant = 'B'"}]
    path = db_experiment(forbidden=forbidden)
    result = run_cli(
        "run", path, agent=fake_agent.fails(500).effect(lambda _: sqlite_db.insert(tenant="B"))
    )
    assert_exit(result.code, 1)
    assert "unreachable" not in result.err


@pytest.mark.security
def test_database_url_is_never_written(db_experiment, run_cli, fake_agent, sqlite_db):
    result = run_cli("run", db_experiment(), agent=fake_agent.answers("ok."))
    assert_not_in_outputs(sqlite_db.url, result.out, result.err, result.report_text())


@pytest.mark.security
def test_invalid_database_url_never_prints_the_password(experiment, tmp_path, run_cli, monkeypatch):
    monkeypatch.setenv("DB_URL", samples.DB_URL_WITH_PASSWORD)
    path = str(experiment.database().write(tmp_path))
    result = run_cli("run", path)
    assert_exit(result.code, 3)
    assert_not_in_outputs(samples.DB_PASSWORD, result.out, result.err)


def test_failed_before_run_exits_3_without_running_trials(
    experiment, tmp_path, run_cli, fake_agent
):
    path = str(experiment.hooks(before_run=[{"command": ["python", "prepara.py"]}]).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent, commands=FakeCommands().exits(1))
    assert_exit(result.code, 3)
    assert "before_run[0] (command) exited with code 1" in result.err
    assert fake_agent.requests == []


def test_reset_failing_in_every_trial_exits_3(experiment, tmp_path, run_cli, fake_agent):
    path = str(experiment.hooks(before_trial=[{"command": ["python", "reset.py"]}]).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent, commands=FakeCommands().exits(2))
    assert_exit(result.code, 3)
    assert "before_trial hook failed in every trial" in result.err
    assert fake_agent.requests == []


def test_failed_reset_then_agent_errors_report_the_agent_error(
    experiment, tmp_path, run_cli, fake_agent
):
    hooks = {"before_trial": [{"command": ["python", "reset.py"]}]}
    path = str(experiment.hooks(**hooks).trials(2).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.fails(500), commands=FakeCommands().exits(1, 0))
    assert_exit(result.code, 3)
    assert "agent unreachable in every trial (agent returned HTTP 500" in result.err
    assert "StopIteration" not in result.err


def test_after_run_failure_exits_3_with_a_warning(experiment, tmp_path, run_cli, fake_agent):
    path = str(experiment.hooks(after_run=[{"command": ["python", "limpia.py"]}]).write(tmp_path))
    result = run_cli("run", path, agent=fake_agent.answers("ok."), commands=FakeCommands().exits(1))
    assert_exit(result.code, 3)
    assert "Warning: hook after_run[0] (command) exited with code 1" in result.err
    assert result.report()["hooks"][-1]["ok"] is False


def test_after_run_failure_keeps_a_no_go(thirty_cases, experiment, run_cli, fake_agent):
    experiment.hooks(after_run=[{"command": ["python", "limpia.py"]}])
    fake_agent.answers("conciliado con datos de TENANT-B")
    result = run_cli("run", thirty_cases(), agent=fake_agent, commands=FakeCommands().exits(1))
    assert_exit(result.code, 1)
    assert "Warning: hook after_run[0] (command) exited with code 1" in result.err


@pytest.mark.security
def test_hook_env_values_are_never_printed(experiment, tmp_path, run_cli, fake_agent, monkeypatch):
    monkeypatch.setenv("TENANT_TOKEN", "tt-55443322")
    path = str(
        experiment.hooks(before_run=[{"command": ["python", "x.py", "${TENANT_TOKEN}"]}]).write(
            tmp_path
        )
    )
    result = run_cli("run", path, agent=fake_agent, commands=FakeCommands().exits(1))
    assert_not_in_outputs("tt-55443322", result.out, result.err)
