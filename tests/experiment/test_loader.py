import re

import pytest

from octoqai.agents import AgentReply
from octoqai.errors import ConfigError
from octoqai.expectations import evaluate
from octoqai.experiment import (
    Case,
    Expectation,
    JudgeCheck,
    SimulatedUser,
    Turn,
    load_experiment,
)
from octoqai.experiment.loader import _reject_llm_text_variables, _render_case, _Variables
from support import samples
from support.assertions import assert_config_error
from support.builders import (
    judge_check,
    simulated,
    turn,
    write_data_set,
    write_env_file,
)
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.mark.smoke
def test_loads_minimal_experiment(experiment, tmp_path):
    exp = load_experiment(experiment.write(tmp_path))
    assert exp.trials == 5 and exp.variants[0].name == "base"


@pytest.mark.security
def test_env_vars_expanded_in_headers(experiment, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "s3cret")
    path = experiment.agent(headers={"Authorization": "Bearer ${AGENT_TOKEN}"}).write(tmp_path)
    assert load_experiment(path).agent.headers["Authorization"] == "Bearer s3cret"


def test_missing_env_var_names_it(experiment, tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    path = experiment.agent(headers={"X": "${NOPE_VAR}"}).write(tmp_path)
    assert_config_error(lambda: load_experiment(path), "NOPE_VAR")


def test_missing_file_is_config_error(tmp_path):
    assert_config_error(lambda: load_experiment(tmp_path / "missing.yaml"), "not found")


@pytest.mark.parametrize("row", cases(__file__, "errors"))
def test_invalid_experiment_is_config_error(row, experiment, tmp_path):
    if "expect" in row:
        experiment.case("c1", turn("Hola", *row["expect"]))
    path = experiment.merge(row.get("overrides", {})).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path), *row["fragments"], absent=row.get("absent", ())
    )


DEV = {"cliente": {"nombre": "Ana", "cuenta": "ES-0001"}, "limite": 100}
QA = {"cliente": {"nombre": "Luis", "cuenta": "ES-0002"}, "limite": 250}


@pytest.fixture
def multienv(experiment, tmp_path):
    write_data_set(tmp_path, "dev", DEV)
    write_data_set(tmp_path, "qa", QA)
    return experiment.environments("dev", "qa").write(tmp_path)


def test_env_required_when_environments_declared(multienv):
    assert_config_error(lambda: load_experiment(multienv, process_env={}), "--env", "dev, qa")


def test_unknown_env_rejected(multienv):
    assert_config_error(
        lambda: load_experiment(multienv, env="prod", process_env={}), "prod", "dev, qa"
    )


@pytest.mark.parametrize("row", cases(__file__, "shapes"))
def test_data_sets_must_share_shape(row, experiment, tmp_path):
    write_data_set(tmp_path, "dev", row["dev"])
    write_data_set(tmp_path, "qa", row["qa"])
    path = experiment.environments("dev", "qa").write(tmp_path)
    assert_config_error(lambda: load_experiment(path, env="dev", process_env={}), *row["fragments"])


@pytest.mark.security
def test_shape_errors_never_show_values(experiment, tmp_path):
    write_data_set(tmp_path, "dev", {"cuenta": "ES-SECRETO-1", "limite": 100})
    write_data_set(tmp_path, "qa", {"cuenta": 42, "limite": "ES-SECRETO-2"})
    path = experiment.environments("dev", "qa").write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, env="qa", process_env={}),
        "data.cuenta",
        absent=["ES-SECRETO-1", "ES-SECRETO-2", "42", "100"],
    )


def test_missing_data_set_for_any_environment_is_rejected(experiment, tmp_path):
    write_data_set(tmp_path, "dev", DEV)
    path = experiment.environments("dev", "qa").write(tmp_path)
    assert_config_error(lambda: load_experiment(path, env="dev", process_env={}), "qa", "data")


@pytest.mark.smoke
def test_data_values_loaded_for_selected_env(multienv):
    exp = load_experiment(multienv, env="qa", process_env={})
    assert exp.environment == "qa" and exp.data_values == QA


def test_data_values_expand_env_vars_only_for_selected_env(experiment, tmp_path):
    write_data_set(tmp_path, "dev", {"token": "${DEV_ONLY}"})
    write_data_set(tmp_path, "qa", {"token": "${QA_ONLY}"})
    path = experiment.environments("dev", "qa").write(tmp_path)
    exp = load_experiment(path, env="qa", process_env={"QA_ONLY": "q"})
    assert exp.data_values == {"token": "q"}


def test_data_without_environments_uses_path_as_is(experiment, tmp_path):
    write_data_set(tmp_path, "shared", {"limite": 1})
    path = experiment.merge({"data": "data/shared.yaml"}).write(tmp_path)
    assert load_experiment(path, process_env={}).data_values == {"limite": 1}


def test_reserved_fields_cannot_be_set_in_yaml(experiment, tmp_path):
    path = experiment.merge({"data_values": {"x": 1}}).write(tmp_path)
    assert_config_error(lambda: load_experiment(path, process_env={}), "data_values")


@pytest.mark.security
def test_url_expanded_from_env_file(experiment, tmp_path):
    write_env_file(tmp_path, ".env", AGENT_URL="http://base.agent.local/chat")
    write_env_file(tmp_path, ".env.qa", AGENT_URL="http://qa.agent.local/chat")
    path = experiment.agent(url="${AGENT_URL}").write(tmp_path)
    assert load_experiment(path, env="qa", process_env={}).agent.url == (
        "http://qa.agent.local/chat"
    )


@pytest.fixture
def with_data(experiment, tmp_path):
    def _write(*turns, values=None):
        write_data_set(tmp_path, "dev", values or DEV)
        write_data_set(tmp_path, "qa", values or QA)
        return experiment.environments("dev", "qa").case("c1", *turns).write(tmp_path)

    return _write


def test_user_message_renders_data(with_data):
    path = with_data(
        turn("Concilia la cuenta {{ data.cliente.cuenta }} en {{ env }}", {"contains": "."})
    )
    exp = load_experiment(path, env="qa", process_env={})
    assert exp.cases[0].turns[0].user == "Concilia la cuenta ES-0002 en qa"


def test_unknown_data_placeholder_fails_at_load(with_data):
    path = with_data(turn("ok", {"contains": "."}), turn("{{ data.nope }}", {"contains": "."}))
    assert_config_error(
        lambda: load_experiment(path, env="dev", process_env={}),
        r"cases\[c1\]\.turns\[1\]",
        "data.nope",
    )


@pytest.mark.security
def test_contains_inserts_data_escaped(with_data):
    path = with_data(
        turn("x", {"contains": "ref {{ data.ref }}"}, {"not_contains": "{{ data.ref }}9"}),
        values={"ref": "(01)"},
    )
    contains, not_contains = (
        load_experiment(path, env="dev", process_env={}).cases[0].turns[0].expect
    )
    reply = AgentReply("la ref (01) existe", None, None, 1.0, None)
    assert evaluate(contains, reply).passed is True
    assert evaluate(not_contains, reply).passed is True


def test_invalid_tag_rejected(experiment, tmp_path):
    path = experiment.case("c1", turn("x", {"contains": "."}), tags=["Smoke Test"]).write(tmp_path)
    assert_config_error(lambda: load_experiment(path, process_env={}), "invalid tag")


def test_hyphenated_data_keys_render(with_data):
    path = with_data(
        turn("cuenta {{ data.numero-cuenta }}", {"contains": "{{ data.numero-cuenta }}"}),
        values={"numero-cuenta": "ES-2"},
    )
    rendered = load_experiment(path, env="dev", process_env={}).cases[0].turns[0]
    assert rendered.user == "cuenta ES-2" and rendered.expect[0].value == r"ES\-2"


@pytest.mark.parametrize(
    "item",
    [
        turn("{{ data.ref | upper }}", {"contains": "."}),
        turn("x", {"contains": "{{ data.ref | upper }}"}),
        turn("{{data}", {"contains": "."}),
    ],
    ids=["filter-in-message", "filter-in-pattern", "unbalanced"],
)
def test_unsupported_placeholder_fails_at_load(item, with_data):
    path = with_data(item, values={"ref": "r"})
    assert_config_error(
        lambda: load_experiment(path, env="dev", process_env={}),
        r"cases\[c1\]\.turns\[0\]",
        "placeholder",
    )


@pytest.mark.security
def test_canary_value_expands_env_and_is_marked_for_redaction(experiment, tmp_path):
    path = experiment.security(canaries=[{"value": "${CANARY_KEY}"}]).write(tmp_path)
    exp = load_experiment(path, process_env={"CANARY_KEY": "tenant-b-secret-01"})
    assert exp.security.canaries[0].value == "tenant-b-secret-01"
    assert "tenant-b-secret-01" in exp.redacted_values


def test_missing_canary_env_names_variable(experiment, tmp_path):
    path = experiment.security(canaries=[{"value": "${CANARY_KEY}"}]).write(tmp_path)
    assert_config_error(lambda: load_experiment(path, process_env={}), "CANARY_KEY")


@pytest.mark.security
def test_short_env_canary_rejected_without_echoing_value(experiment, tmp_path):
    path = experiment.security(canaries=[{"value": "${CANARY_KEY}"}]).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"CANARY_KEY": "zq9"}),
        "shorter than 4",
        absent=["zq9"],
    )


@pytest.mark.parametrize("row", cases(__file__, "db_errors"))
def test_invalid_database_or_hooks_is_config_error(row, experiment, tmp_path):
    path = experiment.merge(row["overrides"]).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"DB_URL": "sqlite:///x.db"}),
        *row["fragments"],
    )


@pytest.mark.parametrize("row", cases(__file__, "judge_errors"))
def test_invalid_judge_or_simulated_user_is_config_error(row, experiment, tmp_path):
    path = experiment.merge(row["overrides"]).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"DB_URL": "sqlite:///x.db"}),
        *row["fragments"],
        absent=row.get("absent", ()),
    )


def test_judge_and_simulated_sections_load_in_short_form(experiment, tmp_path):
    exp = load_experiment(experiment.judge().simulated_user().write(tmp_path))
    assert (exp.judge.provider, exp.judge.max_calls, exp.judge.max_input_chars) == (
        "ollama",
        3,
        32_000,
    )
    assert exp.simulated_user.model == "m"


@pytest.mark.security
def test_llm_api_keys_come_from_the_environment_and_are_redacted(experiment, tmp_path):
    judge_key, user_key = samples.ANTHROPIC_KEY, samples.ANTHROPIC_KEY_NO_PREFIX
    experiment.judge(provider="anthropic", api_key="${JUDGE_KEY}").simulated_user(
        provider="anthropic", api_key="${USER_KEY}"
    )
    exp = load_experiment(
        experiment.write(tmp_path), process_env={"JUDGE_KEY": judge_key, "USER_KEY": user_key}
    )
    assert exp.judge.api_key == judge_key
    assert exp.simulated_user.api_key == user_key
    assert {judge_key, user_key} <= set(exp.redacted_values)
    shown = repr(exp) + repr(exp.judge) + repr(exp.simulated_user)
    assert judge_key not in shown and user_key not in shown


@pytest.mark.security
@pytest.mark.parametrize("section", ["judge", "simulated_user"])
def test_literal_llm_api_key_is_rejected_without_echo(section, experiment, tmp_path):
    key = samples.ANTHROPIC_KEY
    getattr(experiment, section)(provider="anthropic", api_key=key)
    message = assert_config_error(
        lambda: load_experiment(experiment.write(tmp_path), process_env={}),
        rf"{section}\.api_key must come from an environment variable",
        r"\$\{ANTHROPIC_API_KEY\}",
    )
    assert key not in message


@pytest.mark.security
@pytest.mark.parametrize("value", ["sk-ant-${X}", "${X}-tail", "${X}${X}", 7])
def test_partial_llm_api_key_is_rejected_without_echo(value, experiment, tmp_path):
    experiment.judge(provider="anthropic", api_key=value)
    message = assert_config_error(
        lambda: load_experiment(experiment.write(tmp_path), process_env={"X": "secret-value-x"}),
        r"judge\.api_key must come from an environment variable",
    )
    assert "secret-value-x" not in message


def test_missing_llm_api_key_variable_names_it(experiment, tmp_path):
    experiment.simulated_user(provider="anthropic", api_key="${USER_KEY}")
    assert_config_error(
        lambda: load_experiment(experiment.write(tmp_path), process_env={}),
        "environment variable USER_KEY is not set",
    )


def test_llm_model_can_come_from_the_environment(experiment, tmp_path):
    experiment.judge(model="${JUDGE_MODEL}")
    exp = load_experiment(experiment.write(tmp_path), process_env={"JUDGE_MODEL": "llama3"})
    assert exp.judge.model == "llama3"


def test_ollama_with_an_api_key_is_still_rejected_by_the_model(experiment, tmp_path):
    experiment.judge(provider="ollama", api_key="${JUDGE_KEY}")
    assert_config_error(
        lambda: load_experiment(experiment.write(tmp_path), process_env={"JUDGE_KEY": "k"}),
        r"judge",
        "api_key",
    )


def db_case():
    return turn(
        "Crea", db=[{"id": "x", "query": "SELECT count(*) FROM ajustes", "expect": {"delta": 1}}]
    )


@pytest.mark.parametrize(
    "check",
    [
        {"query": "SELECT 1", "expect": {"unchanged": True}},
        {"query": "WITH t AS (SELECT 1) SELECT * FROM t", "expect": {"row_count": 1}},
        {"query": "  with t AS (SELECT 1) SELECT * FROM t", "expect": {"value": None}},
    ],
    ids=["unchanged-true", "with-query", "lowercase-with-null"],
)
def test_valid_database_checks_load(check, experiment, tmp_path):
    experiment.database().case("c1", turn("Hola", db=[{"id": "x", **check}]))
    exp = load_experiment(experiment.write(tmp_path), process_env={"DB_URL": "sqlite:///a.db"})
    assert exp.cases[0].turns[0].db[0].id == "x"


@pytest.mark.smoke
def test_database_url_comes_from_env_and_is_redacted(experiment, tmp_path):
    path = experiment.database().case("c1", db_case()).write(tmp_path)
    exp = load_experiment(path, process_env={"DB_URL": "sqlite:///app.db"})
    assert exp.database.url == "sqlite:///app.db"
    assert "sqlite:///app.db" in exp.redacted_values


@pytest.mark.security
def test_literal_database_url_is_rejected_without_echo(experiment, tmp_path):
    path = experiment.database(url=samples.DB_URL_WITH_PASSWORD).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={}),
        r"database\.url must come from an environment variable",
        absent=[samples.DB_PASSWORD],
    )


@pytest.mark.security
def test_hook_values_expand_env_and_are_redacted(experiment, tmp_path):
    experiment.hooks(
        before_trial=[{"command": ["python", "reset.py", "--tenant", "${TENANT}"]}],
        after_run=[
            {"http": {"url": "http://hooks.local/fin", "headers": {"X-Key": "${HOOK_KEY}"}}}
        ],
    )
    exp = load_experiment(
        experiment.write(tmp_path), process_env={"TENANT": "tenant-a", "HOOK_KEY": "hk-998877"}
    )
    assert exp.hooks.before_trial[0].command[-1] == "tenant-a"
    assert exp.hooks.after_run[0].http.headers == {"X-Key": "hk-998877"}
    assert {"tenant-a", "hk-998877"} <= set(exp.redacted_values)


@pytest.mark.security
@pytest.mark.parametrize(
    "value",
    ["A' OR '1'='1", "A'; DELETE FROM t; --", "tenant-{{-77"],
    ids=["or", "stack", "marker"],
)
def test_sql_hook_rejects_a_variable_before_pasting_its_value(value, experiment, tmp_path):
    experiment.database().hooks(before_trial=[{"sql": ["DELETE FROM t WHERE a = '${TENANT}'"]}])
    path = experiment.write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"DB_URL": "sqlite:///a.db", "TENANT": value}),
        r"sql hooks cannot use \$\{VAR\}: the value would be pasted into the statement unescaped",
        absent=[value, "1'='1", "-77"],
    )


@pytest.mark.parametrize(
    "statement",
    ["SELECT '${ not a variable }'", "SELECT '$TENANT'", "SELECT '${1A}'"],
    ids=["spaces", "no-braces", "invalid-name"],
)
def test_sql_hook_keeps_dollar_text_that_is_not_a_variable(statement, experiment, tmp_path):
    experiment.database().hooks(before_trial=[{"sql": [statement]}])
    path = experiment.write(tmp_path)
    exp = load_experiment(path, process_env={"DB_URL": "sqlite:///a.db", "TENANT": "x"})
    assert exp.hooks.before_trial[0].sql == [statement]


def test_db_params_render_data_values(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"cuenta": "001"})
    check = {
        "id": "x",
        "query": "SELECT 1 FROM ajustes WHERE cuenta = :c",
        "params": {"c": "{{ data.cuenta }}"},
        "expect": {"row_count": 1},
    }
    experiment.environments("qa").database().case("c1", turn("Crea", db=[check]))
    exp = load_experiment(
        experiment.write(tmp_path), env="qa", process_env={"DB_URL": "sqlite:///a.db"}
    )
    assert exp.cases[0].turns[0].db[0].params == {"c": "001"}


def test_render_error_in_global_forbidden_params_names_its_location(experiment, tmp_path):
    forbidden = [
        {"id": "no-b", "query": "SELECT 1 WHERE :t = 1", "params": {"t": "{{ data.nope }}"}}
    ]
    experiment.database(forbidden=forbidden).case("c1", turn("Hola", {"contains": "."}))
    path = experiment.write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"DB_URL": "sqlite:///a.db"}),
        r"database\.forbidden\[no-b\]: ",
    )


def test_env_placeholder_renders_in_check_params(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"cuenta": "001"})
    check = {
        "id": "x",
        "query": "SELECT 1 WHERE :e = 'qa'",
        "params": {"e": "{{ env }}"},
        "expect": {"row_count": 1},
    }
    experiment.environments("qa").database().case("c1", turn("Crea", db=[check]))
    exp = load_experiment(
        experiment.write(tmp_path), env="qa", process_env={"DB_URL": "sqlite:///a.db"}
    )
    assert exp.cases[0].turns[0].db[0].params == {"e": "qa"}


@pytest.mark.security
def test_check_params_expand_env_and_are_redacted(experiment, tmp_path):
    def guarded(check_id):
        return {"id": check_id, "query": "SELECT 1 WHERE :t = :t", "params": {"t": "${TENANT}"}}

    experiment.database().case(
        "c1",
        turn("Crea", db=[{**guarded("turno"), "expect": {"row_count": 1}}]),
        db=[{**guarded("caso"), "expect": {"row_count": 1}}],
        forbidden=[guarded("prohibido")],
    )
    exp = load_experiment(
        experiment.write(tmp_path),
        process_env={"DB_URL": "sqlite:///a.db", "TENANT": "tenant-b-77"},
    )
    case = exp.cases[0]
    params = [case.turns[0].db[0].params, case.db[0].params, case.forbidden[0].params]
    assert params == [{"t": "tenant-b-77"}] * 3
    assert "tenant-b-77" in exp.redacted_values


@pytest.mark.security
def test_base_dir_and_hook_env_are_filled_and_hidden(experiment, tmp_path):
    write_env_file(tmp_path, ".env", DB_URL="sqlite:///app.db")
    exp = load_experiment(
        experiment.database().case("c1", db_case()).write(tmp_path), process_env={}
    )
    assert exp.base_dir == tmp_path and exp.hook_env["DB_URL"] == "sqlite:///app.db"
    assert "sqlite:///app.db" not in repr(exp)


@pytest.mark.security
@pytest.mark.parametrize("database", [{"url": samples.DB_URL_PARTIAL_VARIABLE}, {}])
def test_database_url_must_be_exactly_one_variable(database, experiment, tmp_path):
    path = experiment.merge({"database": database}).write(tmp_path)
    assert_config_error(
        lambda: load_experiment(path, process_env={"DB_NAME": "app"}),
        r"database\.url must come from an environment variable",
        absent=[samples.DB_PASSWORD],
    )


def test_database_forbidden_params_render_data_values(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"cuenta": "001"})
    guard = {
        "id": "g",
        "query": "SELECT 1 FROM ajustes WHERE cuenta = :c",
        "params": {"c": "{{ data.cuenta }}"},
    }
    experiment.environments("qa").database(forbidden=[guard]).case("c1", db_case())
    exp = load_experiment(
        experiment.write(tmp_path), env="qa", process_env={"DB_URL": "sqlite:///a.db"}
    )
    assert exp.database.forbidden[0].params == {"c": "001"}


@pytest.mark.security
def test_secret_with_template_syntax_reaches_params_intact_and_is_not_printed(experiment, tmp_path):
    secret = "tok-{{ x }}-end"
    check = {
        "id": "x",
        "query": "SELECT count(*) FROM ajustes WHERE cuenta = :c",
        "params": {"c": "${CHECK_KEY}"},
        "expect": {"value": 1},
    }
    forbidden = [{"id": "f", "query": "SELECT 1 WHERE :c = 'z'", "params": {"c": "${CHECK_KEY}"}}]
    path = (
        experiment.database(forbidden=forbidden)
        .case("c1", turn("Hola", db=[check]))
        .write(tmp_path)
    )
    exp = load_experiment(path, process_env={"DB_URL": "sqlite:///x.db", "CHECK_KEY": secret})
    assert exp.cases[0].turns[0].db[0].params == {"c": secret}
    assert exp.database.forbidden[0].params == {"c": secret}
    assert secret in exp.redacted_values


@pytest.mark.security
def test_repr_hides_queries_params_and_hook_actions(experiment, tmp_path):
    check = {
        "id": "x",
        "query": "SELECT count(*) FROM ajustes WHERE cuenta = :c",
        "params": {"c": "${CHECK_KEY}"},
        "expect": {"value": 1},
    }
    experiment.database().hooks(
        before_trial=[{"sql": ["DELETE FROM ajustes WHERE cuenta = 'hk-sql-77'"]}],
        after_run=[
            {"command": ["python", "limpia.py", "hk-cmd-77"]},
            {"http": {"url": "http://hooks.local/hk-url-77"}},
        ],
    ).case("c1", turn("Hola", db=[check]))
    env = {"DB_URL": "sqlite:///x.db", "CHECK_KEY": "hk-param-77"}
    exp = load_experiment(experiment.write(tmp_path), process_env=env)
    exp = exp.model_copy(update={"data_values": {"k": "hk-data-77"}})
    text = repr(exp)
    for hidden in ("hk-sql-77", "hk-cmd-77", "hk-url-77", "hk-param-77", "hk-data-77"):
        assert hidden not in text


@pytest.mark.security
def test_database_password_and_secret_query_values_are_redacted_in_every_form(experiment, tmp_path):
    path = experiment.database().case("c1", db_case()).write(tmp_path)
    env = {"DB_URL": samples.DB_URL_WITH_ENCODED_SECRETS}
    values = load_experiment(path, process_env=env).redacted_values
    for expected in (
        samples.DB_SECRET_PASSWORD,
        samples.DB_SECRET_PASSWORD_ENCODED,
        samples.DB_SSL_SECRET,
        samples.DB_KEY_SECRET,
    ):
        assert expected in values
    assert "octoqai" not in values


@pytest.mark.security
@pytest.mark.parametrize("url", [samples.DB_URL_BAD_PORT, "postgresql://[::1/app?token=abc"])
def test_a_malformed_database_url_still_loads_without_raising(url, experiment, tmp_path):
    path = experiment.database().case("c1", db_case()).write(tmp_path)
    exp = load_experiment(path, process_env={"DB_URL": url})
    assert url in exp.redacted_values


@pytest.mark.security
@pytest.mark.parametrize(
    ("name", "expected"), [("after-slash", None), ("plain", samples.SURROGATE_PASSWORDS["plain"])]
)
def test_a_database_password_with_a_lone_surrogate_still_loads(
    name, expected, experiment, tmp_path
):
    url = samples.db_url_with(samples.SURROGATE_PASSWORDS[name])
    path = experiment.database().case("c1", db_case()).write(tmp_path)
    exp = load_experiment(path, process_env={"DB_URL": url})
    assert url in exp.redacted_values
    assert expected is None or expected in exp.redacted_values


@pytest.mark.security
def test_the_pw_query_key_is_redacted(experiment, tmp_path):
    path = experiment.database().case("c1", db_case()).write(tmp_path)
    values = load_experiment(path, process_env={"DB_URL": samples.PW_URL}).redacted_values
    assert samples.PW_SECRET in values


def _params_experiment(experiment, tmp_path, params, forbidden=()):
    check = {
        "id": "x",
        "query": "SELECT count(*) FROM ajustes WHERE cuenta = :c",
        "params": params,
        "expect": {"value": 1},
    }
    return experiment.database().case("c1", turn("Hola", db=[check])).write(tmp_path)


@pytest.mark.security
def test_environment_value_holding_a_placeholder_is_not_expanded_again(experiment, tmp_path):
    secret = "tok-" + "${OTHER}-end"
    path = _params_experiment(experiment, tmp_path, {"c": "${CHECK_KEY}"})
    exp = load_experiment(path, process_env={"DB_URL": "sqlite:///x.db", "CHECK_KEY": secret})
    assert exp.cases[0].turns[0].db[0].params == {"c": secret}


@pytest.mark.security
def test_data_value_holding_a_placeholder_is_not_expanded(experiment, tmp_path):
    value = "cta-" + "${OTHER}"
    write_data_set(tmp_path, "qa", {"cuenta": "${DATA_VAR}"})
    experiment.environments("qa")
    path = _params_experiment(experiment, tmp_path, {"c": "{{ data.cuenta }}"})
    process_env = {"DB_URL": "sqlite:///x.db", "DATA_VAR": value}
    exp = load_experiment(path, env="qa", process_env=process_env)
    assert exp.cases[0].turns[0].db[0].params == {"c": value}


@pytest.mark.security
def test_environment_value_holding_a_template_is_not_rendered(experiment, tmp_path):
    path = _params_experiment(experiment, tmp_path, {"c": "pre-${CHECK_KEY}"})
    process_env = {"DB_URL": "sqlite:///x.db", "CHECK_KEY": "{{ data.nope }}"}
    exp = load_experiment(path, process_env=process_env)
    assert exp.cases[0].turns[0].db[0].params == {"c": "pre-{{ data.nope }}"}


def test_param_mixing_literal_variable_and_data_gets_both(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"cuenta": "001"})
    experiment.environments("qa")
    path = _params_experiment(experiment, tmp_path, {"c": "${TENANT}-{{ data.cuenta }}"})
    process_env = {"DB_URL": "sqlite:///x.db", "TENANT": "tenant-b-77"}
    exp = load_experiment(path, env="qa", process_env=process_env)
    assert exp.cases[0].turns[0].db[0].params == {"c": "tenant-b-77-001"}


LLM_TEXT_LOCATIONS = [
    (
        {"turns": [turn("Hola", judge=[judge_check("j", "Cobra ${TENANT}")])]},
        r"cases\[c1\]\.turns\[0\]\.judge\[0\]",
        "rubric",
    ),
    (
        {"turns": [turn("Hola", judge=[judge_check("j", reference="Cobra ${TENANT}")])]},
        r"cases\[c1\]\.turns\[0\]\.judge\[0\]",
        "reference",
    ),
    ({"judge": [judge_check("j", "Es de ${TENANT}")]}, r"cases\[c1\]\.judge\[0\]", "rubric"),
    (
        {"simulated": simulated(goal="Entrar en ${TENANT}")},
        r"cases\[c1\]\.simulated",
        "goal",
    ),
    (
        {"simulated": simulated(persona="Cliente de ${TENANT}")},
        r"cases\[c1\]\.simulated",
        "persona",
    ),
    (
        {"simulated": simulated(opening="Hola ${TENANT}")},
        r"cases\[c1\]\.simulated",
        "opening",
    ),
    (
        {"simulated": simulated(each_turn_judge=[judge_check("j", "Es ${TENANT}")])},
        r"cases\[c1\]\.simulated\.each_turn_judge\[0\]",
        "rubric",
    ),
]


@pytest.mark.security
@pytest.mark.parametrize(("case", "location", "field"), LLM_TEXT_LOCATIONS)
def test_env_variable_in_llm_text_is_rejected(case, location, field, experiment, tmp_path):
    experiment.judge().simulated_user().merge({"cases": [{"id": "c1", **case}]})
    message = assert_config_error(
        lambda: load_experiment(experiment.write(tmp_path), process_env={"TENANT": "tenant-b-77"}),
        location,
        rf"\$\{{VAR\}} is not allowed in {field}: the text is sent to the LLM provider",
        r"use \{\{ data\.\* \}\} instead",
    )
    assert "tenant-b-77" not in message


def test_simulated_case_renders_data_placeholders_end_to_end(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"cuenta": "ES-0002", "cliente": "Ana"})
    experiment.environments("qa").simulated_user().case(
        "c1",
        simulated=simulated(
            goal="Reclama el cargo de la cuenta {{ data.cuenta }}",
            persona="Eres {{ data.cliente }}",
            opening="Hola, soy {{ data.cliente }}",
            stop_when="cuenta {{ data.cuenta }}",
            each_turn=[{"not_contains": "{{ data.cuenta }}9"}],
        ),
    )
    exp = load_experiment(experiment.write(tmp_path), env="qa", process_env={})
    user = exp.cases[0].simulated
    assert user.goal == "Reclama el cargo de la cuenta ES-0002"
    assert (user.persona, user.opening) == ("Eres Ana", "Hola, soy Ana")
    assert user.stop_when == r"cuenta ES\-0002"
    assert user.each_turn[0].value == r"ES\-00029"


def test_llm_text_check_leaves_data_placeholders_alone():
    data = {"cases": [{"id": "c1", "judge": [judge_check("j", "Cuenta {{ data.cuenta }}")]}]}
    _reject_llm_text_variables(data)


CONTEXT = {"data": {"cuenta": "ES-0002", "raw": "${X}{{ data.nope }}"}, "env": "qa"}


def _bare_case(**fields):
    base = {"id": "c1", "turns": [], "judge": [], "db": [], "forbidden": [], "simulated": None}
    return Case.model_construct(**{**base, **fields})


def _render(case):
    return _render_case(case, CONTEXT, _Variables({"X": "leaked"}))


def test_rubric_and_goal_render_data_and_env():
    user = SimulatedUser(
        goal="Concilia {{ data.cuenta }}",
        persona="Usuario de {{ env }}",
        opening="Hola {{ data.cuenta }}",
        max_turns=2,
        stop_when="fin {{ data.cuenta }}.",
        each_turn=[Expectation.model_validate({"contains": "{{ data.cuenta }}"})],
        each_turn_judge=[JudgeCheck(id="k", rubric="{{ data.cuenta }}")],
    )
    check = JudgeCheck(
        id="j", rubric="Cuenta {{ data.cuenta }} en {{ env }}", reference="{{ data.cuenta }}!"
    )
    rendered = _render(_bare_case(judge=[check], simulated=user))
    assert rendered.judge[0].rubric == "Cuenta ES-0002 en qa"
    assert rendered.judge[0].reference == "ES-0002!"
    assert (rendered.simulated.goal, rendered.simulated.persona, rendered.simulated.opening) == (
        "Concilia ES-0002",
        "Usuario de qa",
        "Hola ES-0002",
    )
    assert rendered.simulated.stop_when == "fin ES\\-0002."
    assert rendered.simulated.each_turn[0].value == "ES\\-0002"
    assert rendered.simulated.each_turn_judge[0].rubric == "ES-0002"


def test_turn_judge_rubric_renders_and_names_its_location():
    good = Turn(user="x", judge=[JudgeCheck(id="j", rubric="{{ data.cuenta }}")])
    bad = Turn(user="x", judge=[JudgeCheck(id="j", rubric="{{ data.nope }}")])
    assert _render(_bare_case(turns=[good])).turns[0].judge[0].rubric == "ES-0002"
    with pytest.raises(ConfigError, match=r"cases\[c1\]\.turns\[1\].*data\.nope"):
        _render(_bare_case(turns=[good, bad]))


def test_undefined_variable_in_goal_names_its_location():
    user = SimulatedUser(goal="{{ data.nope }}", max_turns=2)
    with pytest.raises(ConfigError, match=r"cases\[c1\]\.simulated.*data\.nope"):
        _render(_bare_case(simulated=user))


@pytest.mark.security
def test_data_value_with_placeholder_text_is_neither_expanded_nor_rendered_again():
    check = JudgeCheck(id="j", rubric="Valor {{ data.raw }}")
    rendered = _render(_bare_case(judge=[check]))
    assert rendered.judge[0].rubric == "Valor ${X}{{ data.nope }}"


@pytest.mark.security
def test_data_value_with_placeholders_reaches_the_judge_rubric_literally(experiment, tmp_path):
    write_data_set(tmp_path, "qa", {"v": "Valor ${X} y {{ nope }}"})
    experiment.environments("qa").judge().merge(
        {"cases": [{"id": "c1", "turns": [turn("Hola", judge=[judge_check("j", "{{ data.v }}")])]}]}
    )
    exp = load_experiment(
        experiment.write(tmp_path), env="qa", process_env={"X": "${Y} {{ data.v }}", "Y": "boom"}
    )
    assert exp.cases[0].turns[0].judge[0].rubric == "Valor ${Y} {{ data.v }} y {{ nope }}"
    assert "boom" not in repr(exp.cases)


@pytest.mark.security
@pytest.mark.parametrize("section", ["judge", "simulated_user"])
@pytest.mark.parametrize("scalar", ["plain-scalar-key-9", 12345, True, ["list-scalar-9"]])
def test_non_mapping_llm_section_never_carries_the_scalar(section, scalar, experiment, tmp_path):
    experiment.merge({section: scalar})
    with pytest.raises(ConfigError) as info:
        load_experiment(experiment.write(tmp_path), process_env={})
    assert re.search(rf"{section}\.api_key must come from an environment variable", str(info.value))
    chain = [info.value.__cause__, info.value.__context__]
    for text in [str(info.value), *(repr(link) for link in chain)]:
        assert "plain-scalar-key-9" not in text and "list-scalar-9" not in text
    assert chain == [None, None]
