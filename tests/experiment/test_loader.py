import pytest

from octoqai.agents import AgentReply
from octoqai.expectations import evaluate
from octoqai.experiment import load_experiment
from support import samples
from support.assertions import assert_config_error
from support.builders import turn, write_data_set, write_env_file
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


def db_case():
    return turn(
        "Crea", db=[{"id": "x", "query": "SELECT count(*) FROM ajustes", "expect": {"delta": 1}}]
    )


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
