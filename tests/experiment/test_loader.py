import pytest

from octoqai.agents import AgentReply
from octoqai.expectations import evaluate
from octoqai.experiment import load_experiment
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
