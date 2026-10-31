import pytest

from octoqai.experiment import load_experiment
from support.assertions import assert_config_error
from support.builders import turn
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
    assert_config_error(lambda: load_experiment(path), *row["fragments"])
