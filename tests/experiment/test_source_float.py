import copy
import json
import pickle
from decimal import Decimal

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter

from octoqai.db import SqlDatabase
from octoqai.experiment import AgentConfig, SourceFloat, load_experiment
from support.agents import TURN_CONTEXT
from support.builders import DEFAULT_URL, turn, write_data_set, write_with_numbers

pytestmark = [pytest.mark.unit, pytest.mark.regression]

WIDE = "99999999999999.99"


class Measure(BaseModel):
    temperature: float


def loaded_param(experiment, tmp_path, text):
    experiment.variant("base", baseline=True, temperature="NUMBER")
    return load_experiment(write_with_numbers(experiment, tmp_path, NUMBER=text)).variants[0]


def loaded_float(experiment, tmp_path, text):
    return loaded_param(experiment, tmp_path, text).params["temperature"]


@pytest.mark.smoke
def test_a_yaml_float_keeps_the_text_it_was_written_with(experiment, tmp_path):
    value = loaded_float(experiment, tmp_path, WIDE)
    assert isinstance(value, SourceFloat) and isinstance(value, float)
    assert value.source == WIDE
    assert value == float(WIDE)


@pytest.mark.parametrize("text", [".inf", "-.inf", ".nan", "1_000.5", "1:30.5"])
def test_a_special_yaml_float_keeps_the_behaviour_of_a_plain_float(experiment, tmp_path, text):
    value = loaded_float(experiment, tmp_path, text)
    expected = yaml.safe_load("x: " + text)["x"]
    assert type(value) is float
    assert str(value) == str(expected)


@pytest.mark.parametrize("text", ["7", "'7.5'", "true"])
def test_a_yaml_value_that_is_not_a_float_is_not_touched(experiment, tmp_path, text):
    value = loaded_float(experiment, tmp_path, text)
    assert type(value) is type(yaml.safe_load("x: " + text)["x"])


def test_a_source_float_behaves_as_a_float_everywhere(experiment, tmp_path):
    value = loaded_float(experiment, tmp_path, "0.7")
    assert json.dumps({"t": value}) == json.dumps({"t": 0.7}) == '{"t": 0.7}'
    assert json.loads(json.dumps(value)) == 0.7
    assert hash(value) == hash(0.7) and repr(value) == "0.7" and value + 1 == 1.7
    assert TypeAdapter(float).validate_python(value) == 0.7
    assert TypeAdapter(float | int).validate_python(value) == 0.7
    assert Measure(temperature=value).temperature == 0.7
    assert Decimal(repr(value)) == Decimal("0.7")


def test_a_source_float_survives_copy_and_pickle(experiment, tmp_path):
    value = loaded_float(experiment, tmp_path, WIDE)
    for clone in (copy.copy(value), copy.deepcopy(value), pickle.loads(pickle.dumps(value))):
        assert clone == value and clone.source == WIDE


def test_a_source_float_is_bound_as_a_number_by_a_real_sqlite(experiment, tmp_path, sqlite_db):
    value = loaded_float(experiment, tmp_path, "0.5")
    database = SqlDatabase(sqlite_db.url)
    try:
        rows = database.query("SELECT :p, typeof(:p)", {"p": value})
    finally:
        database.close()
    assert rows.error is None and rows.rows == ((0.5, "real"),)


def test_a_source_float_goes_into_an_agent_request_body_as_a_number(
    experiment, tmp_path, fake_agent
):
    value = loaded_float(experiment, tmp_path, "0.7")
    cfg = AgentConfig(
        url=DEFAULT_URL,
        request={"message": "{{ message }}", "temperature": "{{ variant.temperature }}"},
        response={"answer": "$.answer"},
        timeout_s=5,
    )
    fake_agent.adapter(cfg).send({**TURN_CONTEXT, "variant": {"temperature": value}})
    assert fake_agent.requests[0]["temperature"] == 0.7


def test_a_float_in_a_data_file_stays_a_plain_float(experiment, tmp_path):
    experiment.environments("qa")
    write_data_set(tmp_path, "qa", {"amount": 0.5})
    experiment.case("c1", turn("Hola", {"contains": "."}))
    loaded = load_experiment(experiment.write(tmp_path), env="qa")
    assert type(loaded.data_values["amount"]) is float
