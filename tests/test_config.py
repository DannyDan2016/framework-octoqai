import pytest

from octoqai.config import ConfigError, load_experiment


def test_loads_minimal_experiment(write_yaml):
    exp = load_experiment(write_yaml())
    assert exp.trials == 5 and exp.variants[0].name == "base"


def test_expectation_shorthand_and_hard_flag(write_yaml):
    exp = load_experiment(
        write_yaml(
            expect=[
                {"tool_called": ["buscar_movimientos"]},
                {"not_contains": {"pattern": "TENANT-B", "hard": True}},
            ]
        )
    )
    e0, e1 = exp.cases[0].turns[0].expect
    assert (e0.kind, e0.value, e0.hard) == ("tool_called", ["buscar_movimientos"], False)
    assert (e1.kind, e1.value, e1.hard) == ("not_contains", "TENANT-B", True)


def test_env_vars_expanded_in_headers(write_yaml, monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "s3cret")
    exp = load_experiment(write_yaml(headers={"Authorization": "Bearer ${AGENT_TOKEN}"}))
    assert exp.agent.headers["Authorization"] == "Bearer s3cret"


def test_missing_env_var_names_it(write_yaml, monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    with pytest.raises(ConfigError, match="NOPE_VAR"):
        load_experiment(write_yaml(headers={"X": "${NOPE_VAR}"}))


def test_two_baselines_rejected(write_yaml):
    with pytest.raises(ConfigError, match="baseline"):
        load_experiment(
            write_yaml(variants=[{"name": "a", "baseline": True}, {"name": "b", "baseline": True}])
        )


def test_multiple_variants_need_one_baseline(write_yaml):
    with pytest.raises(ConfigError, match="baseline"):
        load_experiment(write_yaml(variants=[{"name": "a"}, {"name": "b"}]))


def test_unknown_expectation_kind_rejected(write_yaml):
    with pytest.raises(ConfigError, match="teleport"):
        load_experiment(write_yaml(expect=[{"teleport": 1}]))


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_experiment(tmp_path / "missing.yaml")
