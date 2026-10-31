import datetime

import pytest
from pydantic import ValidationError

from octoqai.experiment import EffectiveWait, Experiment
from support.assertions import assert_config_error
from support.builders import turn

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def test_expectation_shorthand_and_hard_flag(experiment):
    experiment.case(
        "c1",
        turn(
            "Hola",
            {"tool_called": ["buscar_movimientos"]},
            {"not_contains": {"pattern": "TENANT-B", "hard": True}},
        ),
    )
    e0, e1 = Experiment.model_validate(experiment.build()).cases[0].turns[0].expect
    assert (e0.kind, e0.value, e0.hard) == ("tool_called", ["buscar_movimientos"], False)
    assert (e1.kind, e1.value, e1.hard) == ("not_contains", "TENANT-B", True)


def test_variant_missing_template_param_rejected(experiment):
    experiment.agent(
        request={"message": "{{ message }}", "temperature": "{{ variant.temperature }}"}
    ).variant("base")
    with pytest.raises(ValidationError, match="base.*temperature"):
        Experiment.model_validate(experiment.build())


def test_non_json_serializable_param_rejected(experiment):
    experiment.variant("base", d=datetime.date(2026, 1, 1))
    with pytest.raises(ValidationError, match="JSON"):
        Experiment.model_validate(experiment.build())


@pytest.fixture
def tagged(experiment):
    experiment.case("c1", turn("a", {"contains": "."}), tags=["smoke"])
    experiment.case("c2", turn("b", {"contains": "."}), tags=["regression"])
    experiment.case("c3", turn("c", {"contains": "."}), tags=["smoke", "pagos"])
    return Experiment.model_validate(experiment.build())


@pytest.mark.smoke
def test_select_keeps_cases_with_any_tag(tagged):
    assert [c.id for c in tagged.select(["smoke"]).cases] == ["c1", "c3"]
    assert [c.id for c in tagged.select(["regression", "pagos"]).cases] == ["c2", "c3"]
    assert tagged.select(["smoke"]).selected_tags == ["smoke"]
    assert len(tagged.cases) == 3


def test_select_without_matches_is_config_error(tagged):
    assert_config_error(lambda: tagged.select(["nightly"]), "no case matches tags", "nightly")


@pytest.mark.smoke
def test_stream_makes_answer_optional(experiment):
    agent = Experiment.model_validate(experiment.streaming().build()).agent
    assert (agent.stream.delta, agent.stream.event, agent.stream.done) == (
        "$.delta",
        None,
        "[DONE]",
    )
    assert agent.response.answer is None


def test_wait_defaults_to_timeout(experiment):
    agent = Experiment.model_validate(experiment.agent(timeout_s=30).build()).agent
    assert agent.effective_wait() == EffectiveWait(30, 30, 30, 30)


def test_wait_fields_override_and_default_to_total(experiment):
    experiment.agent(timeout_s=30, wait={"connect_s": 2, "idle_s": 5, "total_s": 20})
    wait = Experiment.model_validate(experiment.build()).agent.effective_wait()
    assert wait == EffectiveWait(connect_s=2, first_fragment_s=20, idle_s=5, total_s=20)


def test_retry_defaults_to_no_retry(experiment):
    retry = Experiment.model_validate(experiment.build()).agent.retry
    assert (retry.retries, retry.backoff_s) == (0, 0.5)


def test_security_defaults_enable_all_detectors(experiment):
    security = Experiment.model_validate(experiment.build()).security
    assert (security.canaries, security.detectors, security.allow) == ([], ["secrets", "pii"], [])


def test_canary_labels_default_to_position(experiment):
    experiment.security(
        canaries=[{"value": "TEN-B-0042"}, {"value": "CANARY-7f3a9c2e", "label": "mine"}]
    )
    canaries = Experiment.model_validate(experiment.build()).security.canaries
    assert [c.label for c in canaries] == ["canary-1", "mine"]


def test_agent_headers_are_hidden_from_repr(experiment):
    experiment.agent(headers={"Authorization": "Bearer zzz-visible-check"})
    assert "zzz-visible-check" not in repr(Experiment.model_validate(experiment.build()).agent)
