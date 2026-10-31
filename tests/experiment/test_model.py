import datetime

import pytest
from pydantic import ValidationError

from octoqai.experiment import Experiment
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
