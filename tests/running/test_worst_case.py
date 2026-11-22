import pytest

from octoqai.experiment import Experiment
from octoqai.running import worst_case_calls
from support.builders import ExperimentBuilder, judge_check, simulated, turn
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def checks(count, prefix):
    return [judge_check(f"{prefix}{i}") for i in range(count)]


def build_case(builder, spec):
    case_id = spec["id"]
    extra = {"tags": spec.get("tags", ())}
    if spec.get("case_judge"):
        extra["judge"] = checks(spec["case_judge"], f"{case_id}-case")
    if "turns" in spec:
        turns = [
            turn("hola", {"contains": "x"}, judge=checks(n, f"{case_id}-t{i}"))
            for i, n in enumerate(spec["turns"])
        ]
        builder.case(case_id, *turns, **extra)
        return
    settings = {"each_turn": [{"contains": "."}], **spec["simulated"]}
    count = settings.pop("each_turn_judge", 0)
    if count:
        settings["each_turn_judge"] = checks(count, f"{case_id}-each")
    builder.case(case_id, simulated=simulated(**settings), **extra)


def build(row):
    builder = ExperimentBuilder().trials(row.get("trials", 1))
    for index in range(row.get("variants", 1)):
        builder.variant(f"v{index}", baseline=index == 0, temperature=0)
    if row.get("judge"):
        builder.judge()
    if row.get("simulated_user"):
        builder.simulated_user()
    for spec in row["cases"]:
        build_case(builder, spec)
    exp = Experiment.model_validate(builder.build())
    return exp.select(row["tags"]) if "tags" in row else exp


@pytest.mark.parametrize("row", cases(__file__, "worst_case"))
def test_worst_case_calls_table(row):
    assert worst_case_calls(build(row)) == row["expect"]
