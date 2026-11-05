from datetime import date
from decimal import Decimal

import pytest

from octoqai.expectations import evaluate_db, forbidden_check
from octoqai.experiment import DbCheck, ForbiddenChange
from support.databases import result
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def check(**fields):
    return DbCheck.model_validate({"id": "x", "query": "SELECT 1", **fields})


def as_result(data):
    return None if data is None else result(*data["rows"], error=data.get("error"))


@pytest.mark.parametrize("row", cases(__file__, "db_checks"))
def test_evaluates_database_expectations(row):
    outcome = evaluate_db(
        check(**row["check"]), as_result(row.get("before")), as_result(row["after"])
    )
    assert (outcome.kind, outcome.passed, outcome.detail) == ("db", row["passed"], row["detail"])


@pytest.mark.smoke
def test_database_types_match_yaml_values():
    assert evaluate_db(check(expect={"value": 10.1}), None, result([Decimal("10.10")])).passed
    assert evaluate_db(
        check(expect={"value": date(2026, 10, 31)}), None, result([date(2026, 10, 31)])
    ).passed
    assert evaluate_db(
        check(expect={"rows": [["2026-10-31", 3]]}),
        None,
        result([date(2026, 10, 31), Decimal("3")]),
    ).passed
    assert evaluate_db(check(expect={"value": True}), None, result([1])).passed


def test_agent_error_fails_with_the_agent_error_flag():
    outcome = evaluate_db(
        check(expect={"delta": 1}), result([0]), result([0]), agent_error="HTTP 500"
    )
    assert (outcome.passed, outcome.agent_error, outcome.detail) == (
        False,
        True,
        "db 'x': agent error: HTTP 500",
    )


@pytest.mark.security
def test_detail_never_contains_database_values():
    outcome = evaluate_db(check(expect={"value": "otro"}), None, result(["TEN-B-0042"]))
    assert "TEN-B-0042" not in outcome.detail


@pytest.mark.security
def test_forbidden_change_is_a_hard_failure_only_when_rows_change():
    change = ForbiddenChange(
        id="no-tocar-b", query="SELECT count(*) FROM ajustes WHERE tenant = 'B'"
    )
    assert forbidden_check(change, result([0]), result([0]), 1) is None
    failed = forbidden_check(change, result([0]), result([1]), 2)
    assert (failed.kind, failed.passed, failed.hard, failed.detail) == (
        "forbidden_crud",
        False,
        True,
        "forbidden change 'no-tocar-b' after turn 2",
    )
    broken = forbidden_check(change, result([0]), result(error="OperationalError: x"), 1)
    assert (broken.passed, broken.hard) == (None, True)
