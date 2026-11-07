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


NON_FINITE = [
    Decimal("NaN"),
    Decimal("sNaN"),
    Decimal("Infinity"),
    Decimal("-Infinity"),
    float("nan"),
    float("inf"),
    float("-inf"),
]


@pytest.mark.regression
@pytest.mark.parametrize("special", NON_FINITE, ids=repr)
def test_a_non_finite_number_never_crashes_a_check(special):
    for expect in ({"value": 1}, {"rows": [[1]]}, {"row_count": 1}, {"unchanged": True}):
        outcome = evaluate_db(check(expect=expect), result([special]), result([special]))
        assert outcome.passed in (True, False)
    delta = evaluate_db(check(expect={"delta": 1}), result([0]), result([special]))
    assert (
        delta.passed is None and delta.detail == "db 'x': delta needs a number in the first column"
    )


@pytest.mark.security
@pytest.mark.parametrize("special", NON_FINITE, ids=repr)
def test_a_non_finite_value_equal_before_and_after_is_not_a_forbidden_change(special):
    change = ForbiddenChange(id="g", query="SELECT amount FROM ajustes")
    assert forbidden_check(change, result([special]), result([special]), 1) is None


def test_a_non_finite_value_that_appears_is_a_forbidden_change():
    change = ForbiddenChange(id="g", query="SELECT amount FROM ajustes")
    failed = forbidden_check(change, result([1]), result([Decimal("NaN")]), 1)
    assert failed is not None and failed.passed is False


def test_a_non_finite_value_matches_only_its_stable_text_form():
    assert evaluate_db(check(expect={"value": "NaN"}), None, result([Decimal("sNaN")])).passed
    assert evaluate_db(check(expect={"value": "-Infinity"}), None, result([float("-inf")])).passed
    assert not evaluate_db(check(expect={"value": 1}), None, result([Decimal("NaN")])).passed


LONG_A = Decimal("1234567890123456.13")
LONG_B = Decimal("1234567890123456.14")
DIGITS_A = Decimal("1" * 40 + ".5")
DIGITS_B = Decimal("1" * 40 + ".6")
WIDE_A = Decimal("2" * 40)
WIDE_B = Decimal("2" * 39 + "3")


@pytest.mark.security
@pytest.mark.parametrize(
    ("before", "after"),
    [
        (LONG_A, LONG_B),
        (DIGITS_A, DIGITS_B),
        (WIDE_A, WIDE_B),
        (Decimal("100.10"), Decimal("100.20")),
    ],
    ids=["16-digits", "40-digits-fraction", "40-digits-integer", "ordinary"],
)
def test_a_tiny_change_in_a_wide_number_is_still_a_change(before, after):
    change = ForbiddenChange(id="g", query="SELECT amount FROM ajustes")
    failed = forbidden_check(change, result([before]), result([after]), 1)
    assert failed is not None and failed.passed is False
    unchanged = evaluate_db(check(expect={"unchanged": True}), result([before]), result([after]))
    assert unchanged.passed is False


@pytest.mark.security
def test_the_same_wide_number_is_not_a_change_whatever_its_scale():
    change = ForbiddenChange(id="g", query="SELECT amount FROM ajustes")
    assert forbidden_check(change, result([LONG_A]), result([LONG_A]), 1) is None
    assert forbidden_check(change, result([Decimal("1.1")]), result([Decimal("1.10")]), 1) is None


@pytest.mark.parametrize(
    ("expected", "stored"),
    [
        (1.1, Decimal("1.10")),
        (5, Decimal("5.0")),
        (5.0, 5),
        (0.1, Decimal("0.10")),
        (0.1, 0.1),
    ],
    ids=["float-vs-scaled", "int-vs-scaled", "float-vs-int", "tenth", "float-vs-float"],
)
def test_value_and_rows_compare_the_number_not_its_scale(expected, stored):
    assert evaluate_db(check(expect={"value": expected}), None, result([stored])).passed is True
    assert evaluate_db(check(expect={"rows": [[expected]]}), None, result([stored])).passed is True


@pytest.mark.security
def test_value_and_rows_tell_apart_numbers_that_differ_far_below_a_float_digit():
    stored = result([LONG_B])
    assert evaluate_db(check(expect={"value": 1234567890123456.13}), None, stored).passed is False
    assert (
        evaluate_db(check(expect={"rows": [[1234567890123456.13]]}), None, stored).passed is False
    )
    assert (
        evaluate_db(check(expect={"value": 1.2}), None, result([Decimal("1.2000000001")])).passed
        is False
    )


HUGE = Decimal("1E+400")


@pytest.mark.security
@pytest.mark.parametrize(
    ("before", "after", "delta", "passed"),
    [
        (HUGE, Decimal(0), 1, False),
        (Decimal(0), HUGE, 1, False),
        (HUGE, HUGE, 0, True),
        (Decimal(0), HUGE, 0, False),
        (HUGE, HUGE + 1, 1, False),
        (Decimal("1E+999999"), Decimal("1E+999999"), 0, None),
    ],
    ids=["from", "to", "same", "to-zero-delta", "rounded-away", "beyond-exact-limit"],
)
def test_a_numeric_out_of_float_range_is_compared_exactly_and_never_shown(
    before, after, delta, passed
):
    outcome = evaluate_db(check(expect={"delta": delta}), result([before]), result([after]))
    assert outcome.passed is passed
    assert "E+" not in outcome.detail and "0000000" not in outcome.detail


@pytest.mark.security
@pytest.mark.parametrize(
    ("before", "after", "delta", "passed"),
    [
        (1000000000, 2000000001, 1000000000, False),
        (10**12, 10**12 + 500, 500, True),
        (10**12, 10**12 + 500, 0, False),
        (1000000000, 2000000000, 1000000000, True),
        (Decimal("0"), Decimal("1E-10"), 0, False),
        (Decimal("0.3"), Decimal("0.5"), 0.2, True),
        (Decimal("0.1"), Decimal("0.3"), 0.2, True),
        (0, 1, 1.0, True),
        (0, 1, 1.0000001, False),
    ],
    ids=[
        "int-off-by-one",
        "int-exact",
        "int-500-over-zero",
        "int-exact-big",
        "numeric-tiny",
        "decimal-point-two",
        "decimal-tenths",
        "float-expected-int-change",
        "float-expected-off",
    ],
)
def test_delta_is_exact_when_no_operand_is_a_float(before, after, delta, passed):
    outcome = evaluate_db(check(expect={"delta": delta}), result([before]), result([after]))
    assert outcome.passed is passed


@pytest.mark.security
def test_delta_has_no_relative_tolerance():
    big = check(expect={"delta": 1000000000})
    assert evaluate_db(big, result([0]), result([1000000001])).passed is False
    assert evaluate_db(big, result([0]), result([1000000000])).passed is True


@pytest.mark.parametrize(
    ("before", "after", "delta", "passed"),
    [
        (0.1, 0.3, 0.2, True),
        (0.3, 0.1, -0.2, True),
        (0.1, 0.3, 0, False),
        (0.0, 5e-10, 0, True),
        (0.0, 2e-9, 0, False),
        (1e15, 1e15 + 1.0, 0, False),
        (1e15, 1e15, 0, True),
        (Decimal("0.1"), 0.3, 0.2, True),
    ],
    ids=[
        "float-columns",
        "negative",
        "real-change",
        "inside-absolute",
        "outside-absolute",
        "big-no-relative",
        "big-same",
        "mixed-decimal-float",
    ],
)
def test_delta_on_float_columns_allows_an_absolute_tolerance_of_a_billionth(
    before, after, delta, passed
):
    outcome = evaluate_db(check(expect={"delta": delta}), result([before]), result([after]))
    assert outcome.passed is passed


def test_delta_reads_a_float_expected_value_as_its_shortest_decimal():
    outcome = evaluate_db(
        check(expect={"delta": 0.1}), result([Decimal("1.0")]), result([Decimal("1.1")])
    )
    assert outcome.passed is True
    assert outcome.detail == "db 'x': changed by 0.1 (expected 0.1)"


@pytest.mark.security
def test_delta_between_the_extremes_of_a_float_is_exact_and_never_crashes():
    outcome = evaluate_db(check(expect={"delta": 0}), result([1e308]), result([-1e308]))
    assert outcome.passed is False
    assert outcome.detail == "db 'x': changed by a value out of range (expected 0)"
    assert "e+308" not in outcome.detail


@pytest.mark.security
@pytest.mark.parametrize("special", NON_FINITE, ids=repr)
@pytest.mark.parametrize("expected", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_expected_delta_is_never_a_pass(special, expected):
    wrong = DbCheck.model_construct(
        id="x", query="SELECT 1", params={}, kind="delta", value=expected, hard=False
    )
    for before, after in ((1, 2), (1, 1), (special, 1)):
        assert evaluate_db(wrong, result([before]), result([after])).passed is not True
    outcome = evaluate_db(wrong, result([1]), result([2]))
    assert (outcome.passed, outcome.detail) == (None, "db 'x': delta out of range")


@pytest.mark.parametrize("amount", [True, False, "7", None])
def test_delta_needs_a_real_number_not_a_bool_or_text(amount):
    outcome = evaluate_db(check(expect={"delta": 1}), result([0]), result([amount]))
    assert outcome.passed is None
