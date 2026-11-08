"""Pure evaluation of database checks and forbidden changes."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from octoqai.db.port import MAX_ROWS, QueryResult
from octoqai.expectations.checks import CheckResult
from octoqai.experiment import DbCheck, ForbiddenChange, SourceFloat

DB_KIND = "db"
FORBIDDEN_KIND = "forbidden_crud"

_Outcome = tuple[bool | None, str]
_FLOAT_TOLERANCE = Fraction(1, 10**9)
_MAX_EXPONENT = 100_000
_OUT_OF_RANGE = "a value out of range"


def _non_finite(value: Decimal | float) -> str:
    if isinstance(value, Decimal):
        negative = value.is_signed()
        nan = value.is_nan()
    else:
        negative = value < 0
        nan = math.isnan(value)
    if nan:
        return "NaN"
    return "-Infinity" if negative else "Infinity"


def _written(value: float) -> Decimal:
    if isinstance(value, SourceFloat) and value.source is not None:
        try:
            written = Decimal(value.source)
        except InvalidOperation:
            return Decimal(repr(value))
        if written.is_finite():
            return written
    return Decimal(repr(value))


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal | float) and not (
        value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
    ):
        return _non_finite(value)
    if isinstance(value, float):
        return _written(value)
    if isinstance(value, date | datetime | time):
        return value.isoformat()
    return value


def _key(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, date | datetime | time):
        return "time", value.isoformat()
    plain = _plain(value)
    if isinstance(value, Decimal | float) and isinstance(plain, str):
        return "non-finite", plain
    if isinstance(plain, int | Decimal):
        return "number", plain
    return "other", plain


def _normalized(result: QueryResult) -> list[tuple[Any, ...]]:
    return [tuple(_key(item) for item in row) for row in result.rows]


def _same_rows(before: QueryResult, after: QueryResult) -> bool:
    return _normalized(before) == _normalized(after)


def _one_or_zero(stored: Any) -> bool:
    return _number(stored) and stored in (0, 1)


def _same(stored: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return stored is expected or (_one_or_zero(_plain(stored)) and bool(stored) is expected)
    if isinstance(stored, bool):
        return False
    if (
        isinstance(stored, float)
        and isinstance(expected, float)
        and math.isfinite(stored)
        and math.isfinite(expected)
    ):
        return stored == float(expected)
    return bool(_plain(stored) == _plain(expected))


def _row_count(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    count = len(after.rows)
    return count == check.value, f"db {check.id!r}: {count} row(s) (expected {check.value})"


def _value(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    if not after.rows:
        return False, f"db {check.id!r}: query returned no rows"
    matches = _same(after.scalar, check.value)
    return matches, f"db {check.id!r}: " + (
        "value matches" if matches else "value differs from expected"
    )


def _same_row(stored: Any, expected: Any) -> bool:
    return len(stored) == len(expected) and all(
        _same(*pair) for pair in zip(stored, expected, strict=True)
    )


def _rows(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    got = after.rows
    expected = check.value
    if len(got) == len(expected) and all(
        _same_row(*pair) for pair in zip(got, expected, strict=True)
    ):
        return True, f"db {check.id!r}: rows match"
    return False, f"db {check.id!r}: rows differ (got {len(got)}, expected {len(expected)})"


def _number(value: Any) -> bool:
    return isinstance(value, int | Decimal) and not isinstance(value, bool)


def _exact(value: int | float | Decimal) -> Fraction:
    if isinstance(value, float):
        value = _written(value)
    if isinstance(value, Decimal) and (
        abs(value.adjusted()) > _MAX_EXPONENT or len(value.as_tuple().digits) > _MAX_EXPONENT
    ):
        raise OverflowError
    return Fraction(value)


def _shown(value: Fraction | int | float) -> str:
    try:
        return f"{float(value):g}"
    except OverflowError:
        return _OUT_OF_RANGE


def _within(change: Fraction, expected: Fraction, tolerant: bool) -> bool:
    if tolerant:
        return abs(change - expected) <= _FLOAT_TOLERANCE
    return change == expected


def _delta(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    assert before is not None
    raw = (before.scalar, after.scalar)
    start, end = _plain(raw[0]), _plain(raw[1])
    if not (_number(start) and _number(end)):
        return None, f"db {check.id!r}: delta needs a number in the first column"
    try:
        if isinstance(check.value, float) and not math.isfinite(check.value):
            raise OverflowError
        change = _exact(end) - _exact(start)
        matches = _within(change, _exact(check.value), any(isinstance(v, float) for v in raw))
    except OverflowError:
        return None, f"db {check.id!r}: delta out of range"
    shown = f"changed by {_shown(change)} (expected {_shown(check.value)}"
    shown += ")" if matches else ", not equal)"
    return matches, f"db {check.id!r}: {shown}"


def _unchanged(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    assert before is not None
    same = _same_rows(before, after)
    return same, f"db {check.id!r}: " + ("unchanged" if same else "changed")


_DB_CHECKS: dict[str, Callable[[DbCheck, QueryResult | None, QueryResult], _Outcome]] = {
    "row_count": _row_count,
    "value": _value,
    "rows": _rows,
    "delta": _delta,
    "unchanged": _unchanged,
}


def _one_side_overflows(before: QueryResult, after: QueryResult) -> bool:
    return (before.overflow and after.error is None) or (after.overflow and before.error is None)


def _overflow_failure(check: DbCheck, before: QueryResult | None, after: QueryResult) -> str | None:
    label = f"db {check.id!r}"
    if check.kind == "unchanged" and before is not None and _one_side_overflows(before, after):
        return f"{label}: changed (one snapshot has more than {MAX_ROWS} rows)"
    if not after.overflow:
        return None
    if check.kind == "row_count" and check.value <= MAX_ROWS:
        return f"{label}: more than {MAX_ROWS} row(s) (expected {check.value})"
    if check.kind == "rows" and len(check.value) <= MAX_ROWS:
        return f"{label}: rows differ (got more than {MAX_ROWS}, expected {len(check.value)})"
    return None


def _database_error(check: DbCheck, before: QueryResult | None, after: QueryResult) -> str | None:
    if after.error is not None:
        return after.error
    if check.needs_before:
        return "no snapshot taken before the turn" if before is None else before.error
    return None


def evaluate_db(
    check: DbCheck,
    before: QueryResult | None,
    after: QueryResult,
    *,
    agent_error: str | None = None,
) -> CheckResult:
    """Evaluate one database check without touching the database.

    Numbers are compared exactly, never rounded: a ``Decimal`` keeps all its digits, a float
    from the database is read as its shortest decimal text, a float written in the experiment
    is read as the text the user wrote, and only the value counts, not the scale, so ``1.1``
    equals ``Decimal("1.10")``. ``delta`` is computed with exact fractions and must equal the
    expected value; when a database value is a float it may differ by at most ``1e-9``, never
    by a relative amount. A non-finite expected delta, or a number with an exponent or a
    digit count beyond ``100000``, is not evaluable.

    Types are kept apart. Against an expected value (``value``, ``rows``), an expected ``true``
    or ``false`` matches a database ``bool`` or a number equal to ``1`` or ``0``, because
    SQLite and MySQL have no real booleans; an expected number never matches a database
    ``bool``; a date matches its ISO text, since YAML writes dates that way. Between two
    snapshots (``unchanged`` and forbidden changes) a ``bool`` is not a number, a non-finite
    number is not its text and a date is not its ISO text, while ``Decimal("1.10")`` equals
    ``Decimal("1.1")`` and ``1`` equals ``Decimal("1")``, because drivers may vary the type.

    Args:
        check: The check to evaluate.
        before: Snapshot taken before the turn; needed by ``delta`` and ``unchanged``.
        after: Result of the check query after the turn.
        agent_error: Error of the agent reply, if any; it fails the check.

    A result over ``MAX_ROWS`` rows fails only what it proves: ``row_count`` or ``rows``
    expecting fewer rows, and ``unchanged`` when the other snapshot was readable. Anything
    else it touches is not evaluable.

    Returns:
        The check result; ``passed`` is ``None`` when the database could not be read or a
        ``delta`` cannot be computed. Details never include values read from the database,
        except the size of a ``delta``.
    """
    if agent_error is not None:
        detail = f"db {check.id!r}: agent error: {agent_error}"
        return CheckResult(DB_KIND, False, check.hard, detail, True)
    proven = _overflow_failure(check, before, after)
    if proven is not None:
        return CheckResult(DB_KIND, False, check.hard, proven)
    error = _database_error(check, before, after)
    if error is not None:
        return CheckResult(DB_KIND, None, check.hard, f"db {check.id!r}: database error: {error}")
    passed, detail = _DB_CHECKS[check.kind](check, before, after)
    return CheckResult(DB_KIND, passed, check.hard, detail)


def forbidden_check(
    change: ForbiddenChange, before: QueryResult, after: QueryResult, turn_number: int
) -> CheckResult | None:
    """Detect a forbidden change by comparing the query result before and after a turn.

    Args:
        change: The forbidden change definition.
        before: Result of its query before the turn.
        after: Result of its query after the turn.
        turn_number: Number of the turn that was just executed.

    Returns:
        A hard failure when the rows changed, a hard non-evaluable result when the database
        could not be read, and ``None`` when nothing changed. Numbers are compared exactly, so
        a change in the last digit of a wide ``NUMERIC`` is a change. A snapshot over
        ``MAX_ROWS`` rows against a readable one is a change, since the readable one fits.
    """
    changed = CheckResult(
        FORBIDDEN_KIND, False, True, f"forbidden change {change.id!r} after turn {turn_number}"
    )
    if _one_side_overflows(before, after):
        return changed
    error = after.error if after.error is not None else before.error
    if error is not None:
        detail = f"forbidden {change.id!r}: database error: {error}"
        return CheckResult(FORBIDDEN_KIND, None, True, detail)
    return None if _same_rows(before, after) else changed
