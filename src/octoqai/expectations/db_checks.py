"""Pure evaluation of database checks and forbidden changes."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from octoqai.db.port import QueryResult
from octoqai.expectations.checks import CheckResult
from octoqai.experiment import DbCheck, ForbiddenChange

DB_KIND = "db"
FORBIDDEN_KIND = "forbidden_crud"

_Outcome = tuple[bool | None, str]


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


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal | float) and not (
        value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
    ):
        return _non_finite(value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, date | datetime | time):
        return value.isoformat()
    return value


def _normalized(result: QueryResult) -> list[tuple[Any, ...]]:
    return [tuple(_plain(item) for item in row) for row in result.rows]


def _row_count(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    count = len(after.rows)
    return count == check.value, f"db {check.id!r}: {count} row(s) (expected {check.value})"


def _value(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    if not after.rows:
        return False, f"db {check.id!r}: query returned no rows"
    matches = _plain(after.scalar) == _plain(check.value)
    return matches, f"db {check.id!r}: " + (
        "value matches" if matches else "value differs from expected"
    )


def _rows(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    expected = [tuple(_plain(item) for item in row) for row in check.value]
    got = _normalized(after)
    if got == expected:
        return True, f"db {check.id!r}: rows match"
    return False, f"db {check.id!r}: rows differ (got {len(got)}, expected {len(expected)})"


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _delta(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    assert before is not None
    start, end = _plain(before.scalar), _plain(after.scalar)
    if not (_number(start) and _number(end)):
        return None, f"db {check.id!r}: delta needs a number in the first column"
    try:
        float(start), float(end)
        change = end - start
        matches = math.isclose(change, check.value, abs_tol=1e-9)
        return matches, f"db {check.id!r}: changed by {change:g} (expected {check.value:g})"
    except OverflowError:
        return None, f"db {check.id!r}: delta out of range"


def _unchanged(check: DbCheck, before: QueryResult | None, after: QueryResult) -> _Outcome:
    assert before is not None
    same = _normalized(before) == _normalized(after)
    return same, f"db {check.id!r}: " + ("unchanged" if same else "changed")


_DB_CHECKS: dict[str, Callable[[DbCheck, QueryResult | None, QueryResult], _Outcome]] = {
    "row_count": _row_count,
    "value": _value,
    "rows": _rows,
    "delta": _delta,
    "unchanged": _unchanged,
}


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

    Args:
        check: The check to evaluate.
        before: Snapshot taken before the turn; needed by ``delta`` and ``unchanged``.
        after: Result of the check query after the turn.
        agent_error: Error of the agent reply, if any; it fails the check.

    Returns:
        The check result; ``passed`` is ``None`` when the database could not be read.
        Details never include values read from the database.
    """
    if agent_error is not None:
        detail = f"db {check.id!r}: agent error: {agent_error}"
        return CheckResult(DB_KIND, False, check.hard, detail, True)
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
        could not be read, and ``None`` when nothing changed.
    """
    error = after.error or before.error
    if error is not None:
        detail = f"forbidden {change.id!r}: database error: {error}"
        return CheckResult(FORBIDDEN_KIND, None, True, detail)
    if _normalized(before) == _normalized(after):
        return None
    detail = f"forbidden change {change.id!r} after turn {turn_number}"
    return CheckResult(FORBIDDEN_KIND, False, True, detail)
