"""Database snapshots around turns and the checks evaluated from them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from octoqai.agents.port import AgentReply
from octoqai.db.port import DatabasePort, QueryResult
from octoqai.expectations import CheckResult, evaluate_db, forbidden_check
from octoqai.experiment import Case, DbCheck, Experiment, ForbiddenChange, Turn


class _NoDatabase:
    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        return QueryResult(error="no database connection")

    def execute(self, statements: Sequence[str]) -> str | None:
        return "no database connection"

    def close(self) -> None:
        return None


class DbProbe:
    """Takes database snapshots during one trial and evaluates the checks from them."""

    def __init__(self, db: DatabasePort | None, exp: Experiment, case: Case) -> None:
        """Take the snapshots that must precede the first turn.

        Args:
            db: Database to read, or ``None`` when there is no connection.
            exp: The experiment, for its global forbidden changes.
            case: The case being played.
        """
        self._db: DatabasePort = db if db is not None else _NoDatabase()
        self._case = case
        global_forbidden = exp.database.forbidden if exp.database is not None else []
        self._forbidden = [*global_forbidden, *case.forbidden]
        self._forbidden_start = {c.id: self._snapshot(c) for c in self._forbidden}
        self._failed: set[str] = set()
        self._unreadable: set[str] = set()
        self._case_before = self._snapshots(case.db)
        self._turn_before: dict[int, QueryResult] = {}

    def _snapshot(self, check: DbCheck | ForbiddenChange) -> QueryResult:
        return self._db.query(check.query, check.params)

    def _snapshots(self, checks: Sequence[DbCheck]) -> dict[int, QueryResult]:
        return {i: self._snapshot(c) for i, c in enumerate(checks) if c.needs_before}

    def before_turn(self, turn: Turn) -> None:
        """Snapshot the checks of the turn that compare against the previous state."""
        self._turn_before = self._snapshots(turn.db)

    def _evaluate(
        self,
        checks: Sequence[DbCheck],
        before: Mapping[int, QueryResult],
        agent_error: str | None,
    ) -> list[CheckResult]:
        return [
            evaluate_db(c, before.get(i), self._snapshot(c), agent_error=agent_error)
            for i, c in enumerate(checks)
        ]

    def _forbidden_results(self, number: int) -> list[CheckResult]:
        results: list[CheckResult] = []
        for change in self._forbidden:
            if change.id in self._failed:
                continue
            found = forbidden_check(
                change, self._forbidden_start[change.id], self._snapshot(change), number
            )
            if found is None or self._already_reported(change.id, found):
                continue
            results.append(found)
            if found.passed is False:
                self._failed.add(change.id)
            else:
                self._unreadable.add(change.id)
        return results

    def _already_reported(self, change_id: str, found: CheckResult) -> bool:
        return found.passed is None and change_id in self._unreadable

    def after_turn(self, turn: Turn, reply: AgentReply, number: int) -> list[CheckResult]:
        """Evaluate the turn checks and the forbidden changes, even if the agent failed."""
        checks = self._evaluate(turn.db, self._turn_before, reply.error)
        return checks + self._forbidden_results(number)

    def finish(self, agent_error: str | None) -> list[CheckResult]:
        """Evaluate the checks of the whole case."""
        return self._evaluate(self._case.db, self._case_before, agent_error)
