"""Port that every database adapter implements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

MAX_ROWS = 10_000


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by one query, or the error that prevented them.

    A query that returns more than ``MAX_ROWS`` rows keeps no rows: it has an ``error`` and
    ``overflow`` set, so code that ignores ``overflow`` treats it as unreadable, never as a
    pass.
    """

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None
    overflow: bool = False

    @property
    def scalar(self) -> Any:
        """First column of the first row, or ``None`` when there are no rows."""
        return self.rows[0][0] if self.rows else None


class DatabasePort(Protocol):
    """Connection to the database of the system under test."""

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Run a read-only query and never commit.

        A failure is reported in ``error`` as the exception class name only, never as driver
        text, which can repeat stored values, parameters or connection details.
        """
        ...

    def execute(self, statements: Sequence[str]) -> str | None:
        """Run the statements in one transaction; return the exception class name, or ``None``."""
        ...

    def close(self) -> None:
        """Release the resources held by the adapter."""
        ...
