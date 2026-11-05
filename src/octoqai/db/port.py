"""Port that every database adapter implements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by one query, or the error that prevented them."""

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None

    @property
    def scalar(self) -> Any:
        """First column of the first row, or ``None`` when there are no rows."""
        return self.rows[0][0] if self.rows else None


class DatabasePort(Protocol):
    """Connection to the database of the system under test."""

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Run a read-only query and never commit; failures are reported in ``error``."""
        ...

    def execute(self, statements: Sequence[str]) -> str | None:
        """Run the statements in one transaction and return the error, or ``None``."""
        ...

    def close(self) -> None:
        """Release the resources held by the adapter."""
        ...
