"""Hermetic SQLite test database."""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from octoqai.db.port import MAX_ROWS, QueryResult

BANK_SCHEMA = (
    "CREATE TABLE ajustes (id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, "
    "cuenta TEXT NOT NULL, importe NUMERIC NOT NULL)",
)

RECREATE_LINES = (
    "import os, sqlite3, sys",
    "path = sys.argv[1]",
    "if os.path.exists(path):",
    "    os.remove(path)",
    "connection = sqlite3.connect(path)",
    "connection.execute(sys.argv[2])",
    "connection.commit()",
    "connection.close()",
)


def sqlite_url(path: Path) -> str:
    """SQLAlchemy URL of a SQLite file."""
    return f"sqlite:///{path.as_posix()}"


def recreate_command(directory: Path, database: SqliteFile) -> list[str]:
    """Write a script that deletes and recreates the SQLite file; return its argv."""
    script = directory / "recreate.py"
    script.write_text(chr(10).join(RECREATE_LINES), encoding="utf-8")
    return [sys.executable, str(script), str(database.path), BANK_SCHEMA[0]]


class SqliteFile:
    """A SQLite database in a file, opened and closed on every call."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def url(self) -> str:
        """SQLAlchemy URL of the file."""
        return sqlite_url(self.path)

    def run(self, *statements: str) -> None:
        """Run the statements in one transaction and commit."""
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                for statement in statements:
                    connection.execute(statement)
        finally:
            connection.close()

    def scalar(self, sql: str) -> Any:
        """Return the first column of the first row, or ``None``."""
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(sql).fetchone()
        finally:
            connection.close()
        return None if row is None else row[0]

    def insert(self, tenant: str = "A", cuenta: str = "001", importe: float = 10.0) -> None:
        """Insert one row into ``ajustes``."""
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute(
                    "INSERT INTO ajustes (tenant, cuenta, importe) VALUES (?, ?, ?)",
                    (tenant, cuenta, importe),
                )
        finally:
            connection.close()


def result(*rows: Sequence[Any], error: str | None = None) -> QueryResult:
    """Build a ``QueryResult`` with columns ``c0``, ``c1``... sized by the first row."""
    width = len(rows[0]) if rows else 0
    return QueryResult(
        columns=tuple(f"c{index}" for index in range(width)),
        rows=tuple(tuple(row) for row in rows),
        error=error,
    )


def overflowed() -> QueryResult:
    """Build the result of a query that returned more than ``MAX_ROWS`` rows."""
    return QueryResult(error=f"query returned more than {MAX_ROWS} rows", overflow=True)


class FakeDatabase:
    """In-memory ``DatabasePort`` that records the statements and queries it is asked to run."""

    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.executed: list[list[str]] = []
        self.queries: list[str] = []

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Record the query and return an empty result."""
        self.queries.append(sql)
        return QueryResult()

    def execute(self, statements: Sequence[str]) -> str | None:
        """Record the statements and return the configured error."""
        self.executed.append(list(statements))
        return self.error

    def close(self) -> None:
        """Do nothing."""
