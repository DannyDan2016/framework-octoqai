"""Hermetic SQLite test database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

BANK_SCHEMA = (
    "CREATE TABLE ajustes (id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, "
    "cuenta TEXT NOT NULL, importe NUMERIC NOT NULL)",
)


class SqliteFile:
    """A SQLite database in a file, opened and closed on every call."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def url(self) -> str:
        """SQLAlchemy URL of the file."""
        return f"sqlite:///{self.path.as_posix()}"

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
