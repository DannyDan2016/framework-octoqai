"""SQLAlchemy adapter of the database port."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, quote_plus, unquote

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.pool import NullPool

from octoqai.db.port import QueryResult
from octoqai.errors import ConfigError

_SENSITIVE_KEY = re.compile(r"pass|pwd|secret|token|key", re.IGNORECASE)
_NO_RESULT_SET = "query returned no result set"


def _forms(value: str) -> set[str]:
    return {value, quote(value, safe=""), quote_plus(value), unquote(value)}


def _parsed_secrets(url: str) -> list[str]:
    try:
        parsed = make_url(url)
    except Exception:
        return []
    found = [parsed.password]
    for key, item in parsed.query.items():
        if _SENSITIVE_KEY.search(key):
            found.extend((item,) if isinstance(item, str) else item)
    return [value for value in found if value]


def _cut_passwords(url: str) -> list[str]:
    rest = url.partition("://")[2]
    cuts = (rest.find("@"), rest.rfind("@"))
    return [rest[:cut].partition(":")[2] for cut in cuts if cut >= 0]


def connection_secrets(url: str) -> list[str]:
    """List the values of a database URL that must never be shown.

    The password is read as SQLAlchemy reads it and also cut from the raw text, up to the
    first and up to the last ``@``, so a password with an unencoded ``/``, ``#``, ``?`` or
    ``@`` is still found. Values of query keys that look like credentials are included.

    Args:
        url: Database URL; it is never echoed and a malformed one never raises.

    Returns:
        Each value in its plain, unquoted, percent-encoded and plus-encoded form, without
        empty strings or duplicates.
    """
    found = [*_parsed_secrets(url), *_cut_passwords(url)]
    forms = [form for value in found if value for form in _forms(value)]
    return [form for form in dict.fromkeys(forms) if form]


def _explicit_transactions(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _manual(dbapi_connection: Any, record: Any) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN")


class SqlDatabase:
    """Database adapter on top of a SQLAlchemy engine."""

    def __init__(self, url: str) -> None:
        """Create the engine.

        Args:
            url: SQLAlchemy URL of the database.

        Raises:
            ConfigError: When the URL or its driver is not usable; the URL is never echoed.
        """
        try:
            self._engine = create_engine(url, hide_parameters=True, poolclass=NullPool)
        except Exception:
            raise ConfigError(
                "database.url is not a valid SQLAlchemy URL or its driver is not installed"
            ) from None
        if self._engine.dialect.name == "sqlite":
            _explicit_transactions(self._engine)

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Run a query and roll back, so nothing it does is ever committed.

        Args:
            sql: The query.
            params: Values bound to the query.

        Returns:
            The rows, or only the class name of the exception when it fails: a driver message
            can repeat a stored value, a parameter or a connection detail.
        """
        try:
            with self._engine.connect() as conn:
                try:
                    result = conn.execute(text(sql), dict(params))
                    if not result.returns_rows:
                        return QueryResult(error=_NO_RESULT_SET)
                    columns = tuple(result.keys())
                    rows = tuple(tuple(r) for r in result.fetchall())
                finally:
                    conn.rollback()
        except Exception as exc:
            return QueryResult(error=type(exc).__name__)
        return QueryResult(columns, rows, None)

    def execute(self, statements: Sequence[str]) -> str | None:
        """Run the statements in one transaction.

        Statements may hold environment values, so a failure is reported as the exception
        class name only, never as driver text.

        Args:
            statements: SQL statements to run in order.

        Returns:
            ``None`` on success, otherwise the class name of the exception.
        """
        try:
            with self._engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))
        except Exception as exc:
            return type(exc).__name__
        return None

    def close(self) -> None:
        """Dispose of the engine and its connections."""
        self._engine.dispose()
