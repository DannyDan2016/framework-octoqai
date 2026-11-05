"""SQLAlchemy adapter of the database port."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from octoqai.db.port import QueryResult
from octoqai.errors import ConfigError

_MAX_ERROR_CHARS = 200
_MIN_PARAM_CHARS = 4
_NO_PARAMS: Mapping[str, Any] = MappingProxyType({})


def scrub_error(message: str, url: str | URL, params: Mapping[str, Any] = _NO_PARAMS) -> str:
    """Mask connection details and parameter values that a driver message may repeat.

    Args:
        message: Text produced by the driver.
        url: Database URL; its password, user name, host and database are masked.
        params: Query parameters; values of four characters or more are masked.

    Returns:
        The message with every sensitive value replaced by ``[redacted]``.
    """
    parsed = make_url(url)
    secrets = {parsed.password, parsed.username, parsed.host, parsed.database}
    secrets |= {str(v) for v in params.values() if len(str(v)) >= _MIN_PARAM_CHARS}
    values = sorted((s for s in secrets if s), key=len, reverse=True)
    if not values:
        return message
    pattern = re.compile("|".join(re.escape(v) for v in values), re.IGNORECASE)
    return pattern.sub("[redacted]", message)


def _describe(exc: SQLAlchemyError) -> str:
    original = getattr(exc, "orig", None)
    lines = str(original).splitlines() if original is not None else []
    detail = lines[0] if lines else ""
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name


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
            self._engine = create_engine(url, hide_parameters=True)
        except Exception:
            raise ConfigError(
                "database.url is not a valid SQLAlchemy URL or its driver is not installed"
            ) from None

    def _error(self, exc: SQLAlchemyError, params: Mapping[str, Any] = _NO_PARAMS) -> str:
        cleaned = scrub_error(_describe(exc), self._engine.url, params)
        return cleaned[:_MAX_ERROR_CHARS]

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Run a query and roll back, so nothing it does is ever committed."""
        try:
            with self._engine.connect() as conn:
                try:
                    result = conn.execute(text(sql), dict(params))
                    columns = tuple(result.keys()) if result.returns_rows else ()
                    rows = tuple(tuple(r) for r in result.fetchall()) if columns else ()
                finally:
                    conn.rollback()
        except SQLAlchemyError as exc:
            return QueryResult(error=self._error(exc, params))
        return QueryResult(columns, rows, None)

    def execute(self, statements: Sequence[str]) -> str | None:
        """Run the statements in one transaction; return the error or ``None``."""
        try:
            with self._engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))
        except SQLAlchemyError as exc:
            return self._error(exc)
        return None

    def close(self) -> None:
        """Dispose of the engine and its connections."""
        self._engine.dispose()
