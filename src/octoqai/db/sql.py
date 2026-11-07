"""SQLAlchemy adapter of the database port."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, Connection, Engine, make_url
from sqlalchemy.exc import DataError, DBAPIError, IntegrityError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from octoqai.db.port import QueryResult
from octoqai.errors import ConfigError

_MAX_ERROR_CHARS = 200
_MIN_PARAM_CHARS = 4
_NO_PARAMS: Mapping[str, Any] = MappingProxyType({})
_SENSITIVE_KEY = re.compile(r"pass|pwd|secret|token|key", re.IGNORECASE)
_VALUE_ERRORS = (DataError, IntegrityError)
_NO_RESULT_SET = "query returned no result set"
REDACTED = "[redacted]"
_ADDRESS = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])|\[[0-9A-Fa-f]*:[0-9A-Fa-f:.]*\]")


def _forms(value: str) -> set[str]:
    return {value, quote(value, safe=""), quote_plus(value), unquote(value)}


def _raw_secrets(url: str) -> set[str]:
    try:
        parts = urlsplit(url)
        found = {parts.password, parts.username}
        found.update(value for _, value in parse_qsl(parts.query))
        found.update(pair.partition("=")[2] for pair in parts.query.split("&"))
    except ValueError:
        return set()
    return {form for value in found if value for form in _forms(value)}


def _url_secrets(parsed: URL) -> set[str]:
    found = {parsed.password, parsed.username, parsed.host, parsed.database}
    found.add(None if parsed.port is None else str(parsed.port))
    for item in parsed.query.values():
        found.update((item,) if isinstance(item, str) else item)
    return {form for value in found if value for form in _forms(value)}


def scrub_error(
    message: str,
    url: str | URL,
    params: Mapping[str, Any] = _NO_PARAMS,
    secrets: Sequence[str] = (),
) -> str:
    """Mask connection details and parameter values that a driver message may repeat.

    Args:
        message: Text produced by the driver.
        url: Database URL; its password, user name, host, port, database and query values are
            masked, in plain and percent-encoded forms, and so are IP address literals.
        params: Query parameters; values of four characters or more are masked.
        secrets: Further values to mask, such as environment values used by the experiment.

    Returns:
        The message with every sensitive value replaced by ``[redacted]``.
    """
    found = _url_secrets(make_url(url)) | {s for s in secrets if s}
    if isinstance(url, str):
        found |= _raw_secrets(url) | set(connection_secrets(url))
    found |= {str(v) for v in params.values() if len(str(v)) >= _MIN_PARAM_CHARS}
    values = sorted(found, key=len, reverse=True)
    if values:
        message = re.compile("|".join(re.escape(v) for v in values), re.IGNORECASE).sub(
            REDACTED, message
        )
    return _ADDRESS.sub(REDACTED, message)


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


def _describe(exc: SQLAlchemyError) -> str:
    name = type(exc).__name__
    if not isinstance(exc, DBAPIError) or isinstance(exc, _VALUE_ERRORS):
        return name
    lines = str(exc.orig).splitlines() if exc.orig is not None else []
    return f"{name}: {lines[0]}" if lines and lines[0] else name


def _explicit_transactions(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _manual(dbapi_connection: Any, record: Any) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN")


class SqlDatabase:
    """Database adapter on top of a SQLAlchemy engine."""

    def __init__(self, url: str, secrets: Sequence[str] = ()) -> None:
        """Create the engine.

        Args:
            url: SQLAlchemy URL of the database.
            secrets: Values scrubbed from error messages besides the connection details.

        Raises:
            ConfigError: When the URL or its driver is not usable; the URL is never echoed.
        """
        try:
            self._engine = create_engine(url, hide_parameters=True, poolclass=NullPool)
        except Exception:
            raise ConfigError(
                "database.url is not a valid SQLAlchemy URL or its driver is not installed"
            ) from None
        self._url = url
        self._secrets = tuple(secrets)
        if self._engine.dialect.name == "sqlite":
            _explicit_transactions(self._engine)

    def _error(self, exc: SQLAlchemyError, params: Mapping[str, Any] = _NO_PARAMS) -> str:
        cleaned = scrub_error(_describe(exc), self._url, params, self._secrets)
        return cleaned[:_MAX_ERROR_CHARS]

    def query(self, sql: str, params: Mapping[str, Any]) -> QueryResult:
        """Run a query and roll back, so nothing it does is ever committed."""
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
        except SQLAlchemyError as exc:
            return QueryResult(error=self._error(exc, params))
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
