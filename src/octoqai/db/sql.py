"""SQLAlchemy adapter of the database port."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from urllib.parse import quote, quote_plus, unquote, unquote_plus

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.pool import NullPool

from octoqai.db.port import QueryResult
from octoqai.errors import ConfigError
from octoqai.experiment import is_sensitive_key

_UP_TO_SEPARATOR = re.compile(r"[^=;]*")
_BLANK = re.compile(r"\s*")
_MAX_CUTS = 8
_MAX_PAIRS = 32
_NO_RESULT_SET = "query returned no result set"


def _forms(value: str) -> set[str]:
    found = {value, unquote(value)}
    for encode in (lambda text: quote(text, safe=""), quote_plus):
        try:
            found.add(encode(value))
        except UnicodeEncodeError:
            continue
    return found


def _parsed_secrets(url: str) -> list[str]:
    try:
        parsed = make_url(url)
    except Exception:
        return []
    found = [parsed.password]
    for key, item in parsed.query.items():
        if is_sensitive_key(key):
            found.extend((item,) if isinstance(item, str) else item)
    return [value for value in found if value]


def _pairs(query: str) -> Iterator[tuple[str, int]]:
    done = 0
    for pair in query.split("&"):
        name, separator, _ = pair.partition("=")
        if separator:
            yield name, done + len(name) + 1
        done += len(pair) + 1


def _cuts(tail: str) -> list[str]:
    found = [tail]
    end = tail.find("&")
    while end >= 0 and len(found) <= _MAX_CUTS:
        found.append(tail[:end])
        end = tail.find("&", end + 1)
    return found


def _end(pattern: re.Pattern[str], text: str, start: int) -> int:
    found = pattern.match(text, start)
    return found.end() if found else start


def _brace_end(text: str, start: int) -> int:
    index = start + 1
    while True:
        close = text.find("}", index)
        if close < 0:
            return -1
        if text.startswith("}}", close):
            index = close + 2
        else:
            return close + 1


def _odbc_value(text: str, start: int, closable: bool) -> tuple[str, str, int, bool]:
    stop = text.find(";", start)
    stop = len(text) if stop < 0 else stop
    if not text.startswith("{", start):
        raw = text[start:stop]
        return raw, raw.strip(), stop, closable
    end = _brace_end(text, start) if closable else -1
    if end >= 0:
        raw = text[start:end]
        stop = text.find(";", end)
        return raw, raw[1:-1].replace("}}", "}"), len(text) if stop < 0 else stop, True
    raw = text[start:stop]
    return raw, raw[1:].replace("}}", "}"), stop, False


def _odbc_pairs(text: str) -> Iterator[tuple[str, str, str]]:
    position, closable = 0, True
    while position <= len(text):
        key_end = _end(_UP_TO_SEPARATOR, text, position)
        if key_end >= len(text) or text[key_end] != "=":
            position = key_end + 1
            continue
        start = _end(_BLANK, text, key_end + 1)
        raw, inner, stop, closable = _odbc_value(text, start, closable)
        yield text[position:key_end].strip(), raw, inner
        position = stop + 1


def _odbc_values(text: str) -> list[str]:
    found: list[str] = []
    for name, raw, inner in _odbc_pairs(text):
        if is_sensitive_key(name):
            found.extend((raw, inner))
            if raw.startswith("{") and raw.endswith("}"):
                found.append(raw[1:-1])
    return found


def _odbc_secrets(url: str) -> list[str]:
    query = url.partition("?")[2]
    found: list[str] = []
    seen = 0
    for name, start in _pairs(query):
        if unquote_plus(name).lower() != "odbc_connect":
            continue
        seen += 1
        if seen > _MAX_PAIRS:
            break
        for raw in _cuts(query[start:]):
            decoded = unquote_plus(raw)
            for form in dict.fromkeys((decoded, unquote(decoded))):
                found.append(form)
                found.extend(_odbc_values(form))
    return found


def _query_tails(url: str) -> list[str]:
    query = url.partition("?")[2]
    tails: list[str] = []
    seen = 0
    for name, start in _pairs(query):
        if is_sensitive_key(name):
            seen += 1
            if seen > _MAX_PAIRS:
                break
            tails.extend(_cuts(query[start:]))
    return tails


def _cut_passwords(url: str) -> list[str]:
    rest = url.partition("://")[2]
    cuts = (rest.find("@"), rest.rfind("@"))
    return [rest[:cut].partition(":")[2] for cut in cuts if cut >= 0]


def connection_secrets(url: str) -> list[str]:
    """List the values of a database URL that must never be shown.

    The password is read as SQLAlchemy reads it and also cut from the raw text, up to the
    first and up to the last ``@``, so a password with an unencoded ``/``, ``#``, ``?`` or
    ``@`` is still found. Values of query keys that look like credentials are included, and so
    are the whole ``odbc_connect`` value and the credential entries inside it. After an
    unencoded ``&`` in a credential value, the rest of the query is included too.

    Args:
        url: Database URL; it is never echoed and a malformed one never raises.

    Returns:
        Each value in its plain, unquoted, percent-encoded and plus-encoded form, without
        empty strings or duplicates.
    """
    found = [
        *_parsed_secrets(url),
        *_cut_passwords(url),
        *_odbc_secrets(url),
        *_query_tails(url),
    ]
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
