"""Shared mapping of agent payloads onto reply fields."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.parser import parse as parse_jsonpath

from octoqai.errors import ConfigError
from octoqai.experiment import ResponseMap

MISSING = object()


class ReplyError(Exception):
    def __init__(
        self,
        error: str,
        raw: Any = None,
        body: str | None = None,
        alt_body: str | None = None,
    ) -> None:
        super().__init__(error)
        self.error = error
        self.raw = raw
        self.body = body
        self.alt_body = alt_body


@contextmanager
def mapping_guard(payload: Any) -> Iterator[None]:
    try:
        yield
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplyError(f"cannot map agent response: {type(exc).__name__}", payload) from exc


def compile_path(expr: str) -> JSONPath:
    try:
        return parse_jsonpath(expr)
    except JSONPathError as exc:
        raise ConfigError(f"invalid JSONPath {expr!r}: {exc}") from exc


def compile_optional(expr: str | None) -> JSONPath | None:
    return None if expr is None else compile_path(expr)


def first_match(path: JSONPath, payload: Any) -> Any:
    matches = path.find(payload)
    return matches[0].value if matches else MISSING


class ResponseFields:
    def __init__(self, response: ResponseMap) -> None:
        self._response = response
        self._tool_calls = compile_optional(response.tool_calls)
        self._tokens = compile_optional(response.tokens)

    def tools(self, payload: Any) -> list[str] | None:
        expr = self._response.tool_calls
        if self._tool_calls is None or expr is None:
            return None
        matches = self._tool_calls.find(payload)
        if not matches:
            container = self._container_of(expr, payload)
            return [] if isinstance(container, list) and not container else None
        values: list[Any] = []
        for match in matches:
            if isinstance(match.value, list):
                values.extend(match.value)
            else:
                values.append(match.value)
        if not all(isinstance(value, str) for value in values):
            return None
        return values

    def tokens(self, payload: Any) -> int | None:
        if self._tokens is None:
            return None
        value = first_match(self._tokens, payload)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        return int(value)

    @staticmethod
    def _container_of(expr: str, payload: Any) -> Any:
        head = expr.split("[*]", 1)[0] if "[*]" in expr else expr
        matches = compile_path(head).find(payload)
        return matches[0].value if matches else None
