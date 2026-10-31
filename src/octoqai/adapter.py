"""HTTP adapter that talks to the agent under test."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.parser import parse as parse_jsonpath

from octoqai.config import AgentConfig, ConfigError
from octoqai.template import render

HTTP_ERROR_MIN = 400

_MISSING = object()


@dataclass
class AgentReply:
    """What the agent answered to one turn."""

    answer: str | None
    tool_calls: list[str] | None
    tokens: int | None
    latency_ms: float
    error: str | None
    raw: Any = None


class _ReplyError(Exception):
    def __init__(self, error: str, raw: Any = None) -> None:
        super().__init__(error)
        self.error = error
        self.raw = raw


def _compile(expr: str) -> JSONPath:
    try:
        return parse_jsonpath(expr)
    except JSONPathError as exc:
        raise ConfigError(f"invalid JSONPath {expr!r}: {exc}") from exc


def _compile_optional(expr: str | None) -> JSONPath | None:
    return None if expr is None else _compile(expr)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _failure(exc: _ReplyError, latency_ms: float) -> AgentReply:
    return AgentReply(None, None, None, latency_ms, exc.error, exc.raw)


class HttpAgentAdapter:
    """Send rendered requests to an agent exposed over HTTP JSON."""

    def __init__(self, cfg: AgentConfig, transport: httpx.BaseTransport | None = None) -> None:
        """Create the adapter.

        Args:
            cfg: Agent connection settings.
            transport: Optional httpx transport, used to test without a network.
        """
        self._cfg = cfg
        self._answer = _compile(cfg.response.answer)
        self._tool_calls = _compile_optional(cfg.response.tool_calls)
        self._tokens = _compile_optional(cfg.response.tokens)
        self._client = httpx.Client(transport=transport, timeout=cfg.timeout_s)

    def send(self, context: dict[str, Any]) -> AgentReply:
        """Send one turn and map the response.

        Transport and protocol failures are returned as an ``AgentReply`` with ``error`` set.

        Args:
            context: Template context for this turn.

        Returns:
            The mapped agent reply.
        """
        body = render(self._cfg.request, context)
        start = time.perf_counter()
        try:
            response = self._post(body)
        except _ReplyError as exc:
            return _failure(exc, _elapsed_ms(start))
        latency_ms = _elapsed_ms(start)
        try:
            return self._map_payload(self._decode(response), latency_ms)
        except _ReplyError as exc:
            return _failure(exc, latency_ms)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._client.close()

    def _post(self, body: Any) -> httpx.Response:
        try:
            return self._client.request(
                self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
            )
        except httpx.TimeoutException as exc:
            raise _ReplyError(f"timeout after {self._cfg.timeout_s}s") from exc
        except httpx.ConnectError as exc:
            raise _ReplyError(f"cannot connect to agent: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            raise _ReplyError(f"transport error: {type(exc).__name__}") from exc

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        if response.status_code >= HTTP_ERROR_MIN:
            raise _ReplyError(f"agent returned HTTP {response.status_code}")
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _ReplyError("agent response is not valid JSON") from exc

    def _map_payload(self, payload: Any, latency_ms: float) -> AgentReply:
        answer = self._first(self._answer, payload)
        if answer is _MISSING:
            raise _ReplyError(f"answer path {self._cfg.response.answer!r} not found", payload)
        if answer is None:
            raise _ReplyError("agent answer is null", payload)
        try:
            text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            tool_calls = self._tools(payload)
            tokens = self._token_count(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _ReplyError(f"cannot map agent response: {type(exc).__name__}", payload) from exc
        return AgentReply(
            answer=text,
            tool_calls=tool_calls,
            tokens=tokens,
            latency_ms=latency_ms,
            error=None,
            raw=payload,
        )

    @staticmethod
    def _first(path: JSONPath, payload: Any) -> Any:
        matches = path.find(payload)
        return matches[0].value if matches else _MISSING

    def _tools(self, payload: Any) -> list[str] | None:
        expr = self._cfg.response.tool_calls
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

    @staticmethod
    def _container_of(expr: str, payload: Any) -> Any:
        head = expr.split("[*]", 1)[0] if "[*]" in expr else expr
        matches = _compile(head).find(payload)
        return matches[0].value if matches else None

    def _token_count(self, payload: Any) -> int | None:
        if self._tokens is None:
            return None
        value = self._first(self._tokens, payload)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        return int(value)
