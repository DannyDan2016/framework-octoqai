"""HTTP adapter that talks to the agent under test."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jsonpath_ng import parse as parse_jsonpath
from jsonpath_ng.exceptions import JSONPathError

from octoqai.config import AgentConfig, ConfigError
from octoqai.template import render

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


def _compile(expr: str | None):
    if expr is None:
        return None
    try:
        return parse_jsonpath(expr)
    except JSONPathError as exc:
        raise ConfigError(f"invalid JSONPath {expr!r}: {exc}") from exc


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
        self._tool_calls = _compile(cfg.response.tool_calls)
        self._tokens = _compile(cfg.response.tokens)
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
            response = self._client.request(
                self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
            )
        except httpx.TimeoutException:
            return self._failure(start, f"timeout after {self._cfg.timeout_s}s")
        except httpx.ConnectError as exc:
            return self._failure(start, f"cannot connect to agent: {type(exc).__name__}")
        except httpx.HTTPError as exc:
            return self._failure(start, f"transport error: {type(exc).__name__}")
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code >= 400:
            return self._failure(start, f"agent returned HTTP {response.status_code}", latency_ms)
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._failure(start, "agent response is not valid JSON", latency_ms)
        answer = self._first(self._answer, payload)
        if answer is _MISSING:
            return self._failure(
                start, f"answer path {self._cfg.response.answer!r} not found", latency_ms, payload
            )
        if answer is None:
            return self._failure(start, "agent answer is null", latency_ms, payload)
        try:
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            tool_calls = self._tools(payload)
            tokens = self._token_count(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            return self._failure(
                start, f"cannot map agent response: {type(exc).__name__}", latency_ms, payload
            )
        return AgentReply(
            answer=answer,
            tool_calls=tool_calls,
            tokens=tokens,
            latency_ms=latency_ms,
            error=None,
            raw=payload,
        )

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._client.close()

    def _failure(
        self, start: float, error: str, latency_ms: float | None = None, raw: Any = None
    ) -> AgentReply:
        if latency_ms is None:
            latency_ms = (time.perf_counter() - start) * 1000
        return AgentReply(None, None, None, latency_ms, error, raw)

    @staticmethod
    def _first(path, payload: Any) -> Any:
        matches = path.find(payload)
        return matches[0].value if matches else _MISSING

    def _tools(self, payload: Any) -> list[str] | None:
        if self._tool_calls is None:
            return None
        matches = self._tool_calls.find(payload)
        if not matches:
            container = self._container_of(self._cfg.response.tool_calls, payload)
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
