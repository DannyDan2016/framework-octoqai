"""Simulated agents that answer through an in-memory httpx transport."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from octoqai.agents import HttpAgentAdapter
from octoqai.experiment import AgentConfig

RESPONSE_MAP = {
    "answer": "$.answer",
    "tool_calls": "$.tool_calls[*].name",
    "tokens": "$.usage.total_tokens",
}

_Handler = Callable[[httpx.Request], httpx.Response]


class FakeJsonAgent:
    """An HTTP JSON agent whose behaviour is chosen by the test."""

    def __init__(self) -> None:
        """Create an agent that answers ``ok`` until told otherwise."""
        self.requests: list[Any] = []
        self.headers: list[httpx.Headers] = []
        self._handler: _Handler = self._payload({"answer": "ok"})

    def answers(
        self, text: Any, tools: Sequence[str] = (), tokens: float | None = None
    ) -> FakeJsonAgent:
        """Answer every turn with ``text``, the tools called and the token usage."""
        payload: dict[str, Any] = {"answer": text}
        if tools:
            payload["tool_calls"] = [{"name": name} for name in tools]
        if tokens is not None:
            payload["usage"] = {"total_tokens": tokens}
        return self.replies(payload)

    def replies(self, payload: Any) -> FakeJsonAgent:
        """Answer every turn with a raw JSON payload."""
        self._handler = self._payload(payload)
        return self

    def returns(self, content: bytes, status: int = 200) -> FakeJsonAgent:
        """Answer every turn with a raw body, e.g. invalid JSON."""
        self._handler = lambda request: httpx.Response(status, content=content)
        return self

    def fails(self, status: int, body: str = "") -> FakeJsonAgent:
        """Answer every turn with an HTTP error status."""
        self._handler = lambda request: httpx.Response(status, text=body)
        return self

    def times_out(self) -> FakeJsonAgent:
        """Raise a read timeout on every turn."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        self._handler = handler
        return self

    def refuses(self) -> FakeJsonAgent:
        """Refuse every connection."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        self._handler = handler
        return self

    def transport(self) -> httpx.MockTransport:
        """Return a transport that records each request and applies the behaviour."""
        return httpx.MockTransport(self._record)

    def adapter(self, cfg: AgentConfig) -> HttpAgentAdapter:
        """Return a real HTTP adapter wired to this fake agent."""
        return HttpAgentAdapter(cfg, transport=self.transport())

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content) if request.content else None)
        self.headers.append(request.headers)
        return self._handler(request)

    @staticmethod
    def _payload(payload: Any) -> _Handler:
        return lambda request: httpx.Response(200, json=payload)
