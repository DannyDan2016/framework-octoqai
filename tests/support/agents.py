"""Simulated agents that answer through an in-memory httpx transport."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Self

import httpx

from octoqai.agents import HttpAgentAdapter, SseAgentAdapter
from octoqai.experiment import AgentConfig

RESPONSE_MAP = {
    "answer": "$.answer",
    "tool_calls": "$.tool_calls[*].name",
    "tokens": "$.usage.total_tokens",
}

SSE_RESPONSE_MAP = {"tool_calls": "$.tools[*]", "tokens": "$.usage.total"}

TURN_CONTEXT = {
    "message": "Concilia octubre",
    "conversation_id": "c1",
    "turn_index": 0,
    "history": [],
    "variant": {"temperature": 0},
}

DONE = "data: [DONE]\n\n"

_Frame = tuple[float, str]
_Handler = Callable[[httpx.Request], httpx.Response]


class FakeClock:
    """Manual clock shared by a simulated agent and the adapter under test."""

    def __init__(self) -> None:
        """Start at zero with no recorded sleeps."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        """Record a sleep and move the clock forward instead of waiting."""
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeAgent:
    """Behaviour shared by every simulated agent."""

    def __init__(self) -> None:
        """Create an agent that records requests and fails until told otherwise."""
        self.requests: list[Any] = []
        self.headers: list[httpx.Headers] = []
        self.urls: list[httpx.URL] = []
        self.timeouts: list[Any] = []
        self.clock = FakeClock()
        self.jitter: Callable[[], float] = lambda: 0.0
        self._refusals = 0
        self._refusal: type[httpx.TransportError] = httpx.ConnectError
        self._handler: _Handler = lambda request: httpx.Response(500)

    @property
    def sleeps(self) -> list[float]:
        """Backoff sleeps requested by the adapter."""
        return self.clock.sleeps

    def fails(self, status: int, body: str = "") -> Self:
        """Answer every turn with an HTTP error status."""
        self._handler = lambda request: httpx.Response(status, text=body)
        return self

    def times_out(self) -> Self:
        """Raise a read timeout before any response byte."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        self._handler = handler
        return self

    def refuses(self) -> Self:
        """Refuse every connection."""
        return self.refuses_first(10**6)

    def refuses_first(self, times: int, timeout: bool = False) -> Self:
        """Refuse the next ``times`` connections, then behave as configured."""
        self._refusals = times
        self._refusal = httpx.ConnectTimeout if timeout else httpx.ConnectError
        return self

    def transport(self) -> httpx.MockTransport:
        """Return a transport that records each request and applies the behaviour."""
        return httpx.MockTransport(self._record)

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {"clock": self.clock, "sleep": self.clock.sleep, "jitter": self.jitter}

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content) if request.content else None)
        self.headers.append(request.headers)
        self.urls.append(request.url)
        self.timeouts.append(request.extensions.get("timeout"))
        if self._refusals:
            self._refusals -= 1
            raise self._refusal("refused", request=request)
        return self._handler(request)


class FakeJsonAgent(FakeAgent):
    """An HTTP JSON agent whose behaviour is chosen by the test."""

    def __init__(self) -> None:
        """Create an agent that answers ``ok`` until told otherwise."""
        super().__init__()
        self.replies({"answer": "ok"})

    def answers(self, text: Any, tools: Sequence[str] = (), tokens: float | None = None) -> Self:
        """Answer every turn with ``text``, the tools called and the token usage."""
        payload: dict[str, Any] = {"answer": text}
        if tools:
            payload["tool_calls"] = [{"name": name} for name in tools]
        if tokens is not None:
            payload["usage"] = {"total_tokens": tokens}
        return self.replies(payload)

    def replies(self, payload: Any) -> Self:
        """Answer every turn with a raw JSON payload."""
        self._handler = lambda request: httpx.Response(200, json=payload)
        return self

    def returns(self, content: bytes, status: int = 200) -> Self:
        """Answer every turn with a raw body, e.g. invalid JSON."""
        self._handler = lambda request: httpx.Response(status, content=content)
        return self

    def adapter(self, cfg: AgentConfig) -> HttpAgentAdapter:
        """Return a real HTTP adapter wired to this fake agent, its clock and a fixed jitter."""
        return HttpAgentAdapter(cfg, transport=self.transport(), **self._adapter_kwargs())


def sse_event(name: str, payload: Any) -> str:
    """Encode one SSE event whose data is ``payload`` as JSON."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class FakeSseAgent(FakeAgent):
    """An SSE agent whose events, timing and failures are chosen by the test."""

    def __init__(self) -> None:
        """Create an agent that streams ``ok`` until told otherwise."""
        super().__init__()
        self.streams("ok")

    def streams(
        self,
        *deltas: str,
        first_s: float = 0.0,
        gap_s: float = 0.0,
        final: dict[str, Any] | None = None,
    ) -> Self:
        """Stream ``token`` events with a ``delta`` each; ``final`` is merged into the last.

        The fake clock advances ``first_s`` before the first event and ``gap_s`` before each
        later one, only when the adapter reads it.
        """
        payloads: list[dict[str, Any]] = [{"delta": delta} for delta in deltas]
        if final and payloads:
            payloads[-1].update(final)
        frames = [
            (first_s if index == 0 else gap_s, sse_event("token", payload))
            for index, payload in enumerate(payloads)
        ]
        return self._use([*frames, (0.0, DONE)])

    def emits(self, *lines: str) -> Self:
        """Send raw SSE lines, joined with newlines."""
        return self._use([(0.0, "\n".join(lines) + "\n")])

    def leaks(self, value: str) -> Self:
        """Leak ``value`` in the second delta."""
        return self.streams("Te paso ", value, " y nada más.")

    def cuts_after(self, *deltas: str) -> Self:
        """Stream some deltas and then stall until the read times out."""
        return self._use([(0.0, sse_event("token", {"delta": d})) for d in deltas], cut=True)

    def wrong_content_type(self) -> Self:
        """Answer with JSON instead of an event stream."""
        self._handler = lambda request: httpx.Response(200, json={"answer": "x"})
        return self

    def adapter(self, cfg: AgentConfig) -> SseAgentAdapter:
        """Return a real SSE adapter wired to this fake agent, its clock and a fixed jitter."""
        return SseAgentAdapter(cfg, transport=self.transport(), **self._adapter_kwargs())

    def _use(self, frames: list[_Frame], cut: bool = False) -> Self:
        def handler(request: httpx.Request) -> httpx.Response:
            def body() -> Iterator[bytes]:
                for delay, text in frames:
                    self.clock.advance(delay)
                    yield text.encode("utf-8")
                if cut:
                    raise httpx.ReadTimeout("stalled", request=request)

            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body()
            )

        self._handler = handler
        return self
