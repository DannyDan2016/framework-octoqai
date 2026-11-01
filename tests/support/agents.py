"""Simulated agents that answer through an in-memory httpx transport."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Self

import httpx

from octoqai.agents import AdapterKwargs, AgentPort, HttpAgentAdapter, SseAgentAdapter
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

EVENT_STREAM = "text/event-stream"
DONE = "data: [DONE]\n\n"

_Frame = tuple[float, str | bytes]
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


class FakeWatchdog:
    """Watchdog factory that never uses real time: the simulated agent fires it by hand."""

    def __init__(self) -> None:
        """Start with no armed alarm."""
        self.armed: list[float] = []
        self.cancelled = 0
        self._fires: list[Callable[[], None]] = []

    def __call__(self, seconds: float, fire: Callable[[], None]) -> Self:
        """Arm an alarm for ``seconds``; ``fire`` runs only when the test calls ``fire()``."""
        self.armed.append(seconds)
        self._fires.append(fire)
        return self

    def cancel(self) -> None:
        """Count a cancelled alarm."""
        self.cancelled += 1

    def fire(self, index: int = -1) -> None:
        """Fire an armed alarm, the last one by default, as if its time had run out."""
        if self._fires:
            self._fires[index]()


class _FakeTransport(httpx.MockTransport):
    def __init__(self, handler: _Handler, on_close: Callable[[], None]) -> None:
        super().__init__(handler)
        self._on_close = on_close

    def close(self) -> None:
        self._on_close()


def _read_timeout(request: httpx.Request) -> None:
    raise httpx.ReadTimeout("stalled", request=request)


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
        self.watchdog = FakeWatchdog()
        self.closes = 0
        self._adapters: list[AgentPort] = []
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

    def sends(
        self,
        chunk: str,
        times: int,
        *,
        status: int = 200,
        gap_s: float = 0.0,
        content_type: str = "text/plain",
    ) -> Self:
        """Answer with ``chunk`` repeated ``times``; the fake clock advances ``gap_s`` per chunk."""
        data = chunk.encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            def body() -> Iterator[bytes]:
                for _ in range(times):
                    self.clock.advance(gap_s)
                    yield data

            return httpx.Response(status, headers={"content-type": content_type}, content=body())

        self._handler = handler
        return self

    def stalls(
        self,
        while_connecting: bool = False,
        status: int | None = None,
        body: str = "boom",
        timeout: bool = False,
    ) -> Self:
        """Block until the watchdog fires; the read then fails as on a closed socket.

        With ``status``, the agent first answers that status and ``body``, then blocks; with
        ``timeout`` the blocked read times out instead of being closed by the watchdog.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if status is None:
                self._closed_by_watchdog(request, while_connecting)

            def content() -> Iterator[bytes]:
                yield body.encode("utf-8")
                if timeout:
                    _read_timeout(request)
                self._closed_by_watchdog(request)

            return httpx.Response(status or 500, content=content())

        self._handler = handler
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

    def fires_stale_watchdog(self) -> Self:
        """Fire the first attempt's alarm while each later request is in flight.

        The request fails, as on a closed socket, only if that firing closed its connection.
        """
        answer = self._handler

        def handler(request: httpx.Request) -> httpx.Response:
            closes = self.closes
            self.watchdog.fire(0)
            if self.closes > closes:
                raise httpx.ReadError("closed by a stale watchdog", request=request)
            return answer(request)

        self._handler = handler
        return self

    def transport(self) -> httpx.MockTransport:
        """Return a transport that records each request and counts its closes in ``closes``."""
        return _FakeTransport(self._record, self._count_close)

    def _count_close(self) -> None:
        self.closes += 1

    def close(self) -> None:
        """Close every adapter created through this fake agent."""
        for adapter in self._adapters:
            adapter.close()

    def adapter_kwargs(self) -> AdapterKwargs:
        """Clock, sleep, jitter and watchdog that make an adapter use this agent's fake time."""
        return {
            "clock": self.clock,
            "sleep": self.clock.sleep,
            "jitter": self.jitter,
            "watchdog": self.watchdog,
        }

    def _closed_by_watchdog(self, request: httpx.Request, while_connecting: bool = False) -> None:
        self.watchdog.fire()
        error = httpx.ConnectError if while_connecting else httpx.ReadError
        raise error("closed by the watchdog", request=request)

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
        adapter = HttpAgentAdapter(cfg, transport=self.transport(), **self.adapter_kwargs())
        self._adapters.append(adapter)
        return adapter


def sse_event(name: str, payload: Any) -> str:
    """Encode one SSE event whose data is ``payload`` as JSON."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _token_frames(deltas: Sequence[str], partial: str) -> list[_Frame]:
    frames: list[_Frame] = [(0.0, sse_event("token", {"delta": d})) for d in deltas]
    return [*frames, (0.0, partial)] if partial else frames


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

    def emits(self, *lines: str, content_type: str = EVENT_STREAM) -> Self:
        """Send raw SSE lines, joined with newlines."""
        return self._use([(0.0, "\n".join(lines) + "\n")], content_type=content_type)

    def script(self, *frames: _Frame) -> Self:
        """Send raw frames; the fake clock advances by each delay as the frame is read."""
        return self._use(list(frames))

    def heartbeats(self, count: int, gap_s: float) -> Self:
        """Send ``count`` comment lines, one every ``gap_s``, and no event."""
        return self._use([(gap_s, ": ping\n\n")] * count)

    def fails_reading(self, status: int) -> Self:
        """Answer with an error status whose body times out while being read."""

        def handler(request: httpx.Request) -> httpx.Response:
            def body() -> Iterator[bytes]:
                raise httpx.ReadTimeout("stalled", request=request)
                yield b""

            return httpx.Response(status, content=body())

        self._handler = handler
        return self

    def leaks(self, value: str) -> Self:
        """Leak ``value`` in the second delta."""
        return self.streams("Te paso ", value, " y nada más.")

    def cuts_after(self, *deltas: str, partial: str = "") -> Self:
        """Stream some deltas and an unterminated ``partial`` line, then time out reading."""
        return self._use(_token_frames(deltas, partial), then=_read_timeout)

    def stalls_after(self, *deltas: str, partial: str = "") -> Self:
        """Stream some deltas and an unterminated ``partial`` line, then block until the
        watchdog closes the connection."""
        return self._use(_token_frames(deltas, partial), then=self._closed_by_watchdog)

    def wrong_content_type(self) -> Self:
        """Answer with JSON instead of an event stream."""
        self._handler = lambda request: httpx.Response(200, json={"answer": "x"})
        return self

    def adapter(self, cfg: AgentConfig) -> SseAgentAdapter:
        """Return a real SSE adapter wired to this fake agent, its clock and a fixed jitter."""
        adapter = SseAgentAdapter(cfg, transport=self.transport(), **self.adapter_kwargs())
        self._adapters.append(adapter)
        return adapter

    def _use(
        self,
        frames: list[_Frame],
        then: Callable[[httpx.Request], None] | None = None,
        content_type: str = EVENT_STREAM,
    ) -> Self:
        def handler(request: httpx.Request) -> httpx.Response:
            def body() -> Iterator[bytes]:
                for delay, text in frames:
                    self.clock.advance(delay)
                    yield text.encode("utf-8") if isinstance(text, str) else text
                if then is not None:
                    then(request)

            return httpx.Response(200, headers={"content-type": content_type}, content=body())

        self._handler = handler
        return self
