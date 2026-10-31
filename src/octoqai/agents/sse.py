"""Server-sent events adapter that talks to a streaming agent."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from octoqai.agents._delivery import (
    HTTP_ERROR_MIN,
    NotDelivered,
    classify_delivery,
    client_timeout,
    transport_failure,
    with_retries,
)
from octoqai.agents._mapping import (
    ReplyError,
    ResponseFields,
    compile_path,
    first_match,
    mapping_guard,
)
from octoqai.agents.port import AgentReply
from octoqai.agents.sse_events import SseEvent, parse_sse
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig, StreamConfig, render

EVENT_STREAM = "text/event-stream"


@dataclass
class _Stream:
    start: float = 0.0
    fragments: list[str] = field(default_factory=list)
    deltas: list[str] = field(default_factory=list)
    tool_calls: list[str] | None = None
    tokens: int | None = None
    first_at: float | None = None
    last_at: float | None = None
    body: str | None = None


@dataclass
class _Block:
    lines: list[str] = field(default_factory=list)

    def add(self, raw: str) -> None:
        if raw and not raw.startswith(":"):
            self.lines.append(raw)

    def reset(self) -> None:
        self.lines.clear()


class SseAgentAdapter:
    """Send rendered requests to an agent that answers with server-sent events."""

    def __init__(
        self,
        cfg: AgentConfig,
        transport: httpx.BaseTransport | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Create the adapter.

        Args:
            cfg: Agent connection settings; ``stream`` is required.
            transport: Optional httpx transport, used to test without a network.
            clock: Monotonic clock in seconds, injectable for tests.
            sleep: Sleep function used between retries, injectable for tests.
            jitter: Random fraction in [0, 1) that shortens each backoff by up to 25%.

        Raises:
            ConfigError: When the configuration has no ``stream`` section.
        """
        if cfg.stream is None:
            raise ConfigError("agent.stream is required for the SSE adapter")
        self._cfg = cfg
        self._stream: StreamConfig = cfg.stream
        self._delta = compile_path(self._stream.delta)
        self._fields = ResponseFields(cfg.response)
        self._wait = cfg.effective_wait()
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._read_s = max(self._wait.first_fragment_s, self._wait.idle_s)
        self._client = httpx.Client(
            transport=transport, timeout=client_timeout(self._wait, self._read_s)
        )

    def send(self, context: dict[str, Any]) -> AgentReply:
        """Send one turn and read the event stream.

        Transport and protocol failures are returned as an ``AgentReply`` with ``error`` set.

        Args:
            context: Template context for this turn.

        Returns:
            The reply built from the events received so far.
        """
        body = render(self._cfg.request, context)
        retries: list[str] = []
        state = _Stream()
        try:
            response = with_retries(
                lambda: self._open(body, state),
                self._cfg.retry,
                self._sleep,
                retries,
                self._jitter,
            )
        except NotDelivered as exc:
            return self._reply(state, exc.reason, retries)
        except ReplyError as exc:
            return self._reply(state, exc.error, retries)
        try:
            self._consume(response, state)
        except ReplyError as exc:
            return self._reply(state, exc.error, retries)
        finally:
            response.close()
        return self._reply(state, None, retries)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._client.close()

    def _open(self, body: Any, state: _Stream) -> httpx.Response:
        request = self._client.build_request(
            self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
        )
        state.start = self._clock()
        with classify_delivery(self._wait, self._no_fragment(self._read_s)):
            return self._client.send(request, stream=True)

    def _consume(self, response: httpx.Response, state: _Stream) -> None:
        try:
            self._check_response(response, state)
            for event in parse_sse(self._watched(response.iter_lines(), state)):
                if self._on_event(event, state):
                    return
        except httpx.TimeoutException as exc:
            raise ReplyError(self._stalled(state)) from exc
        except httpx.HTTPError as exc:
            raise transport_failure(exc) from exc

    def _watched(self, lines: Iterable[str], state: _Stream) -> Iterator[str]:
        block = _Block()
        for raw in lines:
            text = raw.removesuffix("\r")
            block.add(text)
            self._guard(block, state)
            if not text:
                block.reset()
            yield raw
        self._guard(block, state)

    def _guard(self, block: _Block, state: _Stream) -> None:
        now = self._clock()
        error = self._check_deadlines(now, state)
        if error is None:
            return
        for event in self._events(block):
            self._record(event, now, state)
        raise ReplyError(error)

    def _events(self, block: _Block) -> list[SseEvent]:
        events = parse_sse([*block.lines, ""])
        return [event for event in events if event.data != self._stream.done]

    def _check_response(self, response: httpx.Response, state: _Stream) -> None:
        content_type = response.headers.get("content-type", "")
        if response.status_code >= HTTP_ERROR_MIN:
            error = f"agent returned HTTP {response.status_code}"
        elif not content_type.lower().startswith(EVENT_STREAM):
            error = f"expected {EVENT_STREAM}, got {content_type}"
        else:
            return
        try:
            response.read()
        except httpx.HTTPError:
            raise ReplyError(error) from None
        state.body = response.text
        raise ReplyError(error)

    def _on_event(self, event: SseEvent, state: _Stream) -> bool:
        if event.data == self._stream.done:
            return True
        self._record(event, self._clock(), state)
        return False

    def _record(self, event: SseEvent, now: float, state: _Stream) -> None:
        state.fragments.append(event.data)
        state.first_at = now if state.first_at is None else state.first_at
        state.last_at = now
        self._extract(event, state)

    def _extract(self, event: SseEvent, state: _Stream) -> None:
        try:
            payload = json.loads(event.data)
        except ValueError:
            return
        with mapping_guard(payload):
            tools = self._fields.tools(payload)
            tokens = self._fields.tokens(payload)
        state.tool_calls = state.tool_calls if tools is None else tools
        state.tokens = state.tokens if tokens is None else tokens
        if self._stream.event in (None, event.event):
            delta = first_match(self._delta, payload)
            if isinstance(delta, str):
                state.deltas.append(delta)

    def _check_deadlines(self, now: float, state: _Stream) -> str | None:
        wait = self._wait
        if state.last_at is None and now - state.start > wait.first_fragment_s:
            return self._no_fragment(wait.first_fragment_s)
        if state.last_at is not None and now - state.last_at > wait.idle_s:
            return f"stream idle for more than {wait.idle_s:g}s"
        return self._total_error(now, state)

    def _total_error(self, now: float, state: _Stream) -> str | None:
        if now - state.start > self._wait.total_s:
            return f"stream exceeded {self._wait.total_s:g}s"
        return None

    @staticmethod
    def _no_fragment(seconds: float) -> str:
        return f"no stream fragment within {seconds:g}s"

    def _stalled(self, state: _Stream) -> str:
        if state.first_at is None:
            return self._no_fragment(self._read_s)
        return f"stream idle for more than {self._wait.idle_s:g}s"

    def _reply(self, state: _Stream, error: str | None, retries: list[str]) -> AgentReply:
        answer = "".join(state.deltas) if state.deltas else None
        if answer is None and error is None:
            error = f"no stream event matched stream.delta {self._stream.delta!r}"
        ttft = None if state.first_at is None else (state.first_at - state.start) * 1000
        return AgentReply(
            answer=answer,
            tool_calls=state.tool_calls,
            tokens=state.tokens,
            latency_ms=(self._clock() - state.start) * 1000,
            error=error,
            ttft_ms=ttft,
            fragments=state.fragments,
            body=state.body,
            retries=retries,
        )
