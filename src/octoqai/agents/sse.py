"""Server-sent events adapter that talks to a streaming agent."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from octoqai.agents._delivery import NotDelivered, client_timeout, not_delivered, with_retries
from octoqai.agents._mapping import ReplyError, ResponseFields, compile_path, first_match
from octoqai.agents.port import AgentReply
from octoqai.agents.sse_events import SseEvent, parse_sse
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig, StreamConfig, render

HTTP_ERROR_MIN = 400
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
        read_s = max(self._wait.first_fragment_s, self._wait.idle_s)
        self._client = httpx.Client(transport=transport, timeout=client_timeout(self._wait, read_s))

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
        try:
            return self._client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise not_delivered(exc, self._wait) from exc
        except httpx.TimeoutException as exc:
            raise ReplyError(self._no_fragment()) from exc
        except httpx.HTTPError as exc:
            raise ReplyError(f"transport error: {type(exc).__name__}") from exc

    def _consume(self, response: httpx.Response, state: _Stream) -> None:
        try:
            self._check_response(response, state)
            for event in parse_sse(response.iter_lines()):
                if self._on_event(event, state):
                    return
        except httpx.TimeoutException as exc:
            raise ReplyError(self._stalled(state)) from exc
        except httpx.HTTPError as exc:
            raise ReplyError(f"transport error: {type(exc).__name__}") from exc

    def _check_response(self, response: httpx.Response, state: _Stream) -> None:
        content_type = response.headers.get("content-type", "")
        if response.status_code >= HTTP_ERROR_MIN:
            error = f"agent returned HTTP {response.status_code}"
        elif not content_type.startswith(EVENT_STREAM):
            error = f"expected {EVENT_STREAM}, got {content_type}"
        else:
            return
        response.read()
        state.body = response.text
        raise ReplyError(error)

    def _on_event(self, event: SseEvent, state: _Stream) -> bool:
        now = self._clock()
        if event.data == self._stream.done:
            return True
        previous = state.last_at
        state.fragments.append(event.data)
        if state.first_at is None:
            state.first_at = now
        state.last_at = now
        self._extract(event, state)
        error = self._check_deadlines(now, previous, state)
        if error:
            raise ReplyError(error)
        return False

    def _extract(self, event: SseEvent, state: _Stream) -> None:
        try:
            payload = json.loads(event.data)
        except ValueError:
            return
        try:
            tools = self._fields.tools(payload)
            tokens = self._fields.tokens(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReplyError(f"cannot map agent response: {type(exc).__name__}", payload) from exc
        state.tool_calls = state.tool_calls if tools is None else tools
        state.tokens = state.tokens if tokens is None else tokens
        if self._stream.event in (None, event.event):
            delta = first_match(self._delta, payload)
            if isinstance(delta, str):
                state.deltas.append(delta)

    def _check_deadlines(self, now: float, previous: float | None, state: _Stream) -> str | None:
        wait = self._wait
        if previous is None and now - state.start > wait.first_fragment_s:
            return self._no_fragment()
        if previous is not None and now - previous > wait.idle_s:
            return f"stream idle for more than {wait.idle_s:g}s"
        if now - state.start > wait.total_s:
            return f"stream exceeded {wait.total_s:g}s"
        return None

    def _no_fragment(self) -> str:
        return f"no stream fragment within {self._wait.first_fragment_s:g}s"

    def _stalled(self, state: _Stream) -> str:
        if state.first_at is None:
            return self._no_fragment()
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
