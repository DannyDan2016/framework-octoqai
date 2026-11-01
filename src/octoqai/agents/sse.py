"""Server-sent events adapter that talks to a streaming agent."""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Unpack

import httpx

from octoqai.agents._delivery import (
    HTTP_ERROR_MIN,
    AdapterKwargs,
    AgentAnswered,
    NotDelivered,
    Seams,
    TooLarge,
    WatchedClient,
    classify_delivery,
    client_timeout,
    read_bounded,
    read_capped,
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
_CR = "\r"
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


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


class _Splitter:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._parts: list[str] = []
        self._held = ""

    @property
    def pending(self) -> str:
        return "".join(self._parts)

    def feed(self, chunk: bytes) -> list[str]:
        return self._split(self._decoder.decode(chunk))

    def flush(self) -> list[str]:
        lines = self._split(self._decoder.decode(b"", final=True))
        rest = self.pending
        if self._held or rest:
            lines.append(rest)
        self._parts, self._held = [], ""
        return lines

    def _split(self, text: str) -> list[str]:
        if not text:
            return []
        joined = self._held + text
        self._held = _CR if joined.endswith(_CR) else ""
        pieces = _LINE_BREAK.split(joined.removesuffix(_CR) if self._held else joined)
        if len(pieces) == 1:
            self._parts.append(pieces[0])
            return []
        lines = ["".join([*self._parts, pieces[0]]), *pieces[1:-1]]
        self._parts = [pieces[-1]]
        return lines


class SseAgentAdapter:
    """Send rendered requests to an agent that answers with server-sent events."""

    def __init__(
        self,
        cfg: AgentConfig,
        transport: httpx.BaseTransport | None = None,
        **seams: Unpack[AdapterKwargs],
    ) -> None:
        """Create the adapter.

        Args:
            cfg: Agent connection settings; ``stream`` is required.
            transport: Optional httpx transport, used to test without a network.
            **seams: Optional ``clock``, ``sleep``, ``jitter`` and ``watchdog`` (see
                ``AdapterKwargs``), injectable for tests.

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
        seam = Seams.of(seams)
        self._clock, self._sleep, self._jitter = seam.clock, seam.sleep, seam.jitter
        self._read_s = max(self._wait.first_fragment_s, self._wait.idle_s)
        self._http = WatchedClient(
            transport, client_timeout(self._wait, self._read_s), seam.watchdog
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
            with_retries(
                lambda: self._attempt(body, state),
                self._cfg.retry,
                self._sleep,
                retries,
                self._jitter,
            )
        except NotDelivered as exc:
            return self._reply(state, exc.reason, retries)
        except ReplyError as exc:
            return self._reply(state, exc.error, retries)
        return self._reply(state, None, retries)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._http.close()

    def _attempt(self, body: Any, state: _Stream) -> None:
        with self._http.attempt(self._wait.total_s, self._over_total()) as client:
            response = self._open(client, body, state)
            try:
                self._consume(response, state)
            finally:
                response.close()

    def _open(self, client: httpx.Client, body: Any, state: _Stream) -> httpx.Response:
        request = client.build_request(
            self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
        )
        state.start = self._clock()
        with classify_delivery(self._wait, self._no_fragment(self._read_s)):
            return client.send(request, stream=True)

    def _consume(self, response: httpx.Response, state: _Stream) -> None:
        self._check_response(response, state)
        block = _Block()
        for event in parse_sse(self._watched(self._lines(response, block, state), block, state)):
            if self._on_event(event, state):
                return

    def _lines(self, response: httpx.Response, block: _Block, state: _Stream) -> Iterator[str]:
        splitter = _Splitter()
        chunks = read_bounded(response.iter_bytes(), lambda: self._guard(block, state, splitter))
        try:
            for chunk in chunks:
                yield from splitter.feed(chunk)
        except (TooLarge, httpx.HTTPError) as exc:
            error = self._read_failure(exc, state)
            self._keep(block, splitter.pending, self._clock(), state)
            raise error from exc
        yield from splitter.flush()

    def _read_failure(self, exc: TooLarge | httpx.HTTPError, state: _Stream) -> ReplyError:
        if isinstance(exc, TooLarge):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return ReplyError(self._stalled(state))
        return transport_failure(exc)

    def _watched(self, lines: Iterable[str], block: _Block, state: _Stream) -> Iterator[str]:
        for raw in lines:
            text = raw.removesuffix(_CR)
            block.add(text)
            self._guard(block, state)
            if not text:
                block.reset()
            yield raw
        self._guard(block, state)

    def _guard(self, block: _Block, state: _Stream, splitter: _Splitter | None = None) -> None:
        now = self._clock()
        error = self._check_deadlines(now, state)
        if error is None:
            return
        self._keep(block, "" if splitter is None else splitter.pending, now, state)
        raise ReplyError(error)

    def _keep(self, block: _Block, partial: str, now: float, state: _Stream) -> None:
        block.add(partial)
        for event in self._events(block):
            self._record(event, now, state)

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
        state.body = read_capped(
            response.iter_bytes(),
            lambda: self._check_deadlines(self._clock(), state) is not None,
            response.charset_encoding or "utf-8",
        )
        raise AgentAnswered(error)

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
            return self._over_total()
        return None

    def _over_total(self) -> str:
        return f"stream exceeded {self._wait.total_s:g}s"

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
