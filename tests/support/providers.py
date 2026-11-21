"""Simulated language models that answer from a script and never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NoReturn, Self

import httpx
import pytest

from octoqai.providers.port import LlmReply, LlmRequest, LlmUsage
from support.agents import FakeWatchdog

EXHAUSTED_SCRIPT = "fake script exhausted"
DEFAULT_MODEL = "fake-judge-1"
TRUNCATED_TEXT = '{"verdict": "pa'
LATENCY_MS = 5.0


@dataclass(frozen=True)
class _Planned:
    text: str | None = None
    stop_reason: str | None = "end_turn"
    error: str | None = None


class FakeLlm:
    """An ``LlmPort`` that replies from a queue and records every request."""

    def __init__(self) -> None:
        """Start with an empty script, the default model and no reported usage."""
        self.model = DEFAULT_MODEL
        self.usage = LlmUsage()
        self.requests: list[LlmRequest] = []
        self.closed = False
        self._queue: list[_Planned | LlmReply] = []

    def judges(self, verdict: str, rationale: str = "no reason given") -> Self:
        """Queue a judge reply with a JSON verdict."""
        return self.returns_text(json.dumps({"verdict": verdict, "rationale": rationale}))

    def returns_text(self, raw: str) -> Self:
        """Queue a reply with exactly this text."""
        self._queue.append(_Planned(text=raw))
        return self

    def script(self, *replies: str | LlmReply) -> Self:
        """Replace the queue with these replies; a string is a plain text reply."""
        self._queue = [_Planned(text=r) if isinstance(r, str) else r for r in replies]
        return self

    def fails(self, error_class_name: str) -> Self:
        """Queue a reply that failed with this error class name."""
        self._queue.append(_Planned(stop_reason=None, error=error_class_name))
        return self

    def times_out(self) -> Self:
        """Queue a timeout, reported as the SDK timeout class name without any waiting."""
        return self.fails("APITimeoutError")

    def refuses(self) -> Self:
        """Queue a reply the model refused to give."""
        self._queue.append(_Planned(stop_reason="refusal"))
        return self

    def truncates(self) -> Self:
        """Queue a reply cut by the token limit."""
        self._queue.append(_Planned(text=TRUNCATED_TEXT, stop_reason="max_tokens"))
        return self

    def serves_model(self, name: str) -> Self:
        """Report this model name in every reply from now on."""
        self.model = name
        return self

    def uses(self, tokens_in: int | None, tokens_out: int | None) -> Self:
        """Report this token usage in every reply from now on."""
        self.usage = LlmUsage(tokens_in, tokens_out)
        return self

    def complete(self, request: LlmRequest) -> LlmReply:
        """Record the request and return the next scripted reply."""
        self.requests.append(request)
        if not self._queue:
            return self._reply(_Planned(stop_reason=None, error=EXHAUSTED_SCRIPT))
        planned = self._queue.pop(0)
        return planned if isinstance(planned, LlmReply) else self._reply(planned)

    def close(self) -> None:
        """Remember that the port was closed."""
        self.closed = True

    def _reply(self, planned: _Planned) -> LlmReply:
        return LlmReply(
            text=planned.text,
            model=None if planned.error else self.model,
            stop_reason=planned.stop_reason,
            usage=LlmUsage() if planned.error else self.usage,
            latency_ms=LATENCY_MS,
            error=planned.error,
        )


class ScriptedUser(FakeLlm):
    """A simulated user whose messages are scripted; it can say it has nothing more to add."""

    def __init__(self) -> None:
        """Start with no messages and no end of conversation."""
        super().__init__()
        self._done_after: int | None = None
        self._said = 0

    def says(self, *messages: str, done_after: int | None = None) -> Self:
        """Queue these messages; after ``done_after`` replies every reply is an empty text."""
        for message in messages:
            self.returns_text(message)
        if done_after is not None:
            self._done_after = done_after
        return self

    def complete(self, request: LlmRequest) -> LlmReply:
        """Return the next message, or an empty text once ``done_after`` replies were given."""
        if self._done_after is not None and self._said >= self._done_after:
            self.requests.append(request)
            return self._reply(_Planned(text=""))
        self._said += 1
        return super().complete(request)


class FakeClock:
    """A clock that only moves when told to, so deadlines never wait."""

    def __init__(self, now: float = 100.0) -> None:
        """Start at ``now`` seconds."""
        self.now = now

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the fake time forward."""
        self.now += seconds


class _Chunks(httpx.SyncByteStream):
    def __init__(self, parts: list[bytes], clock: FakeClock, seconds_per_chunk: float) -> None:
        self._parts = parts
        self._clock = clock
        self._seconds = seconds_per_chunk

    def __iter__(self) -> Iterator[bytes]:
        for part in self._parts:
            self._clock.advance(self._seconds)
            yield part


class _Stalled(httpx.SyncByteStream):
    def __init__(self, first: bytes, block: Callable[[], NoReturn]) -> None:
        self._first = first
        self._block = block

    def __iter__(self) -> Iterator[bytes]:
        yield self._first
        self._block()


class FakeOllama:
    """A fake Ollama server over ``httpx.MockTransport`` that records what it was sent."""

    def __init__(self) -> None:
        """Answer every request with an empty chat reply until told otherwise."""
        self.clock = FakeClock()
        self.watchdog = FakeWatchdog()
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)
        self._seconds_per_request = 0.0
        self._respond: Callable[[httpx.Request], httpx.Response] = lambda _: self._json({})

    def answers(self, content: str, **fields: Any) -> Self:
        """Answer with a chat reply whose ``message.content`` is ``content``."""
        reply: dict[str, Any] = {"message": {"role": "assistant", "content": content}}
        reply.update(fields)
        return self.body(json.dumps(reply).encode())

    def status(self, code: int, body: bytes = b"") -> Self:
        """Answer with this status and body, without following redirects."""
        return self.body(body, status=code)

    def raises(self, exc: BaseException) -> Self:
        """Raise this exception from the transport."""

        def respond(_: httpx.Request) -> httpx.Response:
            raise exc

        self._respond = respond
        return self

    def body(self, raw: bytes, headers: dict[str, str] | None = None, *, status: int = 200) -> Self:
        """Answer with these exact bytes and headers."""
        return self.chunks([raw], headers, status=status)

    def chunks(
        self,
        parts: list[bytes],
        headers: dict[str, str] | None = None,
        *,
        seconds_per_chunk: float = 0.0,
        status: int = 200,
    ) -> Self:
        """Answer with these chunks, advancing the fake clock before each one."""
        self._respond = lambda _: httpx.Response(
            status, headers=headers, stream=_Chunks(parts, self.clock, seconds_per_chunk)
        )
        return self

    def stalls(self, after: bytes | None = None) -> Self:
        """Block until the watchdog fires, before any header or after ``after`` of the body.

        The blocked read then fails as on a socket closed by the alarm.
        """
        watchdog = self.watchdog

        def blocked(request: httpx.Request) -> NoReturn:
            watchdog.fire()
            raise httpx.ReadError("closed", request=request)

        def respond(request: httpx.Request) -> httpx.Response:
            if after is None:
                blocked(request)
            return httpx.Response(200, stream=_Stalled(after, lambda: blocked(request)))

        self._respond = respond
        return self

    def takes(self, seconds: float) -> Self:
        """Advance the fake clock this much on every request."""
        self._seconds_per_request = seconds
        return self

    @property
    def sent(self) -> dict[str, Any]:
        """The JSON body of the last request received."""
        parsed: dict[str, Any] = json.loads(self.requests[-1].content)
        return parsed

    def record_default_transports(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Replace the transport httpx builds for itself; return the arguments each one got."""
        built: list[dict[str, Any]] = []
        fake = self

        class Recording(httpx.BaseTransport):
            def __init__(self, **kwargs: Any) -> None:
                built.append(kwargs)

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return fake.transport.handle_request(request)

        monkeypatch.setattr(httpx._client, "HTTPTransport", Recording)
        return built

    def _handle(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        self.clock.advance(self._seconds_per_request)
        return self._respond(request)

    @staticmethod
    def _json(payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json=payload)
