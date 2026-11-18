"""Simulated language models that answer from a script and never touch the network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self

from octoqai.providers.port import LlmReply, LlmRequest, LlmUsage

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
