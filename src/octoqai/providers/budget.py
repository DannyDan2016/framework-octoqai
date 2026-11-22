"""Hard cap on the number of calls made to a language model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from octoqai.providers.port import LlmPort, LlmReply, LlmRequest, LlmUsage

BUDGET_EXHAUSTED = "call budget exhausted"


@dataclass(frozen=True)
class LlmStats:
    """What one role asked of its model during a run."""

    calls: int
    errors_by_class: dict[str, int]
    input_tokens: int | None
    output_tokens: int | None
    served_models: dict[str, int]


def _add(total: int | None, extra: int | None) -> int | None:
    if extra is None:
        return total
    return extra if total is None else total + extra


class BudgetedLlm:
    """Wraps a port and refuses every call after ``max_calls`` without reaching it."""

    def __init__(self, inner: LlmPort, max_calls: int) -> None:
        """Wrap ``inner``; ``max_calls`` of 0 means the inner port is never called."""
        if max_calls < 0:
            raise ValueError("max_calls cannot be negative")
        self._inner = inner
        self._max_calls = max_calls
        self.calls = 0
        self.errors: Counter[str] = Counter()
        self.usage_total = LlmUsage()
        self.served_models: Counter[str] = Counter()

    def complete(self, request: LlmRequest) -> LlmReply:
        """Forward the call while budget remains; later calls get a fixed error reply."""
        if self.calls >= self._max_calls:
            self.errors[BUDGET_EXHAUSTED] += 1
            return LlmReply(None, None, None, LlmUsage(), 0.0, error=BUDGET_EXHAUSTED)
        self.calls += 1
        reply = self._inner.complete(request)
        if reply.error is not None:
            self.errors[reply.error] += 1
        if reply.model is not None:
            self.served_models[reply.model] += 1
        self.usage_total = LlmUsage(
            _add(self.usage_total.input_tokens, reply.usage.input_tokens),
            _add(self.usage_total.output_tokens, reply.usage.output_tokens),
        )
        return reply

    def stats(self) -> LlmStats:
        """Return a snapshot of the calls, errors, tokens and served models so far."""
        return LlmStats(
            self.calls,
            dict(self.errors),
            self.usage_total.input_tokens,
            self.usage_total.output_tokens,
            dict(self.served_models),
        )

    def close(self) -> None:
        """Close the wrapped port."""
        self._inner.close()
