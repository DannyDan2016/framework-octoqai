"""Port that every language model adapter implements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

LlmRole = Literal["judge", "simulated_user"]


@dataclass(frozen=True)
class LlmMessage:
    """One turn of the conversation sent to the model."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LlmRequest:
    """One call to the model.

    The model, ``max_tokens`` and ``effort`` belong to the adapter configuration, not to the
    request, so the identity of the judge is the same for the whole run.
    """

    role: LlmRole
    system: str
    messages: tuple[LlmMessage, ...]
    schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LlmUsage:
    """Tokens a call consumed; ``None`` when the provider did not report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class LlmReply:
    """What the model answered, or the error that prevented it.

    ``error`` is an exception class name or a fixed message, never the text of an exception,
    which can repeat headers, bodies or the key.
    """

    text: str | None
    model: str | None
    stop_reason: str | None
    usage: LlmUsage
    latency_ms: float
    error: str | None = None
    request_id: str | None = None


class LlmPort(Protocol):
    """Connection to a language model, independent of the provider."""

    def complete(self, request: LlmRequest) -> LlmReply:
        """Run one call and return the reply; it never raises, failures go in ``error``."""
        ...

    def close(self) -> None:
        """Release the resources held by the adapter."""
        ...
