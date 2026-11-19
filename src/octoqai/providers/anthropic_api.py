"""Adapter that talks to the Anthropic Messages API behind the language model port."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from octoqai.experiment import ProviderSettings
from octoqai.providers.port import LlmReply, LlmRequest, LlmUsage

PROVIDER_FAILURE = "provider failure"


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    messages: _Messages

    def close(self) -> None: ...


def _sdk_client(**kwargs: Any) -> Any:
    import anthropic

    return anthropic.Anthropic(open_telemetry=False, **kwargs)


def _sdk_classes() -> tuple[type[Exception], type[Exception]] | None:
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.AnthropicError, anthropic.APIStatusError


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_request_id(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _usage(message: Any) -> LlmUsage:
    usage = getattr(message, "usage", None)
    return LlmUsage(
        _as_int(getattr(usage, "input_tokens", None)),
        _as_int(getattr(usage, "output_tokens", None)),
    )


def _text(message: Any) -> str | None:
    parts = [
        block.text
        for block in getattr(message, "content", None) or ()
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    return "".join(parts) if parts else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


class AnthropicProvider:
    """Language model port over the Anthropic SDK; it never raises and never repeats a secret."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        client_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Build the client with the key, timeout and retries stated explicitly.

        Without ``client_factory`` the SDK is imported here, so a missing extra fails now.
        """
        self._settings = settings
        self._clock = clock
        factory = client_factory or _sdk_client
        self._client: _Client = factory(
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )

    def complete(self, request: LlmRequest) -> LlmReply:
        """Run one call; an SDK failure is returned as its class name, never its text."""
        start = self._clock()
        try:
            message = self._client.messages.create(**self._arguments(request))
            return self._reply(message, start)
        except Exception as exc:
            return self._failure(exc, self._elapsed_ms(start))

    def close(self) -> None:
        """Close the SDK client."""
        self._client.close()

    def _reply(self, message: Any, start: float) -> LlmReply:
        return LlmReply(
            text=_text(message),
            model=_optional_str(getattr(message, "model", None)),
            stop_reason=_optional_str(getattr(message, "stop_reason", None)),
            usage=_usage(message),
            latency_ms=self._elapsed_ms(start),
            request_id=_as_request_id(getattr(message, "_request_id", None)),
        )

    def _arguments(self, request: LlmRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self._settings.effort}
        if request.schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.schema}
        return {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "system": request.system,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "output_config": output_config,
        }

    def _elapsed_ms(self, start: float) -> float:
        return (self._clock() - start) * 1000

    @staticmethod
    def _failure(exc: Exception, latency_ms: float) -> LlmReply:
        classes = _sdk_classes()
        error = PROVIDER_FAILURE
        request_id = None
        if classes is not None and isinstance(exc, classes[0]):
            error = type(exc).__name__
            if isinstance(exc, classes[1]):
                request_id = _as_request_id(getattr(exc, "request_id", None))
        return LlmReply(
            None, None, None, LlmUsage(), latency_ms, error=error, request_id=request_id
        )
