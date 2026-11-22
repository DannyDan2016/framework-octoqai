"""Adapter that talks to a local Ollama server behind the language model port."""

from __future__ import annotations

import ipaddress
import json
import threading
import time
from collections.abc import Callable
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Protocol

import httpx

from octoqai.errors import ConfigError
from octoqai.experiment import ProviderSettings
from octoqai.providers.port import LlmReply, LlmRequest, LlmUsage

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
CHAT_PATH = "/api/chat"
MAX_RESPONSE_BYTES = 1_048_576
SHORT_TIMEOUT_S = 5.0
IDENTITY = "identity"
HTTP_REDIRECT_MIN = 300
MAX_PORT = 65535
PROVIDER_FAILURE = "provider failure"
INVALID_RESPONSE = "invalid response"
DEADLINE_EXCEEDED = "deadline exceeded"
TOO_LARGE = "response too large"
UNSUPPORTED_ENCODING = "unsupported content-encoding"
LOOPBACK_HOST = "localhost"
LOOPBACK_MESSAGE = (
    "ollama base_url must be a loopback address (127.0.0.1, localhost or [::1]); "
    "set allow_remote: true to use a remote server"
)


class Watchdog(Protocol):
    """A wall-clock alarm armed for one call."""

    def cancel(self) -> None:
        """Disarm the alarm so that it never fires."""


WatchdogFactory = Callable[[float, Callable[[], None]], Watchdog]


def timer_watchdog(seconds: float, fire: Callable[[], None]) -> Watchdog:
    """Call ``fire`` from a daemon thread after ``seconds`` of real time.

    Args:
        seconds: Wall-clock time before the alarm fires.
        fire: Called once when the time runs out, unless the alarm was cancelled.

    Returns:
        The started ``threading.Timer``.
    """
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


class _Failed(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def _is_loopback(host: str) -> bool:
    if host == LOOPBACK_HOST:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_url(raw: str) -> httpx.URL:
    try:
        url = httpx.URL(raw)
        _ = url.host
        return url
    except Exception:
        raise ConfigError("ollama base_url is not a valid url") from None


def _check_shape(url: httpx.URL) -> None:
    if url.scheme not in ("http", "https") or not url.host:
        raise ConfigError("ollama base_url must use http or https and name a host")
    if url.port is not None and not 0 < url.port <= MAX_PORT:
        raise ConfigError("ollama base_url is not a valid url")
    if url.userinfo:
        raise ConfigError("ollama base_url must not contain credentials")
    if url.path not in ("", "/") or url.query or url.fragment:
        raise ConfigError("ollama base_url must not contain a path, query or fragment")


def _endpoint(settings: ProviderSettings) -> httpx.URL:
    raw = DEFAULT_BASE_URL if settings.base_url is None else settings.base_url
    url = _parse_url(raw)
    _check_shape(url)
    if not settings.allow_remote and not _is_loopback(url.host):
        raise ConfigError(LOOPBACK_MESSAGE)
    return url.copy_with(path=CHAT_PATH)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _reject_encoded(response: httpx.Response) -> None:
    codings = response.headers.get_list("content-encoding", split_commas=True)
    if any(coding.strip().lower() not in ("", IDENTITY) for coding in codings):
        raise _Failed(UNSUPPORTED_ENCODING)


def _parse(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        raise _Failed(INVALID_RESPONSE) from None
    if not isinstance(payload, dict):
        raise _Failed(INVALID_RESPONSE)
    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise _Failed(INVALID_RESPONSE)
    return payload


class OllamaProvider:
    """Language model port over a local Ollama server; it never raises and never repeats a body.

    Ollama has no authentication and runs on the local machine, so ``api_key`` does not exist,
    ``effort`` and ``max_retries`` are not used and a failed call is not retried. The client
    ignores the proxy variables of the environment, so a prompt never leaves through a proxy.
    """

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        watchdog: WatchdogFactory = timer_watchdog,
    ) -> None:
        """Validate the base URL and build the client with explicit timeouts.

        Raises:
            ConfigError: The base URL is not loopback (unless ``allow_remote``), carries
                credentials, a path, a query or a fragment, or cannot be parsed.
        """
        self._settings = settings
        self._clock = clock
        self._url = _endpoint(settings)
        self._transport = transport
        self._watchdog = watchdog
        self._closed = False
        self._client = self._new_client()

    def __repr__(self) -> str:
        """Show only the model, never the address or any prompt."""
        return f"OllamaProvider(model={self._settings.model!r})"

    def complete(self, request: LlmRequest) -> LlmReply:
        """Run one call; a failure is returned as a fixed value, never as the server's text."""
        start = self._clock()
        try:
            return self._reply(self._watched(request, start), start)
        except _Failed as exc:
            error = exc.error
        except httpx.HTTPError as exc:
            error = type(exc).__name__
        except Exception:
            error = PROVIDER_FAILURE
        return LlmReply(None, None, None, LlmUsage(), self._elapsed_ms(start), error=error)

    def close(self) -> None:
        """Close the HTTP client for good."""
        self._closed = True
        self._client.close()

    def _new_client(self) -> httpx.Client:
        limit = self._settings.timeout_s
        short = min(SHORT_TIMEOUT_S, limit)
        return httpx.Client(
            transport=self._transport,
            trust_env=False,
            follow_redirects=False,
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            timeout=httpx.Timeout(connect=short, read=limit, write=short, pool=short),
        )

    def _watched(self, request: LlmRequest, start: float) -> dict[str, Any]:
        if self._client.is_closed and not self._closed:
            self._client = self._new_client()
        client = self._client
        fired = threading.Event()
        done = threading.Event()
        lock = threading.Lock()

        def fire() -> None:
            with lock:
                if done.is_set():
                    return
                fired.set()
                client.close()

        alarm = self._watchdog(self._settings.timeout_s, fire)
        try:
            return self._exchange(client, request, start)
        except Exception:
            if fired.is_set():
                raise _Failed(DEADLINE_EXCEEDED) from None
            raise
        finally:
            with lock:
                done.set()
            alarm.cancel()

    def _exchange(self, client: httpx.Client, request: LlmRequest, start: float) -> dict[str, Any]:
        with client.stream(
            "POST",
            self._url,
            json=self._body(request),
            headers={"Accept-Encoding": IDENTITY},
        ) as response:
            if response.status_code >= HTTP_REDIRECT_MIN:
                raise _Failed(f"HTTP {response.status_code}")
            _reject_encoded(response)
            return _parse(self._read(response, start))

    def _read(self, response: httpx.Response, start: float) -> bytes:
        kept = bytearray()
        chunks = response.iter_raw()
        while True:
            if self._clock() - start > self._settings.timeout_s:
                raise _Failed(DEADLINE_EXCEEDED)
            chunk = next(chunks, None)
            if chunk is None:
                return bytes(kept)
            kept += chunk
            if len(kept) > MAX_RESPONSE_BYTES:
                raise _Failed(TOO_LARGE)

    def _reply(self, payload: dict[str, Any], start: float) -> LlmReply:
        return LlmReply(
            text=payload["message"]["content"],
            model=_optional_str(payload.get("model")),
            stop_reason=_optional_str(payload.get("done_reason")),
            usage=LlmUsage(
                _as_int(payload.get("prompt_eval_count")), _as_int(payload.get("eval_count"))
            ),
            latency_ms=self._elapsed_ms(start),
        )

    def _body(self, request: LlmRequest) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        body: dict[str, Any] = {
            "model": self._settings.model,
            "stream": False,
            "think": False,
            "messages": messages,
            "options": {"temperature": 0, "num_predict": self._settings.max_tokens},
        }
        if request.schema is not None:
            body["format"] = request.schema
        return body

    def _elapsed_ms(self, start: float) -> float:
        return (self._clock() - start) * 1000
