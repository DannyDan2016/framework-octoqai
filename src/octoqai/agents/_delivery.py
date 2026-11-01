"""Delivery of a request to the agent: retries before the first byte and wait deadlines."""

from __future__ import annotations

import codecs
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import httpx

from octoqai.agents._mapping import ReplyError
from octoqai.experiment import AgentConfig, EffectiveWait, RetryPolicy

JITTER_SPREAD = 0.25
HTTP_ERROR_MIN = 400
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 64 * 1024
NOT_CHARSETS = frozenset({"idna", "punycode", "unicode-escape", "raw-unicode-escape", "undefined"})
IDENTITY = "identity"
KNOWN_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate", "br", "zstd", "compress", "x-compress"})
MAX_SHOWN_ENCODINGS = 3


class Watchdog(Protocol):
    """A wall-clock alarm armed for one attempt."""

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


class AdapterKwargs(TypedDict, total=False):
    """Time and randomness sources an adapter accepts, injectable for tests.

    Attributes:
        clock: Monotonic clock in seconds.
        sleep: Sleep function used between retries.
        jitter: Random fraction in [0, 1) that shortens each backoff by up to 25%.
        watchdog: Arms a wall-clock alarm of ``total_s`` for each attempt; when it fires the
            connection is closed, so a read blocked inside the HTTP client still ends.
    """

    clock: Callable[[], float]
    sleep: Callable[[float], None]
    jitter: Callable[[], float]
    watchdog: WatchdogFactory


@dataclass(frozen=True)
class Seams:
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    jitter: Callable[[], float]
    watchdog: WatchdogFactory

    @classmethod
    def of(cls, kwargs: AdapterKwargs) -> Seams:
        return cls(
            clock=kwargs.get("clock", time.perf_counter),
            sleep=kwargs.get("sleep", time.sleep),
            jitter=kwargs.get("jitter", random.random),
            watchdog=kwargs.get("watchdog", timer_watchdog),
        )


class WatchedClient:
    def __init__(
        self,
        transport: httpx.BaseTransport | None,
        timeout: httpx.Timeout,
        watchdog: WatchdogFactory,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._watchdog = watchdog
        self._closed = False
        self._client = self._new()

    @contextmanager
    def attempt(self, seconds: float, message: str) -> Iterator[httpx.Client]:
        if self._client.is_closed and not self._closed:
            self._client = self._new()
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

        alarm = self._watchdog(seconds, fire)
        try:
            yield client
        except AgentAnswered:
            raise
        except Exception as exc:
            if fired.is_set():
                raise _interrupted(exc, message) from exc
            raise
        finally:
            with lock:
                done.set()
            alarm.cancel()

    def close(self) -> None:
        self._closed = True
        self._client.close()

    def _new(self) -> httpx.Client:
        return httpx.Client(transport=self._transport, timeout=self._timeout)


def _interrupted(exc: Exception, message: str) -> ReplyError:
    if isinstance(exc, ReplyError):
        return ReplyError(message, exc.raw, exc.body, exc.alt_body)
    return ReplyError(message)


class NotDelivered(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TooLarge(ReplyError):
    pass


class AgentAnswered(ReplyError):
    pass


class _Expired(Exception):
    pass


def agent_request(client: httpx.Client, cfg: AgentConfig, body: Any) -> httpx.Request:
    request = client.build_request(cfg.method, cfg.url, json=body, headers=cfg.headers)
    request.headers["Accept-Encoding"] = IDENTITY
    return request


def _shown_encodings(codings: list[str]) -> str:
    shown = [c if c in KNOWN_ENCODINGS else "other" for c in codings[:MAX_SHOWN_ENCODINGS]]
    if len(codings) > MAX_SHOWN_ENCODINGS:
        shown.append("...")
    return ", ".join(shown)


def reject_encoded(response: httpx.Response) -> None:
    codings = [
        coding.strip().lower()
        for coding in response.headers.get_list("content-encoding", split_commas=True)
    ]
    unsupported = [coding for coding in codings if coding not in ("", IDENTITY)]
    if unsupported:
        shown = _shown_encodings(unsupported)
        raise AgentAnswered(f"agent response uses unsupported content-encoding {shown}")


def not_delivered(exc: httpx.TransportError, wait: EffectiveWait) -> NotDelivered:
    if isinstance(exc, httpx.ConnectTimeout):
        return NotDelivered(f"cannot connect to agent within {wait.connect_s:g}s")
    return NotDelivered(f"cannot connect to agent: {type(exc).__name__}")


def transport_failure(exc: httpx.HTTPError) -> ReplyError:
    return ReplyError(f"transport error: {type(exc).__name__}")


@contextmanager
def classify_delivery(wait: EffectiveWait, timeout_message: str) -> Iterator[None]:
    try:
        yield
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise not_delivered(exc, wait) from exc
    except httpx.TimeoutException as exc:
        raise ReplyError(timeout_message) from exc
    except httpx.HTTPError as exc:
        raise transport_failure(exc) from exc


def with_retries[T](
    attempt: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    retries: list[str],
    jitter: Callable[[], float],
) -> T:
    for n in range(policy.retries):
        try:
            return attempt()
        except NotDelivered as exc:
            retries.append(exc.reason)
            sleep(policy.backoff_s * 2**n * (1 - JITTER_SPREAD * jitter()))
    return attempt()


def client_timeout(wait: EffectiveWait, read_s: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=wait.connect_s, read=read_s, write=wait.connect_s, pool=wait.connect_s
    )


def _declared_text(data: bytes, encoding: str) -> str | None:
    try:
        if codecs.lookup(encoding).name not in NOT_CHARSETS:
            return data.decode(encoding, errors="replace")
    except (LookupError, UnicodeError):
        pass
    return None


@dataclass(frozen=True)
class Decoded:
    text: str
    alt: str | None = None


def decode_body(data: bytes, encoding: str = "utf-8") -> Decoded:
    utf8 = data.decode("utf-8", errors="replace")
    declared = _declared_text(data, encoding)
    if declared is None or declared == utf8:
        return Decoded(utf8)
    return Decoded(declared, utf8)


def keep_body(exc: ReplyError, body: Decoded | None) -> ReplyError:
    exc.body = None if body is None else body.text
    exc.alt_body = None if body is None else body.alt
    return exc


def read_bounded(
    chunks: Iterable[bytes], check: Callable[[], None], limit: int = MAX_RESPONSE_BYTES
) -> Iterator[bytes]:
    size = 0
    for chunk in chunks:
        check()
        size += len(chunk)
        if size > limit:
            raise TooLarge(f"stream exceeded {limit} bytes")
        yield chunk


def read_capped(
    chunks: Iterable[bytes],
    expired: Callable[[], bool],
    encoding: str = "utf-8",
    cap: int = MAX_ERROR_BODY_BYTES,
) -> Decoded | None:
    kept = bytearray()

    def check() -> None:
        if expired():
            raise _Expired

    try:
        for chunk in read_bounded(chunks, check):
            kept += chunk[: cap - len(kept)]
            if len(kept) >= cap:
                break
    except (_Expired, httpx.HTTPError):
        if not kept:
            return None
    return decode_body(bytes(kept), encoding)
