"""Delivery of a request to the agent: retries before the first byte and wait deadlines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import httpx

from octoqai.agents._mapping import ReplyError
from octoqai.experiment import EffectiveWait, RetryPolicy

JITTER_SPREAD = 0.25
HTTP_ERROR_MIN = 400
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 64 * 1024


class NotDelivered(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TooLarge(ReplyError):
    pass


class _Expired(Exception):
    pass


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


def text_of(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


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
    chunks: Iterable[bytes], expired: Callable[[], bool], cap: int = MAX_ERROR_BODY_BYTES
) -> str | None:
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
    return text_of(bytes(kept))
