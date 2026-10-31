"""Delivery of a request to the agent: retries before the first byte and wait deadlines."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from octoqai.experiment import EffectiveWait, RetryPolicy

JITTER_SPREAD = 0.25


class NotDelivered(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def not_delivered(exc: httpx.TransportError, wait: EffectiveWait) -> NotDelivered:
    if isinstance(exc, httpx.ConnectTimeout):
        return NotDelivered(f"cannot connect to agent within {wait.connect_s:g}s")
    return NotDelivered(f"cannot connect to agent: {type(exc).__name__}")


def with_retries[T](
    attempt: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    jitter: Callable[[], float],
    retries: list[str],
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
