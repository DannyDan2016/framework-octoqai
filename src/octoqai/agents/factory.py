"""Choice of agent adapter from the experiment configuration."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypedDict

import httpx

from octoqai.agents.http_json import HttpAgentAdapter
from octoqai.agents.port import AgentPort
from octoqai.agents.sse import SseAgentAdapter
from octoqai.experiment import AgentConfig


class AdapterKwargs(TypedDict, total=False):
    """Time and randomness sources an adapter accepts, injectable for tests."""

    clock: Callable[[], float]
    sleep: Callable[[float], None]
    jitter: Callable[[], float]


def adapter_for(
    cfg: AgentConfig,
    transport: httpx.BaseTransport | None = None,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> AgentPort:
    """Build the adapter that matches the agent configuration.

    Args:
        cfg: Agent connection settings.
        transport: Optional httpx transport, used to test without a network.
        clock: Monotonic clock in seconds, injectable for tests.
        sleep: Sleep function used between retries, injectable for tests.
        jitter: Random fraction in [0, 1) that shortens each backoff by up to 25%.

    Returns:
        An SSE adapter when ``cfg.stream`` is set, otherwise an HTTP JSON adapter.
    """
    if cfg.stream is not None:
        return SseAgentAdapter(cfg, transport, clock=clock, sleep=sleep, jitter=jitter)
    return HttpAgentAdapter(cfg, transport, clock=clock, sleep=sleep, jitter=jitter)
