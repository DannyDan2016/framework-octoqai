"""Choice of agent adapter from the experiment configuration."""

from __future__ import annotations

from typing import Unpack

import httpx

from octoqai.agents._delivery import AdapterKwargs
from octoqai.agents.http_json import HttpAgentAdapter
from octoqai.agents.port import AgentPort
from octoqai.agents.sse import SseAgentAdapter
from octoqai.experiment import AgentConfig


def adapter_for(
    cfg: AgentConfig,
    transport: httpx.BaseTransport | None = None,
    **seams: Unpack[AdapterKwargs],
) -> AgentPort:
    """Build the adapter that matches the agent configuration.

    Args:
        cfg: Agent connection settings.
        transport: Optional httpx transport, used to test without a network.
        **seams: Optional ``clock``, ``sleep``, ``jitter`` and ``watchdog`` (see
            ``AdapterKwargs``), injectable for tests.

    Returns:
        An SSE adapter when ``cfg.stream`` is set, otherwise an HTTP JSON adapter.
    """
    if cfg.stream is not None:
        return SseAgentAdapter(cfg, transport, **seams)
    return HttpAgentAdapter(cfg, transport, **seams)
