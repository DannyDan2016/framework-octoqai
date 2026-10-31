"""Port that every agent adapter implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class AgentReply:
    """What the agent answered to one turn."""

    answer: str | None
    tool_calls: list[str] | None
    tokens: int | None
    latency_ms: float
    error: str | None
    raw: Any = None


class AgentPort(Protocol):
    """Connection to the agent under test, independent of the transport."""

    def send(self, context: dict[str, Any]) -> AgentReply:
        """Send one turn and return the reply; failures are reported in ``error``."""
        ...

    def close(self) -> None:
        """Release the resources held by the adapter."""
        ...
