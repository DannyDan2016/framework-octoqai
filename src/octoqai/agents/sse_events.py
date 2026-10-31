"""Server-sent events parser."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class SseEvent:
    """One dispatched server-sent event.

    Attributes:
        event: Event name, ``message`` when the stream did not name it.
        data: Data lines joined with newlines.
    """

    event: str
    data: str


def _field(line: str) -> tuple[str, str]:
    name, _, value = line.partition(":")
    return name, value.removeprefix(" ")


def parse_sse(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Parse SSE lines lazily, yielding each event as soon as it completes.

    A pending event is dispatched when the input ends, so content is not lost
    if the agent closes the connection early.

    Args:
        lines: Stream lines, with or without a trailing carriage return.

    Yields:
        Each event that carries at least one data line.
    """
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.removesuffix("\r")
        if not line:
            if data:
                yield SseEvent(event, "\n".join(data))
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        name, value = _field(line)
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value or "message"
    if data:
        yield SseEvent(event, "\n".join(data))
