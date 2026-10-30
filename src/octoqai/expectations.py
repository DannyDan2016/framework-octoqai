"""Deterministic checks applied to agent replies."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from octoqai.adapter import AgentReply
from octoqai.config import Expectation


@dataclass
class CheckResult:
    """Outcome of one expectation; ``passed`` is ``None`` when it cannot be evaluated."""

    kind: str
    passed: bool | None
    hard: bool
    detail: str
    agent_error: bool = False


_Outcome = tuple[bool | None, str]


def _tool_called(value: list[str], reply: AgentReply) -> _Outcome:
    if reply.tool_calls is None:
        return None, "agent does not expose tool calls"
    missing = [name for name in value if name not in reply.tool_calls]
    return not missing, f"missing tools: {missing}" if missing else "all tools called"


def _no_tool_called(value: list[str], reply: AgentReply) -> _Outcome:
    if reply.tool_calls is None:
        return None, "agent does not expose tool calls"
    forbidden = [name for name in value if name in reply.tool_calls]
    return not forbidden, f"forbidden tools called: {forbidden}" if forbidden else "none called"


def _format(value: str, reply: AgentReply) -> _Outcome:
    try:
        parsed = json.loads(reply.answer or "")
    except json.JSONDecodeError:
        return False, "answer is not pure JSON"
    if not isinstance(parsed, dict | list):
        return False, "answer is JSON but not an object or array"
    return True, "answer is pure JSON"


def _contains(value: str, reply: AgentReply) -> _Outcome:
    found = re.search(value, reply.answer or "") is not None
    return found, "pattern found" if found else "pattern not found"


def _not_contains(value: str, reply: AgentReply) -> _Outcome:
    found = re.search(value, reply.answer or "") is not None
    return not found, "forbidden pattern found" if found else "pattern absent"


def _max_tokens(value: float, reply: AgentReply) -> _Outcome:
    if reply.tokens is None:
        return None, "agent does not expose token usage"
    return reply.tokens <= value, f"{reply.tokens} tokens (max {value})"


def _max_latency_ms(value: float, reply: AgentReply) -> _Outcome:
    return reply.latency_ms <= value, f"{reply.latency_ms:.0f} ms (max {value})"


_CHECKS: dict[str, Callable[..., _Outcome]] = {
    "tool_called": _tool_called,
    "no_tool_called": _no_tool_called,
    "format": _format,
    "contains": _contains,
    "not_contains": _not_contains,
    "max_tokens": _max_tokens,
    "max_latency_ms": _max_latency_ms,
}


def evaluate(exp: Expectation, reply: AgentReply) -> CheckResult:
    """Evaluate one expectation against an agent reply.

    Args:
        exp: The expectation to check.
        reply: The agent reply for the turn.

    Returns:
        The check result; a reply with an error fails every check.
    """
    if reply.error is not None:
        return CheckResult(exp.kind, False, exp.hard, f"agent error: {reply.error}", True)
    passed, detail = _CHECKS[exp.kind](exp.value, reply)
    return CheckResult(exp.kind, passed, exp.hard, detail)
