import pytest

from octoqai.adapter import AgentReply
from octoqai.config import Expectation
from octoqai.expectations import evaluate


def E(kind, value, hard=False):
    return Expectation(kind=kind, value=value, hard=hard)


def reply(answer="ok", tool_calls=(), tokens=10, latency_ms=100.0, error=None):
    return AgentReply(
        answer=answer,
        tool_calls=None if tool_calls is None else list(tool_calls),
        tokens=tokens,
        latency_ms=latency_ms,
        error=error,
    )


@pytest.mark.parametrize(
    ("exp", "rep", "expected"),
    [
        (E("tool_called", ["a", "b"]), reply(tool_calls=["b", "x", "a"]), True),
        (E("tool_called", ["a", "b"]), reply(tool_calls=["a"]), False),
        (E("no_tool_called", ["borrar"]), reply(tool_calls=["buscar"]), True),
        (E("no_tool_called", ["borrar"]), reply(tool_calls=["borrar"]), False),
        (E("format", "json"), reply(answer=' {"a": 1} '), True),
        (E("format", "json"), reply(answer="[1, 2]"), True),
        (E("format", "json"), reply(answer="42"), False),
        (E("contains", r"\d+,\d{2}"), reply(answer="Total 10,50"), True),
        (E("contains", "conciliado"), reply(answer="pendiente"), False),
        (E("not_contains", "TENANT-B"), reply(answer="solo TENANT-A"), True),
        (E("max_tokens", 100), reply(tokens=100), True),
        (E("max_tokens", 100), reply(tokens=101), False),
        (E("max_latency_ms", 500), reply(latency_ms=499.9), True),
        (E("max_latency_ms", 500), reply(latency_ms=800), False),
    ],
)
def test_each_kind_pass_and_fail(exp, rep, expected):
    assert evaluate(exp, rep).passed is expected


def test_tool_called_without_exposed_tools_is_not_evaluable():
    assert evaluate(E("tool_called", ["buscar"]), reply(tool_calls=None)).passed is None


def test_no_tool_called_without_exposed_tools_is_not_evaluable():
    assert evaluate(E("no_tool_called", ["borrar"]), reply(tool_calls=None)).passed is None


def test_max_tokens_without_usage_is_not_evaluable():
    assert evaluate(E("max_tokens", 10), reply(tokens=None)).passed is None


def test_json_with_chatty_prefix_fails():
    assert evaluate(E("format", "json"), reply(answer='Claro! {"a": 1}')).passed is False


def test_not_contains_keeps_hard_flag():
    r = evaluate(E("not_contains", "TENANT-B", hard=True), reply(answer="dato de TENANT-B"))
    assert (r.passed, r.hard) == (False, True)


def test_agent_error_fails_every_check():
    r = evaluate(E("contains", "x"), reply(answer=None, error="HTTP 500"))
    assert r.passed is False and "HTTP 500" in r.detail
