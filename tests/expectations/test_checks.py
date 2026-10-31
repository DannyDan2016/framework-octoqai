import pytest

from octoqai.agents import AgentReply
from octoqai.expectations import evaluate
from octoqai.experiment import Expectation
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]


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


@pytest.mark.parametrize("row", cases(__file__, "checks"))
def test_each_kind_pass_fail_or_not_evaluable(row):
    result = evaluate(E(row["kind"], row["value"]), reply(**row["reply"]))
    assert result.passed is row["passed"]


@pytest.mark.smoke
def test_not_contains_keeps_hard_flag():
    r = evaluate(E("not_contains", "TENANT-B", hard=True), reply(answer="dato de TENANT-B"))
    assert (r.passed, r.hard) == (False, True)


def test_agent_error_fails_every_check():
    r = evaluate(E("contains", "x"), reply(answer=None, error="HTTP 500"))
    assert r.passed is False and "HTTP 500" in r.detail
