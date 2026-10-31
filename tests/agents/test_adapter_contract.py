import pytest

from octoqai.experiment import Experiment
from support.agents import TURN_CONTEXT, FakeJsonAgent, FakeSseAgent
from support.builders import ExperimentBuilder

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture(params=["json", "sse"])
def kind(request):
    return request.param


def build(kind, **agent):
    builder = ExperimentBuilder()
    if kind == "sse":
        builder.streaming()
    exp = Experiment.model_validate(builder.agent(**agent).build())
    fake = FakeSseAgent() if kind == "sse" else FakeJsonAgent()
    return fake, fake.adapter(exp.agent)


@pytest.mark.security
def test_sends_rendered_body_and_headers(kind):
    fake, adapter = build(kind, headers={"Authorization": "Bearer s3cret"})
    adapter.send(TURN_CONTEXT)
    assert fake.requests[0] == {"message": "Concilia octubre"}
    assert fake.headers[0]["Authorization"] == "Bearer s3cret"


def test_http_error_keeps_body(kind):
    fake, adapter = build(kind)
    fake.fails(500, "boom")
    r = adapter.send(TURN_CONTEXT)
    assert (r.answer, r.error, r.body) == (None, "agent returned HTTP 500", "boom")


def test_retries_before_the_first_byte(kind):
    fake, adapter = build(kind, retry={"retries": 2, "backoff_s": 0})
    fake.refuses_first(1)
    r = adapter.send(TURN_CONTEXT)
    assert r.answer == "ok" and r.retries == ["cannot connect to agent: ConnectError"]
