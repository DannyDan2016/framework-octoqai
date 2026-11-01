import pytest

from octoqai.experiment import Experiment
from support.agents import TURN_CONTEXT, FakeJsonAgent, FakeSseAgent
from support.builders import ExperimentBuilder

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture(params=["json", "sse"])
def kind(request):
    return request.param


@pytest.fixture
def build(kind, request):
    def _build(**agent):
        builder = ExperimentBuilder()
        if kind == "sse":
            builder.streaming()
        exp = Experiment.model_validate(builder.agent(**agent).build())
        fake = FakeSseAgent() if kind == "sse" else FakeJsonAgent()
        adapter = fake.adapter(exp.agent)
        request.addfinalizer(adapter.close)
        return fake, adapter

    return _build


@pytest.mark.security
def test_sends_rendered_body_and_headers(build):
    fake, adapter = build(headers={"Authorization": "Bearer s3cret"})
    adapter.send(TURN_CONTEXT)
    assert fake.requests[0] == {"message": "Concilia octubre"}
    assert fake.headers[0]["Authorization"] == "Bearer s3cret"


def test_http_error_keeps_body(build):
    fake, adapter = build()
    fake.fails(500, "boom")
    r = adapter.send(TURN_CONTEXT)
    assert (r.answer, r.error, r.body) == (None, "agent returned HTTP 500", "boom")


def test_retries_before_the_first_byte(build):
    fake, adapter = build(retry={"retries": 2, "backoff_s": 0})
    fake.refuses_first(1)
    r = adapter.send(TURN_CONTEXT)
    assert r.answer == "ok" and r.retries == ["cannot connect to agent: ConnectError"]


@pytest.mark.parametrize("behaviour", ["http_502", "http_503", "http_504", "read_timeout"])
def test_never_retries_once_the_agent_may_have_the_request(build, behaviour):
    fake, adapter = build(retry={"retries": 3, "backoff_s": 0.5})
    if behaviour == "read_timeout":
        fake.times_out()
    else:
        fake.fails(int(behaviour.removeprefix("http_")))
    r = adapter.send(TURN_CONTEXT)
    assert r.error is not None and r.retries == []
    assert len(fake.requests) == 1 and fake.sleeps == []


@pytest.mark.parametrize(("gap_s", "kib"), [(1.0, 5), (0.0, 64)], ids=["deadline", "cap"])
def test_endless_error_body_stops_at_the_deadline_or_cap_and_keeps_the_status(build, gap_s, kib):
    fake, adapter = build(wait={"total_s": 5})
    fake.sends("x" * 1024, 2000, status=500, gap_s=gap_s)
    r = adapter.send(TURN_CONTEXT)
    assert (r.error, len(r.body)) == ("agent returned HTTP 500", kib * 1024)
