import pytest

from octoqai.agents import HttpAgentAdapter, SseAgentAdapter, adapter_for
from octoqai.errors import ConfigError
from octoqai.experiment import Experiment
from support.agents import TURN_CONTEXT, FakeSseAgent
from support.builders import ExperimentBuilder

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def stream_cfg(**agent):
    return Experiment.model_validate(ExperimentBuilder().streaming().agent(**agent).build()).agent


def send(fake, **agent):
    return fake.adapter(stream_cfg(**agent)).send(TURN_CONTEXT)


@pytest.fixture
def sse():
    return FakeSseAgent()


def test_requires_stream_section():
    cfg = Experiment.model_validate(ExperimentBuilder().build()).agent
    with pytest.raises(ConfigError, match="agent.stream is required"):
        SseAgentAdapter(cfg)


@pytest.mark.smoke
def test_joins_deltas_and_keeps_every_fragment(sse):
    r = send(sse.streams("Ho", "la"))
    assert (r.answer, r.error) == ("Hola", None)
    assert r.fragments == ['{"delta": "Ho"}', '{"delta": "la"}']


def test_ttft_is_time_to_first_event(sse):
    r = send(sse.streams("Ho", "la", first_s=0.2, gap_s=0.1))
    assert r.ttft_ms == pytest.approx(200)
    assert r.latency_ms == pytest.approx(300)


def test_ttft_and_latency_exclude_backoff(sse):
    sse.streams("ok", first_s=0.2).refuses_first(1)
    r = send(sse, retry={"retries": 1, "backoff_s": 0.5})
    assert r.ttft_ms == pytest.approx(200)
    assert r.latency_ms == pytest.approx(200)
    assert sse.sleeps == [0.5]


def test_stops_at_done_sentinel(sse):
    sse.emits('data: {"delta": "a"}', "", "data: [DONE]", "", 'data: {"delta": "z"}', "")
    r = send(sse)
    assert r.answer == "a" and r.fragments == ['{"delta": "a"}']


def test_tools_and_tokens_from_last_event_that_has_them(sse):
    r = send(sse.streams("a", "b", final={"tools": ["buscar"], "usage": {"total": 12}}))
    assert (r.tool_calls, r.tokens) == (["buscar"], 12)


def test_named_progress_events_are_fragments_not_answer(sse):
    sse.emits(
        "event: progress",
        'data: {"delta": "buscando"}',
        "",
        "event: token",
        'data: {"delta": "ok"}',
        "",
    )
    r = send(sse, stream={"delta": "$.delta", "event": "token"})
    assert r.answer == "ok" and len(r.fragments) == 2


def test_non_json_events_are_fragments_only(sse):
    r = send(sse.emits("data: pensando...", "", 'data: {"delta": "ok"}', ""))
    assert r.answer == "ok" and r.fragments[0] == "pensando..."


def test_wrong_content_type_is_error(sse):
    r = send(sse.wrong_content_type())
    assert r.error == "expected text/event-stream, got application/json"


def test_timeout_mid_stream_keeps_partial_fragments_and_answer(sse):
    r = send(sse.cuts_after("Hola"))
    assert r.error.startswith("stream idle for more than")
    assert r.fragments == ['{"delta": "Hola"}'] and r.answer == "Hola"


def test_first_fragment_deadline(sse):
    r = send(sse.streams("Hola", first_s=40), wait={"first_fragment_s": 30})
    assert r.error == "no stream fragment within 30s" and len(r.fragments) == 1


def test_idle_deadline_between_fragments(sse):
    r = send(sse.streams("Ho", "la", gap_s=20), wait={"idle_s": 5})
    assert r.error == "stream idle for more than 5s"
    assert len(r.fragments) == 2 and r.answer == "Hola"


def test_total_deadline_keeps_received_fragments_and_stops_reading(sse):
    sse.streams("a", "b", "c", "d", first_s=4, gap_s=4)
    r = send(sse, wait={"idle_s": 5, "total_s": 10})
    assert r.error == "stream exceeded 10s" and len(r.fragments) == 3
    assert sse.clock.now == 12


def test_without_matching_delta_is_error(sse):
    r = send(sse.emits('data: {"text": "x"}', ""))
    assert r.error == "no stream event matched stream.delta '$.delta'"


def test_never_retries_after_the_stream_started(sse):
    r = send(sse.cuts_after("Hola"), retry={"retries": 2})
    assert r.retries == [] and len(sse.requests) == 1


def test_stream_timeouts_follow_wait_policy(sse):
    send(sse, wait={"connect_s": 2, "first_fragment_s": 20, "idle_s": 5, "total_s": 30})
    assert sse.timeouts[0] == {"connect": 2, "read": 20, "write": 2, "pool": 2}


def test_adapter_for_picks_the_transport_from_the_config():
    sse_adapter = adapter_for(stream_cfg())
    json_adapter = adapter_for(Experiment.model_validate(ExperimentBuilder().build()).agent)
    assert isinstance(sse_adapter, SseAgentAdapter)
    assert isinstance(json_adapter, HttpAgentAdapter)
    sse_adapter.close()
    json_adapter.close()
