import importlib
from contextlib import closing

import pytest

from octoqai.agents import HttpAgentAdapter, SseAgentAdapter, adapter_for
from octoqai.errors import ConfigError
from octoqai.experiment import Experiment, SecurityConfig
from octoqai.security import Detector, scan_reply
from support.agents import DONE, EVENT_STREAM, TURN_CONTEXT, FakeSseAgent, sse_event
from support.builders import ExperimentBuilder
from support.samples import HOSTILE_JSON

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def stream_cfg(**agent):
    return Experiment.model_validate(ExperimentBuilder().streaming().agent(**agent).build()).agent


def send(fake, **agent):
    with closing(fake.adapter(stream_cfg(**agent))) as adapter:
        return adapter.send(TURN_CONTEXT)


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
    assert r.error == "no stream fragment within 30s" and r.fragments == []


def test_idle_deadline_between_fragments(sse):
    r = send(sse.streams("Ho", "la", gap_s=20), wait={"idle_s": 5})
    assert r.error == "stream idle for more than 5s"
    assert len(r.fragments) == 1 and r.answer == "Ho"


def test_total_deadline_keeps_received_fragments_and_stops_reading(sse):
    sse.streams("a", "b", "c", "d", first_s=4, gap_s=4)
    r = send(sse, wait={"idle_s": 5, "total_s": 10})
    assert r.error == "stream exceeded 10s" and len(r.fragments) == 2
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


def test_wait_policy_read_timeout_is_the_longest_of_first_fragment_and_idle(sse):
    send(sse, wait={"connect_s": 2, "first_fragment_s": 4, "idle_s": 8, "total_s": 30})
    assert sse.timeouts[0] == {"connect": 2, "read": 8, "write": 2, "pool": 2}


def test_timeout_before_response_names_the_timeout_httpx_applied(sse):
    r = send(sse.times_out(), wait={"first_fragment_s": 2, "idle_s": 8, "total_s": 30})
    assert r.error == "no stream fragment within 8s"


def test_heartbeats_alone_hit_the_first_fragment_deadline(sse):
    r = send(sse.heartbeats(10, gap_s=1), wait={"first_fragment_s": 3, "total_s": 30})
    assert r.error == "no stream fragment within 3s"
    assert r.fragments == [] and sse.clock.now == 4


def test_heartbeats_do_not_reset_the_idle_deadline(sse):
    ping = (2.0, ": ping\n\n")
    sse.script((0.0, sse_event("token", {"delta": "a"})), ping, ping, ping, ping)
    r = send(sse, wait={"idle_s": 5, "total_s": 100})
    assert r.error == "stream idle for more than 5s"
    assert r.answer == "a" and sse.clock.now == 6


def test_done_after_the_total_deadline_is_an_error_and_keeps_fragments(sse):
    sse.script((4.0, sse_event("token", {"delta": "a"})), (7.0, DONE))
    r = send(sse, wait={"idle_s": 10, "total_s": 10})
    assert r.error == "stream exceeded 10s"
    assert r.fragments == ['{"delta": "a"}'] and r.answer == "a"


def test_error_body_timeout_keeps_the_http_status(sse):
    r = send(sse.fails_reading(500))
    assert r.error == "agent returned HTTP 500" and r.body is None


def test_content_type_is_case_insensitive(sse):
    r = send(
        sse.emits('data: {"delta": "ok"}', "", content_type="Text/Event-Stream; charset=utf-8")
    )
    assert (r.answer, r.error) == ("ok", None)


def test_unterminated_event_is_capped_by_total_and_keeps_its_data(sse):
    sse.script((3.0, sse_event("token", {"delta": "x"})), *[(1.0, "data: a\n")] * 10)
    r = send(sse, wait={"total_s": 5})
    assert r.error == "stream exceeded 5s"
    assert len(r.fragments) == 2 and r.fragments[1].startswith("a\na")
    assert sse.clock.now == 6


def test_deadline_breached_in_a_dataless_block_is_reported_at_end_of_stream(sse):
    sse.script((10.0, "event: ping\n\n"))
    r = send(sse, wait={"total_s": 5})
    assert r.error == "no stream fragment within 5s" and r.fragments == []


@pytest.mark.parametrize("field", ["event: ping", "id: 1", "retry: 5"])
def test_dataless_blocks_cannot_outlive_the_total_deadline(sse, field):
    sse.script(*[(1.0, f"{field}\n\n")] * 200)
    r = send(sse, wait={"total_s": 5})
    assert r.error == "no stream fragment within 5s"
    assert sse.clock.now <= 6


def test_frames_without_a_line_break_are_capped_by_total_and_keep_their_data(sse):
    sse.script((3.0, sse_event("token", {"delta": "a"})), (1.0, "data: x"), *[(1.0, "x")] * 100)
    r = send(sse, wait={"total_s": 5})
    assert r.error == "stream exceeded 5s"
    assert r.fragments == ['{"delta": "a"}', "xx"] and sse.clock.now == 6


def test_wrong_content_type_body_is_capped_and_kept(sse):
    r = send(sse.sends("x" * 1024, 2000, content_type="text/html"))
    assert r.error == "expected text/event-stream, got text/html"
    assert len(r.body) == 64 * 1024


def test_stream_over_the_byte_cap_is_an_error(sse):
    r = send(sse.sends(": " + "x" * (2**20 - 3) + "\n", 17, content_type=EVENT_STREAM))
    assert r.error == "stream exceeded 16777216 bytes"


def test_stream_over_the_byte_cap_keeps_the_pending_data(sse):
    r = send(sse.script((0.0, "data: a"), *[(0.0, "x" * 2**20)] * 17))
    assert r.error == "stream exceeded 16777216 bytes"
    assert len(r.fragments) == 1 and r.fragments[0].startswith("axx")


def test_end_of_stream_after_a_deadline_is_an_error(sse):
    sse.script((1.0, sse_event("token", {"delta": "a"})), (10.0, ""))
    r = send(sse, wait={"total_s": 5})
    assert r.error == "stream idle for more than 5s" and r.answer == "a"


def test_line_break_split_across_chunks_is_a_single_break(sse):
    sse.script((0.0, "event: token\r"), (0.0, '\ndata: {"delta": "a"}\r\n\r\n'))
    r = send(sse, stream={"delta": "$.delta", "event": "token"})
    assert (r.answer, r.error) == ("a", None)


def test_multibyte_character_split_across_chunks_is_kept(sse):
    data = 'data: {"delta": "ñ"}\n\n'.encode()
    cut = data.index(b"\xb1")
    r = send(sse.script((0.0, data[:cut]), (0.0, data[cut:])))
    assert (r.answer, r.error) == ("ñ", None)


def test_adapter_for_picks_the_transport_from_the_config():
    sse_adapter = adapter_for(stream_cfg())
    json_adapter = adapter_for(Experiment.model_validate(ExperimentBuilder().build()).agent)
    assert isinstance(sse_adapter, SseAgentAdapter)
    assert isinstance(json_adapter, HttpAgentAdapter)
    sse_adapter.close()
    json_adapter.close()


def test_watchdog_mid_stream_keeps_the_received_fragments(sse):
    r = send(sse.stalls_after("Hola"), wait={"total_s": 7})
    assert r.error == "stream exceeded 7s"
    assert r.fragments == ['{"delta": "Hola"}'] and r.answer == "Hola"


def test_watchdog_keeps_the_pending_event(sse):
    r = send(sse.stalls_after("Ho", partial='data: {"delta": "la"}'), wait={"total_s": 7})
    assert r.error == "stream exceeded 7s"
    assert r.fragments == ['{"delta": "Ho"}', '{"delta": "la"}'] and r.answer == "Hola"


def test_read_timeout_keeps_the_pending_event_and_names_the_deadline(sse):
    r = send(sse.cuts_after(partial='data: {"delta": "a"}'), wait={"idle_s": 4, "total_s": 9})
    assert r.error == "no stream fragment within 9s"
    assert r.fragments == ['{"delta": "a"}']


def test_pending_line_is_only_joined_on_a_breach_or_at_the_end(sse, monkeypatch):
    splitter = importlib.import_module("octoqai.agents.sse")._Splitter
    joins = []
    pending = splitter.pending.fget

    def counted(self):
        joins.append(1)
        return pending(self)

    monkeypatch.setattr(splitter, "pending", property(counted))
    sse.script(*[(0.0, "x")] * 50, (0.0, '\ndata: {"delta": "ok"}\n\n'))
    assert send(sse).answer == "ok"
    assert len(joins) <= 1


@pytest.mark.security
@pytest.mark.parametrize("data", HOSTILE_JSON.values(), ids=HOSTILE_JSON.keys())
def test_hostile_json_event_is_kept_as_a_fragment_and_scanned(sse, data):
    r = send(sse.emits(f"data: {data}", ""))
    assert (r.answer, r.fragments) == (None, [data])
    assert r.error == "no stream event matched stream.delta '$.delta'"
    detector = Detector(SecurityConfig.model_validate({"canaries": [{"value": "TEN-B-0042"}]}))
    assert [leak.location for leak in scan_reply(detector, r, [])] == ["fragment 1"]


@pytest.mark.security
def test_mapping_that_recurses_too_deep_is_a_mapping_error(sse, monkeypatch):
    def too_deep(self, payload):
        raise RecursionError

    monkeypatch.setattr(
        importlib.import_module("octoqai.agents._mapping").ResponseFields, "tools", too_deep
    )
    r = send(sse.streams("ok"))
    assert (r.answer, r.error) == (None, "cannot map agent response: RecursionError")
