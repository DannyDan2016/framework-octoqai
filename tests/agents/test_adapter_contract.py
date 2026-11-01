import pytest

from octoqai.experiment import Experiment, SecurityConfig
from octoqai.security import Detector, scan_reply
from support.agents import (
    DONE,
    EVENT_STREAM,
    TURN_CONTEXT,
    FakeJsonAgent,
    FakeSseAgent,
    sse_event,
)
from support.builders import ExperimentBuilder
from support.samples import NON_ASCII_CANARY

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


TOTAL_TIMEOUT = {"json": "timeout after 7.0s", "sse": "stream exceeded 7s"}


def test_watchdog_is_armed_with_total_and_cancelled_on_every_attempt(build):
    fake, adapter = build(retry={"retries": 1, "backoff_s": 0}, wait={"total_s": 7})
    fake.refuses_first(1)
    adapter.send(TURN_CONTEXT)
    assert fake.watchdog.armed == [7, 7] and fake.watchdog.cancelled == 2


def test_watchdog_interrupts_a_blocked_read_with_the_total_timeout(build, kind):
    fake, adapter = build(wait={"total_s": 7})
    fake.stalls()
    r = adapter.send(TURN_CONTEXT)
    assert (r.answer, r.error) == (None, TOTAL_TIMEOUT[kind])
    assert fake.watchdog.cancelled == 1


def test_watchdog_while_connecting_is_not_retried(build, kind):
    fake, adapter = build(retry={"retries": 2, "backoff_s": 0}, wait={"total_s": 7})
    fake.stalls(while_connecting=True)
    r = adapter.send(TURN_CONTEXT)
    assert (r.error, r.retries, len(fake.requests)) == (TOTAL_TIMEOUT[kind], [], 1)


def test_adapter_works_again_after_the_watchdog_fired(build):
    fake, adapter = build(wait={"total_s": 7})
    fake.stalls()
    adapter.send(TURN_CONTEXT)
    fake.fails(500, "boom")
    r = adapter.send(TURN_CONTEXT)
    assert (r.error, r.body) == ("agent returned HTTP 500", "boom")


def test_watchdog_during_an_error_body_keeps_the_status(build):
    fake, adapter = build(wait={"total_s": 7})
    fake.stalls(status=500)
    r = adapter.send(TURN_CONTEXT)
    assert (r.error, r.body) == ("agent returned HTTP 500", "boom")


def test_a_late_watchdog_of_an_earlier_attempt_never_breaks_the_next(build):
    fake, adapter = build(wait={"total_s": 7})
    adapter.send(TURN_CONTEXT)
    fake.fires_stale_watchdog()
    r = adapter.send(TURN_CONTEXT)
    assert (r.answer, r.error) == ("ok", None)
    assert fake.closes == 0


@pytest.mark.security
def test_error_body_is_decoded_with_its_declared_charset(build):
    fake, adapter = build()
    fake.fails(500, NON_ASCII_CANARY, charset="iso-8859-1")
    r = adapter.send(TURN_CONTEXT)
    assert (r.error, r.body) == ("agent returned HTTP 500", NON_ASCII_CANARY)


@pytest.mark.security
@pytest.mark.parametrize("charset", ["idna", "punycode"])
def test_error_body_with_an_unusable_charset_is_kept_as_utf8(build, charset):
    fake, adapter = build()
    fake.fails(500, "dato TEN-B-0042", charset=charset, encoding="ascii")
    r = adapter.send(TURN_CONTEXT)
    detector = Detector(SecurityConfig.model_validate({"canaries": [{"value": "TEN-B-0042"}]}))
    assert (r.error, r.body) == ("agent returned HTTP 500", "dato TEN-B-0042")
    assert [leak.location for leak in scan_reply(detector, r, [])] == ["body"]


def _canary_detector(value):
    return Detector(SecurityConfig.model_validate({"canaries": [{"value": value}]}))


@pytest.mark.security
@pytest.mark.parametrize("charset", ["utf-16", "utf-32", "cp037"])
def test_error_body_with_a_mismatched_charset_still_shows_the_canary(build, charset):
    fake, adapter = build()
    fake.fails(500, "dato TEN-B-0042", charset=charset, encoding="ascii")
    r = adapter.send(TURN_CONTEXT)
    leaks = scan_reply(_canary_detector("TEN-B-0042"), r, [])
    assert r.error == "agent returned HTTP 500"
    assert [leak.location for leak in leaks] == ["body"]


@pytest.mark.security
def test_error_body_in_its_declared_charset_counts_one_leak(build):
    fake, adapter = build()
    fake.fails(500, NON_ASCII_CANARY, charset="iso-8859-1")
    r = adapter.send(TURN_CONTEXT)
    leaks = scan_reply(_canary_detector(NON_ASCII_CANARY), r, [])
    assert [leak.location for leak in leaks] == ["body"]


ESCAPED_CANARY_JSON = '{"e": "\\u0054EN-B-0042 \u00e9"}'


@pytest.mark.security
@pytest.mark.parametrize("charset", ["iso-8859-1", "utf-16"], ids=["declared", "mismatched"])
def test_json_error_body_with_an_escaped_canary_is_scanned_in_every_decode(build, charset):
    fake, adapter = build()
    fake.fails(500, ESCAPED_CANARY_JSON, charset=charset, encoding="iso-8859-1")
    r = adapter.send(TURN_CONTEXT)
    leaks = scan_reply(_canary_detector("TEN-B-0042"), r, [])
    assert [leak.location for leak in leaks] == ["body"]


CONTENT_TYPES = {"json": "application/json", "sse": EVENT_STREAM}
OK_BODIES = {
    "json": b'{"answer": "ok"}',
    "sse": (sse_event("token", {"delta": "ok"}) + DONE).encode("utf-8"),
}


@pytest.mark.security
@pytest.mark.parametrize(
    "headers",
    [{}, {"Accept-Encoding": "gzip"}, {"accept-encoding": "br, gzip"}],
    ids=["default", "user-gzip", "user-lowercase"],
)
def test_every_attempt_asks_for_an_uncompressed_response(build, headers):
    fake, adapter = build(headers=headers, retry={"retries": 1, "backoff_s": 0})
    fake.refuses_first(1)
    adapter.send(TURN_CONTEXT)
    assert [h.get_list("accept-encoding") for h in fake.headers] == [["identity"], ["identity"]]


@pytest.mark.security
@pytest.mark.parametrize(
    ("encoding", "shown"),
    [
        ("gzip", "gzip"),
        ("gzip, gzip", "gzip, gzip"),
        ("br", "br"),
        ("deflate", "deflate"),
        ("identity, GZIP", "gzip"),
        ("x-TEN-B-0042", "other"),
        (", ".join(["gzip"] * 50), "gzip, gzip, gzip, ..."),
    ],
    ids=["gzip", "chained", "br", "deflate", "mixed", "unknown", "many"],
)
@pytest.mark.parametrize("status", [200, 500])
def test_compressed_response_is_refused_unread_and_not_retried(
    build, kind, encoding, shown, status
):
    fake, adapter = build(retry={"retries": 2, "backoff_s": 0})
    fake.encodes(encoding, content_type=CONTENT_TYPES[kind], status=status)
    r = adapter.send(TURN_CONTEXT)
    assert r.error == f"agent response uses unsupported content-encoding {shown}"
    assert (r.answer, r.body, r.retries, len(fake.requests)) == (None, None, [], 1)
    assert fake.body_reads == 0


@pytest.mark.parametrize("encoding", ["identity", "Identity", ""])
def test_identity_response_is_read(build, kind, encoding):
    fake, adapter = build()
    fake.encodes(encoding, OK_BODIES[kind], CONTENT_TYPES[kind])
    r = adapter.send(TURN_CONTEXT)
    assert (r.answer, r.error) == ("ok", None)
