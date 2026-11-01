import json
from dataclasses import asdict

import pytest

from octoqai.agents import AgentReply
from octoqai.experiment import SecurityConfig
from octoqai.security import Detector, scan_reply
from support import samples

pytestmark = [pytest.mark.unit, pytest.mark.regression, pytest.mark.security]

CANARY = "TEN-B-0042"
DETECTOR = Detector(
    SecurityConfig.model_validate({"canaries": [{"value": CANARY, "label": "cuenta-b"}]})
)


def reply(answer=None, fragments=(), body=None, error=None):
    return AgentReply(answer, None, None, 1.0, error, fragments=list(fragments), body=body)


@pytest.mark.smoke
def test_canary_in_fragment_reported_with_location_and_mask():
    r = reply("ok", fragments=['{"delta": "ok"}', f'{{"step": "{CANARY}"}}'])
    [leak] = scan_reply(DETECTOR, r, ["hola"])
    assert (leak.kind, leak.label, leak.location, leak.masked) == (
        "canary",
        "cuenta-b",
        "fragment 2",
        "T…",
    )
    assert CANARY not in str(asdict(leak))


def test_secret_split_across_fragments_found_in_answer():
    head, tail = samples.SPLIT_SECRET
    r = reply(head + tail, fragments=[f'{{"delta": "{head}"}}', f'{{"delta": "{tail}"}}'])
    [leak] = scan_reply(DETECTOR, r, [])
    assert (leak.kind, leak.label, leak.location) == ("secret", "llm api key", "answer")


def test_canary_echoed_from_user_message_is_not_a_leak():
    r = reply(f"La cuenta {CANARY} no es tuya")
    assert scan_reply(DETECTOR, r, [f"Dame los datos de {CANARY.lower()}"]) == []


def test_leak_in_error_body_detected():
    r = reply(body=f'{{"trace": "{CANARY}"}}', error="agent returned HTTP 500")
    assert [leak.location for leak in scan_reply(DETECTOR, r, [])] == ["body"]


def test_same_value_reported_once_at_first_location():
    r = reply(
        f"dato {CANARY}",
        fragments=[f'{{"delta": "dato {CANARY}"}}'],
        body=f'{{"answer": "dato {CANARY}"}}',
    )
    assert [leak.location for leak in scan_reply(DETECTOR, r, [])] == ["fragment 1"]


def test_env_values_are_not_leaks():
    detector = Detector(SecurityConfig.model_validate({}), env_values=["valor-de-entorno"])
    assert scan_reply(detector, reply("tengo valor-de-entorno"), []) == []


ESCAPED = Detector(
    SecurityConfig.model_validate(
        {
            "canaries": [
                {"value": samples.NON_ASCII_CANARY, "label": "acento"},
                {"value": samples.SLASH_CANARY, "label": "barra"},
            ]
        }
    )
)


def found(r):
    return [(leak.kind, leak.label, leak.location) for leak in scan_reply(ESCAPED, r, [])]


def escape_slashes(text):
    return text.replace("/", "\\/")


def test_non_ascii_canary_escaped_by_default_json_dumps_is_found_in_body():
    body = json.dumps({"a": "ok", "dbg": samples.NON_ASCII_CANARY})
    assert found(reply(body=body)) == [("canary", "acento", "body")]


def test_canary_with_escaped_slashes_is_found_in_fragment():
    fragment = escape_slashes(json.dumps({"dbg": samples.SLASH_CANARY}))
    assert found(reply(fragments=[fragment])) == [("canary", "barra", "fragment 1")]


def test_credentials_url_with_escaped_slashes_is_found():
    body = '{"dbg": "' + samples.ESCAPED_DATABASE_URL + '"}'
    assert ("secret", "url with credentials", "body") in found(reply(body=body))


def test_json_nested_in_the_answer_is_decoded():
    answer = json.dumps({"dbg": samples.NON_ASCII_CANARY})
    assert found(reply(answer=answer)) == [("canary", "acento", "answer")]


def test_decoded_raw_payload_is_scanned_as_body():
    r = AgentReply("ok", None, None, 1.0, None, raw={samples.NON_ASCII_CANARY: 1})
    assert found(r) == [("canary", "acento", "body")]


def test_leak_found_raw_and_decoded_counts_once():
    body = json.dumps({"dbg": f"x {CANARY} y"})
    assert [leak.location for leak in scan_reply(DETECTOR, reply(body=body), [])] == ["body"]


def test_non_json_text_is_scanned_raw_only():
    assert found(reply(body="{no es json " + samples.SLASH_CANARY)) == [("canary", "barra", "body")]


def test_deeply_nested_json_does_not_break_the_scan():
    body = "[" * 100_000 + "]" * 100_000
    assert found(reply(body=body)) == []


@pytest.mark.parametrize("body", ["line one\nline two", '{"e": "\\u0054EN-B-0042"}'])
def test_each_decode_of_the_body_is_scanned_alone_and_counted_once(body):
    alt = '{"e": "\\u0054EN-B-0042"}'
    r = AgentReply(None, None, None, 1.0, "agent returned HTTP 500", body=body, alt_body=alt)
    assert [leak.location for leak in scan_reply(DETECTOR, r, [])] == ["body"]
