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
