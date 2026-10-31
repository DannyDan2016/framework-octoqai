import re
import time

import pytest

from octoqai.experiment import SecurityConfig
from octoqai.security import Detector, Finding, mask
from support import samples
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression, pytest.mark.security]


def found(text, env_values=(), **settings):
    detector = Detector(SecurityConfig.model_validate(settings), env_values=env_values)
    return [(f.kind, f.label) for f in detector.find(text)]


@pytest.mark.smoke
def test_canary_found_case_insensitive():
    canary = {"value": "TEN-B-0042", "label": "cuenta-b"}
    assert found("dato ten-b-0042 aquí", canaries=[canary]) == [("canary", "cuenta-b")]


def test_canary_wins_over_overlapping_email():
    text = f"escribe a {samples.EMAIL}"
    assert found(text, canaries=[{"value": samples.EMAIL}]) == [("canary", "canary-1")]


@pytest.mark.parametrize("label", sorted(samples.SECRET_SAMPLES))
def test_secret_patterns(label):
    assert ("secret", label) in found(f"valor {samples.SECRET_SAMPLES[label]} fin")


@pytest.mark.smoke
def test_card_requires_luhn():
    assert found(f"tarjeta {samples.CARD}") == [("pii", "card")]
    assert found(f"tarjeta {samples.CARD_BAD_LUHN}") == []


def test_iban_requires_checksum():
    assert found(f"cuenta {samples.IBAN}.") == [("pii", "iban")]
    assert found(f"cuenta {samples.IBAN_BAD_CHECKSUM}.") == []


def test_iban_followed_by_a_word_is_still_found():
    assert found(f"cuenta {samples.IBAN} OK") == [("pii", "iban")]


def test_email_detected():
    assert found(f"contacto {samples.EMAIL}") == [("pii", "email")]


@pytest.mark.parametrize("row", cases(__file__, "false_positives"))
def test_bank_figures_are_not_sensitive(row):
    assert found(row["text"]) == []


def test_allowlist_ignores_matching_values():
    allow = [re.escape(samples.SUPPORT_EMAIL)]
    assert found(f"escribe a {samples.SUPPORT_EMAIL}", allow=allow) == []


def test_allowlist_never_hides_a_canary():
    settings = {"canaries": [{"value": samples.SUPPORT_EMAIL}], "allow": [".*"]}
    assert found(f"escribe a {samples.SUPPORT_EMAIL}", **settings) == [("canary", "canary-1")]


def test_allowlist_never_hides_an_env_value():
    env = ["ES-REAL-998877"]
    assert found("cuenta ES-REAL-998877", env_values=env, allow=[".*"]) == [("env", "env")]


def test_detectors_can_be_disabled():
    text = f"{samples.EMAIL} {samples.SECRET_SAMPLES['llm api key']} TEN-B-0042"
    settings = {"detectors": [], "canaries": [{"value": "TEN-B-0042"}]}
    assert found(text, **settings) == [("canary", "canary-1")]


def test_env_values_are_found_only_when_given_and_long_enough():
    assert found("cuenta ES-REAL-998877 abc") == []
    env = ["ES-REAL-998877", "abc"]
    assert found("cuenta ES-REAL-998877 abc", env_values=env) == [("env", "env")]


def test_mask_never_reveals_full_value():
    assert mask("ABCDEFGHIJKLMNOP") == "ABC…NOP"
    assert mask("TEN-B-42") == "T…"


def test_findings_sorted_and_non_overlapping():
    text = f"{samples.CARD} y {samples.EMAIL}"
    findings = Detector(SecurityConfig()).find(text)
    assert [f.label for f in findings] == ["card", "email"]
    assert findings[0].end <= findings[1].start
    assert text[findings[1].start : findings[1].end] == samples.EMAIL


def test_overlap_priority_is_secret_then_env_then_pii():
    assert found(f"a {samples.EMAIL}", env_values=[samples.EMAIL]) == [("env", "env")]
    secret = samples.SECRET_SAMPLES["llm api key"]
    assert found(f"a {secret}", env_values=[secret]) == [("secret", "llm api key")]


def test_card_followed_by_extra_digits_is_still_found():
    assert found(f"tarjeta {samples.CARD_WITH_TRAILING_DIGITS}") == [("pii", "card")]


def test_bearer_scheme_is_case_insensitive():
    assert ("secret", "bearer token") in found(samples.LOWERCASE_BEARER)


def test_candidates_keep_overlapping_findings_for_redaction():
    detector = Detector(SecurityConfig(), env_values=[samples.DATABASE_URL])
    text = f"url {samples.DATABASE_URL} fin"
    kinds = {(f.kind, f.value) for f in detector.candidates(text)}
    assert ("secret", samples.DATABASE_URL[: samples.DATABASE_URL.index("@") + 1]) in kinds
    assert ("env", samples.DATABASE_URL) in kinds
    assert [f.kind for f in detector.find(text)] == ["secret"]


def test_candidates_respect_allow_but_not_for_canaries():
    settings = SecurityConfig.model_validate(
        {"canaries": [{"value": samples.SUPPORT_EMAIL}], "allow": [".*"]}
    )
    found_kinds = [f.kind for f in Detector(settings).candidates(f"a {samples.SUPPORT_EMAIL}")]
    assert found_kinds == ["canary"]


def test_repr_never_shows_sensitive_values():
    config = SecurityConfig.model_validate({"canaries": [{"value": "TEN-B-0042"}]})
    finding = Finding("pii", "email", samples.EMAIL, 0, len(samples.EMAIL))
    assert "TEN-B-0042" not in repr(config)
    assert samples.EMAIL not in repr(finding)


def test_long_alphanumeric_text_is_scanned_in_linear_time():
    text = ("a1b2c3d4e5" * 20) * 1000
    started = time.perf_counter()
    Detector(SecurityConfig()).find(text)
    assert time.perf_counter() - started < 2
