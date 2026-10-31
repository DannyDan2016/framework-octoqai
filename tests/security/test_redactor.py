import pytest

from octoqai.experiment import SecurityConfig
from octoqai.security import Detector, Redactor
from support import samples

pytestmark = [pytest.mark.unit, pytest.mark.regression, pytest.mark.security]


def redactor(env_values=(), **settings):
    security = SecurityConfig.model_validate(
        {"canaries": [{"value": "TEN-B-0042", "label": "cuenta-b"}], **settings}
    )
    return Redactor(Detector(security, env_values=env_values))


@pytest.mark.smoke
def test_redacts_every_finding_with_its_kind_and_label():
    secret = samples.SECRET_SAMPLES["llm api key"]
    text = f"cuenta TEN-B-0042, clave {secret}, correo {samples.EMAIL}"
    assert redactor().redact(text) == (
        "cuenta [canary:cuenta-b], clave [secret:llm api key], correo [pii:email]"
    )


def test_env_values_become_the_redacted_placeholder():
    assert redactor(["ES-REAL-998877"]).redact("ok ES-REAL-998877.") == "ok [redacted]."


def test_canary_from_env_keeps_its_label():
    assert redactor(["TEN-B-0042"]).redact("dato TEN-B-0042") == "dato [canary:cuenta-b]"


def test_redact_tree_redacts_values_not_keys():
    tree = {"k": ["x TEN-B-0042", 3, None, {"TEN-B-0042": "TEN-B-0042"}]}
    assert redactor().redact_tree(tree) == {
        "k": ["x [canary:cuenta-b]", 3, None, {"TEN-B-0042": "[canary:cuenta-b]"}]
    }


def test_overlapping_findings_leave_no_partial_value():
    url = samples.DATABASE_URL
    assert redactor([url]).redact(f"uri {url} fin") == "uri [secret:url with credentials] fin"


def test_merged_span_uses_the_highest_priority_kind():
    text = f"x {samples.EMAIL} y"
    assert redactor([samples.EMAIL]).redact(text) == "x [redacted] y"


def test_always_runs_every_detector_regardless_of_config():
    security = SecurityConfig.model_validate({"detectors": []})
    key = samples.SECRET_SAMPLES["llm api key"]
    result = Redactor.for_report(security, []).redact(f"mail {samples.EMAIL} clave {key}")
    assert result == "mail [pii:email] clave [secret:llm api key]"


def test_report_redactor_ignores_allow_patterns():
    security = SecurityConfig.model_validate({"allow": [".*"]})
    key = samples.SECRET_SAMPLES["llm api key"]
    text = f"{samples.EMAIL} {samples.DATABASE_URL} {key}"
    result = Redactor.for_report(security, []).redact(text)
    assert samples.EMAIL not in result and "hunter2" not in result and key not in result


def test_touching_spans_are_not_merged():
    assert redactor(["abcd", "efgh"]).redact("abcdefgh") == "[redacted][redacted]"
