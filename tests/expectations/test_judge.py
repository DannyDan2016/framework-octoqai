import pytest

from octoqai.expectations import (
    JUDGE_KIND,
    JUDGE_PROMPT_VERSION,
    JUDGE_SCHEMA,
    JudgeInputError,
    judge_input_failed,
    judge_prompt_sha256,
    judge_request,
    judge_result,
    judge_skipped,
)
from octoqai.experiment import JudgeCheck
from octoqai.providers.port import LlmReply, LlmRequest, LlmUsage
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]

NONCE = "NONCE"
PINNED_PROMPT_SHA256 = "75dc8bb81d98bf4dec4eee9ce7b826981e26b3851983b26e5746ff25575c977d"
CHECK = JudgeCheck(id="q1", rubric="Responde con cortesia", reference="Saludo formal", hard=True)
VERDICT = '{"verdict": "pass", "rationale": "ok"}'


def reply(text=None, model="m1", stop_reason="end_turn", error=None, **extra):
    return LlmReply(
        text=text,
        model=model,
        stop_reason=stop_reason,
        usage=LlmUsage(11, 7),
        latency_ms=5.0,
        error=error,
        **extra,
    )


def build(transcript, check=CHECK, *, focus_last=True, nonce=NONCE, max_input_chars=2_000_000):
    return judge_request(
        check, transcript, focus_last=focus_last, nonce=nonce, max_input_chars=max_input_chars
    )


@pytest.mark.parametrize("row", cases(__file__, "judge_verdicts"))
def test_reply_maps_to_a_safe_result(row):
    result = judge_result(CHECK, reply(**row["reply"]), expected_model="m1", rubric_sha256="h")
    assert (result.passed, result.verdict) == (row["passed"], row["verdict"])
    assert result.kind == JUDGE_KIND and result.hard is True and result.agent_error is False
    if row["detail"] is not None:
        assert result.detail == row["detail"]
    assert (result.check_id, result.prompt_version, result.rubric_sha256) == (
        "q1",
        JUDGE_PROMPT_VERSION,
        "h",
    )


def test_provider_metadata_is_kept_on_a_verdict():
    result = judge_result(
        CHECK, reply(VERDICT, request_id="req-1"), expected_model="m1", rubric_sha256="h"
    )
    assert (result.model, result.input_tokens, result.output_tokens, result.request_id) == (
        "m1",
        11,
        7,
        "req-1",
    )


def test_rationale_is_truncated_and_never_in_detail():
    long = "SECRETO-" + "x" * 2000
    result = judge_result(
        CHECK,
        reply(f'{{"verdict": "unsure", "rationale": "{long}"}}'),
        expected_model="m1",
        rubric_sha256="h",
    )
    assert result.rationale == long[:1000]
    assert "SECRETO" not in result.detail


def test_details_never_contain_rubric_or_agent_text():
    check = JudgeCheck(id="q1", rubric="RUBRICA-PRIVADA")
    texts = ("AGENT-TEXT RUBRICA-PRIVADA", '{"verdict": "unsure", "rationale": "RUBRICA-PRIVADA"}')
    for text in texts:
        result = judge_result(check, reply(text), expected_model="m1", rubric_sha256="h")
        assert "RUBRICA-PRIVADA" not in result.detail and "AGENT-TEXT" not in result.detail


def test_skipped_judge_fails_without_a_call():
    result = judge_skipped(CHECK, True)
    assert (result.passed, result.agent_error, result.hard, result.kind) == (
        False,
        True,
        True,
        JUDGE_KIND,
    )
    assert result.check_id == "q1" and result.verdict is None


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        ("too_long", "judge 'q1': transcript too long"),
        ("nonce_collision", "judge 'q1': delimiter collision"),
    ],
)
def test_input_failures_are_not_evaluable(reason, detail):
    result = judge_input_failed(
        CHECK, JudgeInputError(reason), expected_model="m1", rubric_sha256="h"
    )
    assert (result.passed, result.detail, result.rubric_sha256) == (None, detail, "h")


def test_request_shape():
    request = build([("hola", "buenas")])
    assert isinstance(request, LlmRequest)
    assert request.role == "judge" and request.schema == JUDGE_SCHEMA
    assert len(request.messages) == 1 and request.messages[0].role == "user"
    body = request.messages[0].content
    assert "Responde con cortesia" in body and "Saludo formal" in body
    assert "Responde con cortesia" not in request.system
    assert NONCE in request.system


def test_rubric_and_reference_live_outside_the_transcript_block():
    body = build([("hola", "buenas")]).messages[0].content
    start = body.index(f"<<<BEGIN-TRANSCRIPT-{NONCE}>>>")
    end = body.index(f"<<<END-TRANSCRIPT-{NONCE}>>>")
    inside = body[start:end]
    assert "Responde con cortesia" not in inside and "Saludo formal" not in inside
    assert "hola" in inside and "buenas" in inside
    assert body.count(f"<<<BEGIN-TRANSCRIPT-{NONCE}>>>") == 1


def test_reference_block_is_omitted_when_absent():
    body = build([("a", "b")], JudgeCheck(id="q2", rubric="r")).messages[0].content
    assert "REFERENCE" not in body


def test_focus_last_marks_the_final_turn():
    turns = [("u1", "a1"), ("u2", "a2")]
    focused = build(turns, focus_last=True).messages[0].content
    whole = build(turns, focus_last=False).messages[0].content
    assert "ONLY the last turn" in focused and "ONLY the last turn" not in whole
    assert "whole conversation" in whole and "whole conversation" not in focused


@pytest.mark.parametrize("row", cases(__file__, "judge_injection"))
def test_agent_output_stays_data(row):
    output = row["output"]
    request = build([("pregunta", output)])
    if NONCE in output:
        assert request == JudgeInputError("nonce_collision")
        return
    assert isinstance(request, LlmRequest)
    body = request.messages[0].content
    assert (output == "" or output not in request.system) and "pregunta" not in request.system
    assert body.count(f"<<<END-TRANSCRIPT-{NONCE}>>>") == 1
    start = body.index(f"<<<BEGIN-TRANSCRIPT-{NONCE}>>>")
    end = body.index(f"<<<END-TRANSCRIPT-{NONCE}>>>")
    assert output in body[start:end]


@pytest.mark.parametrize("field", ["rubric", "reference", "user", "agent"])
def test_nonce_in_any_input_is_a_collision(field):
    check = JudgeCheck(
        id="q1",
        rubric=f"r {NONCE}" if field == "rubric" else "r",
        reference=f"x {NONCE}" if field == "reference" else "x",
    )
    turn = (
        f"u {NONCE}" if field == "user" else "u",
        f"a {NONCE}" if field == "agent" else "a",
    )
    assert build([turn], check) == JudgeInputError("nonce_collision")


def test_nonce_check_is_case_sensitive():
    assert isinstance(build([("u", "a nonce")]), LlmRequest)


def test_one_mebibyte_output_is_data_within_the_limit_and_too_long_above():
    big = "A" * (1 << 20)
    assert isinstance(build([("u", big)]), LlmRequest)
    assert build([("u", big)], max_input_chars=1000) == JudgeInputError("too_long")


def test_too_long_counts_rubric_reference_and_every_text():
    total = len(CHECK.rubric) + len(CHECK.reference) + 2
    assert isinstance(build([("u", "a")], max_input_chars=total + 1), LlmRequest)
    assert build([("u", "a")], max_input_chars=total - 1) == JudgeInputError("too_long")


def test_system_prompt_does_not_depend_on_inputs():
    one = build([("u", "a")]).system
    other = build([("otra", "cosa")], JudgeCheck(id="z", rubric="otro")).system
    assert one == other


def test_prompt_hash_is_stable_and_pinned():
    assert judge_prompt_sha256() == judge_prompt_sha256()
    assert judge_prompt_sha256() == PINNED_PROMPT_SHA256, (
        "the judge template or schema changed: bump JUDGE_PROMPT_VERSION and update the pin"
    )


def test_schema_is_strict():
    assert JUDGE_SCHEMA["additionalProperties"] is False
    assert set(JUDGE_SCHEMA["required"]) == {"verdict", "rationale"}
    assert JUDGE_SCHEMA["properties"]["verdict"]["enum"] == ["pass", "fail", "unsure"]
