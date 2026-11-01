import pytest

from octoqai.agents import AgentReply, HttpAgentAdapter
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig, RetryPolicy, SecurityConfig, WaitPolicy
from octoqai.security import Detector, scan_reply
from support.agents import RESPONSE_MAP, TURN_CONTEXT
from support.builders import DEFAULT_URL
from support.samples import NON_ASCII_CANARY

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture
def agent_cfg():
    return AgentConfig(
        url=DEFAULT_URL,
        headers={"Authorization": "Bearer s3cret"},
        request={"message": "{{ message }}", "temperature": "{{ variant.temperature }}"},
        response=RESPONSE_MAP,
        timeout_s=5,
    )


def _with_response(cfg, **paths):
    return cfg.model_copy(update={"response": cfg.response.model_copy(update=paths)})


def test_reply_defaults_carry_no_stream_data():
    r = AgentReply("ok", None, None, 1.0, None)
    assert (r.ttft_ms, r.fragments, r.body, r.retries) == (None, [], None, [])


def test_json_adapter_requires_answer_path(agent_cfg):
    with pytest.raises(ConfigError, match="answer"):
        HttpAgentAdapter(_with_response(agent_cfg, answer=None))


@pytest.mark.smoke
def test_extracts_answer_tools_tokens(agent_cfg, fake_agent):
    fake_agent.answers("ok", tools=["buscar"], tokens=42)
    r = fake_agent.adapter(agent_cfg).send(TURN_CONTEXT)
    assert (r.answer, r.tool_calls, r.tokens, r.error) == ("ok", ["buscar"], 42, None)
    assert r.latency_ms >= 0


@pytest.mark.security
def test_sends_rendered_body_with_variant_params_and_headers(agent_cfg, fake_agent):
    fake_agent.adapter(agent_cfg).send(TURN_CONTEXT)
    body = fake_agent.requests[0]
    assert body["temperature"] == 0 and body["message"] == TURN_CONTEXT["message"]
    assert fake_agent.headers[0]["Authorization"] == "Bearer s3cret"


def test_missing_tool_calls_path_is_none_not_empty(agent_cfg, fake_agent):
    fake_agent.answers("x")
    assert fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).tool_calls is None


def test_empty_tool_calls_list_is_empty_not_none(agent_cfg, fake_agent):
    fake_agent.replies({"answer": "x", "tool_calls": []})
    assert fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).tool_calls == []


def test_missing_answer_path_is_error(agent_cfg, fake_agent):
    fake_agent.replies({"other": 1})
    assert "answer" in fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).error


def test_structured_answer_is_serialized(agent_cfg, fake_agent):
    fake_agent.answers({"monto": "1.000,50 €"})
    assert fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).answer == '{"monto": "1.000,50 €"}'


def test_invalid_jsonpath_is_config_error(agent_cfg):
    with pytest.raises(ConfigError, match="JSONPath"):
        HttpAgentAdapter(_with_response(agent_cfg, answer="%%"))


def test_tool_call_objects_are_not_evaluable_instead_of_stringified(agent_cfg, fake_agent):
    cfg = _with_response(agent_cfg, tool_calls="$.tool_calls[*]")
    fake_agent.answers("x", tools=["borrar"])
    assert fake_agent.adapter(cfg).send(TURN_CONTEXT).tool_calls is None


def test_null_tool_calls_is_none(agent_cfg, fake_agent):
    cfg = _with_response(agent_cfg, tool_calls="$.tool_calls")
    fake_agent.replies({"answer": "x", "tool_calls": None})
    assert fake_agent.adapter(cfg).send(TURN_CONTEXT).tool_calls is None


def test_non_finite_tokens_become_none(agent_cfg, fake_agent):
    fake_agent.returns(b'{"answer": "x", "usage": {"total_tokens": Infinity}}')
    r = fake_agent.adapter(agent_cfg).send(TURN_CONTEXT)
    assert r.error is None and r.tokens is None


def test_null_answer_is_error(agent_cfg, fake_agent):
    fake_agent.replies({"answer": None})
    assert "null" in fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).error


def _with(cfg, **update):
    return cfg.model_copy(update=update)


def test_json_reply_keeps_body_for_scanning(agent_cfg, fake_agent):
    fake_agent.replies({"answer": "ok", "debug": "traza"})
    assert '"debug"' in fake_agent.adapter(agent_cfg).send(TURN_CONTEXT).body


def test_non_json_body_is_kept(agent_cfg, fake_agent):
    r = fake_agent.returns(b"<html>").adapter(agent_cfg).send(TURN_CONTEXT)
    assert (r.error, r.body) == ("agent response is not valid JSON", "<html>")


@pytest.mark.smoke
def test_retries_connection_failures_and_records_each(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=2, backoff_s=0.5))
    r = fake_agent.answers("ok").refuses_first(2).adapter(cfg).send(TURN_CONTEXT)
    assert (r.answer, r.error) == ("ok", None)
    assert r.retries == ["cannot connect to agent: ConnectError"] * 2
    assert fake_agent.sleeps == [0.5, 1.0]
    assert len(fake_agent.requests) == 3


def test_backoff_jitter_shortens_each_wait_by_up_to_a_quarter(agent_cfg, fake_agent):
    fake_agent.jitter = lambda: 1.0
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=2, backoff_s=0.5))
    fake_agent.answers("ok").refuses_first(2).adapter(cfg).send(TURN_CONTEXT)
    assert fake_agent.sleeps == [0.375, 0.75]


def test_latency_excludes_backoff(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=1, backoff_s=0.5))
    r = fake_agent.refuses_first(1).adapter(cfg).send(TURN_CONTEXT)
    assert r.latency_ms == 0 and fake_agent.sleeps == [0.5]


def test_connect_timeout_is_retried(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=1), wait=WaitPolicy(connect_s=2))
    r = fake_agent.answers("ok").refuses_first(1, timeout=True).adapter(cfg).send(TURN_CONTEXT)
    assert r.error is None and r.retries == ["cannot connect to agent within 2s"]


def test_gives_up_after_the_last_retry(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=2, backoff_s=0))
    r = fake_agent.refuses().adapter(cfg).send(TURN_CONTEXT)
    assert r.error == "cannot connect to agent: ConnectError"
    assert len(r.retries) == 2 and len(fake_agent.requests) == 3


@pytest.mark.parametrize("behaviour", ["http_502", "http_503", "http_504", "read_timeout"])
def test_never_retries_once_the_agent_may_have_the_request(agent_cfg, fake_agent, behaviour):
    if behaviour == "read_timeout":
        fake_agent.times_out()
    else:
        fake_agent.fails(int(behaviour.removeprefix("http_")))
    r = fake_agent.adapter(_with(agent_cfg, retry=RetryPolicy(retries=3))).send(TURN_CONTEXT)
    assert r.error is not None and r.retries == [] and len(fake_agent.requests) == 1


def test_no_retry_by_default(agent_cfg, fake_agent):
    fake_agent.refuses().adapter(agent_cfg).send(TURN_CONTEXT)
    assert len(fake_agent.requests) == 1 and fake_agent.sleeps == []


def test_wait_policy_sets_connect_and_total_timeouts(agent_cfg, fake_agent):
    fake_agent.adapter(_with(agent_cfg, wait=WaitPolicy(connect_s=2, total_s=4))).send(TURN_CONTEXT)
    assert fake_agent.timeouts[0] == {"connect": 2, "read": 4, "write": 2, "pool": 2}


def test_read_timeout_names_total_deadline(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, wait=WaitPolicy(total_s=4))
    assert fake_agent.times_out().adapter(cfg).send(TURN_CONTEXT).error == "timeout after 4.0s"


@pytest.mark.parametrize(
    "payload",
    [{"answer": None, "debug": "trace-xyz"}, {"other": 1, "debug": "trace-xyz"}],
    ids=["null-answer", "answer-path-missing"],
)
def test_mapping_failure_keeps_body(agent_cfg, fake_agent, payload):
    r = fake_agent.replies(payload).adapter(agent_cfg).send(TURN_CONTEXT)
    assert r.error is not None and "trace-xyz" in r.body


def test_final_failure_latency_starts_at_the_last_attempt(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, retry=RetryPolicy(retries=1, backoff_s=0.5))
    r = fake_agent.times_out().refuses_first(1).adapter(cfg).send(TURN_CONTEXT)
    assert r.error.startswith("timeout") and r.latency_ms == 0
    assert fake_agent.sleeps == [0.5]


def test_endless_body_is_capped_by_the_total_deadline_and_kept(agent_cfg, fake_agent):
    cfg = _with(agent_cfg, wait=WaitPolicy(total_s=4))
    r = fake_agent.sends("x" * 1024, 2000, gap_s=1.0).adapter(cfg).send(TURN_CONTEXT)
    assert (r.error, len(r.body)) == ("timeout after 4.0s", 4 * 1024)
    assert fake_agent.clock.now == 5


def test_body_over_the_byte_cap_is_an_error(agent_cfg, fake_agent):
    r = fake_agent.sends("x" * 2**20, 17).adapter(agent_cfg).send(TURN_CONTEXT)
    assert r.error == "stream exceeded 16777216 bytes"


@pytest.mark.security
@pytest.mark.parametrize("timeout", [False, True], ids=["watchdog", "read-timeout"])
def test_partial_body_cut_mid_read_is_kept_and_scanned(agent_cfg, fake_agent, timeout):
    cfg = _with(agent_cfg, wait=WaitPolicy(total_s=7))
    partial = '{"answer": "ok", "dbg": "' + NON_ASCII_CANARY
    fake_agent.stalls(status=200, body=partial, timeout=timeout)
    r = fake_agent.adapter(cfg).send(TURN_CONTEXT)
    detector = Detector(SecurityConfig.model_validate({"canaries": [{"value": NON_ASCII_CANARY}]}))
    assert (r.error, r.body) == ("timeout after 7.0s", partial)
    assert [leak.location for leak in scan_reply(detector, r, [])] == ["body"]


@pytest.mark.security
def test_non_json_success_body_is_decoded_with_its_declared_charset(agent_cfg, fake_agent):
    content = NON_ASCII_CANARY.encode("iso-8859-1")
    fake_agent.returns(content, content_type="text/html; charset=iso-8859-1")
    r = fake_agent.adapter(agent_cfg).send(TURN_CONTEXT)
    assert (r.error, r.body) == ("agent response is not valid JSON", NON_ASCII_CANARY)
