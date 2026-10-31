import pytest

from octoqai.agents import HttpAgentAdapter
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig
from support.agents import RESPONSE_MAP
from support.builders import DEFAULT_URL

pytestmark = [pytest.mark.unit, pytest.mark.regression]

CTX = {
    "message": "Concilia octubre",
    "conversation_id": "c1",
    "turn_index": 0,
    "history": [],
    "variant": {"temperature": 0},
}


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


@pytest.mark.smoke
def test_extracts_answer_tools_tokens(agent_cfg, fake_agent):
    fake_agent.answers("ok", tools=["buscar"], tokens=42)
    r = fake_agent.adapter(agent_cfg).send(CTX)
    assert (r.answer, r.tool_calls, r.tokens, r.error) == ("ok", ["buscar"], 42, None)
    assert r.latency_ms >= 0


@pytest.mark.security
def test_sends_rendered_body_with_variant_params_and_headers(agent_cfg, fake_agent):
    fake_agent.adapter(agent_cfg).send(CTX)
    body = fake_agent.requests[0]
    assert body["temperature"] == 0 and body["message"] == CTX["message"]
    assert fake_agent.headers[0]["Authorization"] == "Bearer s3cret"


def test_missing_tool_calls_path_is_none_not_empty(agent_cfg, fake_agent):
    fake_agent.answers("x")
    assert fake_agent.adapter(agent_cfg).send(CTX).tool_calls is None


def test_empty_tool_calls_list_is_empty_not_none(agent_cfg, fake_agent):
    fake_agent.replies({"answer": "x", "tool_calls": []})
    assert fake_agent.adapter(agent_cfg).send(CTX).tool_calls == []


def test_http_500_becomes_error_reply(agent_cfg, fake_agent):
    r = fake_agent.fails(500, "boom").adapter(agent_cfg).send(CTX)
    assert r.answer is None and "500" in r.error


def test_non_json_body_becomes_error_reply(agent_cfg, fake_agent):
    fake_agent.returns(b"<html>")
    assert "JSON" in fake_agent.adapter(agent_cfg).send(CTX).error


def test_timeout_becomes_error_reply(agent_cfg, fake_agent):
    assert "timeout" in fake_agent.times_out().adapter(agent_cfg).send(CTX).error.lower()


def test_connection_refused_becomes_error_reply(agent_cfg, fake_agent):
    assert "connect" in fake_agent.refuses().adapter(agent_cfg).send(CTX).error.lower()


def test_missing_answer_path_is_error(agent_cfg, fake_agent):
    fake_agent.replies({"other": 1})
    assert "answer" in fake_agent.adapter(agent_cfg).send(CTX).error


def test_structured_answer_is_serialized(agent_cfg, fake_agent):
    fake_agent.answers({"monto": "1.000,50 €"})
    assert fake_agent.adapter(agent_cfg).send(CTX).answer == '{"monto": "1.000,50 €"}'


def test_invalid_jsonpath_is_config_error(agent_cfg):
    with pytest.raises(ConfigError, match="JSONPath"):
        HttpAgentAdapter(_with_response(agent_cfg, answer="%%"))


def test_tool_call_objects_are_not_evaluable_instead_of_stringified(agent_cfg, fake_agent):
    cfg = _with_response(agent_cfg, tool_calls="$.tool_calls[*]")
    fake_agent.answers("x", tools=["borrar"])
    assert fake_agent.adapter(cfg).send(CTX).tool_calls is None


def test_null_tool_calls_is_none(agent_cfg, fake_agent):
    cfg = _with_response(agent_cfg, tool_calls="$.tool_calls")
    fake_agent.replies({"answer": "x", "tool_calls": None})
    assert fake_agent.adapter(cfg).send(CTX).tool_calls is None


def test_non_finite_tokens_become_none(agent_cfg, fake_agent):
    fake_agent.returns(b'{"answer": "x", "usage": {"total_tokens": Infinity}}')
    r = fake_agent.adapter(agent_cfg).send(CTX)
    assert r.error is None and r.tokens is None


def test_null_answer_is_error(agent_cfg, fake_agent):
    fake_agent.replies({"answer": None})
    assert "null" in fake_agent.adapter(agent_cfg).send(CTX).error
