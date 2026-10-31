import json

import httpx
import pytest

from octoqai.agents import HttpAgentAdapter
from octoqai.experiment import AgentConfig

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
        url="http://agent.local/chat",
        headers={"Authorization": "Bearer s3cret"},
        request={"message": "{{ message }}", "temperature": "{{ variant.temperature }}"},
        response={
            "answer": "$.answer",
            "tool_calls": "$.tool_calls[*].name",
            "tokens": "$.usage.total_tokens",
        },
        timeout_s=5,
    )


def mock(handler):
    return httpx.MockTransport(handler)


def test_extracts_answer_tools_tokens(agent_cfg):
    t = mock(
        lambda req: httpx.Response(
            200,
            json={
                "answer": "ok",
                "tool_calls": [{"name": "buscar"}],
                "usage": {"total_tokens": 42},
            },
        )
    )
    r = HttpAgentAdapter(agent_cfg, transport=t).send(CTX)
    assert (r.answer, r.tool_calls, r.tokens, r.error) == ("ok", ["buscar"], 42, None)
    assert r.latency_ms >= 0


def test_sends_rendered_body_with_variant_params_and_headers(agent_cfg):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        seen["auth"] = req.headers["Authorization"]
        return httpx.Response(200, json={"answer": "x"})

    HttpAgentAdapter(agent_cfg, transport=mock(handler)).send(CTX)
    assert seen["temperature"] == 0 and seen["message"] == CTX["message"]
    assert seen["auth"] == "Bearer s3cret"


def test_missing_tool_calls_path_is_none_not_empty(agent_cfg):
    t = mock(lambda q: httpx.Response(200, json={"answer": "x"}))
    assert HttpAgentAdapter(agent_cfg, transport=t).send(CTX).tool_calls is None


def test_empty_tool_calls_list_is_empty_not_none(agent_cfg):
    t = mock(lambda q: httpx.Response(200, json={"answer": "x", "tool_calls": []}))
    assert HttpAgentAdapter(agent_cfg, transport=t).send(CTX).tool_calls == []


def test_http_500_becomes_error_reply(agent_cfg):
    t = mock(lambda q: httpx.Response(500, text="boom"))
    r = HttpAgentAdapter(agent_cfg, transport=t).send(CTX)
    assert r.answer is None and "500" in r.error


def test_non_json_body_becomes_error_reply(agent_cfg):
    t = mock(lambda q: httpx.Response(200, text="<html>"))
    assert "JSON" in HttpAgentAdapter(agent_cfg, transport=t).send(CTX).error


def test_timeout_becomes_error_reply(agent_cfg):
    def handler(q):
        raise httpx.ReadTimeout("slow", request=q)

    assert "timeout" in HttpAgentAdapter(agent_cfg, transport=mock(handler)).send(CTX).error.lower()


def test_connection_refused_becomes_error_reply(agent_cfg):
    def handler(q):
        raise httpx.ConnectError("refused", request=q)

    assert "connect" in HttpAgentAdapter(agent_cfg, transport=mock(handler)).send(CTX).error.lower()


def test_missing_answer_path_is_error(agent_cfg):
    t = mock(lambda q: httpx.Response(200, json={"other": 1}))
    assert "answer" in HttpAgentAdapter(agent_cfg, transport=t).send(CTX).error


def test_structured_answer_is_serialized(agent_cfg):
    t = mock(lambda q: httpx.Response(200, json={"answer": {"monto": "1.000,50 €"}}))
    assert HttpAgentAdapter(agent_cfg, transport=t).send(CTX).answer == '{"monto": "1.000,50 €"}'


def test_invalid_jsonpath_is_config_error(agent_cfg):
    from octoqai.errors import ConfigError

    bad = agent_cfg.model_copy(
        update={"response": agent_cfg.response.model_copy(update={"answer": "%%"})}
    )
    with pytest.raises(ConfigError, match="JSONPath"):
        HttpAgentAdapter(bad)


def test_tool_call_objects_are_not_evaluable_instead_of_stringified(agent_cfg):
    cfg = agent_cfg.model_copy(
        update={"response": agent_cfg.response.model_copy(update={"tool_calls": "$.tool_calls[*]"})}
    )
    t = mock(
        lambda q: httpx.Response(200, json={"answer": "x", "tool_calls": [{"name": "borrar"}]})
    )
    assert HttpAgentAdapter(cfg, transport=t).send(CTX).tool_calls is None


def test_null_tool_calls_is_none(agent_cfg):
    cfg = agent_cfg.model_copy(
        update={"response": agent_cfg.response.model_copy(update={"tool_calls": "$.tool_calls"})}
    )
    t = mock(lambda q: httpx.Response(200, json={"answer": "x", "tool_calls": None}))
    assert HttpAgentAdapter(cfg, transport=t).send(CTX).tool_calls is None


def test_non_finite_tokens_become_none(agent_cfg):
    t = mock(
        lambda q: httpx.Response(
            200, content=b'{"answer": "x", "usage": {"total_tokens": Infinity}}'
        )
    )
    r = HttpAgentAdapter(agent_cfg, transport=t).send(CTX)
    assert r.error is None and r.tokens is None


def test_null_answer_is_error(agent_cfg):
    t = mock(lambda q: httpx.Response(200, json={"answer": None}))
    assert "null" in HttpAgentAdapter(agent_cfg, transport=t).send(CTX).error
