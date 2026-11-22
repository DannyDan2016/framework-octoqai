import builtins
import dataclasses
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from octoqai.errors import ConfigError
from octoqai.experiment import ProviderSettings
from octoqai.providers import LlmMessage, LlmReply, LlmRequest
from octoqai.providers.anthropic_api import AnthropicProvider
from octoqai.providers.factory import provider_for
from support import samples
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]

ERRORS = [case.values[0] for case in cases(__file__, "anthropic_errors")]
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "seed", "tool_choice", "tools", "fallbacks"}
HOST = "http://anthropic.example"


def leaks(text: str) -> bool:
    return samples.ANTHROPIC_KEY in text or samples.ANTHROPIC_KEY_NO_PREFIX in text


def make_settings(**overrides: Any) -> ProviderSettings:
    values: dict[str, Any] = {
        "provider": "anthropic",
        "model": "fake-judge-1",
        "api_key": samples.ANTHROPIC_KEY,
        "max_calls": 5,
    }
    return ProviderSettings(**{**values, **overrides})


def request(schema: dict[str, Any] | None = None) -> LlmRequest:
    return LlmRequest(
        role="judge",
        system="Rate the answer.\nBe strict.",
        messages=(
            LlmMessage(role="user", content="first ñ turn"),
            LlmMessage(role="assistant", content="second"),
        ),
        schema=schema,
    )


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="private reasoning")


def message(
    blocks: list[Any],
    *,
    usage: Any = None,
    request_id: Any = "req_ok",
    stop_reason: str | None = "end_turn",
    model: str = "fake-judge-1-served",
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=11, output_tokens=7) if usage is None else usage
    return SimpleNamespace(
        content=blocks, usage=usage, _request_id=request_id, stop_reason=stop_reason, model=model
    )


class FakeClient:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.closed = True


def provider_with(client: FakeClient, **overrides: Any) -> AnthropicProvider:
    clock = iter([10.0, 10.25])
    return AnthropicProvider(
        make_settings(**overrides), client_factory=lambda **_: client, clock=lambda: next(clock)
    )


def all_fields(reply: LlmReply) -> str:
    return json.dumps(dataclasses.asdict(reply), default=str)


def test_request_carries_configured_values_and_untouched_text():
    client = FakeClient(message([text_block("ok")]))
    provider_with(client, max_tokens=999, effort="high").complete(request())
    (sent,) = client.calls
    assert sent["model"] == "fake-judge-1"
    assert sent["max_tokens"] == 999
    assert sent["system"] == "Rate the answer.\nBe strict."
    assert sent["messages"] == [
        {"role": "user", "content": "first ñ turn"},
        {"role": "assistant", "content": "second"},
    ]
    assert sent["output_config"] == {"effort": "high"}


def test_structured_output_shares_the_output_config_with_effort():
    client = FakeClient(message([text_block("{}")]))
    provider_with(client).complete(request(SCHEMA))
    assert client.calls[0]["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }


@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_request_never_has_sampling_forced_tools_or_fallbacks(schema):
    client = FakeClient(message([text_block("ok")]))
    provider_with(client).complete(request(schema))
    assert SAMPLING_KEYS.isdisjoint(client.calls[0])
    assert set(client.calls[0]) == {"model", "max_tokens", "system", "messages", "output_config"}


def test_client_is_built_with_explicit_key_timeout_and_retries():
    seen: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeClient:
        seen.update(kwargs)
        return FakeClient(message([text_block("ok")]))

    AnthropicProvider(make_settings(timeout_s=12.5, max_retries=4), client_factory=factory)
    key_matches = seen.pop("api_key") == samples.ANTHROPIC_KEY
    assert key_matches
    assert seen == {"timeout": 12.5, "max_retries": 4}


def test_reply_copies_text_model_stop_reason_usage_request_id_and_latency():
    client = FakeClient(message([text_block('{"verdict": "pass"}')], stop_reason="max_tokens"))
    reply = provider_with(client).complete(request())
    assert reply == LlmReply(
        text='{"verdict": "pass"}',
        model="fake-judge-1-served",
        stop_reason="max_tokens",
        usage=reply.usage,
        latency_ms=250.0,
        error=None,
        request_id="req_ok",
    )
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (11, 7)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        ([text_block("a"), text_block("b")], "ab"),
        ([thinking_block(), text_block("a"), thinking_block(), text_block("b")], "ab"),
        ([thinking_block(), text_block("")], ""),
        ([thinking_block()], None),
        ([], None),
    ],
    ids=["two-text", "thinking-between", "empty-text-block", "only-thinking", "no-blocks"],
)
def test_text_is_the_concatenation_of_text_blocks_only(blocks, expected):
    reply = provider_with(FakeClient(message(blocks))).complete(request())
    assert reply.text == expected
    assert reply.error is None


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(input_tokens=None, output_tokens=None),
        SimpleNamespace(),
        SimpleNamespace(input_tokens="3", output_tokens=2.5),
    ],
    ids=["none", "absent", "wrong-types"],
)
def test_missing_or_malformed_usage_is_none(usage):
    reply = provider_with(FakeClient(message([text_block("x")], usage=usage))).complete(request())
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (None, None)


def test_partial_usage_keeps_the_known_half():
    usage = SimpleNamespace(input_tokens=9, output_tokens=None)
    reply = provider_with(FakeClient(message([text_block("x")], usage=usage))).complete(request())
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (9, None)


@pytest.mark.parametrize("request_id", [None, 5, ""], ids=["none", "not-a-string", "empty"])
def test_request_id_is_none_unless_a_non_empty_string(request_id):
    client = FakeClient(message([text_block("x")], request_id=request_id))
    assert provider_with(client).complete(request()).request_id is None


def test_response_without_request_id_attribute_or_stop_reason():
    bare = SimpleNamespace(content=[text_block("x")], usage=SimpleNamespace(), model=None)
    reply = provider_with(FakeClient(bare)).complete(request())
    assert (reply.request_id, reply.stop_reason, reply.model, reply.text) == (None, None, None, "x")


@pytest.mark.parametrize("stop_reason", ["end_turn", "refusal", "max_tokens", "pause_turn", "odd"])
def test_stop_reason_passes_through_unchanged(stop_reason):
    client = FakeClient(message([text_block("x")], stop_reason=stop_reason))
    assert provider_with(client).complete(request()).stop_reason == stop_reason


def build_exception(row: dict[str, Any]) -> BaseException:
    name = row["exception"]
    secret = f"leaked {samples.ANTHROPIC_KEY} and {samples.ANTHROPIC_KEY_NO_PREFIX}"
    if hasattr(builtins, name):
        return getattr(builtins, name)(secret)
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("POST", f"{HOST}/v1/messages")
    headers = {"request-id": row["request_id"]} if row["request_id"] else {}
    resp = httpx2.Response(row.get("status", 500), request=req, headers=headers)
    cls = getattr(anthropic, name)
    if name == "APIConnectionError":
        return cls(message=secret, request=req)
    if name == "APITimeoutError":
        return cls(request=req)
    if name == "APIResponseValidationError":
        return cls(resp, None, message=secret)
    if name == "CredentialsError":
        return cls(secret)
    return cls(secret, response=resp, body={"echo": secret})


@pytest.mark.parametrize("row", cases(__file__, "anthropic_errors"))
def test_exceptions_become_error_values_without_any_secret(row):
    client = FakeClient(build_exception(row))
    reply = provider_with(client).complete(request())
    assert reply.error == row["expected"]
    assert reply.request_id == row["request_id"]
    assert (reply.text, reply.model, reply.stop_reason) == (None, None, None)
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (None, None)
    assert reply.latency_ms == 250.0
    blob = all_fields(reply)
    leaked = leaks(blob) or "leaked" in blob
    assert not leaked


def test_status_error_with_a_non_string_request_id_has_none():
    anthropic = pytest.importorskip("anthropic")
    exc = build_exception(ERRORS[0])
    assert isinstance(exc, anthropic.APIStatusError)
    exc.request_id = 5
    assert provider_with(FakeClient(exc)).complete(request()).request_id is None


def test_close_closes_the_client():
    client = FakeClient(message([text_block("x")]))
    provider_with(client).close()
    assert client.closed


def test_repr_never_shows_the_key():
    provider = provider_with(FakeClient(message([text_block("x")])))
    settings = make_settings()
    shown = " ".join([repr(provider), repr(settings), str(settings), repr(vars(provider))])
    leaked = leaks(shown)
    assert not leaked


def test_port_and_budget_import_without_the_sdk(monkeypatch):
    real_import = builtins.__import__

    def refuse_sdk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in ("anthropic", "httpx2"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.split(".")[0] in ("anthropic", "octoqai")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", refuse_sdk)
    from octoqai.providers import budget, factory, port

    assert budget.BudgetedLlm and port.LlmReply and factory.provider_for
    assert "anthropic" not in sys.modules


def test_provider_for_without_the_extra_raises_the_fixed_config_error(monkeypatch):
    real_import = builtins.__import__

    def refuse_sdk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "anthropic":
            raise ModuleNotFoundError(f"no {samples.ANTHROPIC_KEY}", name="anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_sdk)
    with pytest.raises(ConfigError) as caught:
        provider_for(make_settings())
    assert str(caught.value) == "the anthropic provider needs: pip install 'octoqai[anthropic]'"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("variable", ["ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"])
def test_provider_for_refuses_endpoint_overrides_naming_only_the_variable(monkeypatch, variable):
    monkeypatch.setenv(variable, "http://hijack.example/x")
    with pytest.raises(ConfigError) as caught:
        provider_for(make_settings(), client_factory=lambda **_: FakeClient(None))
    assert variable in str(caught.value)
    assert "hijack" not in str(caught.value)


@pytest.mark.parametrize("variable", ["ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"])
def test_provider_for_accepts_an_empty_variable(monkeypatch, variable):
    monkeypatch.setenv(variable, "")
    built = provider_for(make_settings(), client_factory=lambda **_: FakeClient(None))
    assert isinstance(built, AnthropicProvider)


def test_provider_for_passes_seams_to_the_adapter():
    client = FakeClient(message([text_block("ok")]))
    built = provider_for(make_settings(), client_factory=lambda **_: client)
    assert built.complete(request()).text == "ok"
    assert len(client.calls) == 1


def test_provider_for_unknown_provider_is_a_config_error():
    settings = SimpleNamespace(provider="mystery")
    with pytest.raises(ConfigError, match="unknown provider"):
        provider_for(settings)


def test_real_sdk_over_a_mock_transport_sends_the_expected_request_and_maps_a_401():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: list[Any] = []

    def handler(req: Any) -> Any:
        seen.append(req)
        if len(seen) == 1:
            body = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "fake-judge-1-served",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": '{"verdict": "pass"}'},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 6},
            }
            return httpx2.Response(200, json=body, headers={"request-id": "req_mock_ok"})
        return httpx2.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "bad"}},
            headers={"request-id": "req_mock_401"},
        )

    def factory(**kwargs: Any) -> Any:
        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        return anthropic.Anthropic(
            base_url=HOST, http_client=client, open_telemetry=False, **{**kwargs, "max_retries": 0}
        )

    provider = AnthropicProvider(make_settings(max_tokens=512), client_factory=factory)
    ok = provider.complete(request(SCHEMA))
    assert (ok.text, ok.model, ok.stop_reason, ok.request_id) == (
        '{"verdict": "pass"}',
        "fake-judge-1-served",
        "end_turn",
        "req_mock_ok",
    )
    assert (ok.usage.input_tokens, ok.usage.output_tokens) == (4, 6)
    body = json.loads(seen[0].content)
    assert body["model"] == "fake-judge-1" and body["max_tokens"] == 512
    assert body["system"] == "Rate the answer.\nBe strict."
    assert body["messages"][0] == {"role": "user", "content": "first ñ turn"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert SAMPLING_KEYS.isdisjoint(body)
    bearer_matches = seen[0].headers["authorization"] == "Bearer " + samples.ANTHROPIC_KEY
    assert bearer_matches
    assert "x-api-key" not in seen[0].headers

    failed = provider.complete(request())
    assert (failed.error, failed.request_id, failed.text) == (
        "AuthenticationError",
        "req_mock_401",
        None,
    )
    leaked = leaks(all_fields(failed))
    assert not leaked


class ExplodingMessage:
    content: list[Any] = []

    @property
    def usage(self) -> Any:
        raise RuntimeError(f"boom {samples.ANTHROPIC_KEY}")


def test_a_response_whose_attribute_raises_is_an_error_value_not_an_exception():
    reply = provider_with(FakeClient(ExplodingMessage())).complete(request())
    assert reply.error == "provider failure"
    assert (reply.text, reply.model, reply.stop_reason) == (None, None, None)
    leaked = leaks(all_fields(reply))
    assert not leaked


def test_base_exceptions_still_propagate():
    with pytest.raises(KeyboardInterrupt):
        provider_with(FakeClient(KeyboardInterrupt())).complete(request())


def sdk_provider(handler: Any) -> AnthropicProvider:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def factory(**kwargs: Any) -> Any:
        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        return anthropic.Anthropic(
            base_url=HOST, http_client=client, open_telemetry=False, **{**kwargs, "max_retries": 0}
        )

    return AnthropicProvider(make_settings(), client_factory=factory)


@pytest.mark.parametrize("row", cases(__file__, "anthropic_odd_bodies"))
def test_odd_200_bodies_through_the_real_sdk_never_raise(row):
    httpx2 = pytest.importorskip("httpx2")

    def handler(req: Any) -> Any:
        if "raw" in row:
            return httpx2.Response(200, content=row["raw"], headers={"content-type": "text/html"})
        return httpx2.Response(200, json=row["body"])

    reply = sdk_provider(handler).complete(request())
    assert (reply.error, reply.text) == (row["error"], row["text"])
    leaked = leaks(all_fields(reply))
    assert not leaked


def test_default_client_is_built_with_telemetry_off_and_explicit_settings(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    AnthropicProvider(make_settings())
    assert seen["open_telemetry"] is False
    assert set(seen) == {"open_telemetry", "api_key", "timeout", "max_retries"}


def test_provider_for_sdk_import_error_inside_the_sdk_is_a_fixed_config_error(monkeypatch):
    anthropic = pytest.importorskip("anthropic")

    def boom(**_: Any) -> Any:
        raise ImportError(f"socksio missing {samples.ANTHROPIC_KEY}", name="socksio")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    with pytest.raises(ConfigError) as caught:
        provider_for(make_settings())
    assert str(caught.value) == "the anthropic client could not be built: ImportError"
    assert caught.value.__suppress_context__


def test_provider_for_injected_factory_import_error_propagates_unchanged():
    def factory(**_: Any) -> Any:
        raise ImportError("from test code")

    with pytest.raises(ImportError, match="from test code"):
        provider_for(make_settings(), client_factory=factory)


def recording_provider(trace: Any, recorded: list[str]) -> Any:
    class Span(trace.NonRecordingSpan):
        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: Any) -> None:
            recorded.append(f"{key}={value}")

        def add_event(self, name: str, attributes: Any = None, timestamp: Any = None) -> None:
            recorded.append(f"{name}={attributes}")

    class Tracer(trace.Tracer):
        def start_span(self, name: str, *args: Any, **kwargs: Any) -> Any:
            recorded.append(f"span:{name}")
            for key, value in (kwargs.get("attributes") or {}).items():
                recorded.append(f"{key}={value}")
            return Span(trace.INVALID_SPAN_CONTEXT)

        def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class Provider(trace.TracerProvider):
        def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
            return Tracer()

    return Provider()


def test_telemetry_is_silent_even_with_content_mode_and_a_recording_provider(monkeypatch):
    trace = pytest.importorskip("opentelemetry.trace")
    httpx2 = pytest.importorskip("httpx2")
    recorded: list[str] = []
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", recording_provider(trace, recorded))
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    seen: list[Any] = []

    def handler(req: Any) -> Any:
        seen.append(req)
        body = {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "fake-judge-1",
            "content": [{"type": "text", "text": "REPLY-CANARY"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return httpx2.Response(200, json=body)

    ask = LlmRequest(
        role="judge", system="SYSTEM-CANARY", messages=(LlmMessage("user", "PROMPT-CANARY"),)
    )
    reply = sdk_provider(handler).complete(ask)
    assert reply.text == "REPLY-CANARY"
    assert recorded == []
    assert "traceparent" not in seen[0].headers and "baggage" not in seen[0].headers
