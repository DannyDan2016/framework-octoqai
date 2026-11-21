import dataclasses
import json
import zlib
from typing import Any

import httpx
import pytest

from octoqai.errors import ConfigError
from octoqai.experiment import ProviderSettings
from octoqai.providers import (
    LlmMessage,
    LlmReply,
    LlmRequest,
    LlmUsage,
    OllamaProvider,
    provider_for,
)
from octoqai.providers.ollama_api import DEFAULT_BASE_URL, MAX_RESPONSE_BYTES
from support import samples
from support.paths import cases
from support.providers import FakeOllama

pytestmark = [pytest.mark.unit, pytest.mark.regression]

ERRORS = cases(__file__, "ollama_errors")
URLS = cases(__file__, "ollama_urls") + [
    pytest.param(row, id=row["id"]) for row in samples.BUILT_URL_ROWS
]
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}
FORBIDDEN_KEYS = {"seed", "num_ctx", "effort", "api_key", "tools", "keep_alive", "top_p"}
LOOPBACK_MESSAGE = (
    "ollama base_url must be a loopback address (127.0.0.1, localhost or [::1]); "
    "set allow_remote: true to use a remote server"
)
URL_MESSAGES = {
    LOOPBACK_MESSAGE,
    "ollama base_url is not a valid url",
    "ollama base_url must use http or https and name a host",
    "ollama base_url must not contain credentials",
    "ollama base_url must not contain a path, query or fragment",
}
ENDPOINT = "http://127.0.0.1:11434/api/chat"
BUILTIN_ERRORS: dict[str, type[Exception]] = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
}


def make_settings(**overrides: Any) -> ProviderSettings:
    values: dict[str, Any] = {"provider": "ollama", "model": "fake-local-1", "max_calls": 5}
    return ProviderSettings(**{**values, **overrides})


def settings_with_url(base_url: str, allow_remote: bool) -> ProviderSettings:
    return make_settings().model_copy(update={"base_url": base_url, "allow_remote": allow_remote})


def request(
    schema: dict[str, Any] | None = None, system: str = "Rate the answer.\nBe strict."
) -> LlmRequest:
    return LlmRequest(
        role="judge",
        system=system,
        messages=(
            LlmMessage(role="user", content="first ñ turn"),
            LlmMessage(role="assistant", content="second"),
            LlmMessage(role="user", content=""),
        ),
        schema=schema,
    )


def provider_on(fake: FakeOllama, **overrides: Any) -> OllamaProvider:
    return OllamaProvider(
        make_settings(**overrides),
        transport=fake.transport,
        clock=fake.clock,
        watchdog=fake.watchdog,
    )


def all_fields(reply: LlmReply) -> str:
    return json.dumps(dataclasses.asdict(reply), default=str)


@pytest.mark.parametrize(
    "system", ["Rate the answer.\nBe strict.", ""], ids=["system", "no-system"]
)
@pytest.mark.parametrize("schema", [None, SCHEMA], ids=["free-text", "schema"])
def test_request_body_has_exactly_the_documented_shape(system, schema):
    fake = FakeOllama().answers("ok")
    provider_on(fake, max_tokens=999).complete(request(schema, system))
    expected_messages = [
        {"role": "user", "content": "first ñ turn"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": ""},
    ]
    if system:
        expected_messages.insert(0, {"role": "system", "content": system})
    expected: dict[str, Any] = {
        "model": "fake-local-1",
        "stream": False,
        "think": False,
        "messages": expected_messages,
        "options": {"temperature": 0, "num_predict": 999},
    }
    if schema is not None:
        expected["format"] = SCHEMA
    assert fake.sent == expected
    assert FORBIDDEN_KEYS.isdisjoint(fake.sent)
    assert FORBIDDEN_KEYS.isdisjoint(fake.sent["options"])


def test_request_without_messages_sends_an_empty_list():
    fake = FakeOllama().answers("ok")
    ask = LlmRequest(role="simulated_user", system="", messages=())
    provider_on(fake).complete(ask)
    assert fake.sent["messages"] == []


def test_request_goes_to_the_chat_endpoint_asking_for_identity_encoding():
    fake = FakeOllama().answers("ok")
    provider_on(fake).complete(request())
    (sent,) = fake.requests
    assert sent.method == "POST"
    assert str(sent.url) == ENDPOINT
    assert sent.headers["accept-encoding"] == "identity"
    assert sent.headers["content-type"] == "application/json"


def test_default_base_url_is_the_loopback_constant():
    assert DEFAULT_BASE_URL == "http://127.0.0.1:11434"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:9999", "http://localhost:9999/api/chat"),
        ("http://127.0.0.1:11434/", ENDPOINT),
        ("http://" + "[::1]:8080", "http://" + "[::1]:8080/api/chat"),
    ],
    ids=["port", "trailing-slash", "ipv6"],
)
def test_configured_base_url_decides_where_the_request_goes(base_url, expected):
    fake = FakeOllama().answers("ok")
    provider = OllamaProvider(
        make_settings(base_url=base_url), transport=fake.transport, clock=fake.clock
    )
    provider.complete(request())
    assert str(fake.requests[0].url) == expected


def test_remote_base_url_keeps_its_scheme_and_host_when_allowed():
    fake = FakeOllama().answers("ok")
    settings = make_settings(base_url="https://ollama.example", allow_remote=True)
    OllamaProvider(settings, transport=fake.transport, clock=fake.clock).complete(request())
    assert str(fake.requests[0].url) == "https://ollama.example/api/chat"


def test_timeouts_are_short_for_connecting_and_configurable_for_reading():
    fake = FakeOllama().answers("ok")
    provider_on(fake, timeout_s=30).complete(request())
    assert fake.requests[0].extensions["timeout"] == {
        "connect": 5.0,
        "read": 30.0,
        "write": 5.0,
        "pool": 5.0,
    }


def test_every_timeout_is_capped_by_a_short_configured_one():
    fake = FakeOllama().answers("ok")
    provider_on(fake, timeout_s=2).complete(request())
    assert fake.requests[0].extensions["timeout"] == {
        "connect": 2.0,
        "read": 2.0,
        "write": 2.0,
        "pool": 2.0,
    }


@pytest.mark.security
def test_environment_proxies_are_ignored_even_for_loopback(monkeypatch):
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(variable, "http://proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "")
    fake = FakeOllama().answers("ok")
    built = fake.record_default_transports(monkeypatch)
    reply = OllamaProvider(make_settings(), clock=fake.clock).complete(request())
    assert reply.text == "ok"
    assert built
    assert all(kwargs.get("proxy") is None for kwargs in built)


def test_redirects_are_not_followed():
    fake = FakeOllama().body(b"", {"location": "http://127.0.0.1:11434/elsewhere"}, status=302)
    reply = provider_on(fake).complete(request())
    assert reply.error == "HTTP 302"
    assert len(fake.requests) == 1


def test_reply_copies_text_model_stop_reason_usage_and_latency():
    fake = FakeOllama().answers(
        '{"verdict": "pass"}',
        model="fake-local-1:served",
        done=True,
        done_reason="stop",
        prompt_eval_count=11,
        eval_count=7,
    )
    fake.takes(0.25)
    reply = provider_on(fake).complete(request())
    assert reply == LlmReply(
        text='{"verdict": "pass"}',
        model="fake-local-1:served",
        stop_reason="stop",
        usage=LlmUsage(11, 7),
        latency_ms=250.0,
        error=None,
        request_id=None,
    )


@pytest.mark.parametrize("content", ["", " ", "plain ñ text\n", '{"a": 1}'])
def test_text_is_the_content_as_given(content):
    reply = provider_on(FakeOllama().answers(content)).complete(request())
    assert reply.text == content
    assert reply.error is None


def test_a_bare_reply_has_no_model_stop_reason_or_usage():
    reply = provider_on(FakeOllama().answers("ok")).complete(request())
    assert (reply.model, reply.stop_reason, reply.usage) == (None, None, LlmUsage())


ODD_VALUES = {
    "none": None,
    "true": True,
    "false": False,
    "float": 1.5,
    "string": "5",
    "list": [1],
    "object": {"a": 1},
}


@pytest.mark.parametrize("value", ODD_VALUES.values(), ids=ODD_VALUES.keys())
def test_odd_model_and_done_reason_become_none(value):
    fake = FakeOllama().answers("ok", model=value, done_reason=value)
    reply = provider_on(fake).complete(request())
    if isinstance(value, str):
        assert (reply.model, reply.stop_reason) == (value, value)
    else:
        assert (reply.model, reply.stop_reason) == (None, None)
    assert reply.error is None


@pytest.mark.parametrize("value", [0, 7, -3], ids=["zero", "positive", "negative"])
def test_int_counts_are_kept_even_when_odd(value):
    fake = FakeOllama().answers("ok", prompt_eval_count=value, eval_count=value)
    assert provider_on(fake).complete(request()).usage == LlmUsage(value, value)


@pytest.mark.parametrize("value", ODD_VALUES.values(), ids=ODD_VALUES.keys())
def test_non_int_counts_become_none(value):
    fake = FakeOllama().answers("ok", prompt_eval_count=value, eval_count=value)
    reply = provider_on(fake).complete(request())
    assert reply.usage == LlmUsage()
    assert reply.error is None


def test_each_count_is_judged_on_its_own():
    fake = FakeOllama().answers("ok", prompt_eval_count=12, eval_count=True)
    assert provider_on(fake).complete(request()).usage == LlmUsage(12, None)


def special_payload(name: str) -> bytes:
    return {
        "huge-integer": b'{"message": {"content": "x"}, "eval_count": ' + b"1" * 5000 + b"}",
        "too-deep": b"[" * 100_000 + b"]" * 100_000,
        "invalid-utf8": b'{"message": {"content": "\xff\xfe'
        + samples.OLLAMA_CANARY.encode()
        + b'"}}',
        "gzip-bomb": zlib.compress(b"0" * 5_000_000),
    }[name]


def payload_bytes(row: dict[str, Any]) -> bytes:
    payload: str = row["payload"]
    if payload.startswith("special:"):
        return special_payload(payload.removeprefix("special:"))
    return payload.replace("{canary}", samples.OLLAMA_CANARY).encode()


def headers_of(row: dict[str, Any]) -> dict[str, str]:
    return {
        key: value.replace("{canary}", samples.OLLAMA_CANARY)
        for key, value in row.get("headers", {}).items()
    }


def exception_for(name: str) -> Exception:
    message = f"detail {samples.OLLAMA_CANARY}"
    if name.startswith("httpx."):
        return getattr(httpx, name.removeprefix("httpx."))(message)
    return BUILTIN_ERRORS[name](message)


def fake_for(row: dict[str, Any]) -> FakeOllama:
    fake = FakeOllama()
    kind = row["kind"]
    if kind == "raises":
        return fake.raises(exception_for(row["exception"]))
    if kind == "status":
        return fake.body(payload_bytes(row), headers_of(row), status=row["status"])
    if kind == "body":
        return fake.body(payload_bytes(row), headers_of(row))
    canary = samples.OLLAMA_CANARY.encode()
    if kind == "oversize":
        total = MAX_RESPONSE_BYTES + 1
        size = -(-total // row["parts"])
        data = canary + b"x" * (total - len(canary))
        return fake.chunks([data[i : i + size] for i in range(0, total, size)])
    fake.takes(row.get("seconds_per_request", 0))
    return fake.chunks(
        [b'{"message": {"content": "' + canary + b'"}}'] * row["parts"],
        seconds_per_chunk=row["seconds_per_chunk"],
    )


@pytest.mark.security
@pytest.mark.parametrize("row", [case.values[0] for case in ERRORS], ids=[c.id for c in ERRORS])
def test_failures_are_fixed_values_and_never_repeat_what_the_server_said(row):
    fake = fake_for(row)
    reply = provider_on(fake, timeout_s=10).complete(request())
    assert reply.error == row["expected"]
    assert (reply.text, reply.model, reply.stop_reason) == (None, None, None)
    assert reply.usage == LlmUsage()
    assert reply.request_id is None
    assert reply.latency_ms >= 0
    assert samples.OLLAMA_CANARY not in all_fields(reply)


def test_failures_still_measure_latency():
    fake = FakeOllama().takes(0.5).status(500, b"")
    assert provider_on(fake).complete(request()).latency_ms == 500.0


def test_a_body_of_exactly_the_cap_is_accepted():
    head = b'{"message": {"content": "'
    tail = b'"}}'
    data = head + b"x" * (MAX_RESPONSE_BYTES - len(head) - len(tail)) + tail
    assert len(data) == MAX_RESPONSE_BYTES
    reply = provider_on(FakeOllama().body(data)).complete(request())
    assert reply.error is None
    assert reply.text is not None
    assert len(reply.text) == MAX_RESPONSE_BYTES - len(head) - len(tail)


def test_the_deadline_is_a_total_not_per_chunk():
    chunks = [b'{"message": ', b'{"content": ', b'"ok"}}']
    fake = FakeOllama().chunks(chunks, seconds_per_chunk=2)
    reply = provider_on(fake, timeout_s=10).complete(request())
    assert reply.text == "ok"
    slow = FakeOllama().chunks(chunks, seconds_per_chunk=5)
    assert provider_on(slow, timeout_s=10).complete(request()).error == "deadline exceeded"


@pytest.mark.parametrize("header", ["identity", "", "Identity, identity"])
def test_identity_content_encoding_is_accepted(header):
    fake = FakeOllama().body(b'{"message": {"content": "ok"}}', {"content-encoding": header})
    assert provider_on(fake).complete(request()).text == "ok"


def test_an_encoded_body_is_rejected_without_decoding_it(monkeypatch):
    bomb = zlib.compress(b"0" * 5_000_000)
    decoded: list[int] = []
    original = httpx.Response.iter_bytes

    def spy(self: httpx.Response, *args: Any, **kwargs: Any) -> Any:
        decoded.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Response, "iter_bytes", spy)
    fake = FakeOllama().body(bomb, {"content-encoding": "deflate"})
    assert provider_on(fake).complete(request()).error == "unsupported content-encoding"
    assert decoded == []


def test_keyboard_interrupt_is_not_swallowed():
    fake = FakeOllama().raises(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        provider_on(fake).complete(request())


@pytest.mark.security
@pytest.mark.parametrize("row", [case.values[0] for case in URLS], ids=[c.id for c in URLS])
def test_base_url_is_validated_when_the_provider_is_built(row):
    settings = settings_with_url(row["base_url"], row["allow_remote"])
    if row["accepted"]:
        provider_for(settings, transport=FakeOllama().transport).close()
        return
    with pytest.raises(ConfigError) as caught:
        provider_for(settings, transport=FakeOllama().transport)
    message = str(caught.value)
    assert message in URL_MESSAGES
    assert row["base_url"].strip() == "" or row["base_url"] not in message
    assert caught.value.__suppress_context__ or caught.value.__cause__ is None


@pytest.mark.security
def test_remote_host_is_rejected_with_the_loopback_message():
    settings = settings_with_url("http://ollama.example:11434", False)
    with pytest.raises(ConfigError) as caught:
        OllamaProvider(settings)
    assert str(caught.value) == LOOPBACK_MESSAGE


@pytest.mark.security
def test_unparsable_url_error_hides_the_parser_text():
    with pytest.raises(ConfigError) as caught:
        OllamaProvider(settings_with_url("http://127.0.0.1:abc", False))
    assert caught.value.__suppress_context__
    assert "abc" not in str(caught.value)


@pytest.mark.security
def test_repr_shows_neither_the_url_nor_the_prompt():
    fake = FakeOllama().answers("ok")
    settings = make_settings(base_url="https://ollama.example:4321", allow_remote=True)
    provider = OllamaProvider(settings, transport=fake.transport)
    provider.complete(request(system="SYSTEM-" + samples.OLLAMA_CANARY))
    shown = repr(provider)
    assert "ollama.example" not in shown
    assert samples.OLLAMA_CANARY not in shown
    assert "fake-local-1" in shown


def test_close_closes_the_client():
    fake = FakeOllama().answers("ok")
    provider = provider_on(fake)
    provider.close()
    reply = provider.complete(request())
    assert reply.error == "provider failure"
    assert fake.requests == []


def test_provider_for_builds_an_ollama_provider_and_forwards_the_seams():
    fake = FakeOllama().takes(1).answers("hello")
    built = provider_for(make_settings(), transport=fake.transport, clock=fake.clock)
    assert isinstance(built, OllamaProvider)
    reply = built.complete(request())
    assert (reply.text, reply.latency_ms) == ("hello", 1000.0)
    built.close()


def test_provider_for_does_not_hide_an_import_error_from_the_constructor(monkeypatch):
    def boom(*_: Any, **__: Any) -> Any:
        raise ImportError("from the client")

    monkeypatch.setattr(httpx, "Client", boom)
    with pytest.raises(ImportError, match="from the client"):
        provider_for(make_settings())


@pytest.mark.parametrize("after", [None, b'{"message": '], ids=["headers", "mid-body"])
def test_the_alarm_ends_a_blocked_call_with_deadline_exceeded(after):
    fake = FakeOllama().stalls(after)
    reply = provider_on(fake, timeout_s=7).complete(request())
    assert reply.error == "deadline exceeded"
    assert reply.text is None
    assert fake.watchdog.armed == [7]
    assert fake.watchdog.cancelled == 1


def test_the_next_call_after_an_alarm_works():
    fake = FakeOllama().stalls()
    provider = provider_on(fake)
    assert provider.complete(request()).error == "deadline exceeded"
    fake.answers("ok")
    reply = provider.complete(request())
    assert (reply.error, reply.text) == (None, "ok")
    assert len(fake.requests) == 2


def test_a_normal_call_arms_and_cancels_the_alarm():
    fake = FakeOllama().answers("ok")
    assert provider_on(fake, timeout_s=9).complete(request()).error is None
    assert fake.watchdog.armed == [9]
    assert fake.watchdog.cancelled == 1


def test_a_late_alarm_of_an_earlier_call_never_breaks_the_next():
    fake = FakeOllama().answers("one")
    provider = provider_on(fake)
    provider.complete(request())
    fake.watchdog.fire()
    fake.answers("two")
    assert provider.complete(request()).text == "two"


@pytest.mark.security
def test_cookies_set_by_the_server_are_never_sent_back():
    fake = FakeOllama().body(b'{"message": {"content": "ok"}}', {"set-cookie": "track=abc; Path=/"})
    provider = provider_on(fake)
    provider.complete(request())
    provider.complete(request())
    assert [r.headers.get("cookie") for r in fake.requests] == [None, None]
