import json

import pytest

from octoqai.providers.budget import BUDGET_EXHAUSTED, BudgetedLlm
from octoqai.providers.port import LlmMessage, LlmPort, LlmReply, LlmRequest, LlmUsage
from support.providers import EXHAUSTED_SCRIPT, FakeLlm, ScriptedUser

pytestmark = [pytest.mark.unit, pytest.mark.regression]

REQUEST = LlmRequest(
    role="judge", system="rate it", messages=(LlmMessage(role="user", content="hello"),)
)


def test_fake_llm_and_budget_satisfy_the_port():
    fake: LlmPort = FakeLlm()
    budgeted: LlmPort = BudgetedLlm(FakeLlm(), max_calls=1)
    assert fake.close() is None and budgeted.close() is None


def test_request_and_usage_are_frozen():
    with pytest.raises(AttributeError):
        REQUEST.system = "other"
    with pytest.raises(AttributeError):
        LlmUsage().input_tokens = 3


def test_fake_records_requests_and_answers_in_script_order():
    fake = FakeLlm().judges("pass", "ok").judges("fail", "bad")
    first, second = fake.complete(REQUEST), fake.complete(REQUEST)
    assert fake.requests == [REQUEST, REQUEST]
    assert json.loads(first.text or "")["verdict"] == "pass"
    assert json.loads(second.text or "") == {"verdict": "fail", "rationale": "bad"}
    assert (first.model, first.stop_reason, first.error) == ("fake-judge-1", "end_turn", None)


@pytest.mark.parametrize("extra_calls", [0, 1, 3])
def test_exhausted_script_is_a_fixed_error_never_a_text(extra_calls):
    fake = FakeLlm().judges("pass")
    fake.complete(REQUEST)
    replies = [fake.complete(REQUEST) for _ in range(extra_calls)]
    assert [(r.text, r.error) for r in replies] == [(None, EXHAUSTED_SCRIPT)] * extra_calls
    assert len(fake.requests) == extra_calls + 1


def test_empty_fake_has_no_default_reply():
    reply = FakeLlm().complete(REQUEST)
    assert (reply.text, reply.error) == (None, EXHAUSTED_SCRIPT)


@pytest.mark.parametrize(
    ("build", "text", "stop", "error"),
    [
        (lambda f: f.returns_text("raw"), "raw", "end_turn", None),
        (lambda f: f.returns_text(""), "", "end_turn", None),
        (lambda f: f.fails("APIConnectionError"), None, None, "APIConnectionError"),
        (lambda f: f.times_out(), None, None, "APITimeoutError"),
        (lambda f: f.refuses(), None, "refusal", None),
        (lambda f: f.truncates(), '{"verdict": "pa', "max_tokens", None),
    ],
    ids=["text", "empty-text", "fails", "times-out", "refuses", "truncates"],
)
def test_fake_reply_kinds(build, text, stop, error):
    fake = FakeLlm()
    build(fake)
    reply = fake.complete(REQUEST)
    assert (reply.text, reply.stop_reason, reply.error) == (text, stop, error)


def test_script_replaces_the_queue_with_texts_and_replies():
    custom = LlmReply("x", "m", "end_turn", LlmUsage(1, 2), 3.0)
    fake = FakeLlm().judges("pass").script("one", custom)
    assert fake.complete(REQUEST).text == "one"
    assert fake.complete(REQUEST) is custom
    assert fake.complete(REQUEST).error == EXHAUSTED_SCRIPT


def test_serves_model_and_usage_apply_to_replies():
    fake = FakeLlm().returns_text("a").serves_model("other-model").uses(10, 20)
    reply = fake.complete(REQUEST)
    assert (reply.model, reply.usage) == ("other-model", LlmUsage(10, 20))
    assert FakeLlm().model == "fake-judge-1"


def test_scripted_user_says_messages_then_is_done():
    user = ScriptedUser().says("hi", "more", done_after=2)
    turns = [json.loads(user.complete(REQUEST).text) for _ in range(3)]
    assert turns == [
        {"message": "hi", "done": False},
        {"message": "more", "done": False},
        {"message": "", "done": True},
    ]
    assert user.complete(REQUEST).error is None


def test_scripted_user_without_done_after_exhausts_with_an_error():
    user = ScriptedUser().says("hi")
    assert json.loads(user.complete(REQUEST).text) == {"message": "hi", "done": False}
    assert user.complete(REQUEST).error == EXHAUSTED_SCRIPT


def test_scripted_user_with_no_messages_is_exhausted():
    assert ScriptedUser().says().complete(REQUEST).error == EXHAUSTED_SCRIPT


def test_budget_blocks_the_call_after_max_calls():
    inner = FakeLlm().returns_text("a").returns_text("b").returns_text("c")
    budget = BudgetedLlm(inner, max_calls=2)
    replies = [budget.complete(REQUEST) for _ in range(3)]
    assert [r.text for r in replies] == ["a", "b", None]
    assert replies[2].error == BUDGET_EXHAUSTED == "call budget exhausted"
    assert len(inner.requests) == 2
    assert budget.calls == 2


def test_budget_zero_never_calls_the_inner_port():
    inner = FakeLlm().returns_text("a")
    budget = BudgetedLlm(inner, max_calls=0)
    reply = budget.complete(REQUEST)
    assert (reply.text, reply.error) == (None, BUDGET_EXHAUSTED)
    assert inner.requests == [] and budget.calls == 0
    assert dict(budget.errors) == {BUDGET_EXHAUSTED: 1}


def test_budget_rejects_a_negative_limit():
    with pytest.raises(ValueError, match="max_calls"):
        BudgetedLlm(FakeLlm(), max_calls=-1)


def test_budget_counts_errors_by_text():
    inner = FakeLlm().times_out().times_out().returns_text("ok")
    budget = BudgetedLlm(inner, max_calls=5)
    for _ in range(3):
        budget.complete(REQUEST)
    assert dict(budget.errors) == {"APITimeoutError": 2}


def test_budget_usage_total_skips_none():
    inner = FakeLlm().uses(5, 7).returns_text("a").returns_text("b")
    budget = BudgetedLlm(inner, max_calls=3)
    assert budget.usage_total == LlmUsage(None, None)
    budget.complete(REQUEST)
    inner.uses(None, 4)
    budget.complete(REQUEST)
    assert budget.usage_total == LlmUsage(5, 11)


def test_budget_usage_total_stays_unknown_when_nothing_reports():
    budget = BudgetedLlm(FakeLlm().returns_text("a").uses(None, None), max_calls=1)
    budget.complete(REQUEST)
    assert budget.usage_total == LlmUsage(None, None)


def test_budget_close_closes_the_inner_port():
    inner = FakeLlm()
    BudgetedLlm(inner, max_calls=1).close()
    assert inner.closed


def test_stats_summarise_calls_errors_tokens_and_served_models():
    fake = FakeLlm().uses(3, 4).judges("pass").judges("pass").fails("APITimeoutError")
    budget = BudgetedLlm(fake, max_calls=3)
    for _ in range(4):
        budget.complete(REQUEST)
    stats = budget.stats()
    assert stats.calls == 3
    assert stats.errors_by_class == {"APITimeoutError": 1, BUDGET_EXHAUSTED: 1}
    assert (stats.input_tokens, stats.output_tokens) == (6, 8)
    assert stats.served_models == {"fake-judge-1": 2}


def test_stats_are_a_frozen_copy():
    budget = BudgetedLlm(FakeLlm().judges("pass"), max_calls=1)
    budget.complete(REQUEST)
    stats = budget.stats()
    budget.complete(REQUEST)
    assert stats.errors_by_class == {}
    with pytest.raises(AttributeError):
        stats.calls = 9


def test_stats_of_an_unused_budget_are_empty():
    stats = BudgetedLlm(FakeLlm(), max_calls=1).stats()
    assert (stats.calls, stats.errors_by_class, stats.served_models) == (0, {}, {})
    assert (stats.input_tokens, stats.output_tokens) == (None, None)
