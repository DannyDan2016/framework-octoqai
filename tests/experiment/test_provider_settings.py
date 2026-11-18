import pytest
from pydantic import ValidationError

from octoqai.experiment import JudgeConfig, ProviderSettings, SimulatedUserConfig
from support.samples import SPLIT_SECRET

pytestmark = [pytest.mark.unit, pytest.mark.regression]

KEY = "".join(SPLIT_SECRET)


def anthropic(**over):
    return {"provider": "anthropic", "model": "m", "api_key": KEY, "max_calls": 3, **over}


def ollama(**over):
    return {"provider": "ollama", "model": "m", "max_calls": 3, **over}


def without(data, name):
    return {k: v for k, v in data.items() if k != name}


def test_defaults():
    s = ProviderSettings.model_validate(anthropic())
    assert (s.effort, s.max_tokens, s.timeout_s, s.max_retries) == ("low", 4096, 60, 2)
    assert (s.base_url, s.allow_remote) == (None, False)


def test_ollama_accepts_base_url_and_remote_flag():
    data = ollama(base_url="http://localhost:11434", allow_remote=True)
    s = ProviderSettings.model_validate(data)
    assert (s.base_url, s.allow_remote, s.api_key) == ("http://localhost:11434", True, None)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (anthropic(api_key=None), "api_key is required"),
        (anthropic(api_key=""), "api_key is required"),
        (anthropic(api_key="   "), "api_key is required"),
        (without(anthropic(), "api_key"), "api_key is required"),
        (ollama(api_key=KEY), "api_key is not accepted"),
        (ollama(api_key=""), "api_key is not accepted"),
        (anthropic(base_url="http://localhost:1"), "base_url is only accepted"),
        (ollama(base_url=""), "base_url cannot be empty"),
        (ollama(base_url="  "), "base_url cannot be empty"),
        (anthropic(model=""), "model must not be empty"),
        (anthropic(model="  \t"), "model must not be empty"),
        (without(anthropic(), "model"), "model"),
        (without(anthropic(), "max_calls"), "max_calls"),
        (anthropic(max_calls=0), "max_calls"),
        (anthropic(provider="openai"), "provider"),
        (anthropic(effort="extreme"), "effort"),
        (anthropic(max_tokens=255), "max_tokens"),
        (anthropic(max_tokens=32001), "max_tokens"),
        (anthropic(timeout_s=0), "timeout_s"),
        (anthropic(timeout_s=601), "timeout_s"),
        (anthropic(max_retries=-1), "max_retries"),
        (anthropic(max_retries=6), "max_retries"),
        (ollama(allow_remote="yes"), "allow_remote"),
        (anthropic(extra=1), "extra"),
    ],
)
def test_invalid_settings_are_rejected_without_the_key(data, message):
    with pytest.raises(ValidationError, match=message) as info:
        ProviderSettings.model_validate(data)
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("max_tokens", 256, 32000),
        ("timeout_s", 0.5, 600),
        ("max_retries", 0, 5),
        ("max_calls", 1, 99),
    ],
)
def test_bounds_are_inclusive(field, low, high):
    for value in (low, high):
        assert getattr(ProviderSettings.model_validate(anthropic(**{field: value})), field) == value


@pytest.mark.security
@pytest.mark.parametrize("cls", [ProviderSettings, JudgeConfig, SimulatedUserConfig])
def test_key_never_appears_in_repr_or_str(cls):
    s = cls.model_validate(anthropic())
    assert KEY not in repr(s) and KEY not in str(s)
    assert s.api_key == KEY


def test_judge_config_input_cap():
    assert JudgeConfig.model_validate(anthropic()).max_input_chars == 32_000
    for ok in (1_000, 200_000):
        assert JudgeConfig.model_validate(anthropic(max_input_chars=ok)).max_input_chars == ok
    for bad in (0, 999, 200_001):
        with pytest.raises(ValidationError, match="max_input_chars"):
            JudgeConfig.model_validate(anthropic(max_input_chars=bad))


def test_simulated_user_has_no_input_cap():
    with pytest.raises(ValidationError, match="max_input_chars"):
        SimulatedUserConfig.model_validate(anthropic(max_input_chars=5000))
    assert SimulatedUserConfig.model_validate(ollama()).provider == "ollama"
