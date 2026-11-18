"""Builds the language model adapter that matches the provider settings."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from octoqai.errors import ConfigError
from octoqai.experiment import ProviderSettings
from octoqai.providers.port import LlmPort

ENDPOINT_VARIABLES = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS")


def _anthropic(settings: ProviderSettings, **seams: Any) -> LlmPort:
    for variable in ENDPOINT_VARIABLES:
        if os.environ.get(variable):
            raise ConfigError(
                f"{variable} is set in the environment: unset it, "
                "the judge only talks to the default Anthropic endpoint"
            )
    try:
        from octoqai.providers.anthropic_api import AnthropicProvider

        return AnthropicProvider(settings, **seams)
    except ImportError:
        raise ConfigError(
            "the anthropic provider needs: pip install 'octoqai[anthropic]'"
        ) from None


def _ollama(settings: ProviderSettings, **seams: Any) -> LlmPort:
    raise ConfigError("the ollama provider is not available yet")


_BUILDERS: dict[str, Callable[..., LlmPort]] = {"anthropic": _anthropic, "ollama": _ollama}


def provider_for(settings: ProviderSettings, **seams: Any) -> LlmPort:
    """Return the adapter for ``settings.provider``; ``seams`` reach the adapter for tests."""
    builder = _BUILDERS.get(settings.provider)
    if builder is None:
        raise ConfigError("unknown provider")
    return builder(settings, **seams)
