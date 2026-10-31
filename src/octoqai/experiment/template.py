"""Request templating that injects variant parameters into each request."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from octoqai.errors import ConfigError

_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")
_MISSING = object()


def _lookup(expr: str, context: dict[str, Any]) -> Any:
    value: Any = context
    for part in expr.split("."):
        value = value.get(part, _MISSING) if isinstance(value, dict) else _MISSING
        if value is _MISSING:
            raise ConfigError(f"unknown template variable: {expr}")
    return value


def render_text(
    text: str, context: dict[str, Any], escape: Callable[[str], str] | None = None
) -> str:
    """Interpolate every ``{{ path }}`` placeholder of a text as a string.

    Args:
        text: Text containing placeholders, e.g. a user message or a regex pattern.
        context: Values available to the placeholders.
        escape: Optional function applied to each inserted value, e.g. ``re.escape``.

    Returns:
        The text with every placeholder replaced.

    Raises:
        ConfigError: If a placeholder references an unknown variable.
    """

    def replace(match: re.Match[str]) -> str:
        value = str(_lookup(match.group(1), context))
        return escape(value) if escape else value

    return _PLACEHOLDER.sub(replace, text)


def render(template: Any, context: dict[str, Any]) -> Any:
    """Render a request template against a context.

    A string that is exactly one placeholder is replaced by the raw value, keeping its type.
    Placeholders embedded in longer strings are interpolated as text.

    Args:
        template: Any JSON-like structure containing ``{{ path }}`` placeholders.
        context: Values available to the placeholders.

    Returns:
        A new structure with every placeholder resolved.

    Raises:
        ConfigError: If a placeholder references an unknown variable.
    """
    if isinstance(template, dict):
        return {key: render(value, context) for key, value in template.items()}
    if isinstance(template, list):
        return [render(item, context) for item in template]
    if not isinstance(template, str):
        return template
    whole = _PLACEHOLDER.fullmatch(template.strip())
    if whole:
        return _lookup(whole.group(1), context)
    return _PLACEHOLDER.sub(lambda m: str(_lookup(m.group(1), context)), template)
