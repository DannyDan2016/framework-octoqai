"""Reusable assertions with messages that explain the failure."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import pytest

from octoqai.errors import ConfigError


def assert_exit(code: int, expected: int) -> None:
    """Check a CLI exit code."""
    assert code == expected, f"exit code {code}, expected {expected}"


def assert_config_error(
    fn: Callable[[], object], *fragments: str, absent: Sequence[str] = ()
) -> str:
    """Check that ``fn`` raises ``ConfigError`` matching every regex fragment.

    Returns:
        The error message, for further checks.
    """
    with pytest.raises(ConfigError) as info:
        fn()
    message = str(info.value)
    for fragment in fragments:
        assert re.search(fragment, message), f"{fragment!r} not in {message!r}"
    for text in absent:
        assert text not in message, f"{text!r} must not appear in {message!r}"
    return message


def assert_not_in_outputs(value: str, *texts: str) -> None:
    """Check that a sensitive value does not leak into any output."""
    for text in texts:
        assert value not in text, f"{value!r} leaked into an output"
