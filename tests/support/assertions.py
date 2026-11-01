"""Reusable assertions with messages that explain the failure."""

from __future__ import annotations

import re
import time
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


def assert_linear_growth(
    run: Callable[[str], object],
    make: Callable[[int], str],
    size: int,
    factor: int = 4,
    bound: float = 8.0,
) -> None:
    """Check that ``run`` on an input ``factor`` times larger costs less than ``bound`` times more.

    Each size keeps its best of three timings, so a pause of the machine cannot fail the check,
    while a quadratic cost grows ``factor ** 2`` times and does.
    """
    small, large = make(size), make(size * factor)

    def best(text: str) -> float:
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            run(text)
            timings.append(time.perf_counter() - started)
        return min(timings)

    ratio = best(large) / max(best(small), 1e-6)
    assert ratio < bound, f"cost grew {ratio:.1f}x for a {factor}x larger input"
