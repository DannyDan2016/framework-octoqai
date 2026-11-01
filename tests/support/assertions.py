"""Reusable assertions with messages that explain the failure."""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Sequence
from types import FrameType
from typing import Any

import pytest

from octoqai.agents import AgentReply
from octoqai.errors import ConfigError
from octoqai.experiment import SecurityConfig
from octoqai.security import Detector, scan_reply


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


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\(.)", re.DOTALL)


def _unescape_once(match: re.Match[str]) -> str:
    code, char = match.groups()
    return chr(int(code, 16)) if code else char


def _unescaped(text: str, rounds: int = 3) -> list[str]:
    forms = [text]
    for _ in range(rounds):
        forms.append(_ESCAPE.sub(_unescape_once, forms[-1]))
    return forms


def assert_canary_hidden(canary: str, *texts: str) -> None:
    """Check that ``canary`` is in no text: raw, behind JSON escapes, or behind backslash escapes.

    Texts are scanned as JSON at any depth, and also unescaped a few times by hand, so a value
    inside text that is no longer valid JSON is still caught.
    """
    security = SecurityConfig.model_validate({"detectors": [], "canaries": [{"value": canary}]})
    detector = Detector(security)
    for text in texts:
        for form in _unescaped(text):
            assert canary.casefold() not in form.casefold(), f"{canary!r} leaked into an output"
        reply = AgentReply(text, None, None, 1.0, None)
        assert scan_reply(detector, reply, []) == [], f"{canary!r} hidden in an output"


def line_events(run: Callable[[str], object], text: str, module_file: str) -> int:
    """Count the Python lines executed in ``module_file`` while ``run(text)`` runs.

    The count is deterministic: it does not depend on the speed or the load of the machine.
    """
    count = 0

    def trace(frame: FrameType, event: str, arg: Any) -> Callable[..., Any] | None:
        nonlocal count
        if frame.f_code.co_filename != module_file:
            return None
        if event == "line":
            count += 1
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        run(text)
    finally:
        sys.settrace(previous)
    return count


def assert_linear_steps(
    run: Callable[[str], object],
    make: Callable[[int], str],
    size: int,
    module_file: str,
) -> None:
    """Check that the lines run in ``module_file`` grow less than 5 times for a 4 times input.

    A quadratic loop grows 16 times; the check never reads a clock.
    """
    small = line_events(run, make(size), module_file)
    large = line_events(run, make(size * 4), module_file)
    ratio = large / max(small, 1)
    assert ratio < 5, f"steps grew {ratio:.1f}x ({small} to {large}) for a 4x larger input"
