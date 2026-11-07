"""Reusable assertions with messages that explain the failure."""

from __future__ import annotations

import gc
import json
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


_ATTEMPTS = 3


def _timed(run: Callable[[str], object], text: str) -> float:
    started = time.perf_counter()
    run(text)
    return time.perf_counter() - started


def _paused_collector[T](measure: Callable[[], T]) -> T:
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return measure()
    finally:
        if was_enabled:
            gc.enable()


def _best_without_collector(run: Callable[[str], object], text: str) -> float:
    return _paused_collector(lambda: min(_timed(run, text) for _ in range(3)))


def _growth(run: Callable[[str], object], small: str, large: str) -> float:
    def measure() -> float:
        fast, slow = [], []
        for _ in range(3):
            fast.append(_timed(run, small))
            slow.append(_timed(run, large))
        return min(slow) / max(min(fast), 1e-6)

    return _paused_collector(measure)


def assert_linear_growth(
    run: Callable[[str], object],
    make: Callable[[int], str],
    size: int,
    factor: int = 4,
    bound: float = 8.0,
) -> None:
    """Check that ``run`` on an input ``factor`` times larger costs less than ``bound`` times more.

    The garbage collector is collected and paused while timing, the two sizes are timed in turn
    and each keeps its best of three timings. The check is tried up to three times and fails
    only when every attempt exceeds ``bound``, so a pause of the machine is unlikely to fail it,
    while a quadratic cost grows ``factor ** 2`` times in every attempt.
    """
    small, large = make(size), make(size * factor)
    ratios = []
    for _ in range(_ATTEMPTS):
        ratio = _growth(run, small, large)
        if ratio < bound:
            return
        ratios.append(ratio)
    shown = ", ".join(f"{ratio:.1f}x" for ratio in ratios)
    raise AssertionError(f"cost grew {shown} in {_ATTEMPTS} attempts for a {factor}x larger input")


def assert_runs_within(run: Callable[[str], object], text: str, seconds: float) -> None:
    """Check that ``run(text)`` finishes in less than ``seconds`` in at least one of 3 attempts.

    The garbage collector is collected and paused while timing.
    """
    timings = []
    for _ in range(_ATTEMPTS):
        timings.append(_best_without_collector(run, text))
        if timings[-1] < seconds:
            return
    shown = ", ".join(f"{timing:.2f}s" for timing in timings)
    raise AssertionError(f"took {shown} in {_ATTEMPTS} attempts, the limit is {seconds}s")


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


def _strings(text: str) -> list[str]:
    try:
        stack: list[Any] = [json.loads(text)]
    except (ValueError, RecursionError):
        return [text]
    found = [text]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(part for pair in item.items() for part in pair)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            found.append(item)
    return found


def assert_value_hidden(value: str, *texts: str) -> None:
    """Check that ``value`` is in no text: raw, in any string of its JSON, or unescaped by hand.

    Each text and every string or key of it, when it is JSON, is also unescaped a few times,
    so a value behind JSON escapes in text that is not valid JSON is still caught.
    """
    for text in texts:
        for string in _strings(text):
            for form in _unescaped(string):
                assert value.casefold() not in form.casefold(), f"{value!r} leaked into an output"


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
    assert small > 0, f"no line of {module_file} ran: the module path is wrong"
    large = line_events(run, make(size * 4), module_file)
    ratio = large / small
    assert ratio < 5, f"steps grew {ratio:.1f}x ({small} to {large}) for a 4x larger input"
