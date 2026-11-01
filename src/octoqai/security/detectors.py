"""Find canaries, secrets, environment values and personal data in text."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from types import MappingProxyType
from typing import Literal

from octoqai.experiment import SecurityConfig
from octoqai.security import _patterns

FindingKind = Literal["canary", "secret", "env", "pii"]

MIN_ENV_VALUE_LENGTH = 4
PRIORITY: Mapping[FindingKind, int] = MappingProxyType(
    {"canary": 0, "secret": 1, "env": 2, "pii": 3}
)
_MASK_THRESHOLD = 12


@dataclass(frozen=True)
class Finding:
    """A sensitive value located in a text."""

    kind: FindingKind
    label: str
    value: str = field(repr=False)
    start: int
    end: int


def mask(value: str) -> str:
    """Shorten a sensitive value so it can be shown without revealing it.

    Args:
        value: The value to hide.

    Returns:
        The first and last three characters for long values, only the first one otherwise.
    """
    if len(value) >= _MASK_THRESHOLD:
        return value[:3] + "…" + value[-3:]
    return value[:1] + "…"


class Detector:
    """Locates leaks according to a security configuration."""

    def __init__(self, security: SecurityConfig, env_values: Sequence[str] = ()) -> None:
        """Create a detector.

        Args:
            security: Canaries, enabled detectors and allow patterns.
            env_values: Values taken from the environment that must not appear; only the
                report redactor passes them.
        """
        self._security = security
        self._env_values = [v for v in env_values if len(v) >= MIN_ENV_VALUE_LENGTH]
        self._allow = [re.compile(pattern) for pattern in security.allow]
        self._canary_patterns = [
            (canary.label, re.compile(re.escape(canary.value), re.IGNORECASE))
            for canary in security.canaries
        ]

    def find(self, text: str) -> list[Finding]:
        """Return the non-overlapping findings in ``text``, ordered by position.

        Args:
            text: The text to scan.

        Returns:
            Findings where canaries beat secrets, secrets beat environment values and these
            beat personal data when they overlap.
        """
        return _without_overlaps(self.candidates(text))

    def candidates(self, text: str) -> list[Finding]:
        """Return every finding in ``text``, overlapping ones included.

        Args:
            text: The text to scan.

        Returns:
            Canaries, secrets, environment values and personal data that survive the allow
            patterns, unpruned and in detection order; the redactor masks their union.
        """
        found = [
            *self._canaries(text),
            *self._secrets(text),
            *self._env(text),
            *self._pii(text),
        ]
        return [item for item in found if not self._allowed(item)]

    def _canaries(self, text: str) -> Iterator[Finding]:
        for label, pattern in self._canary_patterns:
            for match in pattern.finditer(text):
                yield _finding("canary", label, match.group(), match.start())

    def _secrets(self, text: str) -> Iterator[Finding]:
        if "secrets" not in self._security.detectors:
            return
        for label, pattern in _patterns.SECRET_PATTERNS.items():
            group = "value" if "value" in pattern.groupindex else 0
            for match in pattern.finditer(text):
                yield _finding("secret", label, match.group(group), match.start(group))

    def _env(self, text: str) -> Iterator[Finding]:
        for value in self._env_values:
            for match in re.finditer(re.escape(value), text):
                yield _finding("env", "env", match.group(), match.start())

    def _pii(self, text: str) -> Iterator[Finding]:
        if "pii" not in self._security.detectors:
            return
        for match in _patterns.EMAIL.finditer(text):
            yield _finding("pii", "email", match.group(), match.start())
        yield from _cards(text)
        yield from _ibans(text)

    def _allowed(self, finding: Finding) -> bool:
        if finding.kind in ("canary", "env"):
            return False
        return any(pattern.fullmatch(finding.value) for pattern in self._allow)


def _finding(kind: FindingKind, label: str, value: str, start: int) -> Finding:
    return Finding(kind, label, value, start, start + len(value))


def _cards(text: str) -> Iterator[Finding]:
    for match in _patterns.CARD_CANDIDATE.finditer(text):
        span = _patterns.card_span(match.group())
        if span is not None:
            yield _finding("pii", "card", match.group()[:span], match.start())


def _ibans(text: str) -> Iterator[Finding]:
    for match in _patterns.IBAN_CANDIDATE.finditer(text):
        span = _patterns.iban_span(match.group())
        if span is not None:
            yield _finding("pii", "iban", match.group()[:span], match.start())


_Ranked = tuple[int, Finding]


class _Spans:
    def __init__(self, items: Sequence[_Ranked] = ()) -> None:
        self.items = list(items)
        self._starts = [finding.start for _, finding in self.items]
        self._reach = list(accumulate((finding.end for _, finding in self.items), max))

    def blocks(self, finding: Finding) -> bool:
        before = bisect_left(self._starts, finding.end)
        return before > 0 and self._reach[before - 1] > finding.start

    def add(self, item: _Ranked) -> None:
        end = item[1].end
        self.items.append(item)
        self._starts.append(item[1].start)
        self._reach.append(max(self._reach[-1], end) if self._reach else end)


def _accept_free(group: list[_Ranked], accepted: _Spans) -> _Spans:
    fresh = _Spans()
    for item in group:
        if not accepted.blocks(item[1]) and not fresh.blocks(item[1]):
            fresh.add(item)
    return _Spans(sorted([*accepted.items, *fresh.items], key=lambda item: item[1].start))


def _without_overlaps(candidates: list[Finding]) -> list[Finding]:
    ranked = sorted(candidates, key=lambda c: (PRIORITY[c.kind], c.start, c.start - c.end))
    accepted = _Spans()
    for kind in sorted(PRIORITY, key=PRIORITY.__getitem__):
        group = [(rank, c) for rank, c in enumerate(ranked) if c.kind == kind]
        accepted = _accept_free(group, accepted)
    return [finding for _, finding in sorted(accepted.items, key=lambda i: (i[1].start, i[0]))]
