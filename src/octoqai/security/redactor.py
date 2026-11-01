"""Mask every sensitive value in a report with a placeholder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from octoqai.experiment import SecurityConfig
from octoqai.security._json_text import decode_json, string_leaves
from octoqai.security.detectors import PRIORITY, Detector, Finding

REDACTED = "[redacted]"


def _placeholder(finding: Finding) -> str:
    if finding.kind == "env":
        return REDACTED
    return f"[{finding.kind}:{finding.label}]"


def _merged(candidates: list[Finding]) -> list[list[Finding]]:
    groups: list[list[Finding]] = []
    end = -1
    for finding in sorted(candidates, key=lambda c: (c.start, c.start - c.end)):
        if groups and finding.start < end:
            groups[-1].append(finding)
            end = max(end, finding.end)
        else:
            groups.append([finding])
            end = finding.end
    return groups


class Redactor:
    """Replaces sensitive values in text and in report trees."""

    def __init__(self, detector: Detector) -> None:
        """Create a redactor.

        Args:
            detector: Finds the values to mask; it should run every detector.
        """
        self._detector = detector

    @classmethod
    def for_report(cls, security: SecurityConfig, env_values: Sequence[str]) -> Redactor:
        """Create the report redactor: every detector runs and ``allow`` is ignored.

        Args:
            security: Canaries of the experiment; its detectors and allow patterns only decide
                what counts as a leak, so they are ignored here.
            env_values: Values taken from the environment that must never be written.

        Returns:
            A redactor that masks canaries, secrets, environment values and personal data.
        """
        every = security.model_copy(update={"detectors": ["secrets", "pii"], "allow": []})
        return cls(Detector(every, env_values=env_values))

    def redact(self, text: str) -> str:
        """Replace the union of all findings in ``text`` with placeholders.

        Overlapping findings are merged into one span whose placeholder comes from the
        highest-priority kind: canary, secret, environment value, personal data. JSON text
        with a finding in any decoded string is decoded, redacted and written again without
        escapes, so a unicode escape or an escaped slash cannot hide a value.

        Args:
            text: The text to clean.

        Returns:
            The text with every sensitive span masked.
        """
        decoded = decode_json(text)
        if decoded is None or not self._hides(decoded):
            return self._mask(text)
        try:
            cleaned = self.redact_tree(decoded)
        except RecursionError:
            return REDACTED
        return self._mask(json.dumps(cleaned, ensure_ascii=False))

    def _hides(self, decoded: Any) -> bool:
        return any(self._detector.candidates(leaf) for leaf in string_leaves(decoded))

    def _mask(self, text: str) -> str:
        parts: list[str] = []
        done = 0
        for group in _merged(self._detector.candidates(text)):
            lead = min(group, key=lambda f: (PRIORITY[f.kind], f.start, f.start - f.end))
            start, end = group[0].start, max(f.end for f in group)
            parts += [text[done:start], _placeholder(lead)]
            done = end
        parts.append(text[done:])
        return "".join(parts)

    def redact_tree(self, obj: Any) -> Any:
        """Redact every string value in nested dicts and lists.

        Dictionary keys are kept as they are: no report key is controlled by the agent.

        Args:
            obj: The structure to clean.

        Returns:
            A copy with strings redacted and other values unchanged.
        """
        if isinstance(obj, dict):
            return {key: self.redact_tree(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.redact_tree(item) for item in obj]
        if isinstance(obj, str):
            return self.redact(obj)
        return obj
