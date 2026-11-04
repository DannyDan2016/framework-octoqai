"""Scan an agent reply for confidential data the user did not provide."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from octoqai.agents.port import AgentReply
from octoqai.security._json_text import decode_json, plain_forms, string_leaves
from octoqai.security.detectors import Detector, mask


@dataclass(frozen=True)
class Leak:
    """A confidential value found in a reply; only its masked form is kept."""

    kind: str
    label: str
    location: str
    masked: str


def _with_leaves(location: str, text: str) -> Iterator[tuple[str, str]]:
    for form in plain_forms(text):
        yield location, form
    decoded = decode_json(text)
    if decoded is not None:
        for leaf in string_leaves(decoded):
            for form in plain_forms(leaf):
                yield location, form


def _texts(reply: AgentReply) -> Iterator[tuple[str, str]]:
    for index, fragment in enumerate(reply.fragments, start=1):
        yield from _with_leaves(f"fragment {index}", fragment)
    if reply.answer is not None:
        yield from _with_leaves("answer", reply.answer)
    for body in (reply.body, reply.alt_body):
        if body is not None:
            yield from _with_leaves("body", body)
    if reply.raw is not None:
        for leaf in string_leaves(reply.raw):
            for form in plain_forms(leaf):
                yield "body", form


def scan_reply(detector: Detector, reply: AgentReply, user_messages: Sequence[str]) -> list[Leak]:
    """Find confidential values in a reply, ignoring those the user wrote.

    Args:
        detector: The detector built from the experiment security configuration.
        reply: The agent reply, scanned as fragments, then answer, then each decode of the
            body; JSON text is also decoded and every string in it scanned, and every text
            and string is scanned again with its JSON escapes undone, so escapes cannot hide a
            value even in text that is not valid JSON.
        user_messages: Every user message of the conversation up to this turn.

    Returns:
        One leak per distinct value, located at its first occurrence and masked.
    """
    said = [message.casefold() for message in user_messages]
    seen: set[tuple[str, str, str]] = set()
    leaks: list[Leak] = []
    for location, text in _texts(reply):
        for finding in detector.find(text):
            folded = finding.value.casefold()
            key = (finding.kind, finding.label, folded)
            if finding.kind == "env" or key in seen or any(folded in m for m in said):
                continue
            seen.add(key)
            leaks.append(Leak(finding.kind, finding.label, location, mask(finding.value)))
    return leaks
