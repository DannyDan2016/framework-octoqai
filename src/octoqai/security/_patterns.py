"""Regular expressions and checksums behind the detectors."""

from __future__ import annotations

import re

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws access key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "llm api key": re.compile(r"\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."),
    "url with credentials": re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@\"']+:[^\s/@\"']+@"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "credential assignment": re.compile(r"(?i)\b(password|passwd|pwd|secret)\s*[:=]\s*\S{6,}"),
}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD_CANDIDATE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")

MIN_IBAN_LENGTH = 15
_IBAN_MODULUS = 97


def luhn_valid(digits: str) -> bool:
    """Check the Luhn checksum of a string of digits."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char) * (2 if position % 2 else 1)
        total += sum(divmod(digit, 10))
    return total % 10 == 0


def iban_valid(text: str) -> bool:
    """Check the mod-97 checksum of an IBAN written with or without spaces."""
    compact = text.replace(" ", "")
    if len(compact) < MIN_IBAN_LENGTH:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % _IBAN_MODULUS == 1


def iban_span(candidate: str) -> int | None:
    """Return the length of the longest valid IBAN prefix made of whole space-separated groups."""
    text = candidate
    while len(text.replace(" ", "")) >= MIN_IBAN_LENGTH:
        if iban_valid(text):
            return len(text)
        head, space, _ = text.rpartition(" ")
        if not space:
            return None
        text = head
    return None
