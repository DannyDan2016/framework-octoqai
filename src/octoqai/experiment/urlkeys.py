"""Which keys of a connection string hold a secret."""

from __future__ import annotations

import re

__all__ = ["is_sensitive_key"]

_SECRET_KEY = re.compile(r"pass|pwd|pw|secret|token|key", re.IGNORECASE)
_CREDENTIAL_KEY = re.compile(r"credential|auth", re.IGNORECASE)
_METHOD_KEY = re.compile(
    r"auth(entication)?[_-]?(source|mechanism|plugin|method|type|database|db)|authentication",
    re.IGNORECASE,
)


def is_sensitive_key(name: str) -> bool:
    """Tell whether the value of a connection-string key may be a secret.

    Keys that only name an authentication method or source, such as ``authSource``, hold no
    secret and are not sensitive, so their value is not masked across a report.

    Args:
        name: Key of a URL query or of an ODBC connection string.

    Returns:
        ``True`` when the key looks like a password, token, key or credential.
    """
    if _SECRET_KEY.search(name):
        return True
    return bool(_CREDENTIAL_KEY.search(name)) and not _METHOD_KEY.fullmatch(name.strip())
