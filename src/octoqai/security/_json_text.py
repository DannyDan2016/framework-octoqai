"""Strings hidden inside JSON text, decoded so escapes cannot mask a value."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_JSON_START = ("{", "[", '"')


def decode_json(text: str) -> Any:
    if not text.lstrip().startswith(_JSON_START):
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def string_leaves(value: Any) -> Iterator[str]:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(reversed([part for pair in item.items() for part in pair]))
        elif isinstance(item, list):
            stack.extend(reversed(item))
        elif isinstance(item, str):
            yield item
            nested = decode_json(item)
            if nested is not None:
                stack.append(nested)


_ESCAPE = re.compile(
    r"\\(?:u(?P<high>[dD][89abAB][0-9a-fA-F]{2})\\u(?P<low>[dD][c-fC-F][0-9a-fA-F]{2})"
    r"|u(?P<code>[0-9a-fA-F]{4})|(?P<char>[\"\\/bfnrt]))"
)
_SIMPLE = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_SURROGATES = range(0xD800, 0xE000)


def _unescaped(match: re.Match[str]) -> str:
    high, low, code, char = match.group("high", "low", "code", "char")
    if high:
        return chr(0x10000 + ((int(high, 16) - 0xD800) << 10) + int(low, 16) - 0xDC00)
    if code:
        value = int(code, 16)
        return "\ufffd" if value in _SURROGATES else chr(value)
    return _SIMPLE[char]


def unescape(text: str) -> str:
    if "\\" not in text:
        return text
    return _ESCAPE.sub(_unescaped, text)


def plain_forms(text: str) -> Iterator[str]:
    yield text
    plain = unescape(text)
    if plain != text:
        yield plain
