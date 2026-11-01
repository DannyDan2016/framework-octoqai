"""Strings hidden inside JSON text, decoded so escapes cannot mask a value."""

from __future__ import annotations

import json
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
