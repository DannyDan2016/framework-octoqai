"""A float that remembers the decimal text it was written with."""

from __future__ import annotations

import re
from typing import Any

import yaml

__all__ = ["SourceFloat"]

_DECIMAL = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
_FLOAT_TAG = "tag:yaml.org,2002:float"


class SourceFloat(float):
    """A float read from YAML that keeps its source text.

    It is a ``float`` in every other respect, so JSON, pydantic and database drivers treat it
    as one. The text lets a check compare the number the user wrote, not the nearest double.

    Attributes:
        source: The decimal text as written, or ``None`` when there is none.
    """

    __slots__ = ("source",)

    source: str | None

    def __new__(cls, value: float | str = 0.0, source: str | None = None) -> SourceFloat:
        """Create the float.

        Args:
            value: The number.
            source: The decimal text it was written with.

        Returns:
            The new number.
        """
        number = super().__new__(cls, value)
        number.source = source
        return number


def _construct(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> float:
    value = yaml.SafeLoader.construct_yaml_float(loader, node)
    if _DECIMAL.fullmatch(node.value):
        return SourceFloat(value, node.value)
    return value


class SourceLoader(yaml.SafeLoader):
    """A safe YAML loader whose decimal floats are ``SourceFloat``."""


SourceLoader.add_constructor(_FLOAT_TAG, _construct)


def load_source_yaml(text: str) -> Any:
    """Parse YAML with ``SafeLoader`` rules, keeping the source text of decimal floats.

    Args:
        text: The YAML document.

    Returns:
        The parsed document.
    """
    return yaml.load(text, Loader=SourceLoader)
