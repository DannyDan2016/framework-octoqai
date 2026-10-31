"""Per-environment data sets that must share the same fields and types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from octoqai.errors import ConfigError


def load_data_set(
    base_dir: Path, data: str, environments: list[str] | None, env: str | None
) -> dict[str, Any]:
    """Load the data set of the selected environment after checking every declared one.

    Error messages name environments, files and field paths, never values.

    Args:
        base_dir: Directory the ``data`` path is relative to.
        data: Path template; ``{env}`` is replaced by each environment name.
        environments: Declared environments, or ``None`` to use ``data`` as is.
        env: Selected environment; required when ``environments`` is declared.

    Returns:
        The raw values of the selected data set.

    Raises:
        ConfigError: If a file is missing, unreadable or the sets differ in shape.
    """
    if not environments or env is None:
        return _read(base_dir, data, "data set")
    sets = {
        name: _read(base_dir, data.replace("{env}", name), f"data set for environment '{name}'")
        for name in environments
    }
    _check_shapes(sets)
    return sets[env]


def _read(base_dir: Path, relative: str, label: str) -> dict[str, Any]:
    path = Path(base_dir) / relative
    if not path.is_file():
        raise ConfigError(f"{label} not found: {relative}")
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"cannot parse {label} {relative}{where}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{label} {relative} is not UTF-8") from None
    if not isinstance(values, dict):
        raise ConfigError(f"{label} {relative} must contain a mapping at the top level")
    return values


def _check_shapes(sets: dict[str, dict[str, Any]]) -> None:
    names = list(sets)
    reference = names[0]
    diffs: list[str] = []
    for name in names[1:]:
        diffs += _same_shape(sets[reference], sets[name], "data", ref_env=reference, env=name)
    if diffs:
        raise ConfigError("data sets differ between environments: " + "; ".join(diffs))


def _same_shape(reference: Any, other: Any, path: str, *, ref_env: str, env: str) -> list[str]:
    ref_kind, other_kind = _kind(reference), _kind(other)
    if ref_kind != other_kind:
        return [f"{path} is {ref_kind} in {ref_env} but {other_kind} in {env}"]
    if ref_kind != "mapping":
        return []
    diffs = [f"missing {path}.{key} in {env}" for key in reference if key not in other]
    diffs += [f"extra {path}.{key} in {env}" for key in other if key not in reference]
    for key in reference:
        if key in other:
            diffs += _same_shape(
                reference[key], other[key], f"{path}.{key}", ref_env=ref_env, env=env
            )
    return diffs


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    return type(value).__name__
