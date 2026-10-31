"""Resolve variables from dotenv files and the process environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values


def resolve_env(base_dir: Path, env: str | None, process_env: Mapping[str, str]) -> dict[str, str]:
    """Merge ``.env``, ``.env.<env>`` and the process environment, lowest priority first.

    Missing files are ignored and ``os.environ`` is never modified.

    Args:
        base_dir: Directory that holds the dotenv files, usually the experiment's folder.
        env: Selected environment, or ``None`` to read only ``.env``.
        process_env: Variables of the running process, e.g. CI secrets.

    Returns:
        The resolved variables.
    """
    resolved: dict[str, str] = {}
    for name in _env_file_names(env):
        resolved.update(_read(Path(base_dir) / name))
    resolved.update(process_env)
    return resolved


def _env_file_names(env: str | None) -> list[str]:
    return [".env"] if env is None else [".env", f".env.{env}"]


def _read(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = dotenv_values(path, interpolate=False, encoding="utf-8")
    return {key: value for key, value in values.items() if value is not None}
