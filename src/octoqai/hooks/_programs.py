"""Where a command hook's program lives: the experiment folder or an absolute PATH folder."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

_BATCH_SUFFIXES = (".bat", ".cmd")


class ProgramError(Exception):
    """A program that cannot start; the message never holds its name or the PATH."""


def resolve_program(program: str, base_dir: Path | None, env: Mapping[str, str]) -> str:
    """Return the path to start for a command hook's program.

    A bare name is looked up only in the absolute folders of PATH, in order, so neither the
    current folder nor a relative PATH entry can supply it. PATH comes from ``env``, or from
    the process environment when ``env`` is empty, as the child would receive it. A name with
    a folder is anchored to ``base_dir`` when there is one.

    Args:
        program: The first argument of the command.
        base_dir: The experiment folder, or ``None`` to leave relative paths alone.
        env: The environment the command will run with.

    Returns:
        The program path to pass to the process runner.

    Raises:
        ProgramError: The program is not found or must not be started.
    """
    if not os.path.split(program)[0] and not os.path.splitdrive(program)[0]:
        return _search_path(program, env or os.environ)
    if base_dir is None or Path(program).is_absolute():
        return program
    anchored = Path(base_dir).absolute() / program
    if not anchored.is_absolute():
        raise ProgramError("its path is relative to another drive")
    return str(anchored)


def _search_path(name: str, env: Mapping[str, str]) -> str:
    if not name.strip():
        raise ProgramError("program not found in PATH")
    candidate = _candidate(name)
    for folder in _path_value(env).split(os.pathsep):
        if not Path(folder).is_absolute():
            continue
        path = os.path.join(folder, candidate)
        if _runnable(path):
            return path
    raise ProgramError("program not found in PATH")


def _candidate(name: str) -> str:
    if sys.platform != "win32":
        return name
    if name.rstrip(". ").lower().endswith(_BATCH_SUFFIXES):
        raise ProgramError("only .exe programs are looked up in PATH")
    return name if name.lower().endswith(".exe") else name + ".exe"


def _path_value(env: Mapping[str, str]) -> str:
    if sys.platform != "win32":
        return env.get("PATH", "")
    values = [value for key, value in env.items() if key.upper() == "PATH"]
    return values[-1] if values else ""


def _runnable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    return sys.platform == "win32" or os.access(path, os.X_OK)
