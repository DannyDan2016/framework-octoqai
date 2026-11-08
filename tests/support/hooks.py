"""Simulated process runner and lifecycle port."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from octoqai.experiment import HookStage
from octoqai.hooks import HookResult
from support.agents import FakeClock


class FakeCommands:
    """Process runner that records its calls and never starts a process."""

    def __init__(self, clock: FakeClock | None = None, took_s: float = 0.0) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._clock = clock
        self._took_s = took_s
        self._codes: list[int] = [0]
        self._failure: Exception | None = None

    def exits(self, *codes: int) -> Self:
        """Exit with each code in turn, one per call; the last one repeats."""
        self._codes = list(codes)
        return self

    def times_out(self) -> Self:
        """Raise a timeout on every call."""
        self._failure = subprocess.TimeoutExpired("fake", 0)
        return self

    def missing(self) -> Self:
        """Fail every call as if the program did not exist."""
        self._failure = FileNotFoundError("fake")
        return self

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        """Record the call, advance the clock and answer as configured."""
        self.calls.append((list(args), kwargs))
        if self._clock is not None:
            self._clock.advance(self._took_s)
        if self._failure is not None:
            raise self._failure
        code = self._codes[min(len(self.calls), len(self._codes)) - 1]
        return subprocess.CompletedProcess(args, code)


class RecordingHooks:
    """Lifecycle port that records the stages it runs and fails those it is told to."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.closed = False
        self._failures: dict[HookStage, tuple[int, str]] = {}

    def fails(
        self, stage: HookStage, times: int = 10**6, error: str = "exited with code 1"
    ) -> Self:
        """Fail the next ``times`` runs of ``stage`` with ``error``."""
        self._failures[stage] = (times, error)
        return self

    def run(self, stage: HookStage) -> list[HookResult]:
        """Record the stage and report one command hook, failed if configured."""
        self.events.append(stage)
        times, error = self._failures.get(stage, (0, ""))
        failing = times > 0
        if failing:
            self._failures[stage] = (times - 1, error)
        return [HookResult(stage, 0, "command", not failing, 0.0, error if failing else None)]

    def close(self) -> None:
        """Mark the port as closed."""
        self.closed = True


def wait_until(predicate: Callable[[], bool], limit_s: float = 10.0) -> bool:
    """Poll ``predicate`` until it holds or ``limit_s`` passes, without sleeping."""
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if predicate():
            return True
    return False


def process_alive(pid: int) -> bool:
    """Whether a process exists and is not a zombie waiting to be reaped."""
    if sys.platform == "win32":
        tasklist = os.path.join(os.environ["SYSTEMROOT"], "System32", "tasklist.exe")
        listing = subprocess.run(
            [tasklist, "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return f'"{pid}"' in listing
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().rpartition(")")[2].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def process_tree_code(pid_file: Path) -> str:
    """Python code for a child that starts a grandchild and both block forever.

    The grandchild writes its pid to ``pid_file`` once it runs.
    """
    grandchild = (
        "import os, pathlib, threading; "
        f"pathlib.Path({pid_file.as_posix()!r}).write_text(str(os.getpid())); "
        "threading.Event().wait()"
    )
    return (
        "import subprocess, sys, threading; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "threading.Event().wait()"
    )
