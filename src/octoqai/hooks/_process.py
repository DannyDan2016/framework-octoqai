"""Process runner that never leaves the descendants of a stopped command running."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from typing import Any


def run_tree(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run a command like ``subprocess.run`` and stop its whole process tree on any failure.

    ``subprocess.run`` kills only the direct child on a timeout, so a script that started a
    helper process would leave it running. Here a timeout or an interrupt kills every
    descendant before the exception propagates.

    Args:
        args: The program and its arguments.
        **kwargs: ``subprocess.run`` keyword arguments; ``check`` is ignored.

    Returns:
        The finished process with its exit code.
    """
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("check", None)
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    with subprocess.Popen(args, **kwargs) as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            _kill_tree(process)
            process.wait()
            raise
    return subprocess.CompletedProcess(args, code)


def _kill_tree(process: subprocess.Popen[Any]) -> None:
    if sys.platform == "win32":
        taskkill = os.path.join(os.environ.get("SYSTEMROOT", ""), "System32", "taskkill.exe")
        if os.path.isabs(taskkill):
            subprocess.run(
                [taskkill, "/T", "/F", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.kill()
