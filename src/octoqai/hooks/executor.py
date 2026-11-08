"""Hook executor: http, sql and argv commands, never a shell."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import Any, TypedDict, Unpack, get_args

import httpx

from octoqai.experiment import Hook, HooksConfig, HookStage
from octoqai.hooks._process import run_tree
from octoqai.hooks.port import HookEnvironment, HookResult

RunCommand = Callable[..., subprocess.CompletedProcess[Any]]


class HookKwargs(TypedDict, total=False):
    """Seams that tests replace: the http transport, the process runner and the clock."""

    transport: httpx.BaseTransport | None
    run_command: RunCommand
    clock: Callable[[], float]


class HookExecutor:
    """Runs the hooks of each stage in the declared order.

    A ``before_*`` stage stops at the first failure; an ``after_*`` stage runs every hook, as a
    best-effort cleanup.
    """

    def __init__(
        self, hooks: HooksConfig, environment: HookEnvironment, **seams: Unpack[HookKwargs]
    ) -> None:
        """Create an executor; the http client exists only when some hook needs it."""
        self._hooks = hooks
        self._environment = environment
        self._run_command: RunCommand = seams.get("run_command", run_tree)
        self._clock = seams.get("clock", time.perf_counter)
        self._client: httpx.Client | None = None
        if any(hook.http for stage in get_args(HookStage) for hook in hooks.stage(stage)):
            self._client = httpx.Client(transport=seams.get("transport"))

    def run(self, stage: HookStage) -> list[HookResult]:
        """Run the hooks of ``stage`` and report each outcome."""
        stop_at_failure = stage.startswith("before_")
        results: list[HookResult] = []
        for index, hook in enumerate(self._hooks.stage(stage)):
            results.append(self._execute(stage, index, hook))
            if stop_at_failure and not results[-1].ok:
                break
        return results

    def close(self) -> None:
        """Release the http client, if one was created."""
        if self._client is not None:
            self._client.close()

    def _execute(self, stage: HookStage, index: int, hook: Hook) -> HookResult:
        started = self._clock()
        error = self._dispatch(hook)
        elapsed_ms = (self._clock() - started) * 1000
        return HookResult(stage, index, hook.kind, error is None, elapsed_ms, error)

    def _dispatch(self, hook: Hook) -> str | None:
        if hook.http is not None:
            return self._http(hook)
        if hook.sql is not None:
            return self._sql(hook.sql)
        return self._command(hook)

    def _command(self, hook: Hook) -> str | None:
        argv = list(hook.command or [])
        timeout = hook.effective_timeout_s()
        env = self._environment.env
        try:
            completed = self._run_command(
                argv,
                cwd=self._environment.base_dir,
                env=dict(env) or None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout:g}s"
        except (OSError, ValueError) as exc:
            return f"cannot start the command: {type(exc).__name__}"
        if completed.returncode != 0:
            return f"exited with code {completed.returncode}"
        return None

    def _http(self, hook: Hook) -> str | None:
        spec = hook.http
        client = self._client
        assert spec is not None and client is not None
        timeout = hook.effective_timeout_s()
        try:
            response = client.request(
                spec.method, spec.url, headers=spec.headers, json=spec.body, timeout=timeout
            )
        except httpx.TimeoutException:
            return f"timed out after {timeout:g}s"
        except httpx.TransportError as exc:
            return f"cannot connect: {type(exc).__name__}"
        except (httpx.RequestError, httpx.InvalidURL, ValueError, TypeError) as exc:
            return f"cannot send: {type(exc).__name__}"
        if not response.is_success:
            return f"HTTP {response.status_code}"
        return None

    def _sql(self, statements: list[str]) -> str | None:
        if self._environment.db is None:
            return "no database configured"
        try:
            return self._environment.db.execute(statements)
        except Exception as exc:
            return type(exc).__name__
