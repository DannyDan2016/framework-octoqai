"""Pytest fixtures shared by every test package."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from octoqai.cli import main
from octoqai.db import SqlDatabase
from octoqai.hooks import HookKwargs
from support.agents import FakeAgent, FakeClock, FakeJsonAgent
from support.builders import ExperimentBuilder
from support.databases import BANK_SCHEMA, SqliteFile
from support.hooks import FakeCommands
from support.paths import TOY_SSE_AGENT


@dataclass
class CliRun:
    """Exit code and captured output of one CLI invocation."""

    code: int
    out: str
    err: str
    out_dir: Path

    def report(self) -> dict[str, Any]:
        """Load the JSON report written by the run."""
        data: dict[str, Any] = json.loads(
            (self.out_dir / "report.json").read_text(encoding="utf-8")
        )
        return data

    def report_text(self) -> str:
        """Return the raw JSON report."""
        return (self.out_dir / "report.json").read_text(encoding="utf-8")


@pytest.fixture
def experiment() -> ExperimentBuilder:
    """A builder for a minimal valid experiment."""
    return ExperimentBuilder()


@pytest.fixture
def fake_agent() -> Iterator[FakeJsonAgent]:
    """A simulated HTTP JSON agent; its adapters are closed after the test."""
    agent = FakeJsonAgent()
    yield agent
    agent.close()


@pytest.fixture
def sqlite_db(tmp_path: Path) -> SqliteFile:
    """A SQLite file with the bank schema."""
    database = SqliteFile(tmp_path / "app.db")
    database.run(*BANK_SCHEMA)
    return database


@pytest.fixture
def sql_database(sqlite_db: SqliteFile) -> Iterator[SqlDatabase]:
    """An adapter on the test database; it is closed after the test."""
    database = SqlDatabase(sqlite_db.url)
    yield database
    database.close()


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Directory where the CLI writes its reports."""
    return tmp_path / "out"


@pytest.fixture
def run_cli(out_dir: Path, capsys: pytest.CaptureFixture[str]) -> Callable[..., CliRun]:
    """Run the CLI in process; ``run`` commands write their reports to ``out_dir``."""

    def _run(
        *args: str,
        agent: FakeAgent | None = None,
        commands: FakeCommands | None = None,
        hook_agent: FakeJsonAgent | None = None,
    ) -> CliRun:
        argv = list(args)
        hook_kwargs = HookKwargs()
        if commands is not None:
            hook_kwargs["run_command"] = commands
        if hook_agent is not None:
            hook_kwargs["transport"] = hook_agent.transport()
        if hook_kwargs:
            hook_kwargs["clock"] = FakeClock()
        if argv[:1] == ["run"]:
            argv += ["--out", str(out_dir)]
        code = main(
            argv,
            transport=agent.transport() if agent else None,
            adapter_kwargs=agent.adapter_kwargs() if agent else None,
            hook_kwargs=hook_kwargs,
        )
        captured = capsys.readouterr()
        return CliRun(code, captured.out, captured.err, out_dir)

    return _run


@pytest.fixture
def datadir(request: pytest.FixtureRequest) -> Path:
    """The ``data`` folder next to the requesting test module."""
    return Path(request.path).parent / "data"


def _first_line(process: subprocess.Popen[str]) -> str:
    """Read the first stdout line of ``process`` in a thread, bounded by a queue timeout."""
    stdout = process.stdout
    assert stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: lines.put(stdout.readline()), daemon=True).start()
    try:
        return lines.get(timeout=10).strip()
    except queue.Empty:
        return ""


def _stop(process: subprocess.Popen[str]) -> str:
    """Terminate ``process``, killing it if it ignores the request, and return its stderr."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    assert process.stderr is not None
    return process.stderr.read()


def _close_pipes(process: subprocess.Popen[str]) -> None:
    """Close the stdout and stderr pipes of ``process``, whether or not it still runs."""
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _launch(*args: str) -> Iterator[str]:
    """Run the toy SSE agent with ``args`` and yield its URL; fail fast without a port."""
    process = subprocess.Popen(
        [sys.executable, str(TOY_SSE_AGENT), "--port", "0", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _first_line(process)
        match = re.fullmatch(r"PORT (\d+)", line)
        if match is None:
            stderr = _stop(process)
            pytest.fail(f"toy SSE agent did not announce its port: {line!r}; stderr: {stderr}")
        yield f"http://127.0.0.1:{match.group(1)}/chat/stream"
    finally:
        if process.poll() is None:
            _stop(process)
        _close_pipes(process)


@pytest.fixture
def toy_sse_agent_url() -> Iterator[str]:
    """URL of the toy SSE agent on 127.0.0.1; fails fast if it never announces its port."""
    yield from _launch()


@pytest.fixture
def toy_db_agent(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """URL of the toy SSE agent writing to ``<tmp_path>/app.db``, and the database path."""
    db_path = tmp_path / "app.db"
    for url in _launch("--db", str(db_path)):
        yield url, db_path
