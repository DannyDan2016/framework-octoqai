"""Pytest fixtures shared by every test package."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from octoqai.cli import main
from support.agents import FakeJsonAgent
from support.builders import ExperimentBuilder


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
def fake_agent() -> FakeJsonAgent:
    """A simulated HTTP JSON agent."""
    return FakeJsonAgent()


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Directory where the CLI writes its reports."""
    return tmp_path / "out"


@pytest.fixture
def run_cli(out_dir: Path, capsys: pytest.CaptureFixture[str]) -> Callable[..., CliRun]:
    """Run the CLI in process; ``run`` commands write their reports to ``out_dir``."""

    def _run(*args: str, agent: FakeJsonAgent | None = None) -> CliRun:
        argv = list(args)
        if argv[:1] == ["run"]:
            argv += ["--out", str(out_dir)]
        code = main(argv, transport=agent.transport() if agent else None)
        captured = capsys.readouterr()
        return CliRun(code, captured.out, captured.err, out_dir)

    return _run


@pytest.fixture
def datadir(request: pytest.FixtureRequest) -> Path:
    """The ``data`` folder next to the requesting test module."""
    return Path(request.path).parent / "data"
