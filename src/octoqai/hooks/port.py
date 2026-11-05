"""Port that the runner uses to run lifecycle hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from octoqai.db.port import DatabasePort
from octoqai.experiment import HookKind, HookStage


@dataclass(frozen=True)
class HookResult:
    """Outcome of one hook: whether it succeeded, how long it took and why it failed."""

    stage: HookStage
    index: int
    kind: HookKind
    ok: bool
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True)
class HookEnvironment:
    """What hooks need from the outside: the database, a working folder and environment."""

    db: DatabasePort | None = None
    base_dir: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)


class LifecyclePort(Protocol):
    """Runs the hooks of each lifecycle stage."""

    def run(self, stage: HookStage) -> list[HookResult]:
        """Run the hooks of ``stage`` and report each outcome."""
        ...

    def close(self) -> None:
        """Release the resources held by the implementation."""
        ...
