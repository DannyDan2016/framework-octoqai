"""Fluent builder for experiment definitions used across the test suite."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_URL = "http://agent.local/chat"


def turn(user: str, *expect: dict[str, Any]) -> dict[str, Any]:
    """Build one conversation turn with its expectations."""
    return {"user": user, "expect": list(expect)}


def write_env_file(directory: Path, name: str, **values: str) -> Path:
    """Write a dotenv file such as ``.env`` or ``.env.qa`` into ``directory``."""
    path = Path(directory) / name
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


def write_data_set(directory: Path, env: str, values: dict[str, Any]) -> Path:
    """Write ``data/<env>.yaml`` into ``directory``."""
    path = Path(directory) / "data" / f"{env}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    return path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class ExperimentBuilder:
    """Start from a minimal valid experiment and change only what a test cares about."""

    def __init__(self) -> None:
        """Create a builder with one variant, one case and a JSON agent."""
        self._agent: dict[str, Any] = {
            "url": DEFAULT_URL,
            "headers": {},
            "request": {"message": "{{ message }}"},
            "response": {"answer": "$.answer"},
        }
        self._variants: list[dict[str, Any]] = []
        self._cases: list[dict[str, Any]] = []
        self._trials: int | None = None
        self._threshold = 0.8
        self._overrides: dict[str, Any] = {}

    def agent(self, **settings: Any) -> ExperimentBuilder:
        """Replace agent settings such as ``url``, ``headers``, ``request`` or ``response``."""
        self._agent.update(settings)
        return self

    def variant(self, name: str, baseline: bool = False, **params: Any) -> ExperimentBuilder:
        """Add a variant; ``params`` are sent with every request."""
        self._variants.append({"name": name, "baseline": baseline, "params": params})
        return self

    def case(
        self, case_id: str, *turns: dict[str, Any], tags: Iterable[str] = ()
    ) -> ExperimentBuilder:
        """Add a case made of ``turn(...)`` items."""
        case: dict[str, Any] = {"id": case_id, "turns": list(turns)}
        if tags:
            case["tags"] = list(tags)
        self._cases.append(case)
        return self

    def trials(self, n: int) -> ExperimentBuilder:
        """Set how many times each case is played per variant."""
        self._trials = n
        return self

    def threshold(self, pass_rate: float) -> ExperimentBuilder:
        """Set the pass rate required for a go decision."""
        self._threshold = pass_rate
        return self

    def environments(self, *names: str, data: str | None = "data/{env}.yaml") -> ExperimentBuilder:
        """Declare the environments and, optionally, the per-environment data set path."""
        overrides: dict[str, Any] = {"environments": list(names)}
        if data is not None:
            overrides["data"] = data
        return self.merge(overrides)

    def merge(self, overrides: dict[str, Any]) -> ExperimentBuilder:
        """Deep-merge raw keys into the built document, e.g. rows from a data file."""
        self._overrides = _deep_merge(self._overrides, overrides)
        return self

    def build(self) -> dict[str, Any]:
        """Return the experiment as a plain dictionary."""
        data: dict[str, Any] = {
            "name": "demo",
            "agent": copy.deepcopy(self._agent),
            "variants": copy.deepcopy(self._variants)
            or [{"name": "base", "params": {"temperature": 0}}],
            "thresholds": {"pass_rate": self._threshold},
            "cases": copy.deepcopy(self._cases)
            or [{"id": "c1", "turns": [turn("Hola", {"contains": "."})]}],
        }
        if self._trials is not None:
            data["trials"] = self._trials
        return _deep_merge(data, self._overrides)

    def write(self, directory: Path) -> Path:
        """Write ``experiment.yaml`` into ``directory`` and return its path."""
        path = Path(directory) / "experiment.yaml"
        path.write_text(yaml.safe_dump(self.build(), allow_unicode=True), encoding="utf-8")
        return path
