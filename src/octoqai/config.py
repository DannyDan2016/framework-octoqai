"""Experiment configuration model and loader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ExpectationKind = Literal[
    "tool_called",
    "no_tool_called",
    "format",
    "contains",
    "not_contains",
    "max_tokens",
    "max_latency_ms",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised when an experiment file cannot be loaded or is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expectation(_Model):
    """A single check applied to an agent reply."""

    kind: ExpectationKind
    value: Any
    hard: bool = False

    @model_validator(mode="before")
    @classmethod
    def _from_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "kind" in data:
            return data
        if len(data) != 1:
            raise ValueError(f"expectation must have exactly one key, got {sorted(data)}")
        kind, raw = next(iter(data.items()))
        hard = False
        if isinstance(raw, dict):
            raw = dict(raw)
            hard = bool(raw.pop("hard", False))
            if "pattern" in raw:
                raw = raw.pop("pattern")
            elif "value" in raw:
                raw = raw.pop("value")
        return {"kind": kind, "value": raw, "hard": hard}

    @model_validator(mode="after")
    def _check_value(self) -> Expectation:
        if self.kind in ("tool_called", "no_tool_called"):
            if not isinstance(self.value, list) or not all(isinstance(v, str) for v in self.value):
                raise ValueError(f"{self.kind} expects a list of tool names")
        elif self.kind == "format":
            if self.value != "json":
                raise ValueError(f"format only supports 'json', got {self.value!r}")
        elif self.kind in ("contains", "not_contains"):
            if not isinstance(self.value, str):
                raise ValueError(f"{self.kind} expects a regex string")
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError(f"invalid regex {self.value!r} in {self.kind}: {exc}") from exc
        elif isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ValueError(f"{self.kind} expects a number")
        return self


class Turn(_Model):
    """One user message and the expectations on the agent reply."""

    user: str
    expect: list[Expectation] = Field(default_factory=list)


class Case(_Model):
    """A business scenario played as a multi-turn conversation."""

    id: str
    turns: list[Turn] = Field(min_length=1)


class ResponseMap(_Model):
    """JSONPath expressions that locate data in the agent response."""

    answer: str
    tool_calls: str | None = None
    tokens: str | None = None


class AgentConfig(_Model):
    """How to reach the agent under test over HTTP."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    request: Any
    response: ResponseMap
    timeout_s: float = Field(default=60, gt=0)


class Variant(_Model):
    """A configuration of the agent sent with every request."""

    name: str
    baseline: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class Thresholds(_Model):
    """Minimum values required for a go decision."""

    pass_rate: float = Field(ge=0, le=1)


class Experiment(_Model):
    """A complete experiment definition."""

    name: str
    agent: AgentConfig
    variants: list[Variant] = Field(min_length=1)
    trials: int = Field(default=5, ge=1)
    thresholds: Thresholds
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_variants(self) -> Experiment:
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        baselines = sum(v.baseline for v in self.variants)
        if baselines > 1:
            raise ValueError("only one variant can be the baseline")
        if len(self.variants) > 1 and baselines == 0:
            raise ValueError("experiments with several variants need exactly one baseline")
        case_ids = [c.id for c in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case ids must be unique")
        for variant in self.variants:
            for path in _variant_paths(self.agent.request):
                if not _has_path(variant.params, path):
                    raise ValueError(
                        f"variant {variant.name!r} does not define params.{path} "
                        f"used by the request template"
                    )
        return self

    @property
    def baseline(self) -> Variant:
        """The baseline variant, or the only variant."""
        return next((v for v in self.variants if v.baseline), self.variants[0])


_VARIANT_REF = re.compile(r"\{\{\s*variant\.([\w.]+)\s*\}\}")


def _variant_paths(template: Any) -> set[str]:
    if isinstance(template, dict):
        return set().union(*(_variant_paths(v) for v in template.values()))
    if isinstance(template, list):
        return set().union(*(_variant_paths(v) for v in template))
    if isinstance(template, str):
        return set(_VARIANT_REF.findall(template))
    return set()


def _has_path(params: dict[str, Any], path: str) -> bool:
    value: Any = params
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"environment variable {name} is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, text)


def load_experiment(path: Path) -> Experiment:
    """Load and validate an experiment YAML file.

    Args:
        path: Location of the experiment file.

    Returns:
        The validated experiment.

    Raises:
        ConfigError: If the file is missing, unreadable or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"experiment file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    agent = data.get("agent")
    if isinstance(agent, dict):
        if isinstance(agent.get("url"), str):
            agent["url"] = _expand_env(agent["url"])
        if isinstance(agent.get("headers"), dict):
            agent["headers"] = {k: _expand_env(str(v)) for k, v in agent["headers"].items()}
    try:
        return Experiment.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        detail = error["msg"]
        if error["type"] == "literal_error":
            detail = f"{detail} (got {error['input']!r})"
        lines.append(f"{location}: {detail}")
    return "invalid experiment: " + "; ".join(lines)
