"""Experiment configuration model and loader."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

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
    hard: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _from_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "kind" in data:
            return data
        if len(data) != 1:
            raise ValueError(f"expectation must have exactly one key, got {sorted(data)}")
        kind, raw = next(iter(data.items()))
        hard: Any = False
        if isinstance(raw, dict):
            unknown = sorted(set(raw) - {"pattern", "value", "hard"})
            if unknown:
                raise ValueError(f"unknown keys in {kind}: {unknown}")
            hard = raw.get("hard", False)
            raw = raw.get("pattern", raw.get("value"))
        return {"kind": kind, "value": raw, "hard": hard}

    @model_validator(mode="after")
    def _check_value(self) -> Expectation:
        _VALUE_CHECKS[self.kind](self.kind, self.value)
        return self


def _tool_names(kind: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{kind} expects a list of tool names")


def _json_format(kind: str, value: Any) -> None:
    if value != "json":
        raise ValueError(f"format only supports 'json', got {value!r}")


def _regex(kind: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{kind} expects a regex string")
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"invalid regex {value!r} in {kind}: {exc}") from exc


def _number(kind: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{kind} expects a number")


_VALUE_CHECKS: dict[str, Callable[[str, Any], None]] = {
    "tool_called": _tool_names,
    "no_tool_called": _tool_names,
    "format": _json_format,
    "contains": _regex,
    "not_contains": _regex,
    "max_tokens": _number,
    "max_latency_ms": _number,
}


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
        for check in _EXPERIMENT_CHECKS:
            check(self)
        return self

    @property
    def baseline(self) -> Variant:
        """The baseline variant, or the only variant."""
        return next((v for v in self.variants if v.baseline), self.variants[0])


def _unique_names(exp: Experiment) -> None:
    names = [v.name for v in exp.variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    case_ids = [c.id for c in exp.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")


def _one_baseline(exp: Experiment) -> None:
    baselines = sum(v.baseline for v in exp.variants)
    if baselines > 1:
        raise ValueError("only one variant can be the baseline")
    if len(exp.variants) > 1 and baselines == 0:
        raise ValueError("experiments with several variants need exactly one baseline")


def _cases_have_expectations(exp: Experiment) -> None:
    for case in exp.cases:
        if not any(turn.expect for turn in case.turns):
            raise ValueError(f"case {case.id!r} has no expectation: it would always pass")


def _json_serializable(exp: Experiment) -> None:
    labelled: list[tuple[str, Any]] = [("agent.request", exp.agent.request)]
    labelled += [(f"variant {v.name!r} params", v.params) for v in exp.variants]
    for label, value in labelled:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be JSON serializable: {exc}") from exc


def _variants_cover_template(exp: Experiment) -> None:
    paths = _variant_paths(exp.agent.request)
    for variant in exp.variants:
        for path in sorted(paths):
            if not _has_path(variant.params, path):
                raise ValueError(
                    f"variant {variant.name!r} does not define params.{path} "
                    f"used by the request template"
                )


_EXPERIMENT_CHECKS: tuple[Callable[[Experiment], None], ...] = (
    _unique_names,
    _one_baseline,
    _cases_have_expectations,
    _json_serializable,
    _variants_cover_template,
)

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
