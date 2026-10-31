"""Load experiment YAML files, expanding environment variables."""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from octoqai.errors import ConfigError
from octoqai.experiment.model import Experiment

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_QUOTED = re.compile(r"'([^']+)'")


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
    return "invalid experiment: " + "; ".join(_describe(error) for error in exc.errors())


def _describe(error: ErrorDetails) -> str:
    location = ".".join(str(part) for part in error["loc"]) or "<root>"
    detail = error["msg"]
    if error["type"] == "literal_error":
        detail = f"{detail} (got {error['input']!r})"
        expected = str(error.get("ctx", {}).get("expected", ""))
        detail += _suggestion(error["input"], _QUOTED.findall(expected))
    elif error["type"] == "extra_forbidden":
        detail += _suggestion(error["loc"][-1], _allowed_keys(error["loc"][:-1]))
    return f"{location}: {detail}"


def _suggestion(word: Any, candidates: list[str]) -> str:
    matches = difflib.get_close_matches(str(word), candidates, n=1, cutoff=0.6)
    return f" (did you mean '{matches[0]}'?)" if matches else ""


def _allowed_keys(loc: tuple[int | str, ...]) -> list[str]:
    model: type[BaseModel] | None = Experiment
    for part in loc:
        if model is None:
            return []
        if isinstance(part, str):
            field = model.model_fields.get(part)
            model = _model_in(field.annotation) if field else None
    return list(model.model_fields) if model else []


def _model_in(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None
