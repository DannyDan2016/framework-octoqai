"""Load experiment YAML files, expanding environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from octoqai.errors import ConfigError
from octoqai.experiment.model import Experiment

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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
