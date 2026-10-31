"""Load experiment YAML files, expanding environment variables."""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from octoqai.errors import ConfigError
from octoqai.experiment.datasets import load_data_set
from octoqai.experiment.envfiles import resolve_env
from octoqai.experiment.model import ENV_NAME, Case, Expectation, Experiment, Turn
from octoqai.experiment.template import render_text

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_QUOTED = re.compile(r"'([^']+)'")
_PATTERN_KINDS = ("contains", "not_contains")
_RESERVED = ("data_values", "environment", "selected_tags")


def load_experiment(
    path: Path, env: str | None = None, process_env: Mapping[str, str] | None = None
) -> Experiment:
    """Load and validate an experiment YAML file for one environment.

    ``${VAR}`` placeholders are resolved from the process environment, then ``.env.<env>``,
    then ``.env``, all read from the experiment's folder.

    Args:
        path: Location of the experiment file.
        env: Environment to run in; required when the experiment declares ``environments``.
        process_env: Process variables; defaults to ``os.environ``, which is never modified.

    Returns:
        The validated experiment, with ``data_values`` and ``environment`` filled in.

    Raises:
        ConfigError: If a file is missing, unreadable or invalid, or the environment is wrong.
    """
    path = Path(path)
    data = _read_experiment(path)
    _check_environment(data.get("environments"), env)
    variables = resolve_env(path.parent, env, os.environ if process_env is None else process_env)
    _expand_agent(data, variables)
    exp = _validate(data)
    values: dict[str, Any] = {}
    if exp.data is not None:
        values = _expand_values(
            load_data_set(path.parent, exp.data, exp.environments, env), variables
        )
    exp = _render_cases(exp, _template_context(values, env))
    return exp.model_copy(update={"data_values": values, "environment": env})


def _template_context(values: dict[str, Any], env: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {"data": values}
    if env is not None:
        context["env"] = env
    return context


def _render_cases(exp: Experiment, context: dict[str, Any]) -> Experiment:
    return exp.model_copy(update={"cases": [_render_case(case, context) for case in exp.cases]})


def _render_case(case: Case, context: dict[str, Any]) -> Case:
    turns: list[Turn] = []
    for index, turn in enumerate(case.turns):
        try:
            turns.append(_render_turn(turn, context))
        except ConfigError as exc:
            raise ConfigError(f"cases[{case.id}].turns[{index}]: {exc}") from None
    return case.model_copy(update={"turns": turns})


def _render_turn(turn: Turn, context: dict[str, Any]) -> Turn:
    expect = [_render_expectation(item, context) for item in turn.expect]
    return turn.model_copy(update={"user": render_text(turn.user, context), "expect": expect})


def _render_expectation(item: Expectation, context: dict[str, Any]) -> Expectation:
    if item.kind not in _PATTERN_KINDS:
        return item
    return item.model_copy(update={"value": render_text(item.value, context, escape=re.escape)})


def _read_experiment(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"experiment file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    reserved = [key for key in _RESERVED if key in data]
    if reserved:
        raise ConfigError(f"{', '.join(reserved)} is filled in by octoqai and cannot be set")
    return data


def _check_environment(declared: Any, env: str | None) -> None:
    if env is not None and not ENV_NAME.fullmatch(env):
        raise ConfigError("environment names may only use letters, digits, '-' and '_'")
    if not isinstance(declared, list):
        return
    names = ", ".join(str(name) for name in declared)
    if env is None:
        raise ConfigError(f"this experiment declares environments ({names}): choose one with --env")
    if env not in declared:
        raise ConfigError(f"unknown environment {env!r}; declared environments: {names}")


def _expand_agent(data: dict[str, Any], variables: Mapping[str, str]) -> None:
    agent = data.get("agent")
    if not isinstance(agent, dict):
        return
    if isinstance(agent.get("url"), str):
        agent["url"] = _expand_env(agent["url"], variables)
    if isinstance(agent.get("headers"), dict):
        agent["headers"] = {
            key: _expand_env(str(value), variables) for key, value in agent["headers"].items()
        }


def _expand_values(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_values(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_values(item, variables) for item in value]
    if isinstance(value, str):
        return _expand_env(value, variables)
    return value


def _expand_env(text: str, variables: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ConfigError(f"environment variable {name} is not set")
        return variables[name]

    return _ENV_PATTERN.sub(replace, text)


def _validate(data: dict[str, Any]) -> Experiment:
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
    return [key for key in model.model_fields if key not in _RESERVED] if model else []


def _model_in(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None
