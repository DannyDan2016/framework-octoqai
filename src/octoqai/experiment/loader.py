"""Load experiment YAML files, expanding environment variables."""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, get_args
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from octoqai.errors import ConfigError
from octoqai.experiment.datasets import load_data_set
from octoqai.experiment.envfiles import resolve_env
from octoqai.experiment.model import (
    ENV_NAME,
    Case,
    DbCheck,
    Expectation,
    Experiment,
    ForbiddenChange,
    Turn,
)
from octoqai.experiment.template import render, render_text

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_KEY = re.compile(r"pass|pwd|secret|token|key", re.IGNORECASE)
_QUOTED = re.compile(r"'([^']+)'")
_PATTERN_KINDS = ("contains", "not_contains")
_RESERVED = (
    "data_values",
    "environment",
    "selected_tags",
    "redacted_values",
    "base_dir",
    "hook_env",
)


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
    hook_env = resolve_env(path.parent, env, os.environ if process_env is None else process_env)
    variables = _Variables(hook_env)
    _expand_agent(data, variables)
    _expand_security(data, variables)
    _expand_database(data, variables)
    _expand_hooks(data, variables)
    exp = _validate(data)
    values: dict[str, Any] = {}
    if exp.data is not None:
        values = _expand_values(
            load_data_set(path.parent, exp.data, exp.environments, env), variables
        )
    exp = _render_cases(exp, _template_context(values, env), variables)
    return exp.model_copy(
        update={
            "data_values": values,
            "environment": env,
            "redacted_values": variables.used,
            "base_dir": path.parent,
            "hook_env": hook_env,
        }
    )


def _template_context(values: dict[str, Any], env: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {"data": values}
    if env is not None:
        context["env"] = env
    return context


def _render_cases(exp: Experiment, context: dict[str, Any], variables: _Variables) -> Experiment:
    update: dict[str, Any] = {
        "cases": [_render_case(case, context, variables) for case in exp.cases]
    }
    if exp.database is not None:
        forbidden = _render_params(exp.database.forbidden, context, variables)
        update["database"] = exp.database.model_copy(update={"forbidden": forbidden})
    return exp.model_copy(update=update)


def _render_case(case: Case, context: dict[str, Any], variables: _Variables) -> Case:
    turns: list[Turn] = []
    for index, turn in enumerate(case.turns):
        try:
            turns.append(_render_turn(turn, context, variables))
        except ConfigError as exc:
            raise ConfigError(f"cases[{case.id}].turns[{index}]: {exc}") from None
    return case.model_copy(
        update={
            "turns": turns,
            "db": _render_params(case.db, context, variables),
            "forbidden": _render_params(case.forbidden, context, variables),
        }
    )


def _render_params[C: (DbCheck, ForbiddenChange)](
    items: list[C], context: dict[str, Any], variables: _Variables
) -> list[C]:
    return [
        item.model_copy(update={"params": _expand_values(render(item.params, context), variables)})
        for item in items
    ]


def _render_turn(turn: Turn, context: dict[str, Any], variables: _Variables) -> Turn:
    expect = [_render_expectation(item, context) for item in turn.expect]
    return turn.model_copy(
        update={
            "user": render_text(turn.user, context),
            "expect": expect,
            "db": _render_params(turn.db, context, variables),
        }
    )


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


class _Variables:
    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = values
        self.used: list[str] = []

    def expand(self, text: str) -> str:
        return _ENV_PATTERN.sub(self._replace, text)

    def remember(self, values: Iterable[str]) -> None:
        for value in values:
            if value and value not in self.used:
                self.used.append(value)

    def _replace(self, match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in self._values:
            raise ConfigError(f"environment variable {name} is not set")
        value = self._values[name]
        if value not in self.used:
            self.used.append(value)
        return value


def _expand_agent(data: dict[str, Any], variables: _Variables) -> None:
    agent = data.get("agent")
    if not isinstance(agent, dict):
        return
    if isinstance(agent.get("url"), str):
        agent["url"] = variables.expand(agent["url"])
    if isinstance(agent.get("headers"), dict):
        agent["headers"] = {
            key: variables.expand(str(value)) for key, value in agent["headers"].items()
        }


def _expand_security(data: dict[str, Any], variables: _Variables) -> None:
    security = data.get("security")
    canaries = security.get("canaries") if isinstance(security, dict) else None
    if not isinstance(canaries, list):
        return
    for canary in canaries:
        if isinstance(canary, dict) and isinstance(canary.get("value"), str):
            canary["value"] = variables.expand(canary["value"])


def _expand_database(data: dict[str, Any], variables: _Variables) -> None:
    database = data.get("database")
    if not isinstance(database, dict):
        return
    url = database.get("url")
    if not isinstance(url, str) or not _ENV_PATTERN.fullmatch(url):
        raise ConfigError(
            "database.url must come from an environment variable, e.g. ${DB_URL}: "
            "never write credentials in the experiment"
        )
    database["url"] = variables.expand(url)
    variables.remember(_url_secrets(database["url"]))


def _forms(value: str) -> list[str]:
    return [value, unquote(value), quote(value, safe=""), quote_plus(value)]


def _url_secrets(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
        found = [parts.password or ""]
        found += [value for key, value in parse_qsl(parts.query) if _SENSITIVE_KEY.search(key)]
        found += [
            pair.partition("=")[2]
            for pair in parts.query.split("&")
            if _SENSITIVE_KEY.search(pair.partition("=")[0])
        ]
    except ValueError:
        return []
    return [form for value in found if value for form in _forms(value)]


def _expand_hooks(data: dict[str, Any], variables: _Variables) -> None:
    if "hooks" in data:
        data["hooks"] = _expand_values(data["hooks"], variables)


def _expand_values(value: Any, variables: _Variables) -> Any:
    if isinstance(value, dict):
        return {key: _expand_values(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_values(item, variables) for item in value]
    if isinstance(value, str):
        return variables.expand(value)
    return value


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
