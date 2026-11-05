"""Experiment configuration model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from octoqai.errors import ConfigError

ExpectationKind = Literal[
    "tool_called",
    "no_tool_called",
    "format",
    "contains",
    "not_contains",
    "max_tokens",
    "max_latency_ms",
    "max_ttft_ms",
]


DetectorName = Literal["secrets", "pii"]
DbExpectKind = Literal["row_count", "value", "rows", "delta", "unchanged"]
HookStage = Literal["before_run", "before_trial", "after_trial", "after_run"]
HookKind = Literal["http", "sql", "command"]
DEFAULT_HOOK_TIMEOUT_S = 30.0
MIN_CANARY_LENGTH = 4

ENV_NAME = re.compile(r"[A-Za-z0-9_-]+")
TAG = re.compile(r"[a-z][a-z0-9-]*")
PLACEHOLDER_MARKERS = ("{{", "${")
READ_QUERY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
PATH_SEPARATORS = re.compile(r"[\\/]")
BATCH_SUFFIXES = (".bat", ".cmd")
HOOK_STAGES: tuple[HookStage, ...] = ("before_run", "before_trial", "after_trial", "after_run")


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
    "max_ttft_ms": _number,
}


def _non_negative_int(kind: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{kind} expects a non-negative integer")


def _list_of_rows(kind: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
        raise ValueError(f"{kind} expects a list of rows")


def _only_true(kind: str, value: Any) -> None:
    if value is not True:
        raise ValueError(f"{kind} expects true")


def _scalar(kind: str, value: Any) -> None:
    if value is not None and not isinstance(value, str | int | float | bool | date):
        raise ValueError(f"{kind} expects a scalar value")


_DB_VALUE_CHECKS: dict[str, Callable[[str, Any], None]] = {
    "row_count": _non_negative_int,
    "value": _scalar,
    "rows": _list_of_rows,
    "delta": _number,
    "unchanged": _only_true,
}


def _has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in PLACEHOLDER_MARKERS)
    if isinstance(value, dict):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_placeholder(item) for item in value)
    return False


def _read_only(check_id: str, query: str) -> None:
    if any(marker in query for marker in PLACEHOLDER_MARKERS):
        raise ValueError(
            f"db check {check_id!r}: query cannot use placeholders; pass values through params"
        )
    if not READ_QUERY.match(query):
        raise ValueError(
            f"db check {check_id!r}: query must start with SELECT or WITH; change data with hooks"
        )


class DbCheck(_Model):
    """A read-only query whose result is compared with an expectation."""

    id: str
    query: str
    params: dict[str, Any] = Field(default_factory=dict, repr=False)
    kind: DbExpectKind
    value: Any
    hard: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _from_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "expect" not in data:
            return data
        expect = data["expect"]
        if not isinstance(expect, dict) or len(expect) != 1:
            keys = sorted(expect) if isinstance(expect, dict) else expect
            raise ValueError(f"db check expect must have exactly one key, got {keys}")
        kind, value = next(iter(expect.items()))
        rest = {key: item for key, item in data.items() if key != "expect"}
        return {**rest, "kind": kind, "value": value}

    @model_validator(mode="after")
    def _check_check(self) -> DbCheck:
        _read_only(self.id, self.query)
        if _has_placeholder(self.value):
            raise ValueError(f"db check {self.id!r}: expected values cannot use placeholders")
        _DB_VALUE_CHECKS[self.kind](self.kind, self.value)
        return self

    @property
    def needs_before(self) -> bool:
        """Whether the check compares against a snapshot taken before the turn."""
        return self.kind in ("delta", "unchanged")


class ForbiddenChange(_Model):
    """A read-only query whose result must not change from the snapshot taken after the reset."""

    id: str
    query: str
    params: dict[str, Any] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _check_query(self) -> ForbiddenChange:
        _read_only(self.id, self.query)
        return self


class DatabaseConfig(_Model):
    """The database that checks and sql hooks run against."""

    url: str = Field(repr=False)
    forbidden: list[ForbiddenChange] = Field(default_factory=list)


class HttpHook(_Model):
    """An HTTP request sent by a hook."""

    url: str = Field(repr=False)
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    body: Any = Field(default=None, repr=False)


class Hook(_Model):
    """A lifecycle action: exactly one of an HTTP call, SQL statements or a command."""

    http: HttpHook | None = None
    sql: list[str] | None = Field(default=None, min_length=1, repr=False)
    command: list[str] | None = Field(default=None, min_length=1, repr=False)
    timeout_s: float | None = Field(default=None, gt=0, le=600)

    @model_validator(mode="before")
    @classmethod
    def _command_is_a_list(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("command"), str):
            raise ValueError("command must be a list of arguments: hooks never run through a shell")
        return data

    @model_validator(mode="after")
    def _sql_without_templates(self) -> Hook:
        if self.sql and any("{{" in statement for statement in self.sql):
            raise ValueError("sql hooks cannot use {{ }} placeholders")
        return self

    @model_validator(mode="after")
    def _no_batch_file(self) -> Hook:
        if self.command:
            name = PATH_SEPARATORS.split(self.command[0])[-1].rstrip(". ").lower()
            if name.endswith(BATCH_SUFFIXES):
                raise ValueError(
                    "command cannot run a .bat or .cmd file: "
                    "Windows would pass its arguments through cmd.exe"
                )
        return self

    @model_validator(mode="after")
    def _one_kind(self) -> Hook:
        if sum(item is not None for item in (self.http, self.sql, self.command)) != 1:
            raise ValueError("a hook needs exactly one of http, sql or command")
        if self.sql is not None and self.timeout_s is not None:
            raise ValueError(
                "timeout_s is not supported for sql hooks: "
                "set the statement timeout in the database"
            )
        return self

    @property
    def kind(self) -> HookKind:
        """Which of the three actions this hook performs."""
        if self.http is not None:
            return "http"
        return "sql" if self.sql is not None else "command"

    def effective_timeout_s(self) -> float:
        """Return ``timeout_s``, or the default when it is not set."""
        return self.timeout_s or DEFAULT_HOOK_TIMEOUT_S


class HooksConfig(_Model):
    """Hooks run at each stage of the lifecycle."""

    before_run: list[Hook] = Field(default_factory=list)
    before_trial: list[Hook] = Field(default_factory=list)
    after_trial: list[Hook] = Field(default_factory=list)
    after_run: list[Hook] = Field(default_factory=list)

    def stage(self, name: HookStage) -> list[Hook]:
        """Return the hooks of a lifecycle stage."""
        hooks: list[Hook] = getattr(self, name)
        return hooks


class Turn(_Model):
    """One user message and the expectations on the agent reply."""

    user: str
    expect: list[Expectation] = Field(default_factory=list)
    db: list[DbCheck] = Field(default_factory=list)


class Case(_Model):
    """A business scenario played as a multi-turn conversation."""

    id: str
    turns: list[Turn] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    db: list[DbCheck] = Field(default_factory=list)
    forbidden: list[ForbiddenChange] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, tags: list[str]) -> list[str]:
        invalid = [tag for tag in tags if not TAG.fullmatch(tag)]
        if invalid:
            raise ValueError(f"invalid tag {invalid[0]!r}: use lowercase letters, digits and '-'")
        return tags


class ResponseMap(_Model):
    """JSONPath expressions that locate data in the agent response."""

    answer: str | None = None
    tool_calls: str | None = None
    tokens: str | None = None


class StreamConfig(_Model):
    """How to read an agent that answers with server-sent events."""

    delta: str
    event: str | None = None
    done: str = "[DONE]"


class WaitPolicy(_Model):
    """Per-phase time limits; each one defaults to the total."""

    connect_s: float | None = Field(default=None, gt=0)
    first_fragment_s: float | None = Field(default=None, gt=0)
    idle_s: float | None = Field(default=None, gt=0)
    total_s: float | None = Field(default=None, gt=0)


class RetryPolicy(_Model):
    """Retries for failures that happen before the agent starts answering."""

    retries: int = Field(default=0, ge=0, le=5)
    backoff_s: float = Field(default=0.5, ge=0, le=30)


@dataclass(frozen=True)
class EffectiveWait:
    """Time limits after applying defaults."""

    connect_s: float
    first_fragment_s: float
    idle_s: float
    total_s: float


class AgentConfig(_Model):
    """How to reach the agent under test over HTTP."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    request: Any = Field(repr=False)
    response: ResponseMap
    timeout_s: float = Field(default=60, gt=0)
    stream: StreamConfig | None = None
    wait: WaitPolicy = Field(default_factory=WaitPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    def effective_wait(self) -> EffectiveWait:
        """Resolve the limits: phases default to ``total_s``, which defaults to ``timeout_s``."""
        total = self.wait.total_s or self.timeout_s
        return EffectiveWait(
            connect_s=self.wait.connect_s or total,
            first_fragment_s=self.wait.first_fragment_s or total,
            idle_s=self.wait.idle_s or total,
            total_s=total,
        )

    @model_validator(mode="after")
    def _answer_matches_transport(self) -> AgentConfig:
        if self.stream is None and self.response.answer is None:
            raise ValueError("agent.response.answer is required unless agent.stream is set")
        if self.stream is not None and self.response.answer is not None:
            raise ValueError(
                "agent.response.answer is not used with agent.stream: "
                "the answer is the joined stream.delta fragments"
            )
        return self

    @model_validator(mode="after")
    def _waits_fit_total(self) -> AgentConfig:
        wait = self.effective_wait()
        for name in ("connect_s", "first_fragment_s", "idle_s"):
            value = getattr(wait, name)
            if value > wait.total_s:
                raise ValueError(
                    f"wait.{name} ({value:g}s) cannot exceed total_s ({wait.total_s:g}s)"
                )
        return self


class Canary(_Model):
    """A marker value that must never appear in an agent reply."""

    value: str = Field(repr=False)
    label: str = ""


def _default_detectors() -> list[DetectorName]:
    return ["secrets", "pii"]


class SecurityConfig(_Model):
    """What counts as a leak of confidential data."""

    canaries: list[Canary] = Field(default_factory=list)
    detectors: list[DetectorName] = Field(default_factory=_default_detectors)
    allow: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_security(self) -> SecurityConfig:
        values = [canary.value.casefold() for canary in self.canaries if canary.value]
        for number, canary in enumerate(self.canaries, 1):
            canary.label = canary.label or f"canary-{number}"
            label = canary.label.casefold()
            if any(value in label for value in values):
                raise ValueError(f"canary label #{number} contains a canary value")
        for canary in self.canaries:
            if len(canary.value) < MIN_CANARY_LENGTH:
                raise ValueError(
                    f"canary {canary.label} is shorter than {MIN_CANARY_LENGTH} characters"
                )
        for number, pattern in enumerate(self.allow, 1):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid security.allow pattern #{number}: {exc}") from exc
        return self


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
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    environments: list[str] | None = Field(default=None, min_length=1)
    data: str | None = None
    data_values: dict[str, Any] = Field(default_factory=dict, repr=False)
    environment: str | None = None
    selected_tags: list[str] = Field(default_factory=list)
    redacted_values: list[str] = Field(default_factory=list, repr=False)
    database: DatabaseConfig | None = None
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    base_dir: Path | None = Field(default=None, repr=False)
    hook_env: dict[str, str] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _check_variants(self) -> Experiment:
        for check in _EXPERIMENT_CHECKS:
            check(self)
        return self

    def select(self, tags: Iterable[str]) -> Experiment:
        """Return a copy with only the cases that carry at least one of ``tags``.

        Args:
            tags: Tags to keep.

        Returns:
            The filtered experiment, with ``selected_tags`` set.

        Raises:
            ConfigError: If no case carries any of the tags.
        """
        wanted = list(dict.fromkeys(tags))
        cases = [case for case in self.cases if set(case.tags) & set(wanted)]
        if not cases:
            raise ConfigError(f"no case matches tags: {', '.join(wanted)}")
        return self.model_copy(update={"cases": cases, "selected_tags": wanted})

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
        asserted = any(turn.expect or turn.db for turn in case.turns)
        if not (asserted or case.db or case.forbidden):
            raise ValueError(f"case {case.id!r} has no expectation: it would always pass")


def _case_db_ids(exp: Experiment, case: Case) -> list[str]:
    ids = [check.id for turn in case.turns for check in turn.db]
    ids += [check.id for check in case.db]
    ids += [change.id for change in case.forbidden]
    if exp.database is not None:
        ids += [change.id for change in exp.database.forbidden]
    return ids


def _database_declared(exp: Experiment) -> None:
    if exp.database is not None:
        return
    for case in exp.cases:
        if _case_db_ids(exp, case):
            raise ValueError(f"case {case.id!r} uses database checks but no database is configured")
    if any(hook.kind == "sql" for stage in HOOK_STAGES for hook in exp.hooks.stage(stage)):
        raise ValueError("sql hooks need a database section")


def _unique_db_ids(exp: Experiment) -> None:
    for case in exp.cases:
        ids = _case_db_ids(exp, case)
        for db_id in ids:
            if ids.count(db_id) > 1:
                raise ValueError(f"duplicate database check id {db_id!r} in case {case.id!r}")


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


def _valid_environments(exp: Experiment) -> None:
    names = exp.environments or []
    if len(set(names)) != len(names):
        raise ValueError("environment names must be unique")
    invalid = [name for name in names if not ENV_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"environment names may only use letters, digits, '-' and '_': {invalid}")


_EXPERIMENT_CHECKS: tuple[Callable[[Experiment], None], ...] = (
    _unique_names,
    _valid_environments,
    _one_baseline,
    _cases_have_expectations,
    _database_declared,
    _unique_db_ids,
    _json_serializable,
    _variants_cover_template,
)

_VARIANT_REF = re.compile(r"\{\{\s*variant\.([\w.-]+)\s*\}\}")


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
