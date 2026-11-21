"""Serial runner that plays every case as an isolated multi-turn conversation."""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from octoqai.agents.port import AgentPort, AgentReply
from octoqai.db.port import DatabasePort
from octoqai.expectations import FORBIDDEN_KIND, CheckResult, evaluate
from octoqai.experiment import Case, Experiment, HookStage, Turn, Variant
from octoqai.hooks.port import HookResult, LifecyclePort
from octoqai.providers.budget import BudgetedLlm, LlmStats
from octoqai.providers.port import LlmPort, LlmRole
from octoqai.running._database import DbProbe
from octoqai.running._judge import JudgeProbe
from octoqai.security import Detector, Leak, Redactor, scan_reply

Outcome = Literal["pass", "fail", "not_evaluable"]


@dataclass
class TurnResult:
    """One user message, the agent reply and the checks applied to it."""

    user: str
    reply: AgentReply
    checks: list[CheckResult]
    leaks: list[Leak] = field(default_factory=list)


@dataclass
class TrialResult:
    """One full conversation of a case for a variant."""

    variant: str
    case_id: str
    trial: int
    turns: list[TurnResult]
    hooks: list[HookResult] = field(default_factory=list)
    setup_error: str | None = None
    db_checks: list[CheckResult] = field(default_factory=list)
    judge_checks: list[CheckResult] = field(default_factory=list)

    @property
    def _checks(self) -> list[CheckResult]:
        turn_checks = [check for turn in self.turns for check in turn.checks]
        return turn_checks + self.db_checks + self.judge_checks

    @property
    def outcome(self) -> Outcome:
        """``fail`` if any check failed, ``not_evaluable`` if any is unknown, else ``pass``."""
        if self.setup_error is not None:
            return "not_evaluable"
        checks = self._checks
        if any(turn.reply.error for turn in self.turns) or any(c.passed is False for c in checks):
            return "fail"
        if any(c.passed is None for c in checks):
            return "not_evaluable"
        return "pass"

    @property
    def hard_violation(self) -> bool:
        """Whether a hard expectation failed on a real agent answer."""
        return any(c.hard and c.passed is False and not c.agent_error for c in self._checks)

    @property
    def agent_error(self) -> bool:
        """Whether a turn failed because the agent could not be reached or mapped."""
        return any(turn.reply.error for turn in self.turns)

    @property
    def tokens(self) -> int | None:
        """Total tokens used, or ``None`` if any turn did not report usage."""
        if not self.turns:
            return None
        total = 0
        for turn in self.turns:
            if turn.reply.tokens is None:
                return None
            total += turn.reply.tokens
        return total

    @property
    def leak_count(self) -> int:
        """Number of confidential values the agent leaked across turns."""
        return sum(len(turn.leaks) for turn in self.turns)

    @property
    def ttft_ms(self) -> float | None:
        """Mean time to first token over the turns that report it, or ``None``."""
        values = [t.reply.ttft_ms for t in self.turns if t.reply.ttft_ms is not None]
        return sum(values) / len(values) if values else None

    @property
    def forbidden_count(self) -> int:
        """Number of forbidden database changes, each counted once per trial."""
        return sum(1 for c in self._checks if c.kind == FORBIDDEN_KIND and c.passed is False)

    @property
    def other_hard_violation(self) -> bool:
        """Whether a hard expectation other than a leak or forbidden change failed."""
        return any(
            c.hard
            and c.passed is False
            and not c.agent_error
            and c.kind not in ("leak", FORBIDDEN_KIND)
            for c in self._checks
        )

    @property
    def latency_ms(self) -> float:
        """Total agent latency across turns."""
        return sum(turn.reply.latency_ms for turn in self.turns)


@dataclass
class RunResult:
    """All trials of an experiment run."""

    experiment: Experiment
    trials: list[TrialResult]
    started_at: str
    finished_at: str
    hooks: list[HookResult] = field(default_factory=list)
    setup_error: str | None = None
    llm: dict[LlmRole, LlmStats] = field(default_factory=dict)


class _NoHooks:
    def run(self, stage: HookStage) -> list[HookResult]:
        return []

    def close(self) -> None:
        return None


def hook_error(result: HookResult) -> str:
    """Describe a failed hook as ``stage[index] (kind) error``."""
    return f"{result.stage}[{result.index}] ({result.kind}) {result.error}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class _Session:
    agent: AgentPort
    exp: Experiment
    detector: Detector
    db: DatabasePort | None
    hooks: LifecyclePort
    redactor: Redactor
    nonce: Callable[[], str]
    judge: BudgetedLlm | None = None
    simulator: BudgetedLlm | None = None


@dataclass
class _Probes:
    db: DbProbe
    judge: JudgeProbe
    transcript: list[tuple[str, str]] = field(default_factory=list)


def _leak_check(leak: Leak) -> CheckResult:
    return CheckResult(
        "leak", False, True, f"{leak.kind} {leak.label!r} in {leak.location}: {leak.masked}"
    )


def _play_turn(
    session: _Session,
    probes: _Probes,
    request: dict[str, Any],
    turn: Turn,
    said: list[str],
) -> TurnResult:
    probes.db.before_turn(turn)
    reply = session.agent.send(request)
    said.append(turn.user)
    probes.transcript.append((turn.user, reply.answer or ""))
    leaks = scan_reply(session.detector, reply, said)
    checks = [evaluate(e, reply) for e in turn.expect] + [_leak_check(x) for x in leaks]
    checks += probes.db.after_turn(turn, reply, len(said))
    checks += probes.judge.after_turn(turn, probes.transcript, reply)
    return TurnResult(turn.user, reply, checks, leaks)


def _play(session: _Session, probes: _Probes, variant: Variant, case: Case) -> list[TurnResult]:
    exp = session.exp
    said: list[str] = []
    conversation_id = str(uuid.uuid4())
    history: list[dict[str, str]] = []
    turns: list[TurnResult] = []
    for turn_index, turn in enumerate(case.turns):
        request = {
            "message": turn.user,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "history": [dict(item) for item in history],
            "variant": dict(variant.params),
            "data": exp.data_values,
            "env": exp.environment,
        }
        result = _play_turn(session, probes, request, turn, said)
        turns.append(result)
        if result.reply.error is not None:
            break
        history.append({"role": "user", "content": turn.user})
        history.append({"role": "assistant", "content": result.reply.answer or ""})
    return turns


def _first_failure(results: list[HookResult]) -> str | None:
    return next((hook_error(r) for r in results if not r.ok), None)


def _run_trial(session: _Session, variant: Variant, case: Case, index: int) -> TrialResult:
    before = session.hooks.run("before_trial")
    setup_error = _first_failure(before)
    turns: list[TurnResult] = []
    db_checks: list[CheckResult] = []
    judge_checks: list[CheckResult] = []
    try:
        if setup_error is None:
            probes = _Probes(DbProbe(session.db, session.exp, case), JudgeProbe(session, case))
            turns = _play(session, probes, variant, case)
            failure = next((t.reply.error for t in turns if t.reply.error), None)
            db_checks = probes.db.finish(failure)
            judge_checks = probes.judge.finish(probes.transcript, failure)
    finally:
        after = session.hooks.run("after_trial")
    return TrialResult(
        variant.name, case.id, index, turns, before + after, setup_error, db_checks, judge_checks
    )


def _run_all(session: _Session) -> list[TrialResult]:
    exp = session.exp
    return [
        _run_trial(session, variant, case, index)
        for variant in exp.variants
        for case in exp.cases
        for index in range(exp.trials)
    ]


def _budgeted(port: LlmPort | None, max_calls: int | None) -> BudgetedLlm | None:
    if port is None or max_calls is None:
        return None
    return BudgetedLlm(port, max_calls)


def _llm_stats(session: _Session) -> dict[LlmRole, LlmStats]:
    roles: dict[LlmRole, BudgetedLlm | None] = {
        "judge": session.judge,
        "simulated_user": session.simulator,
    }
    return {role: budget.stats() for role, budget in roles.items() if budget is not None}


def run_experiment(
    exp: Experiment,
    agent: AgentPort,
    *,
    db: DatabasePort | None = None,
    hooks: LifecyclePort | None = None,
    judge: LlmPort | None = None,
    simulator: LlmPort | None = None,
    nonce: Callable[[], str] = secrets.token_hex,
) -> RunResult:
    """Run every variant, case and trial in series, with a reset before each trial.

    Args:
        exp: The experiment to run.
        agent: Adapter connected to the agent under test.
        db: Database the experiment checks, if any.
        hooks: Lifecycle hooks to run around the run and each trial.
        judge: Model that judges replies, if the experiment configures one; it is wrapped in
            the call budget of ``exp.judge`` and never closed here.
        simulator: Model that plays the user, if the experiment configures one; budgeted by
            ``exp.simulated_user`` and never closed here.
        nonce: Source of the unpredictable token that delimits judge data; lowercase hex.

    Returns:
        The collected trial results, the outcome of the run hooks and the calls made to each
        language model role.

    Raises:
        Exception: Whatever a trial raises, such as an agent adapter failing unexpectedly. The
            ``after_trial`` and ``after_run`` hooks still run, but the trials and hook results
            gathered so far are lost, because no ``RunResult`` is returned.
    """
    started_at = _now()
    session = _Session(
        agent,
        exp,
        Detector(exp.security),
        db,
        hooks if hooks is not None else _NoHooks(),
        Redactor.for_report(exp.security, exp.redacted_values),
        nonce,
        _budgeted(judge, exp.judge.max_calls if exp.judge is not None else None),
        _budgeted(simulator, exp.simulated_user.max_calls if exp.simulated_user else None),
    )
    before = session.hooks.run("before_run")
    setup_error = _first_failure(before)
    trials: list[TrialResult] = []
    try:
        if setup_error is None:
            trials = _run_all(session)
    finally:
        after = session.hooks.run("after_run")
    return RunResult(
        exp, trials, started_at, _now(), before + after, setup_error, _llm_stats(session)
    )
