"""Serial runner that plays every case as an isolated multi-turn conversation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from octoqai.agents.port import AgentPort, AgentReply
from octoqai.db.port import DatabasePort
from octoqai.expectations import FORBIDDEN_KIND, CheckResult, evaluate
from octoqai.experiment import Case, Experiment, HookStage, Turn, Variant
from octoqai.hooks.port import HookResult, LifecyclePort
from octoqai.running._database import DbProbe
from octoqai.security import Detector, Leak, scan_reply

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

    @property
    def _checks(self) -> list[CheckResult]:
        return [check for turn in self.turns for check in turn.checks] + self.db_checks

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


def _leak_check(leak: Leak) -> CheckResult:
    return CheckResult(
        "leak", False, True, f"{leak.kind} {leak.label!r} in {leak.location}: {leak.masked}"
    )


def _play_turn(
    session: _Session,
    probe: DbProbe,
    request: dict[str, Any],
    turn: Turn,
    said: list[str],
) -> TurnResult:
    probe.before_turn(turn)
    reply = session.agent.send(request)
    said.append(turn.user)
    leaks = scan_reply(session.detector, reply, said)
    checks = [evaluate(e, reply) for e in turn.expect] + [_leak_check(x) for x in leaks]
    checks += probe.after_turn(turn, reply, len(said))
    return TurnResult(turn.user, reply, checks, leaks)


def _play(session: _Session, probe: DbProbe, variant: Variant, case: Case) -> list[TurnResult]:
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
        result = _play_turn(session, probe, request, turn, said)
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
    try:
        if setup_error is None:
            probe = DbProbe(session.db, session.exp, case)
            turns = _play(session, probe, variant, case)
            failure = next((t.reply.error for t in turns if t.reply.error), None)
            db_checks = probe.finish(failure)
    finally:
        after = session.hooks.run("after_trial")
    return TrialResult(variant.name, case.id, index, turns, before + after, setup_error, db_checks)


def _run_all(session: _Session) -> list[TrialResult]:
    exp = session.exp
    return [
        _run_trial(session, variant, case, index)
        for variant in exp.variants
        for case in exp.cases
        for index in range(exp.trials)
    ]


def run_experiment(
    exp: Experiment,
    agent: AgentPort,
    *,
    db: DatabasePort | None = None,
    hooks: LifecyclePort | None = None,
) -> RunResult:
    """Run every variant, case and trial in series, with a reset before each trial.

    Args:
        exp: The experiment to run.
        agent: Adapter connected to the agent under test.
        db: Database the experiment checks, if any.
        hooks: Lifecycle hooks to run around the run and each trial.

    Returns:
        The collected trial results and the outcome of the run hooks.
    """
    started_at = _now()
    session = _Session(
        agent, exp, Detector(exp.security), db, hooks if hooks is not None else _NoHooks()
    )
    before = session.hooks.run("before_run")
    setup_error = _first_failure(before)
    trials: list[TrialResult] = []
    try:
        if setup_error is None:
            trials = _run_all(session)
    finally:
        after = session.hooks.run("after_run")
    return RunResult(exp, trials, started_at, _now(), before + after, setup_error)
