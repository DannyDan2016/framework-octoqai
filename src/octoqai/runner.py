"""Serial runner that plays every case as an isolated multi-turn conversation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from octoqai.adapter import AgentReply, HttpAgentAdapter
from octoqai.config import Case, Experiment, Variant
from octoqai.expectations import CheckResult, evaluate

Outcome = Literal["pass", "fail", "not_evaluable"]


@dataclass
class TurnResult:
    """One user message, the agent reply and the checks applied to it."""

    user: str
    reply: AgentReply
    checks: list[CheckResult]


@dataclass
class TrialResult:
    """One full conversation of a case for a variant."""

    variant: str
    case_id: str
    trial: int
    turns: list[TurnResult]

    @property
    def _checks(self) -> list[CheckResult]:
        return [check for turn in self.turns for check in turn.checks]

    @property
    def outcome(self) -> Outcome:
        """``fail`` if any check failed, ``not_evaluable`` if any is unknown, else ``pass``."""
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
        counts = [turn.reply.tokens for turn in self.turns]
        return None if any(c is None for c in counts) else sum(counts)

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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_trial(adapter: HttpAgentAdapter, variant: Variant, case: Case, index: int) -> TrialResult:
    conversation_id = str(uuid.uuid4())
    history: list[dict[str, str]] = []
    turns: list[TurnResult] = []
    for turn_index, turn in enumerate(case.turns):
        reply = adapter.send(
            {
                "message": turn.user,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "history": [dict(item) for item in history],
                "variant": dict(variant.params),
            }
        )
        turns.append(TurnResult(turn.user, reply, [evaluate(e, reply) for e in turn.expect]))
        if reply.error is not None:
            break
        history.append({"role": "user", "content": turn.user})
        history.append({"role": "assistant", "content": reply.answer or ""})
    return TrialResult(variant.name, case.id, index, turns)


def run_experiment(exp: Experiment, adapter: HttpAgentAdapter) -> RunResult:
    """Run every variant, case and trial in series.

    Args:
        exp: The experiment to run.
        adapter: Adapter connected to the agent under test.

    Returns:
        The collected trial results.
    """
    started_at = _now()
    trials = [
        _run_trial(adapter, variant, case, index)
        for variant in exp.variants
        for case in exp.cases
        for index in range(exp.trials)
    ]
    return RunResult(exp, trials, started_at, _now())
