"""Judge calls over redacted transcripts, evaluated per turn and per case."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from octoqai.agents.port import AgentReply
from octoqai.expectations import (
    JUDGE_KIND,
    CheckResult,
    JudgeInputError,
    judge_input_failed,
    judge_request,
    judge_result,
    judge_skipped,
)
from octoqai.experiment import Case, JudgeCheck, Turn
from octoqai.providers.port import LlmRequest

if TYPE_CHECKING:
    from octoqai.running.runner import _Session

NO_JUDGE_DETAIL = "judge '{id}': no judge configured"
_ATTEMPTS = 2


def _rubric_sha256(check: JudgeCheck) -> str:
    return hashlib.sha256(f"{check.rubric}\n{check.reference or ''}".encode()).hexdigest()


class JudgeProbe:
    """Asks the judge about one trial; every text it sends is redacted first."""

    def __init__(self, session: _Session, case: Case) -> None:
        """Start a probe for one trial of ``case``.

        Args:
            session: The run session, with the judge port, the redactor and the nonce source.
            case: The case being played, for its case-level judge checks.
        """
        self._session = session
        self._case = case
        self._clean: list[tuple[str, str]] = []

    def _redacted(self, transcript: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        redact = self._session.redactor.redact
        for user, answer in transcript[len(self._clean) :]:
            self._clean.append((redact(user), redact(answer)))
        return self._clean[: len(transcript)]

    def _safe(self, check: JudgeCheck) -> JudgeCheck:
        redact = self._session.redactor.redact
        reference = None if check.reference is None else redact(check.reference)
        return check.model_copy(update={"rubric": redact(check.rubric), "reference": reference})

    def _request(
        self,
        check: JudgeCheck,
        transcript: Sequence[tuple[str, str]],
        focus_last: bool,
        limit: int,
    ) -> LlmRequest | JudgeInputError:
        for _ in range(_ATTEMPTS):
            built = judge_request(
                check,
                transcript,
                focus_last=focus_last,
                nonce=self._session.nonce(),
                max_input_chars=limit,
            )
            if not (isinstance(built, JudgeInputError) and built.reason == "nonce_collision"):
                break
        return built

    def _evaluate(
        self, check: JudgeCheck, transcript: Sequence[tuple[str, str]], focus_last: bool
    ) -> CheckResult:
        port = self._session.judge
        judge = self._session.exp.judge
        if port is None or judge is None:
            return CheckResult(JUDGE_KIND, None, check.hard, NO_JUDGE_DETAIL.format(id=check.id))
        safe = self._safe(check)
        digest = _rubric_sha256(safe)
        built = self._request(safe, self._redacted(transcript), focus_last, judge.max_input_chars)
        if isinstance(built, JudgeInputError):
            return judge_input_failed(
                check, built, expected_model=judge.model, rubric_sha256=digest
            )
        return judge_result(
            check, port.complete(built), expected_model=judge.model, rubric_sha256=digest
        )

    def after_turn(
        self, turn: Turn, transcript: Sequence[tuple[str, str]], reply: AgentReply
    ) -> list[CheckResult]:
        """Evaluate the judge checks of the turn that was just played.

        Args:
            turn: The turn, with its judge checks.
            transcript: ``(user, answer)`` pairs up to and including this turn, unredacted.
            reply: What the agent answered; an error skips the judge without calling it.

        Returns:
            One result per judge check of the turn.
        """
        if reply.error is not None:
            return [judge_skipped(check, True) for check in turn.judge]
        return [self._evaluate(check, transcript, True) for check in turn.judge]

    def finish(
        self, transcript: Sequence[tuple[str, str]], failure: str | None
    ) -> list[CheckResult]:
        """Evaluate the judge checks of the whole case, once, over the full conversation.

        Args:
            transcript: Every ``(user, answer)`` pair of the trial, unredacted.
            failure: The first agent error of the trial; when set the judge is skipped.

        Returns:
            One result per case-level judge check.
        """
        if failure is not None:
            return [judge_skipped(check, True) for check in self._case.judge]
        return [self._evaluate(check, transcript, False) for check in self._case.judge]
