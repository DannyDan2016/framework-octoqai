"""Upper bound on the language model calls an experiment can make."""

from __future__ import annotations

from octoqai.experiment import Case, Experiment
from octoqai.providers.port import LlmRole


def _judge_per_trial(case: Case) -> int:
    scripted = sum(len(turn.judge) for turn in case.turns)
    each_turn = 0
    if case.simulated is not None:
        each_turn = case.simulated.max_turns * len(case.simulated.each_turn_judge)
    return scripted + each_turn + len(case.judge)


def _simulator_per_trial(case: Case) -> int:
    return case.simulated.max_turns if case.simulated is not None else 0


def worst_case_calls(exp: Experiment) -> dict[LlmRole, int]:
    """Return the most calls each configured role can make in a full run.

    Args:
        exp: The experiment, already filtered by tags if they apply.

    Returns:
        The worst case per role; roles without a configured section are absent.
    """
    runs = len(exp.variants) * exp.trials
    worst: dict[LlmRole, int] = {}
    if exp.judge is not None:
        worst["judge"] = runs * sum(_judge_per_trial(case) for case in exp.cases)
    if exp.simulated_user is not None:
        worst["simulated_user"] = runs * sum(_simulator_per_trial(case) for case in exp.cases)
    return worst
