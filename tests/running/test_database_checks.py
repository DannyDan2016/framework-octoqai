import pytest

from octoqai.experiment import Experiment, HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor
from octoqai.running import run_experiment
from support.builders import ExperimentBuilder, turn

pytestmark = [pytest.mark.integration, pytest.mark.regression]

COUNT_001 = {
    "id": "ajuste-creado",
    "query": "SELECT count(*) FROM ajustes WHERE cuenta = :c",
    "params": {"c": "001"},
}
TENANT_B = {"id": "no-tocar-b", "query": "SELECT count(*) FROM ajustes WHERE tenant = 'B'"}


def build(sqlite_db, *turns, case_db=(), forbidden=(), trials=1):
    builder = ExperimentBuilder().database(url=sqlite_db.url, forbidden=forbidden).trials(trials)
    return Experiment.model_validate(builder.case("c1", *turns, db=case_db).build())


def play(exp, agent, db, hooks=None):
    return run_experiment(exp, agent.adapter(exp.agent), db=db, hooks=hooks).trials


@pytest.mark.smoke
def test_delta_passes_when_the_agent_creates_the_row(sqlite_db, sql_database, fake_agent):
    exp = build(sqlite_db, turn("Crea un ajuste", db=[{**COUNT_001, "expect": {"delta": 1}}]))
    agent = fake_agent.answers("hecho.").effect(lambda _: sqlite_db.insert())
    [trial] = play(exp, agent, sql_database)
    assert trial.outcome == "pass"
    assert trial.turns[0].checks[-1].detail == "db 'ajuste-creado': changed by 1 (expected 1)"


def test_delta_fails_when_nothing_was_written(sqlite_db, sql_database, fake_agent):
    exp = build(sqlite_db, turn("Crea un ajuste", db=[{**COUNT_001, "expect": {"delta": 1}}]))
    [trial] = play(exp, fake_agent.answers("hecho."), sql_database)
    assert trial.outcome == "fail"


@pytest.mark.security
def test_forbidden_change_is_hard_and_counted_once(sqlite_db, sql_database, fake_agent):
    exp = build(
        sqlite_db,
        turn("Uno", {"contains": "."}),
        turn("Dos", {"contains": "."}),
        forbidden=[TENANT_B],
    )
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert(tenant="B"))
    [trial] = play(exp, agent, sql_database)
    assert trial.forbidden_count == 1 and trial.hard_violation and not trial.other_hard_violation
    assert [c.detail for c in trial.turns[0].checks if c.kind == "forbidden_crud"] == [
        "forbidden change 'no-tocar-b' after turn 1"
    ]
    assert all(c.kind != "forbidden_crud" for c in trial.turns[1].checks)


@pytest.mark.security
def test_forbidden_change_counts_even_if_the_agent_then_fails(sqlite_db, sql_database, fake_agent):
    exp = build(sqlite_db, turn("Borra", {"contains": "."}), forbidden=[TENANT_B])
    agent = fake_agent.fails(500).effect(lambda _: sqlite_db.insert(tenant="B"))
    [trial] = play(exp, agent, sql_database)
    assert trial.agent_error and trial.hard_violation and trial.forbidden_count == 1


def test_case_level_check_runs_after_the_last_turn(sqlite_db, sql_database, fake_agent):
    exp = build(
        sqlite_db,
        turn("Uno", {"contains": "."}),
        turn("Dos", {"contains": "."}),
        case_db=[{"id": "total", "query": "SELECT count(*) FROM ajustes", "expect": {"value": 2}}],
    )
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert())
    [trial] = play(exp, agent, sql_database)
    assert [(c.kind, c.passed) for c in trial.db_checks] == [("db", True)]


def test_database_error_makes_the_check_not_evaluable(sqlite_db, sql_database, fake_agent):
    bad = {"id": "x", "query": "SELECT count(*) FROM nope", "expect": {"value": 1}}
    [trial] = play(
        build(sqlite_db, turn("Hola", db=[bad])), fake_agent.answers("ok."), sql_database
    )
    assert trial.outcome == "not_evaluable" and "no such table" in trial.turns[0].checks[-1].detail


def test_checks_without_a_database_connection_are_not_evaluable(sqlite_db, fake_agent):
    exp = build(sqlite_db, turn("Crea", db=[{**COUNT_001, "expect": {"delta": 1}}]))
    [trial] = run_experiment(exp, fake_agent.answers("ok.").adapter(exp.agent)).trials
    assert trial.outcome == "not_evaluable"
    assert (
        trial.turns[0].checks[-1].detail
        == "db 'ajuste-creado': database error: no database connection"
    )


def test_sql_reset_hook_isolates_every_trial(sqlite_db, sql_database, fake_agent):
    exp = build(
        sqlite_db,
        turn("Crea", db=[{**COUNT_001, "expect": {"delta": 1}}]),
        case_db=[{"id": "uno", "query": "SELECT count(*) FROM ajustes", "expect": {"value": 1}}],
        trials=3,
    )
    hooks = HookExecutor(
        HooksConfig.model_validate({"before_trial": [{"sql": ["DELETE FROM ajustes"]}]}),
        HookEnvironment(db=sql_database),
    )
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert())
    trials = play(exp, agent, sql_database, hooks)
    assert [t.outcome for t in trials] == ["pass", "pass", "pass"]
