import pytest

from octoqai.db import QueryResult
from octoqai.experiment import Experiment, HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor
from octoqai.running import run_experiment
from support.builders import ExperimentBuilder, turn
from support.databases import FakeDatabase, recreate_command
from support.hooks import RecordingHooks

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
    assert trial.outcome == "not_evaluable" and trial.turns[0].checks[-1].detail.endswith(
        "OperationalError"
    )


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


@pytest.mark.regression
def test_a_real_reset_that_recreates_the_database_file_runs_every_trial(
    tmp_path, sqlite_db, sql_database, fake_agent
):
    exp = build(sqlite_db, turn("Crea", {"contains": "."}), forbidden=[TENANT_B], trials=3)
    reset = {"before_trial": [{"command": recreate_command(tmp_path, sqlite_db)}]}
    hooks = HookExecutor(HooksConfig.model_validate(reset), HookEnvironment(db=sql_database))
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.insert(tenant="B"))
    trials = play(exp, agent, sql_database, hooks)
    assert [t.setup_error for t in trials] == [None, None, None]
    assert [t.forbidden_count for t in trials] == [1, 1, 1]


@pytest.mark.security
@pytest.mark.parametrize(
    "query",
    [
        "WITH t AS (SELECT 1) DELETE FROM ajustes WHERE tenant = 'B'",
        "WITH t AS (SELECT 1) UPDATE ajustes SET importe = 0",
    ],
    ids=["delete", "update"],
)
def test_a_guard_with_no_result_set_is_never_a_pass(query, sqlite_db, sql_database, fake_agent):
    sqlite_db.insert(tenant="B")
    exp = build(
        sqlite_db, turn("Borra", {"contains": "."}), forbidden=[{"id": "g", "query": query}]
    )
    agent = fake_agent.answers("ok.").effect(lambda _: sqlite_db.run("DELETE FROM ajustes"))
    [trial] = play(exp, agent, sql_database)
    assert trial.outcome == "not_evaluable" and trial.hard_violation is False
    assert "query returned no result set" in trial.turns[0].checks[-1].detail
    assert sqlite_db.scalar("SELECT count(*) FROM ajustes") == 0


def test_a_failed_reset_runs_no_query(fake_agent):
    fake = FakeDatabase()
    exp = Experiment.model_validate(
        ExperimentBuilder()
        .database(url="sqlite:///x.db", forbidden=[TENANT_B])
        .trials(1)
        .case("c1", turn("Crea", db=[{**COUNT_001, "expect": {"delta": 1}}]))
        .build()
    )
    hooks = RecordingHooks().fails("before_trial")
    [trial] = play(exp, fake_agent.answers("ok."), fake, hooks)
    assert trial.setup_error is not None and fake.queries == []
    assert fake_agent.requests == []


class UnreadableAfterStart(FakeDatabase):
    def __init__(self, readable):
        super().__init__()
        self.readable = readable

    def query(self, sql, params):
        super().query(sql, params)
        if len(self.queries) in self.readable:
            return QueryResult(("n",), ((0,),), None)
        return QueryResult(error="OperationalError")


@pytest.mark.security
def test_an_unreadable_forbidden_check_is_reported_once_per_trial(fake_agent):
    exp = Experiment.model_validate(
        ExperimentBuilder()
        .database(url="sqlite:///x.db", forbidden=[TENANT_B])
        .trials(2)
        .case("c1", turn("Uno", {"contains": "."}), turn("Dos", {"contains": "."}))
        .build()
    )
    trials = play(exp, fake_agent.answers("ok."), UnreadableAfterStart({1, 4}))
    for trial in trials:
        found = [c for t in trial.turns for c in t.checks if c.kind == "forbidden_crud"]
        assert [(c.passed, c.hard) for c in found] == [(None, True)]
    assert [t.outcome for t in trials] == ["not_evaluable", "not_evaluable"]


@pytest.mark.security
def test_a_real_change_after_an_unreadable_turn_still_counts(fake_agent):
    class Sequence(FakeDatabase):
        answers = iter(
            [
                QueryResult(("n",), ((0,),), None),
                QueryResult(error="OperationalError"),
                QueryResult(("n",), ((1,),), None),
            ]
        )

        def query(self, sql, params):
            return next(self.answers)

    exp = Experiment.model_validate(
        ExperimentBuilder()
        .database(url="sqlite:///x.db", forbidden=[TENANT_B])
        .trials(1)
        .case("c1", turn("Uno", {"contains": "."}), turn("Dos", {"contains": "."}))
        .build()
    )
    [trial] = play(exp, fake_agent.answers("ok."), Sequence())
    assert [c.passed for t in trial.turns for c in t.checks if c.kind == "forbidden_crud"] == [
        None,
        False,
    ]
    assert trial.forbidden_count == 1
