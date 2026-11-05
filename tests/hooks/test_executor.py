import subprocess

import pytest

from octoqai.experiment import HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor, HookResult
from support.agents import FakeClock
from support.databases import FakeDatabase
from support.hooks import FakeCommands

pytestmark = [pytest.mark.unit, pytest.mark.regression]

RESET = ["python", "reset.py", "--tenant", "a && del *"]


def executor(stages, *, commands=None, agent=None, clock=None, **environment):
    hooks = HooksConfig.model_validate(stages)
    seams = {"run_command": commands or FakeCommands(), "clock": clock or FakeClock()}
    if agent is not None:
        seams["transport"] = agent.transport()
    env = HookEnvironment(env={"DB_URL": "sqlite:///a.db"}, **environment)
    return HookExecutor(hooks, env, **seams)


@pytest.mark.smoke
@pytest.mark.security
def test_command_runs_without_shell_in_the_experiment_folder(tmp_path):
    commands = FakeCommands()
    results = executor(
        {"before_trial": [{"command": RESET}]}, commands=commands, base_dir=tmp_path
    ).run("before_trial")
    assert results == [HookResult("before_trial", 0, "command", True, 0.0, None)]
    [(args, kwargs)] = commands.calls
    assert args == RESET
    assert (kwargs["shell"], kwargs["cwd"], kwargs["timeout"]) == (False, tmp_path, 30.0)
    assert kwargs["stdout"] == kwargs["stderr"] == kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"] == {"DB_URL": "sqlite:///a.db"}


@pytest.mark.parametrize(
    ("commands", "error"),
    [
        (FakeCommands().exits(2), "exited with code 2"),
        (FakeCommands().times_out(), "timed out after 5s"),
        (FakeCommands().missing(), "cannot start 'python': FileNotFoundError"),
    ],
    ids=["exit-code", "timeout", "missing-program"],
)
def test_command_failures(commands, error):
    stages = {"before_run": [{"command": RESET, "timeout_s": 5}]}
    [outcome] = executor(stages, commands=commands).run("before_run")
    assert (outcome.ok, outcome.error) == (False, error)


def test_before_stage_stops_at_first_failure():
    commands = FakeCommands().exits(1, 0)
    stages = {"before_trial": [{"command": ["a"]}, {"command": ["b"]}]}
    assert len(executor(stages, commands=commands).run("before_trial")) == 1
    assert len(commands.calls) == 1


def test_after_stage_runs_every_hook():
    commands = FakeCommands().exits(1, 0)
    stages = {"after_trial": [{"command": ["a"]}, {"command": ["b"]}]}
    results = executor(stages, commands=commands).run("after_trial")
    assert [r.ok for r in results] == [False, True]


def test_duration_comes_from_the_injected_clock():
    clock = FakeClock()
    commands = FakeCommands(clock=clock, took_s=0.25)
    stages = {"after_run": [{"command": ["a"]}]}
    [outcome] = executor(stages, commands=commands, clock=clock).run("after_run")
    assert outcome.duration_ms == pytest.approx(250)


def test_http_hook_posts_the_json_body(fake_agent):
    stages = {
        "before_run": [{"http": {"url": "http://hooks.local/reset", "body": {"tenant": "a"}}}]
    }
    [outcome] = executor(stages, agent=fake_agent).run("before_run")
    assert outcome.ok and fake_agent.requests == [{"tenant": "a"}]


@pytest.mark.parametrize(
    ("behaviour", "error"),
    [
        ("fails", "HTTP 503"),
        ("times_out", "timed out after 30s"),
        ("refuses", "cannot connect: ConnectError"),
    ],
)
def test_http_hook_failures_are_never_retried(fake_agent, behaviour, error):
    getattr(fake_agent, behaviour)(*([503] if behaviour == "fails" else []))
    stages = {"before_run": [{"http": {"url": "http://hooks.local/reset"}}]}
    [outcome] = executor(stages, agent=fake_agent).run("before_run")
    assert (outcome.ok, outcome.error, len(fake_agent.requests)) == (False, error, 1)


def test_sql_hook_runs_on_the_database():
    db = FakeDatabase()
    [outcome] = executor({"before_trial": [{"sql": ["DELETE FROM ajustes"]}]}, db=db).run(
        "before_trial"
    )
    assert outcome.ok and db.executed == [["DELETE FROM ajustes"]]


def test_sql_hook_reports_the_database_error():
    db = FakeDatabase(error="OperationalError: database is locked")
    [outcome] = executor({"before_trial": [{"sql": ["DELETE FROM ajustes"]}]}, db=db).run(
        "before_trial"
    )
    assert (outcome.ok, outcome.error) == (False, "OperationalError: database is locked")
