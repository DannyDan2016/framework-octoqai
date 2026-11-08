import datetime
import string
import subprocess
import sys

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


WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows path syntax")
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path syntax")


def program_run(program, base_dir):
    commands = FakeCommands()
    stages = {"before_trial": [{"command": [program, "--flag"]}]}
    [outcome] = executor(stages, commands=commands, base_dir=base_dir).run("before_trial")
    return outcome, commands.calls


@pytest.mark.security
@pytest.mark.parametrize(
    "program",
    [
        "sub/tool",
        "./tool",
        "../bin/tool",
        pytest.param("sub\\tool.exe", marks=WINDOWS_ONLY),
        pytest.param(".\\tool.exe", marks=WINDOWS_ONLY),
        pytest.param("..\\bin\\tool.exe", marks=WINDOWS_ONLY),
    ],
    ids=["slash", "dot-slash", "parent", "backslash", "dot-backslash", "parent-backslash"],
)
def test_relative_program_runs_from_the_experiment_folder_not_the_current_one(program, tmp_path):
    outcome, [(args, _)] = program_run(program, tmp_path)
    assert outcome.ok
    assert args == [str(tmp_path.absolute() / program), "--flag"]


@pytest.mark.parametrize(
    "program",
    ["tool", "python", pytest.param("sub\\tool", marks=POSIX_ONLY)],
    ids=["bare", "interpreter", "posix-backslash-is-a-name"],
)
def test_bare_program_name_is_left_to_the_system_search(program, tmp_path):
    _, [(args, _)] = program_run(program, tmp_path)
    assert args[0] == program


def test_absolute_program_and_missing_folder_leave_the_path_alone(tmp_path):
    absolute = str(tmp_path.absolute() / "bin" / "tool")
    assert program_run(absolute, tmp_path)[1][0][0][0] == absolute
    assert program_run("sub/tool", None)[1][0][0][0] == "sub/tool"


@WINDOWS_ONLY
@pytest.mark.security
def test_drive_and_root_relative_programs_resolve_against_the_experiment_drive(tmp_path):
    base = tmp_path.absolute()
    drive_relative = program_run(base.drive + "tool.exe", tmp_path)[1][0][0][0]
    root_relative = program_run("\\tools\\tool.exe", tmp_path)[1][0][0][0]
    assert drive_relative == str(base / "tool.exe")
    assert root_relative == base.drive + "\\tools\\tool.exe"


@WINDOWS_ONLY
@pytest.mark.security
def test_program_relative_to_another_drive_never_starts(tmp_path):
    drive = tmp_path.absolute().drive
    other = next(f"{letter}:" for letter in string.ascii_uppercase if f"{letter}:" != drive)
    outcome, calls = program_run(other + "tool.exe", tmp_path)
    assert calls == []
    assert outcome.error == "cannot start the command: its path is relative to another drive"


@pytest.mark.parametrize(
    ("commands", "error"),
    [
        (FakeCommands().exits(2), "exited with code 2"),
        (FakeCommands().times_out(), "timed out after 5s"),
        (FakeCommands().missing(), "cannot start the command: FileNotFoundError"),
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


@pytest.mark.parametrize(
    ("hook", "error"),
    [
        ({"http": {"url": "http://a.local:xx/"}}, "cannot send: InvalidURL"),
        (
            {"http": {"url": "http://hooks.local/x", "body": datetime.date(2026, 1, 1)}},
            "cannot send: TypeError",
        ),
    ],
    ids=["invalid-url", "unserializable-body"],
)
def test_http_hook_that_cannot_be_sent_is_a_failed_result(fake_agent, hook, error):
    [outcome] = executor({"before_run": [hook]}, agent=fake_agent).run("before_run")
    assert (outcome.ok, outcome.error, fake_agent.requests) == (False, error, [])


def test_after_stage_continues_after_a_hook_that_cannot_be_sent(fake_agent):
    stages = {
        "after_run": [{"http": {"url": "http://a.local:xx/"}}, {"http": {"url": "http://a.local/"}}]
    }
    results = executor(stages, agent=fake_agent).run("after_run")
    assert [r.ok for r in results] == [False, True]


def test_sql_hook_without_database_fails():
    [outcome] = executor({"before_run": [{"sql": ["DELETE FROM ajustes"]}]}).run("before_run")
    assert (outcome.ok, outcome.error) == (False, "no database configured")


class RaisingDatabase(FakeDatabase):
    def execute(self, statements):
        raise RuntimeError("password=hunter2 for tenant-b-77")


@pytest.mark.security
def test_sql_hook_reports_only_the_class_of_an_exception_from_the_port():
    stages = {"before_trial": [{"sql": ["DELETE FROM ajustes"]}]}
    [outcome] = executor(stages, db=RaisingDatabase()).run("before_trial")
    assert (outcome.ok, outcome.error) == (False, "RuntimeError")


def test_http_hook_with_an_undecodable_body_is_a_failed_result(fake_agent):
    fake_agent.encodes("gzip", content=b"this is not gzip")
    stages = {"before_run": [{"http": {"url": "http://hooks.local/reset"}}]}
    [outcome] = executor(stages, agent=fake_agent).run("before_run")
    assert (outcome.ok, outcome.error) == (False, "cannot send: DecodingError")


def test_http_hook_never_follows_a_redirect(fake_agent):
    fake_agent.fails(302)
    stages = {"before_run": [{"http": {"url": "http://hooks.local/reset"}}]}
    [outcome] = executor(stages, agent=fake_agent).run("before_run")
    assert (outcome.ok, outcome.error, len(fake_agent.requests)) == (False, "HTTP 302", 1)
