import datetime
import os
import string
import subprocess
import sys

import pytest

from octoqai.experiment import Hook, HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor, HookResult
from support.agents import FakeClock
from support.databases import FakeDatabase
from support.hooks import FakeCommands, fake_program

pytestmark = [pytest.mark.unit, pytest.mark.regression]

RESET = ["python", "reset.py", "--tenant", "a && del *"]
DB_ENV = {"DB_URL": "sqlite:///a.db"}
NOT_FOUND = "cannot start the command: program not found in PATH"


def on_path(path):
    return {**DB_ENV, "PATH": str(path)}


def executor(stages, *, commands=None, agent=None, clock=None, env=None, **environment):
    hooks = stages if isinstance(stages, HooksConfig) else HooksConfig.model_validate(stages)
    seams = {"run_command": commands or FakeCommands(), "clock": clock or FakeClock()}
    if agent is not None:
        seams["transport"] = agent.transport()
    variables = DB_ENV if env is None else env
    return HookExecutor(hooks, HookEnvironment(env=variables, **environment), **seams)


@pytest.fixture
def bin_dir(tmp_path):
    folder = tmp_path / "bin"
    for name in ("python", "a", "b", "tool"):
        fake_program(folder, name)
    return folder


@pytest.mark.smoke
@pytest.mark.security
def test_command_runs_without_shell_in_the_experiment_folder(tmp_path, bin_dir):
    commands = FakeCommands()
    results = executor(
        {"before_trial": [{"command": RESET}]},
        commands=commands,
        base_dir=tmp_path,
        env=on_path(bin_dir),
    ).run("before_trial")
    assert results == [HookResult("before_trial", 0, "command", True, 0.0, None)]
    [(args, kwargs)] = commands.calls
    assert args == [str(fake_program(bin_dir, "python")), *RESET[1:]]
    assert (kwargs["shell"], kwargs["cwd"], kwargs["timeout"]) == (False, tmp_path, 30.0)
    assert kwargs["stdout"] == kwargs["stderr"] == kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"] == {**DB_ENV, "PATH": str(bin_dir)}


WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows path syntax")
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path syntax")


def program_run(program, base_dir, **environment):
    commands = FakeCommands()
    stages = {"before_trial": [{"command": [program, "--flag"]}]}
    hooks = executor(stages, commands=commands, base_dir=base_dir, **environment)
    [outcome] = hooks.run("before_trial")
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


def batch_hook(program):
    hook = Hook.model_construct(command=[program, "--flag"], http=None, sql=None, timeout_s=None)
    return HooksConfig.model_construct(before_trial=[hook])


@pytest.mark.security
class TestBareProgramLookup:
    @pytest.mark.parametrize("base", ["experiment", None], ids=["with-folder", "without-folder"])
    def test_found_only_in_absolute_path_folders_in_order(self, base, tmp_path, monkeypatch):
        cwd = tmp_path / "cwd"
        fake_program(cwd, "tool")
        fake_program(cwd / "bin", "tool")
        monkeypatch.chdir(cwd)
        first, second = tmp_path / "first", tmp_path / "second"
        expected = fake_program(second, "tool")
        fake_program(tmp_path / "third", "tool")
        relative = ["", ".", "bin"] + ([cwd.drive + "bin"] if sys.platform == "win32" else [])
        folders = [*relative, str(first), str(second), str(tmp_path / "third")]
        first.mkdir()
        base_dir = tmp_path / base if base else None
        outcome, [(args, _)] = program_run("tool", base_dir, env=on_path(os.pathsep.join(folders)))
        assert outcome.ok
        assert args == [str(expected), "--flag"]

    @pytest.mark.parametrize(
        "program",
        ["python", "a", pytest.param("sub\\tool", marks=POSIX_ONLY)],
        ids=["interpreter", "short", "posix-backslash-is-a-name"],
    )
    def test_bare_name_resolves_to_the_absolute_program(self, program, tmp_path):
        expected = fake_program(tmp_path / "bin", program)
        _, [(args, _)] = program_run(program, tmp_path, env=on_path(tmp_path / "bin"))
        assert args[0] == str(expected)

    def test_dotted_name_keeps_its_dots(self, tmp_path):
        expected = fake_program(tmp_path / "bin", "python3.12")
        _, [(args, _)] = program_run("python3.12", tmp_path, env=on_path(tmp_path / "bin"))
        assert args[0] == str(expected)

    def test_folder_named_like_the_program_is_skipped(self, tmp_path):
        decoy = fake_program(tmp_path / "bin", "tool")
        decoy.unlink()
        decoy.mkdir()
        expected = fake_program(tmp_path / "later", "tool")
        folders = os.pathsep.join([str(tmp_path / "bin"), str(tmp_path / "later")])
        _, [(args, _)] = program_run("tool", tmp_path, env=on_path(folders))
        assert args[0] == str(expected)

    @pytest.mark.parametrize(
        "env",
        [on_path(""), DB_ENV, on_path("bin")],
        ids=["empty-path", "env-without-path", "only-relative"],
    )
    def test_missing_program_never_starts(self, env, tmp_path, bin_dir, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", str(bin_dir))
        outcome, calls = program_run("tool", tmp_path, env=env)
        assert (outcome.ok, outcome.error, calls) == (False, NOT_FOUND, [])

    def test_empty_hook_env_uses_the_process_path(self, tmp_path, bin_dir, monkeypatch):
        monkeypatch.setenv("PATH", str(bin_dir))
        _, [(args, kwargs)] = program_run("tool", tmp_path, env={})
        assert (args[0], kwargs["env"]) == (str(fake_program(bin_dir, "tool")), None)

    @pytest.mark.parametrize(
        "program",
        ["to\x00ol", "t" * 5000, ".", "..", "tool.", "secret-tool-77"],
        ids=["nul", "long", "dot", "dot-dot", "trailing-dot", "unknown"],
    )
    def test_hostile_or_unknown_name_is_a_fixed_error(self, program, tmp_path, bin_dir):
        outcome, calls = program_run(program, tmp_path, env=on_path(bin_dir))
        assert (outcome.ok, outcome.error, calls) == (False, NOT_FOUND, [])

    @pytest.mark.parametrize("program", ["", "   "], ids=["empty", "blank"])
    def test_empty_name_never_matches_a_bare_suffix_file(self, program, tmp_path, bin_dir):
        (bin_dir / ".exe").write_bytes(b"")
        outcome, calls = program_run(program, tmp_path, env=on_path(bin_dir))
        assert (outcome.ok, outcome.error, calls) == (False, NOT_FOUND, [])

    @WINDOWS_ONLY
    def test_uppercase_exe_suffix_is_not_doubled(self, tmp_path, bin_dir):
        _, [(args, _)] = program_run("TOOL.EXE", tmp_path, env=on_path(bin_dir))
        assert args[0] == str(bin_dir / "TOOL.EXE")

    @WINDOWS_ONLY
    @pytest.mark.parametrize("key", ["Path", "path"])
    def test_path_key_is_case_insensitive_on_windows(self, key, tmp_path, bin_dir):
        _, [(args, _)] = program_run("tool", tmp_path, env={**DB_ENV, key: str(bin_dir)})
        assert args[0] == str(bin_dir / "tool.exe")

    @WINDOWS_ONLY
    def test_last_path_spelling_wins_like_in_the_child(self, tmp_path, bin_dir):
        env = {"PATH": str(tmp_path / "empty"), "Path": str(bin_dir)}
        _, [(args, _)] = program_run("tool", tmp_path, env=env)
        assert args[0] == str(bin_dir / "tool.exe")

    @WINDOWS_ONLY
    @pytest.mark.parametrize("program", ["reset.bat", "RESET.CMD", "reset.bat. "])
    def test_batch_file_is_never_looked_up(self, program, tmp_path):
        fake_program(tmp_path / "bin", program.rstrip(". "))
        commands = FakeCommands()
        hooks = executor(batch_hook(program), commands=commands, env=on_path(tmp_path / "bin"))
        [outcome] = hooks.run("before_trial")
        assert outcome.error == "cannot start the command: only .exe programs are looked up in PATH"
        assert commands.calls == []

    @POSIX_ONLY
    def test_file_without_execute_permission_is_skipped(self, tmp_path):
        fake_program(tmp_path / "first", "tool", mode=0o644)
        expected = fake_program(tmp_path / "second", "tool")
        folders = os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        _, [(args, _)] = program_run("tool", tmp_path, env=on_path(folders))
        assert args[0] == str(expected)


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
def test_command_failures(commands, error, bin_dir):
    stages = {"before_run": [{"command": RESET, "timeout_s": 5}]}
    [outcome] = executor(stages, commands=commands, env=on_path(bin_dir)).run("before_run")
    assert (outcome.ok, outcome.error) == (False, error)


def test_before_stage_stops_at_first_failure(bin_dir):
    commands = FakeCommands().exits(1, 0)
    stages = {"before_trial": [{"command": ["a"]}, {"command": ["b"]}]}
    assert len(executor(stages, commands=commands, env=on_path(bin_dir)).run("before_trial")) == 1
    assert len(commands.calls) == 1


def test_after_stage_runs_every_hook(bin_dir):
    commands = FakeCommands().exits(1, 0)
    stages = {"after_trial": [{"command": ["a"]}, {"command": ["b"]}]}
    results = executor(stages, commands=commands, env=on_path(bin_dir)).run("after_trial")
    assert [r.ok for r in results] == [False, True]


def test_duration_comes_from_the_injected_clock(bin_dir):
    clock = FakeClock()
    commands = FakeCommands(clock=clock, took_s=0.25)
    stages = {"after_run": [{"command": ["a"]}]}
    [outcome] = executor(stages, commands=commands, clock=clock, env=on_path(bin_dir)).run(
        "after_run"
    )
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
