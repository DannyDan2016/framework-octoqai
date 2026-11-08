import subprocess
import sys

import pytest

from octoqai.experiment import HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor
from support.hooks import process_alive, process_tree_code, wait_until

pytestmark = [pytest.mark.integration, pytest.mark.regression]


def run_real(tmp_path, *argv):
    hooks = HooksConfig.model_validate({"before_run": [{"command": [sys.executable, "-c", *argv]}]})
    [outcome] = HookExecutor(hooks, HookEnvironment(base_dir=tmp_path)).run("before_run")
    return outcome


def test_real_exit_code_is_reported(tmp_path):
    assert run_real(tmp_path, "raise SystemExit(3)").error == "exited with code 3"


@pytest.mark.security
def test_metacharacters_reach_the_process_literally(tmp_path):
    code = "import sys, pathlib; pathlib.Path('arg.txt').write_text(sys.argv[1])"
    assert run_real(tmp_path, code, "a && echo pwned | more").ok
    assert (tmp_path / "arg.txt").read_text() == "a && echo pwned | more"


def tree_hook(tmp_path, timeout_s):
    pid_file = tmp_path / "grandchild.pid"
    command = [sys.executable, "-c", process_tree_code(pid_file)]
    hooks = HooksConfig.model_validate(
        {"before_run": [{"command": command, "timeout_s": timeout_s}]}
    )
    return HookExecutor(hooks, HookEnvironment(base_dir=tmp_path)), pid_file


def grandchild_pid(pid_file):
    if not pid_file.exists():
        pytest.fail("the grandchild did not start before the hook stopped")
    return int(pid_file.read_text())


@pytest.mark.security
def test_timeout_kills_the_whole_process_tree(tmp_path):
    executor, pid_file = tree_hook(tmp_path, timeout_s=3)
    [outcome] = executor.run("before_run")
    assert outcome.error == "timed out after 3s"
    pid = grandchild_pid(pid_file)
    assert wait_until(lambda: not process_alive(pid)), "the grandchild survived the timeout"


@pytest.mark.security
def test_interrupt_kills_the_whole_process_tree(tmp_path, monkeypatch):
    executor, pid_file = tree_hook(tmp_path, timeout_s=60)
    original_wait = subprocess.Popen.wait

    def interrupted_wait(self, timeout=None):
        if timeout == 60:
            wait_until(pid_file.exists)
            raise KeyboardInterrupt
        return original_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        executor.run("before_run")
    pid = grandchild_pid(pid_file)
    assert wait_until(lambda: not process_alive(pid)), "the grandchild survived the interrupt"


@pytest.mark.security
def test_argument_the_os_rejects_is_a_failed_result_without_its_content(tmp_path):
    outcome = run_real(tmp_path, "pass", "secret\x00value")
    assert not outcome.ok
    assert outcome.error == "cannot start the command: ValueError"
    assert "secret" not in outcome.error
