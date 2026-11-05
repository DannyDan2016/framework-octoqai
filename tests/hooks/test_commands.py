import sys

import pytest

from octoqai.experiment import HooksConfig
from octoqai.hooks import HookEnvironment, HookExecutor

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
