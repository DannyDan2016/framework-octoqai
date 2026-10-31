import os

import pytest

from octoqai.experiment.envfiles import resolve_env
from support.builders import write_env_file

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def test_process_env_wins_over_env_files(tmp_path):
    write_env_file(tmp_path, ".env", AGENT_URL="http://base.local")
    write_env_file(tmp_path, ".env.qa", AGENT_URL="http://qa.local")
    resolved = resolve_env(tmp_path, "qa", {"AGENT_URL": "http://ci.local"})
    assert resolved["AGENT_URL"] == "http://ci.local"


@pytest.mark.smoke
def test_env_specific_file_wins_over_base(tmp_path):
    write_env_file(tmp_path, ".env", AGENT_URL="http://base.local", ONLY_BASE="1")
    write_env_file(tmp_path, ".env.qa", AGENT_URL="http://qa.local")
    resolved = resolve_env(tmp_path, "qa", {})
    assert resolved == {"AGENT_URL": "http://qa.local", "ONLY_BASE": "1"}


def test_missing_files_are_ignored(tmp_path):
    assert resolve_env(tmp_path, "qa", {"X": "1"}) == {"X": "1"}


def test_without_env_only_base_file_is_read(tmp_path):
    write_env_file(tmp_path, ".env", A="base")
    write_env_file(tmp_path, ".env.qa", A="qa")
    assert resolve_env(tmp_path, None, {}) == {"A": "base"}


@pytest.mark.security
def test_does_not_touch_os_environ(tmp_path):
    write_env_file(tmp_path, ".env", OCTOQAI_ENVFILE_PROBE="1")
    before = dict(os.environ)
    resolve_env(tmp_path, None, os.environ)
    assert dict(os.environ) == before
    assert "OCTOQAI_ENVFILE_PROBE" not in os.environ
