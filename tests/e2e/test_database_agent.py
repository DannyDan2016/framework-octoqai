import shutil
import sys

import pytest
import yaml

from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import write_env_file
from support.paths import EXAMPLES

pytestmark = [pytest.mark.e2e, pytest.mark.regression]


@pytest.fixture
def database_example(tmp_path, toy_db_agent):
    url, db_path = toy_db_agent
    shutil.copy(EXAMPLES / "database" / "reset_db.py", tmp_path / "reset_db.py")
    data = yaml.safe_load((EXAMPLES / "database" / "experiment.yaml").read_text(encoding="utf-8"))
    data["agent"]["url"] = url
    data["hooks"]["before_trial"][0]["command"][0] = sys.executable
    (tmp_path / "experiment.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
    )
    write_env_file(tmp_path, ".env", DB_URL=f"sqlite:///{db_path.as_posix()}")
    return str(tmp_path / "experiment.yaml"), f"sqlite:///{db_path.as_posix()}"


@pytest.mark.smoke
def test_real_reset_and_delta_check(database_example, run_cli):
    path, _ = database_example
    result = run_cli("run", path, "--tags", "smoke")
    assert result.code != 3, result.err
    trials = result.report()["trials"]
    assert [t["outcome"] for t in trials] == ["pass", "pass"]
    assert all(h["ok"] for t in trials for h in t["hooks"])


@pytest.mark.security
def test_real_forbidden_change_is_no_go_without_printing_the_url(database_example, run_cli):
    path, db_url = database_example
    result = run_cli("run", path)
    assert_exit(result.code, 1)
    assert "forbidden=2" in result.out
    assert_not_in_outputs(db_url, result.out, result.err, result.report_text())
