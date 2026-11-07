import importlib.util
import json
import shutil
import sys
import urllib.error
import urllib.request

import pytest
import yaml

from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import write_env_file
from support.databases import BANK_SCHEMA, SqliteFile, sqlite_url
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
    db_url = sqlite_url(db_path)
    write_env_file(tmp_path, ".env", DB_URL=db_url)
    return str(tmp_path / "experiment.yaml"), db_url


@pytest.mark.smoke
def test_real_reset_and_delta_check(database_example, run_cli):
    path, _ = database_example
    result = run_cli("run", path, "--tags", "smoke")
    assert_exit(result.code, 2)
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


def load_reset_db():
    spec = importlib.util.spec_from_file_location(
        "reset_db_example", EXAMPLES / "database" / "reset_db.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "url",
    ["sqlite:///seed.db?mode=ro", "sqlite:///seed.db?", "postgresql://127.0.0.1/app", "sqlite:///"],
    ids=["query", "empty-query", "other-driver", "no-path"],
)
def test_reset_db_accepts_only_a_plain_sqlite_path(url, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_URL", url)
    assert load_reset_db().main() == 2
    assert "DB_URL must be sqlite:/// followed by a plain file path" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_reset_db_recreates_the_table(tmp_path, monkeypatch):
    target = tmp_path / "seed.db"
    monkeypatch.setenv("DB_URL", sqlite_url(target))
    assert load_reset_db().main() == 0
    assert target.is_file()


def post(url, message):
    request = urllib.request.Request(
        url,
        data=json.dumps({"message": message}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_toy_agent_answers_500_when_it_cannot_write(toy_db_agent):
    url, _ = toy_db_agent
    assert post(url, "crea un ajuste") == 500


def test_toy_agent_answers_200_when_it_can_write(toy_db_agent):
    url, db_path = toy_db_agent
    SqliteFile(db_path).run(*BANK_SCHEMA)
    assert post(url, "crea un ajuste") == 200
