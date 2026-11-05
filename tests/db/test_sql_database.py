import pytest

from octoqai.db import QueryResult, SqlDatabase, scrub_error
from support import samples
from support.assertions import assert_config_error

pytestmark = [pytest.mark.integration, pytest.mark.regression]


@pytest.mark.smoke
def test_query_returns_columns_and_rows(sqlite_db, sql_database):
    sqlite_db.insert(tenant="A", cuenta="001", importe=10.5)
    result = sql_database.query("SELECT cuenta, importe FROM ajustes WHERE tenant = :t", {"t": "A"})
    assert result == QueryResult(("cuenta", "importe"), (("001", 10.5),), None)
    assert result.scalar == "001"


def test_scalar_of_empty_result_is_none(sql_database):
    assert sql_database.query("SELECT id FROM ajustes", {}).scalar is None


@pytest.mark.security
def test_query_never_commits(sqlite_db, sql_database):
    sqlite_db.insert()
    sql_database.query("DELETE FROM ajustes", {})
    assert sqlite_db.scalar("SELECT count(*) FROM ajustes") == 1


def test_execute_runs_all_statements_in_one_transaction(sqlite_db, sql_database):
    error = sql_database.execute(
        ["INSERT INTO ajustes (tenant, cuenta, importe) VALUES ('A', '001', 1)", "INSERT INTO"]
    )
    assert error is not None and error.startswith("OperationalError")
    assert sqlite_db.scalar("SELECT count(*) FROM ajustes") == 0


def test_execute_returns_none_on_success(sqlite_db, sql_database):
    assert sql_database.execute(["DELETE FROM ajustes"]) is None


def test_query_error_is_reported_not_raised(sql_database):
    result = sql_database.query("SELECT * FROM nope", {})
    assert result.error == "OperationalError: no such table: nope" and result.rows == ()


@pytest.mark.security
def test_error_never_contains_parameters(sql_database):
    result = sql_database.query("SELECT * FROM nope WHERE x = :x", {"x": "TEN-B-0042"})
    assert "TEN-B-0042" not in (result.error or "")


@pytest.mark.security
def test_error_masks_a_parameter_the_driver_repeats(sql_database):
    result = sql_database.query('SELECT * FROM "TEN-B-0042" WHERE x = :x', {"x": "TEN-B-0042"})
    assert result.error == "OperationalError: no such table: [redacted]"


@pytest.mark.security
def test_scrub_error_removes_connection_details_and_parameters():
    message = (
        f'connection to server at "DB.example" failed: password {samples.DB_PASSWORD} '
        'for user "qa" on app, input "TEN-B-0042", id 7'
    )
    cleaned = scrub_error(message, samples.PG_URL_WITH_PASSWORD, {"x": "TEN-B-0042", "y": 7})
    assert cleaned == (
        'connection to server at "[redacted]" failed: password [redacted] '
        'for user "[redacted]" on [redacted], input "[redacted]", id 7'
    )


@pytest.mark.security
def test_invalid_url_is_config_error_without_echo():
    assert_config_error(
        lambda: SqlDatabase(samples.DB_URL_WITH_PASSWORD),
        r"database\.url is not a valid SQLAlchemy URL",
        absent=[samples.DB_PASSWORD, "db.example"],
    )


@pytest.mark.security
def test_malformed_port_is_config_error_without_echo():
    assert_config_error(
        lambda: SqlDatabase(samples.DB_URL_BAD_PORT),
        r"database\.url is not a valid SQLAlchemy URL",
        absent=[samples.DB_PASSWORD, "db.example", "dbhost"],
    )
