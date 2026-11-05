import pytest
from sqlalchemy.exc import DataError, IntegrityError, ProgrammingError

from octoqai.db import QueryResult, SqlDatabase, scrub_error
from support import samples
from support.assertions import assert_config_error

pytestmark = [pytest.mark.integration, pytest.mark.regression]

NOPE_TABLE = 'relation "nope" does not exist' + chr(10) + "LINE 1"


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


def fail_with(database, monkeypatch, error):
    def _raise(*args, **kwargs):
        raise error

    monkeypatch.setattr(database._engine, "connect", _raise)


@pytest.mark.security
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DataError("SELECT", {}, Exception(samples.PG_DATA_ERROR)), "DataError"),
        (IntegrityError("INSERT", {}, Exception(samples.MYSQL_DUPLICATE)), "IntegrityError"),
        (
            ProgrammingError("SELECT", {}, Exception(NOPE_TABLE)),
            'ProgrammingError: relation "nope" does not exist',
        ),
    ],
    ids=["data-error", "integrity-error", "other-keeps-first-line"],
)
def test_error_gives_only_the_class_when_the_driver_may_echo_a_stored_value(
    error, expected, sql_database, monkeypatch
):
    fail_with(sql_database, monkeypatch, error)
    assert sql_database.query("SELECT 1", {}).error == expected
    assert sql_database.execute(["DELETE FROM ajustes"]) == expected


def test_error_is_cut_to_200_characters(sql_database):
    result = sql_database.query(f'SELECT * FROM "{"t" * 300}"', {})
    assert result.error is not None and len(result.error) == 200
    assert result.error.startswith("OperationalError: no such table: ttt")


@pytest.mark.security
def test_error_without_a_driver_exception_gives_only_the_class(sql_database):
    result = sql_database.query("SELECT * FROM ajustes WHERE id = :ssSecretValue99", {})
    assert result.error == "StatementError"
    assert sql_database.execute(["DELETE FROM ajustes WHERE id = :ssSecretValue99"]) == (
        "StatementError"
    )


@pytest.mark.security
@pytest.mark.parametrize("secret", samples.HARD_SECRETS.values(), ids=samples.HARD_SECRETS)
def test_secrets_are_scrubbed_before_the_message_is_cut(secret, sqlite_db):
    database = SqlDatabase(sqlite_db.url, secrets=[secret])
    try:
        error = database.execute([f'SELECT * FROM "{secret}"'])
    finally:
        database.close()
    assert error is not None and secret not in error


@pytest.mark.security
def test_scrub_error_masks_every_form_of_connection_values():
    message = " | ".join(
        [
            samples.DB_SECRET_PASSWORD,
            samples.DB_SECRET_PASSWORD_ENCODED,
            samples.DB_SSL_SECRET,
            samples.DB_KEY_SECRET,
            "port 5432",
            f"resolved {samples.IPV4} and {samples.IPV6} and [::1]",
            "kept: runner-ok",
        ]
    )
    cleaned = scrub_error(message, samples.DB_URL_WITH_ENCODED_SECRETS)
    for hidden in (
        samples.DB_SECRET_PASSWORD,
        samples.DB_SECRET_PASSWORD_ENCODED,
        samples.DB_SSL_SECRET,
        samples.DB_KEY_SECRET,
        "5432",
        samples.IPV4,
        "2001:db8",
        "::1",
    ):
        assert hidden not in cleaned
    assert "kept: runner-ok" in cleaned


@pytest.mark.security
def test_scrub_error_masks_the_given_secrets_longest_first():
    cleaned = scrub_error(
        "a tok-12345-long b tok-12345 c", "sqlite:///x.db", secrets=["tok-12345", "tok-12345-long"]
    )
    assert cleaned == "a [redacted] b [redacted] c"


def test_query_without_a_result_set_is_an_error_and_never_commits(sqlite_db, sql_database):
    sqlite_db.insert()
    result = sql_database.query("WITH t AS (SELECT 1) DELETE FROM ajustes", {})
    assert result == QueryResult(error="query returned no result set")
    assert sqlite_db.scalar("SELECT count(*) FROM ajustes") == 1
