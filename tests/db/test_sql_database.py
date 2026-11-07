from urllib.parse import quote_plus

import pytest
from sqlalchemy.exc import DataError, IntegrityError, OperationalError, ProgrammingError

from octoqai.db import QueryResult, SqlDatabase, connection_secrets
from support import samples
from support.assertions import assert_config_error, assert_linear_growth

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
    assert result.error == "OperationalError" and result.rows == ()


@pytest.mark.security
def test_error_never_contains_parameters(sql_database):
    result = sql_database.query("SELECT * FROM nope WHERE x = :x", {"x": "TEN-B-0042"})
    assert "TEN-B-0042" not in (result.error or "")


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
        (ProgrammingError("SELECT", {}, Exception(NOPE_TABLE)), "ProgrammingError"),
        (OperationalError("SELECT", {}, Exception(samples.PG_DATA_ERROR)), "OperationalError"),
    ],
    ids=["data-error", "integrity-error", "other", "operational"],
)
def test_error_gives_only_the_class_whatever_the_driver_says(
    error, expected, sql_database, monkeypatch
):
    fail_with(sql_database, monkeypatch, error)
    assert sql_database.query("SELECT 1", {}).error == expected
    assert sql_database.execute(["DELETE FROM ajustes"]) == type(error).__name__


@pytest.mark.security
def test_error_without_a_driver_exception_gives_only_the_class(sql_database):
    result = sql_database.query("SELECT * FROM ajustes WHERE id = :ssSecretValue99", {})
    assert result.error == "StatementError"
    assert sql_database.execute(["DELETE FROM ajustes WHERE id = :ssSecretValue99"]) == (
        "StatementError"
    )


def test_query_without_a_result_set_is_an_error_and_never_commits(sqlite_db, sql_database):
    sqlite_db.insert()
    result = sql_database.query("WITH t AS (SELECT 1) DELETE FROM ajustes", {})
    assert result == QueryResult(error="query returned no result set")
    assert sqlite_db.scalar("SELECT count(*) FROM ajustes") == 1


@pytest.mark.security
@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("DELETE FROM ajustes WHERE cuenta = 'ab'" + "QWERTYPART99'", "OperationalError"),
        ("DELETE FROM ajustes WHERE cuenta = abcd-efgh-" + "KLMNSECRET", "OperationalError"),
        ("DELETE FROM ajustes; DELETE FROM ajustes", "ProgrammingError"),
        ("SELECT '\ud800" + "QWERTYPART99'", "UnicodeEncodeError"),
    ],
    ids=["quote", "unquoted", "multi-statement", "non-driver-exception"],
)
def test_execute_error_is_exactly_the_exception_class(statement, expected, sql_database):
    assert sql_database.execute([statement]) == expected


@pytest.mark.security
@pytest.mark.parametrize("name", samples.PASSWORD_SAMPLES)
def test_connection_secrets_finds_a_password_urlsplit_cannot_read(name):
    password = samples.PASSWORD_SAMPLES[name]
    found = connection_secrets(samples.db_url_with(password))
    assert password in found


@pytest.mark.security
def test_connection_secrets_covers_the_text_before_the_first_and_the_last_at_sign():
    found = connection_secrets(samples.db_url_with(samples.AT_PASSWORD))
    assert {"zq7", samples.AT_PASSWORD} <= set(found)


@pytest.mark.security
def test_connection_secrets_gives_every_form_of_a_percent_encoded_password():
    found = connection_secrets(samples.DB_URL_WITH_ENCODED_SECRETS)
    assert {samples.DB_SECRET_PASSWORD, samples.DB_SECRET_PASSWORD_ENCODED} <= set(found)


@pytest.mark.security
def test_connection_secrets_takes_values_of_sensitive_query_keys_only():
    found = connection_secrets(samples.DB_URL_WITH_ENCODED_SECRETS)
    assert {samples.DB_SSL_SECRET, samples.DB_KEY_SECRET} <= set(found)
    assert "octoqai" not in found


@pytest.mark.security
def test_connection_secrets_of_a_url_the_parser_rejects_still_returns_raw_candidates():
    found = connection_secrets(samples.DB_URL_BAD_PORT)
    assert "p@x:" + samples.DB_PASSWORD in found


@pytest.mark.security
@pytest.mark.parametrize("url", ["postgresql://qa" + "@db.example/app", "sqlite:///a.db", "", "::"])
def test_connection_secrets_without_a_password_is_empty_and_never_raises(url):
    assert connection_secrets(url) == []


@pytest.mark.security
def test_connection_secrets_has_no_empty_or_repeated_value():
    found = connection_secrets(samples.DB_URL_WITH_ENCODED_SECRETS)
    assert "" not in found and len(found) == len(set(found))


@pytest.mark.security
def test_query_error_with_driver_text_equals_the_class_name(sql_database, monkeypatch):
    driver = Exception("Could not decode to UTF-8 column 'cuenta' with text '" + "STORED99'")
    fail_with(sql_database, monkeypatch, OperationalError("SELECT", {}, driver))
    assert sql_database.query("SELECT 1", {}) == QueryResult(error="OperationalError")


@pytest.mark.security
def test_query_with_a_lone_surrogate_in_a_parameter_is_an_error_result(sql_database):
    result = sql_database.query("SELECT :p", {"p": "ab\ud800cd"})
    assert result == QueryResult(error="UnicodeEncodeError")


@pytest.mark.security
def test_query_does_not_swallow_a_base_exception(sql_database, monkeypatch):
    fail_with(sql_database, monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        sql_database.query("SELECT 1", {})


@pytest.mark.security
@pytest.mark.parametrize("secret", samples.HARD_SECRETS.values(), ids=samples.HARD_SECRETS)
def test_a_secret_in_the_statement_never_reaches_the_error(secret, sql_database):
    statement = f'SELECT * FROM "{secret}"'
    errors = [sql_database.query(statement, {}).error, sql_database.execute([statement])]
    assert all(e is not None and e.isidentifier() and e.endswith("Error") for e in errors)


@pytest.mark.security
@pytest.mark.parametrize(
    "password", samples.SURROGATE_PASSWORDS.values(), ids=samples.SURROGATE_PASSWORDS
)
def test_connection_secrets_with_a_lone_surrogate_never_raises(password):
    found = connection_secrets(samples.db_url_with(password))
    assert password in found


@pytest.mark.security
def test_connection_secrets_reads_the_pw_query_key():
    assert samples.PW_SECRET in connection_secrets(samples.PW_URL)


@pytest.mark.security
def test_connection_secrets_reads_the_password_inside_odbc_connect():
    found = connection_secrets(samples.ODBC_URL)
    assert samples.ODBC_PASSWORD in found
    assert "{" + samples.ODBC_PASSWORD + "}" in found
    assert any(samples.ODBC_PASSWORD in value and "DRIVER=" in value for value in found)


@pytest.mark.security
def test_connection_secrets_keeps_the_rest_of_a_query_after_an_unencoded_ampersand():
    found = connection_secrets(samples.AMPERSAND_URL)
    assert {"abcd", "abcd&efgh12"} <= set(found)


@pytest.mark.security
def test_connection_secrets_takes_the_short_cut_when_the_parser_rejects_the_url():
    found = connection_secrets(samples.DB_URL_BAD_PORT)
    assert {"p", "p@x:" + samples.DB_PASSWORD} <= set(found)


@pytest.mark.security
def test_connection_secrets_takes_the_text_before_the_first_at_sign_alone_when_it_is_the_only_one():
    found = connection_secrets(samples.SHORT_CUT_URL)
    assert "ab" in found


@pytest.mark.security
@pytest.mark.parametrize("name", samples.BRACED_ODBC_PASSWORDS)
def test_connection_secrets_reads_a_braced_odbc_password_whole(name):
    password, written = samples.BRACED_ODBC_PASSWORDS[name]
    assert password in connection_secrets(samples.braced_odbc_url(written))


@pytest.mark.security
def test_connection_secrets_reads_a_braced_odbc_password_followed_by_more_pairs():
    url = samples.braced_odbc_url("abcd;efgh12") + "&x=1"
    assert "abcd;efgh12" in connection_secrets(url)
    inner = "DRIVER=x;PW" + "D={abcd;efgh12};UID={qa;qa}"
    assert "abcd;efgh12" in connection_secrets("mssql+pyodbc:///?odbc_connect=" + quote_plus(inner))


@pytest.mark.security
@pytest.mark.parametrize(
    "unit", ["}}", ";a={", ";a={}}", "{}}"], ids=["pairs", "opens", "mixed", "open"]
)
def test_connection_secrets_with_hostile_braces_is_linear(unit):
    def make(n):
        return "mssql+pyodbc:///?odbc_connect=" + quote_plus("PW" + "D={" + unit * n)

    assert_linear_growth(connection_secrets, make, 2000, factor=8, bound=24)


@pytest.mark.security
@pytest.mark.parametrize("name", samples.AMPERSAND_ROWS)
def test_connection_secrets_keeps_every_prefix_of_a_tail_ending_before_an_ampersand(name):
    url, password = samples.AMPERSAND_ROWS[name]
    assert {"abcd", password} <= set(connection_secrets(url))


@pytest.mark.security
def test_connection_secrets_bounds_the_candidates_of_a_url_with_many_ampersands():
    url = "postgresql" + "://qa" + "@db.example/app?pass" + "word=" + "&".join(["ab"] * 20000)
    found = connection_secrets(url)
    assert sum(len(value) for value in found) < 40 * len(url)


@pytest.mark.security
@pytest.mark.parametrize("key", samples.CREDENTIAL_KEYS.values(), ids=samples.CREDENTIAL_KEYS)
def test_connection_secrets_reads_the_credential_and_auth_keys(key):
    value = "cv-" + "4417x"
    query = quote_plus(f"DRIVER=x;{key}={value};")
    assert value in connection_secrets(
        "postgresql" + "://qa" + "@db.example/app?" + f"{key}={value}"
    )
    assert value in connection_secrets("mssql+pyodbc:///?odbc_connect=" + query)


@pytest.mark.security
@pytest.mark.parametrize("key", samples.METHOD_KEYS)
def test_connection_secrets_ignores_keys_that_only_name_an_authentication_method(key):
    query = quote_plus(f"DRIVER=x;{key}=admin;")
    assert connection_secrets("postgresql" + "://qa" + "@db.example/app?" + f"{key}=admin") == []
    assert "admin" not in connection_secrets("mssql+pyodbc:///?odbc_connect=" + query)


@pytest.mark.security
@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///:memory:", "sqlite:///"])
def test_in_memory_sqlite_is_rejected_with_a_fixed_message(url):
    message = assert_config_error(
        lambda: SqlDatabase(url),
        r"^in-memory SQLite cannot be used: each check opens a new connection; use a file$",
    )
    assert url not in message
