"""Secret and personal data samples, built by concatenation so hygiene checks stay green."""

from __future__ import annotations

import json
from urllib.parse import quote_plus

SECRET_SAMPLES: dict[str, str] = {
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "aws access key": "AKIA" + "Q" * 16,
    "github token": "ghp_" + "a1" * 18,
    "slack token": "xoxb-" + "1234567890-abc",
    "llm api key": "sk-ant-" + "x" * 30,
    "google api key": "AIza" + "B" * 35,
    "jwt": "eyJ" + "h" * 12 + ".eyJ" + "p" * 12 + ".sig",
    "url with credentials": "postgres://admin:hunter2" + "@db.internal/x",
    "bearer token": "Bearer " + "t" * 24,
    "credential assignment": "password" + "=hunter22",
}
SPLIT_SECRET = ("sk-ant-", "x" * 30)
EMAIL = "ana" + "@" + "tenant-b.example"
SUPPORT_EMAIL = "soporte" + "@" + "banco.example"
CARD = "4111 1111 " + "1111 1111"
CARD_BAD_LUHN = "4111 1111 " + "1111 1112"
IBAN = "GB82 WEST " + "1234 5698 7654 32"
IBAN_BAD_CHECKSUM = "GB82 WEST " + "1234 5698 7654 33"
CARD_WITH_TRAILING_DIGITS = CARD + " 22"
LOWERCASE_BEARER = "Authorization: bearer " + "t" * 24
DATABASE_URL = "postgres://admin:hunter2" + "@db.internal/prod"
CARD_DASHED_WITH_TRAILING = "4111-1111-" + "1111-1111" + "-22"
CARD_DIGITS_RUN = "4111111111111111" + "222"
LONG_DIGITS_RUN = "4111111111111111" + "2222"
URL_AFTER_DIGIT = "1" + DATABASE_URL
URL_AFTER_DASH = "-" + DATABASE_URL
NON_ASCII_CANARY = "contrase" + "ña-ultra"
SLASH_CANARY = "ZETA</" + "x>KEY"
ESCAPED_DATABASE_URL = DATABASE_URL.replace("/", "\\/")
JWT_RUN = "eyJ" + "a-"
JWT_SHAPED_PARTS = ("eyJ", "a" * 5, "a" * 10, "-", ".", "_", " ", "é", ".eyJ", "e", "J")
JWT_PART_WEIGHTS = (3, 1, 3, 2, 2, 1, 1, 1, 3, 1, 1)
SHORT_DOMAIN = "@" + "x.yz"
HOSTILE_JSON = {
    "huge-int": '{"answer": ' + "1" * 5000 + ', "x": "TEN-B-0042"}',
    "too-deep": "[" * 100_000 + '"TEN-B-0042"' + "]" * 100_000,
}
ESCAPED_KEY = '"TEN' + "\\" + 'u002dB-0042"'
WEAK_CREDENTIAL = "pw" + "d=abcd"
ESCAPED_KEY_ANSWERS = {
    "alone": json.dumps({ESCAPED_KEY: 1}),
    "credential-key": json.dumps({ESCAPED_KEY: 1, WEAK_CREDENTIAL: 2}),
    "credential-value": json.dumps({ESCAPED_KEY: 1, "x": WEAK_CREDENTIAL}),
}
ESCAPED_SECRETS = {
    "canary-dash": ("TEN-B-0042", "TEN" + "\\" + "u002dB-0042", "canary"),
    "canary-quoted": ("TEN-B-0042", '\\"' + "\\" + 'u0054EN-B-0042\\"', "canary"),
    "canary-slash": (SLASH_CANARY, SLASH_CANARY.replace("/", "\\/"), "canary"),
    "email": (EMAIL, EMAIL.replace("@", "\\" + "u0040"), "pii"),
    "credential-url": ("hunter2", ESCAPED_DATABASE_URL, "secret"),
    "credential-assignment": ("hunter22", "password" + "\\" + "u003dhunter22", "secret"),
}
ESCAPED_WRAPPERS = {
    "trailing-garbage": '{"a": "' + "{}" + '"} xx',
    "truncated": '{"a": "' + "{}",
    "huge-int": '{"n": ' + "1" * 5000 + ', "a": "' + "{}" + '"}',
    "too-deep": "[" * 100_000 + '"' + "{}" + '"' + "]" * 100_000,
    "prose": "dato " + "{}" + " fin",
}
ESCAPED_LEAKS = {
    f"{wrapper}-{secret}": (value, shape.replace("{}", escaped), kind)
    for wrapper, shape in ESCAPED_WRAPPERS.items()
    for secret, (value, escaped, kind) in ESCAPED_SECRETS.items()
}
CLEAN_ESCAPED = r"linea\nuna \u00f1and\u00fa \/ruta\/ \"citado\" \\ \ud83d\ude00 fin \u"
DB_PASSWORD = "s3cr3t-" + "pw-77"
DB_URL_WITH_PASSWORD = "nodialect" + "://qa:" + DB_PASSWORD + "@db.example/app"
DB_URL_PARTIAL_VARIABLE = "postgresql" + "://qa:" + DB_PASSWORD + "@db.example/${DB_NAME}"
DB_URL_BAD_PORT = "postgresql" + "://qa:p@x:" + DB_PASSWORD + "@db.example:5432/app"
PG_URL_WITH_PASSWORD = "postgresql" + "://qa:" + DB_PASSWORD + "@db.example/app"
DB_SECRET_PASSWORD = "Sup3r-" + "s3cret!"
DB_SECRET_PASSWORD_ENCODED = "Sup3r%2D" + "s3cret%21"
DB_SSL_SECRET = "qwerty-" + "77"
DB_KEY_SECRET = "k-9x" + "7q"
DB_URL_WITH_ENCODED_SECRETS = (
    "postgresql"
    + "://qa:"
    + DB_SECRET_PASSWORD_ENCODED
    + "@db.example:5432/app?sslpass"
    + "word="
    + DB_SSL_SECRET
    + "&sslkey="
    + DB_KEY_SECRET
    + "&application_name=octoqai"
)
HARD_SECRETS = {
    "slash-colon": "/:x" + "9Zk",
    "percent": "%40" + "ab12",
    "braces": "{{ x " + "}}7Qz",
    "long": "L" + "0ng-" * 80 + "zq9Xtl",
    "quote-split": "ab'" + "QWERTYPART99",
    "dash-split": "abcd-" + "efgh-" + "KLMNSECRET",
}
PG_DATA_ERROR = 'invalid input syntax for type integer: "juan ' + 'perez cliente"'
MYSQL_DUPLICATE = "Duplicate entry 'juan " + "perez' for key 'ajustes.cuenta'"
IPV4 = "10.1" + ".2.3"
IPV6 = "[2001:db8" + "::1]"
SLASH_PASSWORD = "kLm/:x9" + "Zk#Qp"
QUESTION_PASSWORD = "mn5?" + "rt6Vb"
AT_PASSWORD = "zq7@" + "wx8Nc"
PASSWORD_SAMPLES = {
    "slash-colon-hash": SLASH_PASSWORD,
    "question": QUESTION_PASSWORD,
    "unencoded-at": AT_PASSWORD,
}
LONE_SURROGATE = chr(0xDC80)
SURROGATE_REPLY = b'{"answer": "ab' + chr(92).encode() + b'udc80cd. TEN-B-0042 fin"}'
SURROGATE_REPLY_CLEAN = b'{"answer": "ab' + chr(92).encode() + b'udc80cd."}'
SURROGATE_PASSWORDS = {
    "after-slash": "kLm/" + LONE_SURROGATE + "x9Zk",
    "plain": "abc" + LONE_SURROGATE + "def",
}
ODBC_PASSWORD = "Od8c" + "-pw-5531"
ODBC_URL = (
    "mssql+pyodbc"
    + ":///?odbc_connect="
    + quote_plus(
        "DRIVER={ODBC Driver 18 for SQL Server};SERVER=db.example;UID=qa;PW"
        + "D={"
        + ODBC_PASSWORD
        + "};"
    )
)
BRACED_ODBC_PASSWORDS = {
    "semicolon": ("abcd;" + "efgh12", "abcd;" + "efgh12"),
    "closing-brace": ("abcd}" + "efgh12", "abcd}}" + "efgh12"),
}


def braced_odbc_url(written: str) -> str:
    """Database URL whose ODBC password is written inside braces, as ``written``."""
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus("DRIVER=x;PW" + "D={" + written + "};")


AMPERSAND_ROWS = {
    "query-then-key": (
        "postgresql" + "://qa" + "@db.example/app?pass" + "word=abcd&efgh12&sslmode=require",
        "abcd&efgh12",
    ),
    "query-then-secret": (
        "postgresql" + "://qa" + "@db.example/app?pass" + "word=abcd&efgh12&token=tok12345",
        "abcd&efgh12",
    ),
    "odbc-raw": (
        "mssql+pyodbc:///?odbc_connect=DRIVER%3Dx%3BPW" + "D%3Dabcd&efgh12",
        "abcd&efgh12",
    ),
}
CREDENTIAL_KEYS = {
    "credential": "credential",
    "auth": "auth",
    "odbc-credential": "Credential",
}
METHOD_KEYS = ["authSource", "authMechanism", "auth_plugin"]
PW_SECRET = "pw-val" + "-4417"
PW_URL = "postgresql" + "://qa" + "@db.example/app?pw=" + PW_SECRET
AMPERSAND_URL = "postgresql" + "://qa" + "@db.example/app?pass" + "word=abcd&efgh12"
SHORT_CUT_URL = "postgresql" + "://qa:ab@c:" + "80" + "@db.example/app"
ESCAPING_PASSWORDS = {
    "quote": 'abcd@ef"' + "gh12",
    "slash": "root@db/" + "abcd",
}


_BACKSLASH = chr(92)
PREFIX_ESCAPE_ROWS = {
    "slash-prefix": (
        "root@db/" + "abcd",
        "x " + _BACKSLASH + "root@db/abcd y root@db" + _BACKSLASH + "/abcd z",
    ),
    "unicode-prefix": (
        'abcd@ef"' + "gh12",
        "x " + _BACKSLASH + 'u006abcd@ef"gh12 y abcd@ef' + _BACKSLASH + '"gh12 z',
    ),
    "double-escape": (
        'abcd@ef"' + "gh12",
        "dato {"
        + _BACKSLASH
        + '"p'
        + _BACKSLASH
        + '": '
        + _BACKSLASH
        + '"abcd@ef'
        + _BACKSLASH * 3
        + '"gh12'
        + _BACKSLASH
        + '"} fin.',
    ),
    "triple-backslash": ('ab"' + "cdef12", "dato ab" + _BACKSLASH * 3 + '"cdef12 fin.'),
    "literal-backslash": (
        "ab" + _BACKSLASH + "ncd12",
        "dato ab" + _BACKSLASH + "ncd12 y ab" + _BACKSLASH * 2 + "ncd12 fin",
    ),
}
COUNTED_COPIES = (
    'abcd@ef"' + "gh12",
    'x abcd@ef"gh12 y abcd@ef' + _BACKSLASH + '"gh12 z',
)


def escaped_reply(password: str) -> str:
    """Prose around a JSON fragment that repeats ``password`` with its JSON escapes."""
    fragment = json.dumps({"p": password}).replace("/", chr(92) + "/")
    return "dato " + fragment + " fin."


def db_url_with(password: str) -> str:
    """Build a database URL whose password is written as is, never percent-encoded."""
    return "postgresql" + "://qa:" + password + "@db.example/app"
