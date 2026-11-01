"""Secret and personal data samples, built by concatenation so hygiene checks stay green."""

from __future__ import annotations

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
ESCAPED_KEY_ANSWER = '{"' + '\\"TEN' + "\\\\u002dB-0042" + '\\"": 1}'
