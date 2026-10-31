import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CREDENTIALS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws access key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "llm api key": re.compile(r"\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."),
    "url with credentials": re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@\"']+:[^\s/@\"']+@"),
}

NETWORK = {
    "private ipv4": re.compile(
        r"\b(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01])|169\.254)\.\d{1,3}\.\d{1,3}\b"
    ),
    "bind to all interfaces": re.compile(r"\b0\.0\.0\.0\b"),
}

PERSONAL = {
    "windows user path": re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE),
    "unix home path": re.compile(r"(/home/|/Users/)[A-Za-z0-9._-]+"),
}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL = re.compile(r"@users\.noreply\.github\.com$|@example\.(com|org)$")

URL_HOST = re.compile(r"\bhttps?://([^/\s\"'):>]+)")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "a", "www.apache.org", "github.com"}
ALLOWED_HOST_SUFFIXES = (".local", ".example", ".test", ".invalid", "example.com")

SECRET_FILE = re.compile(
    r"(^|/)(\.env(\..+)?|.*\.(pem|key|p12|pfx|jks)|id_(rsa|dsa|ecdsa|ed25519)|\.netrc|\.pypirc)$"
)
ALLOWED_SECRET_FILES = {".env.example"}


def _tracked_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "--cached"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def _texts() -> list[tuple[str, str]]:
    texts = []
    for name in _tracked_files():
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        texts.append((name, data.decode("utf-8", errors="replace")))
    return texts


def _scan(rules: dict[str, re.Pattern[str]], text: str) -> list[str]:
    return [label for label, rule in rules.items() if rule.search(text)]


def _offenders(rules: dict[str, re.Pattern[str]]) -> list[str]:
    found = []
    for name, text in _texts():
        for lineno, line in enumerate(text.splitlines(), 1):
            found += [f"{name}:{lineno} {label}" for label in _scan(rules, line)]
    return found


def _host_allowed(host: str) -> bool:
    host = host.lower()
    return host in ALLOWED_HOSTS or host.endswith(ALLOWED_HOST_SUFFIXES)


def test_no_credentials_in_tracked_files():
    assert _offenders(CREDENTIALS) == []


def test_no_private_network_addresses():
    assert _offenders(NETWORK) == []


def test_no_personal_paths():
    assert _offenders(PERSONAL) == []


def test_only_noreply_emails():
    found = []
    for name, text in _texts():
        found += [f"{name} {m}" for m in EMAIL.findall(text) if not ALLOWED_EMAIL.search(m)]
    assert found == []


def test_urls_point_only_to_safe_hosts():
    found = []
    for name, text in _texts():
        found += [f"{name} {h}" for h in URL_HOST.findall(text) if not _host_allowed(h)]
    assert found == []


def test_no_secret_files_tracked():
    found = [
        name
        for name in _tracked_files()
        if SECRET_FILE.search(name) and Path(name).name not in ALLOWED_SECRET_FILES
    ]
    assert found == []


@pytest.mark.parametrize(
    ("label", "sample"),
    [
        ("private key", "-----BEGIN RSA " + "PRIVATE KEY-----"),
        ("aws access key", "AKIA" + "Q" * 16),
        ("github token", "ghp_" + "a1" * 18),
        ("slack token", "xoxb-" + "1234567890-abc"),
        ("llm api key", "sk-ant-" + "x" * 30),
        ("google api key", "AIza" + "B" * 35),
        ("jwt", "eyJ" + "h" * 12 + ".eyJ" + "p" * 12 + ".sig"),
        ("url with credentials", "postgres://admin:" + "hunter2@db.internal/x"),
    ],
)
def test_credential_detectors_catch_samples(label, sample):
    assert label in _scan(CREDENTIALS, f"value = {sample}")


@pytest.mark.parametrize(
    ("rules", "label", "sample"),
    [
        (NETWORK, "private ipv4", "192.168" + ".1.20"),
        (NETWORK, "private ipv4", "10.0" + ".0.5"),
        (NETWORK, "private ipv4", "172.20" + ".3.4"),
        (NETWORK, "bind to all interfaces", "0.0" + ".0.0:8000"),
        (PERSONAL, "windows user path", "C:" + "\\Users\\someone\\x"),
        (PERSONAL, "unix home path", "/home" + "/someone/x"),
    ],
)
def test_network_and_path_detectors_catch_samples(rules, label, sample):
    assert label in _scan(rules, sample)


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1", True),
        ("agent.local", True),
        ("api.mycompany.internal", False),
        ("evil.com", False),
    ],
)
def test_host_allowlist(host, allowed):
    assert _host_allowed(host) is allowed


def test_secret_file_names_detected():
    for name in [".env", "config/.env.prod", "certs/server.pem", "id_rsa", ".pypirc"]:
        assert SECRET_FILE.search(name), name
