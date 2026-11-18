import asyncio
import importlib
import os
import pkgutil
import socket
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import octoqai
from support import netguard
from support.paths import REPO_ROOT as ROOT
from support.paths import SRC

pytestmark = pytest.mark.quality


def _subpackages() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(octoqai.__path__, "octoqai.") if m.ispkg]


def _installed_script(name: str) -> Path | None:
    folders = (Path(sysconfig.get_path("scripts")), Path(sys.executable).parent)
    suffix = ".exe" if sys.platform == "win32" else ""
    found = [folder / f"{name}{suffix}" for folder in folders]
    return next((path for path in found if path.is_file()), None)


def test_import_contracts_hold():
    exe = _installed_script("lint-imports")
    assert exe is not None, "lint-imports is not installed"
    result = subprocess.run(
        [exe], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_package_exports_public_api():
    packages = _subpackages()
    assert packages, "octoqai has no feature packages"
    for name in packages:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        assert exported, f"{name} does not declare __all__"
        missing = [item for item in exported if not hasattr(module, item)]
        assert missing == [], f"{name}.__all__ lists missing names: {missing}"


@pytest.mark.security
def test_only_hooks_start_processes():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if "subprocess" in p.read_text(encoding="utf-8") and p.parent.name != "hooks"
    ]
    assert offenders == []


@pytest.mark.security
def test_no_shell_in_source():
    needle = "shell" + "=True"
    assert [p.name for p in SRC.rglob("*.py") if needle in p.read_text(encoding="utf-8")] == []


REMOTE_HOSTNAME = "agent.example"
REMOTE_IPV4 = ("93.184.216.34", 80)
REMOTE_IPV6 = ("2001:db8::1", 80, 0, 0)
BLOCKED = pytest.fail.Exception


@pytest.mark.security
def test_no_external_network_in_tests():
    with pytest.raises(BLOCKED, match="external network access is blocked"):
        socket.create_connection((REMOTE_HOSTNAME, 80))


@pytest.mark.security
@pytest.mark.parametrize(
    "attempt",
    [
        lambda: socket.socket().connect((REMOTE_HOSTNAME, 80)),
        lambda: socket.socket().connect(REMOTE_IPV4),
        lambda: socket.socket().connect_ex(REMOTE_IPV4),
        lambda: socket.socket(socket.AF_INET6).connect(REMOTE_IPV6),
        lambda: socket.create_connection(REMOTE_IPV4, timeout=1),
        lambda: socket.getaddrinfo(REMOTE_HOSTNAME, 80),
        lambda: socket.getaddrinfo(REMOTE_HOSTNAME.encode(), 80),
        lambda: asyncio.run(asyncio.open_connection(*REMOTE_IPV4)),
        lambda: asyncio.run(asyncio.open_connection(REMOTE_HOSTNAME, 80)),
    ],
    ids=[
        "connect-name",
        "connect-ipv4",
        "connect_ex",
        "connect-ipv6",
        "create_connection",
        "getaddrinfo",
        "getaddrinfo-bytes",
        "asyncio-ip",
        "asyncio-name",
    ],
)
def test_network_guard_blocks_every_remote_path(attempt):
    with pytest.raises(BLOCKED, match="external network access is blocked"):
        attempt()


@pytest.mark.security
def test_network_guard_lets_loopback_through():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        with socket.create_connection(("localhost", port), timeout=2):
            pass
    assert socket.getaddrinfo("localhost", 80)
    assert socket.getaddrinfo(None, 80)
    assert socket.getaddrinfo("", 80)


@pytest.mark.security
def test_network_guard_lets_asyncio_reach_loopback():
    async def talk() -> None:
        async def accept(reader, writer):
            writer.close()

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        server.close()
        await server.wait_closed()

    asyncio.run(talk())


@pytest.mark.security
def test_network_guard_lets_the_toy_agent_process_serve(toy_sse_agent_url):
    host, port = toy_sse_agent_url.split("/")[2].split(":")
    with socket.create_connection((host, int(port)), timeout=5):
        pass


@pytest.mark.security
def test_network_guard_clears_the_provider_credentials(monkeypatch):
    for name in netguard.PROVIDER_ENV:
        monkeypatch.setenv(name, "value")
    netguard.install(monkeypatch)
    assert [name for name in netguard.PROVIDER_ENV if name in os.environ] == []
