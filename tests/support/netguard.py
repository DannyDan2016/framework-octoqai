"""Hermetic network guard: tests may talk to loopback only, never to the outside world."""

from __future__ import annotations

import asyncio.proactor_events
import asyncio.selector_events
import ipaddress
import socket
from collections.abc import Callable
from typing import Any

import pytest

PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY",
)

LOOPBACK_NAMES = frozenset({"localhost"})
UNRESOLVED = frozenset({"", None})


def _is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


def _host_of(address: Any) -> Any:
    if isinstance(address, tuple):
        host = address[0]
        return host.decode("ascii", "replace") if isinstance(host, bytes) else host
    return None


def _refuse(target: object) -> None:
    pytest.fail(
        f"external network access is blocked in tests: {target!r}; "
        "use FakeLlm, ScriptedUser or a fake transport, or a server on 127.0.0.1",
        pytrace=False,
    )


def _check_address(address: Any) -> None:
    if not isinstance(address, tuple):
        return
    host = _host_of(address)
    if not isinstance(host, str) or not _is_loopback(host):
        _refuse(address)


def _check_host(host: Any) -> None:
    if host in UNRESOLVED:
        return
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str) or not _is_loopback(host):
        _refuse(host)


def _guard_method(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        _check_address(address)
        return original(self, address, *args, **kwargs)

    return guarded


def _guard_loop_method(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: Any, sock: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        _check_address(address)
        return original(self, sock, address, *args, **kwargs)

    return guarded


def _guard_resolver(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check_host(host)
        return original(host, *args, **kwargs)

    return guarded


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block non-loopback connections and clear the provider credentials from the environment."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for method in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, _guard_method(getattr(socket.socket, method)))
    monkeypatch.setattr(socket, "getaddrinfo", _guard_resolver(socket.getaddrinfo))
    for loop in (
        asyncio.selector_events.BaseSelectorEventLoop,
        asyncio.proactor_events.BaseProactorEventLoop,
    ):
        monkeypatch.setattr(loop, "sock_connect", _guard_loop_method(loop.sock_connect))
