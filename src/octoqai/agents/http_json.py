"""HTTP adapter that talks to the agent under test."""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from octoqai.agents._delivery import (
    HTTP_ERROR_MIN,
    MAX_ERROR_BODY_BYTES,
    NotDelivered,
    classify_delivery,
    client_timeout,
    read_bounded,
    read_capped,
    text_of,
    with_retries,
)
from octoqai.agents._mapping import (
    MISSING,
    ReplyError,
    ResponseFields,
    compile_path,
    first_match,
    mapping_guard,
)
from octoqai.agents.port import AgentReply
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig, render


class HttpAgentAdapter:
    """Send rendered requests to an agent exposed over HTTP JSON."""

    def __init__(
        self,
        cfg: AgentConfig,
        transport: httpx.BaseTransport | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Create the adapter.

        Args:
            cfg: Agent connection settings.
            transport: Optional httpx transport, used to test without a network.
            clock: Monotonic clock in seconds, injectable for tests.
            sleep: Sleep function used between retries, injectable for tests.
            jitter: Random fraction in [0, 1) that shortens each backoff by up to 25%.
        """
        self._cfg = cfg
        if cfg.response.answer is None:
            raise ConfigError("agent.response.answer is required for JSON agents")
        self._answer = compile_path(cfg.response.answer)
        self._fields = ResponseFields(cfg.response)
        self._wait = cfg.effective_wait()
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._timeout = f"timeout after {self._wait.total_s}s"
        self._client = httpx.Client(
            transport=transport, timeout=client_timeout(self._wait, read_s=self._wait.total_s)
        )

    def send(self, context: dict[str, Any]) -> AgentReply:
        """Send one turn and map the response.

        Transport and protocol failures are returned as an ``AgentReply`` with ``error`` set.

        Args:
            context: Template context for this turn.

        Returns:
            The mapped agent reply.
        """
        body = render(self._cfg.request, context)
        retries: list[str] = []
        starts: list[float] = []
        try:
            content = with_retries(
                lambda: self._post(body, starts),
                self._cfg.retry,
                self._sleep,
                retries,
                self._jitter,
            )
        except NotDelivered as exc:
            return self._failure(ReplyError(exc.reason), starts[-1], retries)
        except ReplyError as exc:
            return self._failure(exc, starts[-1], retries)
        start = starts[-1]
        text = text_of(content)
        try:
            reply = self._map_payload(self._decode(content), self._elapsed_ms(start))
        except ReplyError as exc:
            exc.body = text
            return self._failure(exc, start, retries)
        return dataclasses.replace(reply, body=text, retries=retries)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._client.close()

    def _elapsed_ms(self, start: float) -> float:
        return (self._clock() - start) * 1000

    def _failure(self, exc: ReplyError, start: float, retries: list[str]) -> AgentReply:
        return AgentReply(
            None,
            None,
            None,
            self._elapsed_ms(start),
            exc.error,
            exc.raw,
            body=exc.body,
            retries=retries,
        )

    def _post(self, body: Any, starts: list[float]) -> bytes:
        request = self._client.build_request(
            self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
        )
        starts.append(self._clock())
        with classify_delivery(self._wait, self._timeout):
            response = self._client.send(request, stream=True)
            try:
                return self._read(response, starts[-1])
            finally:
                response.close()

    def _read(self, response: httpx.Response, start: float) -> bytes:
        if response.status_code >= HTTP_ERROR_MIN:
            body = read_capped(response.iter_bytes(), lambda: self._expired(start))
            raise ReplyError(f"agent returned HTTP {response.status_code}", body=body)
        kept = bytearray()
        try:
            for chunk in read_bounded(response.iter_bytes(), lambda: self._deadline(start)):
                kept += chunk
        except ReplyError as exc:
            exc.body = text_of(bytes(kept[:MAX_ERROR_BODY_BYTES]))
            raise
        return bytes(kept)

    def _expired(self, start: float) -> bool:
        return self._clock() - start > self._wait.total_s

    def _deadline(self, start: float) -> None:
        if self._expired(start):
            raise ReplyError(self._timeout)

    @staticmethod
    def _decode(content: bytes) -> Any:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplyError("agent response is not valid JSON") from exc

    def _map_payload(self, payload: Any, latency_ms: float) -> AgentReply:
        answer = first_match(self._answer, payload)
        if answer is MISSING:
            raise ReplyError(f"answer path {self._cfg.response.answer!r} not found", payload)
        if answer is None:
            raise ReplyError("agent answer is null", payload)
        with mapping_guard(payload):
            text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            tool_calls = self._fields.tools(payload)
            tokens = self._fields.tokens(payload)
        return AgentReply(
            answer=text,
            tool_calls=tool_calls,
            tokens=tokens,
            latency_ms=latency_ms,
            error=None,
            raw=payload,
        )
