"""HTTP adapter that talks to the agent under test."""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from octoqai.agents._delivery import NotDelivered, client_timeout, not_delivered, with_retries
from octoqai.agents._mapping import MISSING, ReplyError, ResponseFields, compile_path, first_match
from octoqai.agents.port import AgentReply
from octoqai.errors import ConfigError
from octoqai.experiment import AgentConfig, render

HTTP_ERROR_MIN = 400


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
            jitter: Random fraction in [0, 1] that shortens each backoff by up to 25%.
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
        first_start = self._clock()
        try:
            response, start = with_retries(
                lambda: self._post(body), self._cfg.retry, self._sleep, self._jitter, retries
            )
        except NotDelivered as exc:
            return self._failure(ReplyError(exc.reason), first_start, retries)
        except ReplyError as exc:
            return self._failure(exc, first_start, retries)
        try:
            reply = self._map_payload(self._decode(response), self._elapsed_ms(start))
        except ReplyError as exc:
            return self._failure(exc, start, retries)
        return dataclasses.replace(reply, body=response.text, retries=retries)

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

    def _post(self, body: Any) -> tuple[httpx.Response, float]:
        start = self._clock()
        try:
            response = self._client.request(
                self._cfg.method, self._cfg.url, json=body, headers=self._cfg.headers
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise not_delivered(exc, self._wait) from exc
        except httpx.TimeoutException as exc:
            raise ReplyError(f"timeout after {self._wait.total_s}s") from exc
        except httpx.HTTPError as exc:
            raise ReplyError(f"transport error: {type(exc).__name__}") from exc
        return response, start

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        if response.status_code >= HTTP_ERROR_MIN:
            raise ReplyError(f"agent returned HTTP {response.status_code}", body=response.text)
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplyError("agent response is not valid JSON", body=response.text) from exc

    def _map_payload(self, payload: Any, latency_ms: float) -> AgentReply:
        answer = first_match(self._answer, payload)
        if answer is MISSING:
            raise ReplyError(f"answer path {self._cfg.response.answer!r} not found", payload)
        if answer is None:
            raise ReplyError("agent answer is null", payload)
        try:
            text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            tool_calls = self._fields.tools(payload)
            tokens = self._fields.tokens(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReplyError(f"cannot map agent response: {type(exc).__name__}", payload) from exc
        return AgentReply(
            answer=text,
            tool_calls=tool_calls,
            tokens=tokens,
            latency_ms=latency_ms,
            error=None,
            raw=payload,
        )
