"""HTTP adapter that talks to the agent under test."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Unpack

import httpx

from octoqai.agents._delivery import (
    HTTP_ERROR_MIN,
    MAX_ERROR_BODY_BYTES,
    AdapterKwargs,
    AgentAnswered,
    Decoded,
    NotDelivered,
    Seams,
    WatchedClient,
    agent_request,
    classify_delivery,
    client_timeout,
    decode_body,
    keep_body,
    read_bounded,
    read_capped,
    reject_encoded,
    transport_failure,
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


def _kept_body(kept: bytearray, encoding: str) -> Decoded:
    return decode_body(bytes(kept[:MAX_ERROR_BODY_BYTES]), encoding)


class HttpAgentAdapter:
    """Send rendered requests to an agent exposed over HTTP JSON.

    Every attempt sends ``Accept-Encoding: identity``, replacing any value set in the
    configured headers. A response with any other ``Content-Encoding`` is refused without
    reading its body, so a compressed body can never expand past the size cap.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        transport: httpx.BaseTransport | None = None,
        **seams: Unpack[AdapterKwargs],
    ) -> None:
        """Create the adapter.

        Args:
            cfg: Agent connection settings.
            transport: Optional httpx transport, used to test without a network.
            **seams: Optional ``clock``, ``sleep``, ``jitter`` and ``watchdog`` (see
                ``AdapterKwargs``), injectable for tests.
        """
        self._cfg = cfg
        if cfg.response.answer is None:
            raise ConfigError("agent.response.answer is required for JSON agents")
        self._answer = compile_path(cfg.response.answer)
        self._fields = ResponseFields(cfg.response)
        self._wait = cfg.effective_wait()
        seam = Seams.of(seams)
        self._clock, self._sleep, self._jitter = seam.clock, seam.sleep, seam.jitter
        self._timeout = f"timeout after {self._wait.total_s}s"
        self._http = WatchedClient(
            transport, client_timeout(self._wait, read_s=self._wait.total_s), seam.watchdog
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
            content, charset = with_retries(
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
        try:
            payload = self._decode(content)
        except ReplyError as exc:
            return self._failure(keep_body(exc, decode_body(content, charset)), start, retries)
        text = decode_body(content).text
        try:
            reply = self._map_payload(payload, self._elapsed_ms(start))
        except ReplyError as exc:
            exc.body = text
            return self._failure(exc, start, retries)
        return dataclasses.replace(reply, body=text, retries=retries)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._http.close()

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
            alt_body=exc.alt_body,
        )

    def _post(self, body: Any, starts: list[float]) -> tuple[bytes, str]:
        starts.append(self._clock())
        with self._http.attempt(self._wait.total_s, self._timeout) as client:
            request = agent_request(client, self._cfg, body)
            with classify_delivery(self._wait, self._timeout):
                response = client.send(request, stream=True)
                try:
                    content = self._read(response, starts[-1])
                    return content, response.charset_encoding or "utf-8"
                finally:
                    response.close()

    def _read(self, response: httpx.Response, start: float) -> bytes:
        reject_encoded(response)
        charset = response.charset_encoding or "utf-8"
        if response.status_code >= HTTP_ERROR_MIN:
            body = read_capped(response.iter_bytes(), lambda: self._expired(start), charset)
            raise keep_body(AgentAnswered(f"agent returned HTTP {response.status_code}"), body)
        kept = bytearray()
        try:
            for chunk in read_bounded(response.iter_bytes(), lambda: self._deadline(start)):
                kept += chunk
        except ReplyError as exc:
            keep_body(exc, _kept_body(kept, charset))
            raise
        except httpx.HTTPError as exc:
            error = ReplyError(self._read_error(exc))
            raise keep_body(error, _kept_body(kept, charset)) from exc
        return bytes(kept)

    def _read_error(self, exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return self._timeout
        return transport_failure(exc).error

    def _expired(self, start: float) -> bool:
        return self._clock() - start > self._wait.total_s

    def _deadline(self, start: float) -> None:
        if self._expired(start):
            raise ReplyError(self._timeout)

    @staticmethod
    def _decode(content: bytes) -> Any:
        try:
            return json.loads(content)
        except (ValueError, RecursionError) as exc:
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
