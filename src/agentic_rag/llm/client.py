"""Fault-tolerant async client for Ollama's REST API.

Talks to the HTTP API directly (httpx) so retry, timeout and error-mapping behaviour is explicit
and the transport can be swapped for a fake in tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import TracebackType
from typing import Any

import httpx
from openinference.semconv.trace import SpanAttributes
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.exceptions import (
    LLMError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaResponseError,
    OllamaServerError,
    OllamaTimeoutError,
)
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.schemas import ChatMessage, ChatResult, EmbedResult, LLMMetrics
from agentic_rag.observability import (
    SpanKind,
    message_attributes,
    set_attributes,
    set_output,
    span,
)

log = get_logger(__name__)

# Either "json" (plain JSON mode) or a JSON Schema dict (structured outputs).
ResponseFormat = str | dict[str, Any]


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.retryable


def _normalize_model_name(name: str) -> str:
    """Ollama treats a bare name as ':latest'."""
    return name if ":" in name else f"{name}:latest"


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return response.text[:200]


class OllamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        s = self._settings
        self._http = httpx.AsyncClient(
            base_url=s.ollama_base_url,
            timeout=httpx.Timeout(s.ollama_timeout_s, connect=s.ollama_connect_timeout_s),
            transport=transport,
        )

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- health / model discovery ---------------------------------------------

    async def health_check(self) -> bool:
        """True if the server answers. Never raises, never retries."""
        try:
            await self._send("GET", "/api/tags")
        except LLMError:
            return False
        return True

    async def list_models(self) -> list[str]:
        body = await self._request("GET", "/api/tags")
        models = body.get("models", [])
        if not isinstance(models, list):
            raise OllamaResponseError("Unexpected /api/tags payload: 'models' is not a list")
        return [m["name"] for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]

    async def has_model(self, name: str) -> bool:
        wanted = _normalize_model_name(name)
        return any(_normalize_model_name(m) == wanted for m in await self.list_models())

    async def ensure_model(self, name: str) -> None:
        if not await self.has_model(name):
            raise ModelNotFoundError(f"Model {name!r} is not pulled. Run: ollama pull {name}")

    # -- chat ----------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        response_format: ResponseFormat | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """One chat completion. With `on_token`, the reply is streamed and each piece of text is
        passed to the callback as it arrives; the returned ChatResult is the same either way.
        A streamed call is not retried, because a retry after partial output would repeat it."""
        model_name = model or self._settings.ollama_model
        attributes: dict[str, Any] = {
            SpanAttributes.LLM_MODEL_NAME: model_name,
            SpanAttributes.LLM_PROVIDER: "ollama",
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(
                {
                    "temperature": self._settings.ollama_temperature
                    if temperature is None
                    else temperature,
                    "num_ctx": num_ctx or self._settings.ollama_num_ctx,
                    "max_tokens": max_tokens,
                    "structured": response_format is not None,
                }
            ),
            **message_attributes(
                SpanAttributes.LLM_INPUT_MESSAGES, [(m.role, m.content) for m in messages]
            ),
        }
        with span("ollama.chat", SpanKind.LLM, attributes=attributes) as sp:
            result = await self._chat(
                messages,
                model=model,
                response_format=response_format,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                on_token=on_token,
            )
            m = result.metrics
            set_attributes(
                sp,
                {
                    **message_attributes(
                        SpanAttributes.LLM_OUTPUT_MESSAGES, [("assistant", result.content)]
                    ),
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: m.prompt_tokens,
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: m.completion_tokens,
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL: m.prompt_tokens + m.completion_tokens,
                    "llm.finish_reason": result.finish_reason,
                    "llm.tokens_per_second": m.tokens_per_second,
                    "llm.load_duration_s": m.load_duration_s,
                },
            )
            set_output(sp, result.content)
            return result

    async def _chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        response_format: ResponseFormat | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        s = self._settings
        model_name = model or s.ollama_model
        options: dict[str, Any] = {
            "temperature": s.ollama_temperature if temperature is None else temperature,
            "num_ctx": num_ctx or s.ollama_num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [m.model_dump() for m in messages],
            "stream": on_token is not None,
            "options": options,
            "keep_alive": s.ollama_keep_alive,
        }
        if response_format is not None:
            payload["format"] = response_format

        started = time.perf_counter()
        if on_token is None:
            body = await self._request("POST", "/api/chat", json=payload)
            message = body.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise OllamaResponseError("Ollama response is missing message.content")
        else:
            content, body = await self._stream_chat(payload, on_token)
        wall = time.perf_counter() - started

        metrics = LLMMetrics.from_ollama(model_name, body, wall)
        finish_reason = body.get("done_reason")
        finish = finish_reason if isinstance(finish_reason, str) else None
        log.info(
            "llm_chat",
            model=model_name,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            tokens_per_second=(
                round(metrics.tokens_per_second, 2) if metrics.tokens_per_second else None
            ),
            wall_latency_s=round(wall, 3),
            finish_reason=finish,
        )
        if finish == "length":
            log.warning("llm_output_truncated", model=model_name, num_ctx=options["num_ctx"])
        return ChatResult(content=content, metrics=metrics, finish_reason=finish)

    # -- embeddings ----------------------------------------------------------

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> EmbedResult:
        """Embed a batch of texts with one /api/embed call (same retry policy as chat)."""
        model_name = model or self._settings.embedding_model
        with span(
            "ollama.embed",
            SpanKind.EMBEDDING,
            attributes={
                SpanAttributes.EMBEDDING_MODEL_NAME: model_name,
                "embedding.batch_size": len(texts),
            },
        ) as sp:
            result = await self._embed(texts, model=model)
            set_attributes(
                sp,
                {
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: result.metrics.prompt_tokens,
                    "embedding.dimensions": len(result.vectors[0]) if result.vectors else 0,
                },
            )
            return result

    async def _embed(self, texts: Sequence[str], *, model: str | None = None) -> EmbedResult:
        """Embed a batch of texts with one /api/embed call (same retry policy as chat)."""
        s = self._settings
        model_name = model or s.embedding_model
        if not texts:
            return EmbedResult(vectors=[], metrics=LLMMetrics(model=model_name, calls=0))

        payload: dict[str, Any] = {
            "model": model_name,
            "input": list(texts),
            "keep_alive": s.ollama_keep_alive,
        }
        started = time.perf_counter()
        body = await self._request("POST", "/api/embed", json=payload)
        wall = time.perf_counter() - started

        raw = body.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise OllamaResponseError(
                f"Expected {len(texts)} embeddings from /api/embed, got "
                f"{len(raw) if isinstance(raw, list) else type(raw).__name__}"
            )
        vectors: list[list[float]] = []
        for row in raw:
            if not isinstance(row, list) or not row:
                raise OllamaResponseError("Embedding row is empty or not a list")
            if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in row):
                raise OllamaResponseError("Embedding row contains non-numeric values")
            vectors.append([float(x) for x in row])

        metrics = LLMMetrics.from_ollama(model_name, body, wall)
        log.debug(
            "llm_embed",
            model=model_name,
            texts=len(texts),
            prompt_tokens=metrics.prompt_tokens,
            wall_latency_s=round(wall, 3),
        )
        return EmbedResult(vectors=vectors, metrics=metrics)

    # -- transport -----------------------------------------------------------

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """`_send` wrapped in the retry policy."""
        s = self._settings
        retrying = AsyncRetrying(
            stop=stop_after_attempt(s.ollama_max_attempts),
            wait=wait_exponential_jitter(
                initial=s.ollama_retry_initial_wait_s,
                max=s.ollama_retry_max_wait_s,
                jitter=0.5,
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._log_retry,
            reraise=True,
        )
        return await retrying(self._send, method, path, json=json)

    @staticmethod
    def _log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "ollama_retry",
            attempt=state.attempt_number,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    @contextmanager
    def _mapped_transport_errors(self) -> Iterator[None]:
        """httpx errors mapped to our hierarchy."""
        base = self._settings.ollama_base_url
        try:
            yield
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(f"Cannot reach Ollama at {base}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama at {base} did not respond within "
                f"{self._settings.ollama_timeout_s:g}s ({type(exc).__name__})"
            ) from exc
        except httpx.TransportError as exc:
            raise OllamaConnectionError(f"Connection to Ollama at {base} failed: {exc}") from exc

    async def _stream_chat(
        self, payload: dict[str, Any], on_token: Callable[[str], None]
    ) -> tuple[str, dict[str, Any]]:
        """POST /api/chat with stream=true; returns (full text, the final 'done' chunk, which
        carries the token counts and durations)."""
        parts: list[str] = []
        final: dict[str, Any] = {}
        with self._mapped_transport_errors():
            async with self._http.stream("POST", "/api/chat", json=payload) as response:
                if not response.is_success:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as exc:
                        raise OllamaResponseError("Ollama sent a malformed stream line") from exc
                    if not isinstance(chunk, dict):
                        raise OllamaResponseError("Ollama stream line is not a JSON object")
                    if isinstance(chunk.get("error"), str):
                        raise OllamaServerError(f"Ollama failed mid-stream: {chunk['error']}")
                    message = chunk.get("message")
                    piece = message.get("content") if isinstance(message, dict) else None
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
                        on_token(piece)
                    if chunk.get("done") is True:
                        final = chunk
        if not final:
            raise OllamaResponseError("Ollama's stream ended before it reported completion")
        return "".join(parts), final

    async def _send(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """One HTTP round trip with httpx errors mapped to our hierarchy."""
        with self._mapped_transport_errors():
            response = await self._http.request(method, path, json=json)

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned a non-JSON body") from exc
        if not isinstance(body, dict):
            raise OllamaResponseError("Ollama returned a JSON body that is not an object")
        return body

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        detail = _error_detail(response)
        if status == 404 and "not found" in detail.lower():
            raise ModelNotFoundError(f"{detail}. Pull it with: ollama pull <model>")
        if status == 429 or status >= 500:
            raise OllamaServerError(f"Ollama server error (HTTP {status}): {detail}")
        raise OllamaRequestError(f"Ollama rejected the request (HTTP {status}): {detail}")
