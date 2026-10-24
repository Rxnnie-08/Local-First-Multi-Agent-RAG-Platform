"""Typed error hierarchy. `retryable` drives the client's retry policy."""

from __future__ import annotations

from typing import ClassVar


class AgenticRAGError(Exception):
    """Base class for every error raised by this package."""


class LLMError(AgenticRAGError):
    """Anything that goes wrong talking to the LLM backend."""

    retryable: ClassVar[bool] = False


class OllamaConnectionError(LLMError):
    """Server unreachable, connection refused/reset, or connect timeout."""

    retryable = True


class OllamaTimeoutError(LLMError):
    """Server accepted the request but did not answer in time.

    Not retried: on CPU-bound laptops a read timeout almost always means "slow
    model", and repeating a multi-minute call three times just burns time.
    """


class OllamaServerError(LLMError):
    """HTTP 5xx or 429 from Ollama."""

    retryable = True


class OllamaRequestError(LLMError):
    """HTTP 4xx: the request itself is wrong. Retrying cannot help."""


class ModelNotFoundError(OllamaRequestError):
    """The requested model has not been pulled."""


class OllamaResponseError(LLMError):
    """HTTP 200 but the body is not the shape we expect."""


class VectorStoreError(AgenticRAGError):
    """Qdrant unreachable, rejected a request, or the collection is misconfigured."""


class EmbeddingDimensionError(AgenticRAGError):
    """Embedding size does not match the collection (usually: embedding model was changed)."""


class SandboxError(AgenticRAGError):
    """The code sandbox is unavailable (Docker missing/stopped, image not pulled) or misbehaved.

    A script that merely fails or times out is NOT an error: it is a normal `SandboxResult`.
    """


class WebSearchError(AgenticRAGError):
    """The web search backend failed (network, rate limit, parsing)."""


class RerankerError(AgenticRAGError):
    """The reranker model could not be loaded or failed while scoring."""


class ParseError(AgenticRAGError):
    """A file could not be read or parsed into text."""


class StructuredOutputError(LLMError):
    """The model never produced output matching the schema, even after repairs."""

    def __init__(
        self,
        message: str,
        *,
        schema_name: str,
        attempts: int,
        last_error: str,
        raw_output: str,
    ) -> None:
        super().__init__(message)
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error
        self.raw_output = raw_output
