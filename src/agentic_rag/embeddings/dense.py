"""Dense embeddings via Ollama. `Embedder` is the interface the rest of the code depends on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agentic_rag.core.exceptions import EmbeddingDimensionError
from agentic_rag.llm.client import OllamaClient


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        client: OllamaClient,
        *,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
        doc_prefix: str | None = None,
        query_prefix: str | None = None,
    ) -> None:
        s = client.settings
        self._client = client
        self._model = model or s.embedding_model
        self._dim = dim or s.embedding_dim
        self._batch_size = batch_size or s.embedding_batch_size
        self._doc_prefix = s.embedding_doc_prefix if doc_prefix is None else doc_prefix
        self._query_prefix = s.embedding_query_prefix if query_prefix is None else query_prefix

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [self._doc_prefix + t for t in texts[start : start + self._batch_size]]
            result = await self._client.embed(batch, model=self._model)
            vectors.extend(result.vectors)
        self._check_dim(vectors)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.embed([self._query_prefix + text], model=self._model)
        self._check_dim(result.vectors)
        return result.vectors[0]

    def _check_dim(self, vectors: Sequence[Sequence[float]]) -> None:
        for vec in vectors:
            if len(vec) != self._dim:
                raise EmbeddingDimensionError(
                    f"Model {self._model!r} returned {len(vec)}-dim vectors but EMBEDDING_DIM is "
                    f"{self._dim}. Fix EMBEDDING_DIM in .env (and re-ingest with --reset)."
                )
