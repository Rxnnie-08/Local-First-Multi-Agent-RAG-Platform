"""Query-side retrieval: embeds the query once and runs dense, sparse or hybrid search."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.core.logging import get_logger
from agentic_rag.embeddings.dense import Embedder
from agentic_rag.embeddings.sparse import Bm25SparseEncoder
from agentic_rag.observability import (
    SpanKind,
    document_attributes,
    set_attributes,
    set_output,
    span,
)
from agentic_rag.vectorstore.models import SearchFilter, SearchHit
from agentic_rag.vectorstore.qdrant_store import QdrantStore

log = get_logger(__name__)

RetrievalMode = Literal["hybrid", "dense", "sparse"]


class RetrievalResult(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[SearchHit]
    embed_s: float
    search_s: float

    @property
    def total_s(self) -> float:
        return self.embed_s + self.search_s


class HybridRetriever:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        sparse_encoder: Bm25SparseEncoder,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        mode: RetrievalMode = "hybrid",
        flt: SearchFilter | None = None,
    ) -> RetrievalResult:
        top_k = k or self._settings.retrieval_top_k
        prefetch = max(top_k * self._settings.retrieval_prefetch_multiplier, 20)

        with span(
            "retrieve",
            SpanKind.RETRIEVER,
            input_value=query,
            attributes={"retrieval.mode": mode, "retrieval.k": top_k},
        ) as sp:
            started = time.perf_counter()
            dense = await self._embedder.embed_query(query) if mode != "sparse" else None
            sparse = self._sparse.encode_query(query) if mode != "dense" else None
            embedded = time.perf_counter()

            hits = await self._store.search(
                dense=dense, sparse=sparse, k=top_k, prefetch_limit=prefetch, flt=flt
            )
            done = time.perf_counter()
            set_attributes(
                sp,
                {
                    "retrieval.embed_s": round(embedded - started, 4),
                    "retrieval.search_s": round(done - embedded, 4),
                    **document_attributes(
                        "retrieval.documents",
                        [
                            {
                                "id": h.id,
                                "score": h.score,
                                "content": h.text,
                                "metadata": {"source": h.source, "chunk_index": h.chunk_index},
                            }
                            for h in hits
                        ],
                    ),
                },
            )
            set_output(sp, [h.source for h in hits])

        result = RetrievalResult(
            query=query,
            mode=mode,
            hits=hits,
            embed_s=embedded - started,
            search_s=done - embedded,
        )
        log.info(
            "retrieve",
            mode=mode,
            k=top_k,
            hits=len(hits),
            embed_s=round(result.embed_s, 3),
            search_s=round(result.search_s, 3),
        )
        return result
