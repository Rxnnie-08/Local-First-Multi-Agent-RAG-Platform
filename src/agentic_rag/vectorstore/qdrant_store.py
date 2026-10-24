"""Qdrant wrapper: one collection holding a dense and a sparse vector per chunk.

Hybrid search uses Qdrant's Query API: two prefetches (dense, sparse) fused server-side with
Reciprocal Rank Fusion. RRF scores are rank-based, depend on the server's RRF constant, and are
NOT comparable with cosine scores or across queries; use them for ordering only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.exceptions import EmbeddingDimensionError, VectorStoreError
from agentic_rag.core.logging import get_logger
from agentic_rag.embeddings.sparse import SparseVec
from agentic_rag.vectorstore.models import ChunkRecord, SearchFilter, SearchHit

log = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
_UPSERT_BATCH = 64
_RESERVED_PAYLOAD_KEYS = frozenset({"text", "source", "chunk_index"})


def _source_filter(source: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    )


def _to_qdrant_filter(flt: SearchFilter | None) -> models.Filter | None:
    if flt is None:
        return None
    must: list[models.Condition] = []
    if flt.sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=flt.sources)))
    if flt.doc_types:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=flt.doc_types)))
    return models.Filter(must=must) if must else None


class QdrantStore:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._collection = self._settings.qdrant_collection
        self._client = client or AsyncQdrantClient(
            url=self._settings.qdrant_url, timeout=self._settings.qdrant_timeout_s
        )

    @property
    def collection(self) -> str:
        return self._collection

    async def aclose(self) -> None:
        await self._client.close()

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        try:
            yield
        except (ResponseHandlingException, ApiException) as exc:
            raise VectorStoreError(
                f"Qdrant {operation} failed ({self._settings.qdrant_url}): {exc}. "
                "Is it running? Try: docker compose up -d"
            ) from exc

    # -- collection management -------------------------------------------------

    async def health_check(self) -> bool:
        try:
            async with self._guard("health check"):
                await self._client.get_collections()
        except VectorStoreError:
            return False
        return True

    async def ensure_collection(self, *, dim: int) -> None:
        """Create the collection if missing; otherwise check it matches `dim` and both vectors."""
        async with self._guard("ensure_collection"):
            if await self._client.collection_exists(self._collection):
                await self._verify_collection(dim)
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                # IDF modifier: Qdrant applies inverse document frequency at query time.
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            for field in ("source", "doc_type"):
                await self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.KEYWORD
                )
            log.info("collection_created", collection=self._collection, dim=dim)

    async def _verify_collection(self, dim: int) -> None:
        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
            raise VectorStoreError(
                f"Collection {self._collection!r} has no named '{DENSE_VECTOR}' vector. "
                "Re-run ingestion with --reset to recreate it."
            )
        existing = vectors[DENSE_VECTOR].size
        if existing != dim:
            raise EmbeddingDimensionError(
                f"Collection {self._collection!r} stores {existing}-dim vectors but the embedder "
                f"produces {dim}. Re-run ingestion with --reset to rebuild it."
            )
        if SPARSE_VECTOR not in sparse:
            raise VectorStoreError(
                f"Collection {self._collection!r} has no '{SPARSE_VECTOR}' vector. "
                "Re-run ingestion with --reset to recreate it."
            )

    async def drop_collection(self) -> None:
        async with self._guard("drop_collection"):
            if await self._client.collection_exists(self._collection):
                await self._client.delete_collection(self._collection)

    async def reset_collection(self, *, dim: int) -> None:
        await self.drop_collection()
        await self.ensure_collection(dim=dim)

    # -- writes ------------------------------------------------------------------

    async def upsert(self, records: Sequence[ChunkRecord]) -> None:
        async with self._guard("upsert"):
            for start in range(0, len(records), _UPSERT_BATCH):
                batch = records[start : start + _UPSERT_BATCH]
                points = [
                    models.PointStruct(
                        id=r.id,
                        vector={
                            DENSE_VECTOR: r.dense,
                            SPARSE_VECTOR: models.SparseVector(
                                indices=r.sparse.indices, values=r.sparse.values
                            ),
                        },
                        payload={**r.payload, "text": r.text},
                    )
                    for r in batch
                ]
                await self._client.upsert(self._collection, points=points, wait=True)

    async def delete_source(self, source: str) -> None:
        async with self._guard("delete_source"):
            await self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(filter=_source_filter(source)),
                wait=True,
            )

    # -- reads -------------------------------------------------------------------

    async def get_source_hash(self, source: str) -> str | None:
        """File hash recorded for `source`, or None if it has never been ingested."""
        async with self._guard("get_source_hash"):
            points, _ = await self._client.scroll(
                self._collection,
                scroll_filter=_source_filter(source),
                limit=1,
                with_payload=["file_hash"],
                with_vectors=False,
            )
        if not points or not points[0].payload:
            return None
        value = points[0].payload.get("file_hash")
        return value if isinstance(value, str) else None

    async def count(self, source: str | None = None) -> int:
        async with self._guard("count"):
            result = await self._client.count(
                self._collection,
                count_filter=_source_filter(source) if source else None,
                exact=True,
            )
        return result.count

    async def source_counts(self) -> dict[str, int]:
        """chunks per source. Scrolls the whole collection: fine for thousands, not millions."""
        counts: dict[str, int] = {}
        offset: Any = None
        async with self._guard("source_counts"):
            while True:
                points, offset = await self._client.scroll(
                    self._collection,
                    limit=256,
                    offset=offset,
                    with_payload=["source"],
                    with_vectors=False,
                )
                for p in points:
                    src = (p.payload or {}).get("source")
                    if isinstance(src, str):
                        counts[src] = counts.get(src, 0) + 1
                if offset is None:
                    break
        return counts

    async def search(
        self,
        *,
        dense: Sequence[float] | None,
        sparse: SparseVec | None,
        k: int,
        prefetch_limit: int,
        flt: SearchFilter | None = None,
    ) -> list[SearchHit]:
        """Dense-only, sparse-only, or (both given) hybrid with RRF fusion.

        An empty sparse vector (e.g. a query made only of stopwords) is treated as absent.
        """
        use_sparse = sparse is not None and not sparse.is_empty
        if dense is None and not use_sparse:
            raise ValueError("search needs a dense vector, a non-empty sparse vector, or both")

        qfilter = _to_qdrant_filter(flt)
        dense_q = list(dense) if dense is not None else None
        sparse_q = (
            models.SparseVector(indices=sparse.indices, values=sparse.values)
            if sparse is not None and use_sparse
            else None
        )

        async with self._guard("search"):
            if dense_q is not None and sparse_q is not None:
                response = await self._client.query_points(
                    self._collection,
                    prefetch=[
                        models.Prefetch(
                            query=dense_q, using=DENSE_VECTOR, limit=prefetch_limit, filter=qfilter
                        ),
                        models.Prefetch(
                            query=sparse_q,
                            using=SPARSE_VECTOR,
                            limit=prefetch_limit,
                            filter=qfilter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=k,
                    with_payload=True,
                )
            elif dense_q is not None:
                response = await self._client.query_points(
                    self._collection,
                    query=dense_q,
                    using=DENSE_VECTOR,
                    limit=k,
                    query_filter=qfilter,
                    with_payload=True,
                )
            else:
                response = await self._client.query_points(
                    self._collection,
                    query=sparse_q,
                    using=SPARSE_VECTOR,
                    limit=k,
                    query_filter=qfilter,
                    with_payload=True,
                )
        return [self._to_hit(p) for p in response.points]

    @staticmethod
    def _to_hit(point: models.ScoredPoint) -> SearchHit:
        payload = dict(point.payload or {})
        chunk_index = payload.get("chunk_index")
        return SearchHit(
            id=str(point.id),
            score=point.score,
            text=str(payload.get("text", "")),
            source=str(payload.get("source", "")),
            chunk_index=chunk_index if isinstance(chunk_index, int) else -1,
            metadata={k: v for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS},
        )
