from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import EmbeddingDimensionError
from agentic_rag.embeddings import Bm25SparseEncoder, SparseVec
from agentic_rag.vectorstore import ChunkRecord, QdrantStore, SearchFilter
from fakes import FakeEmbedder

DOCS = {
    "a": ("ops.md", "OLLAMA_KEEP_ALIVE keeps the model loaded in memory between requests"),
    "b": ("ops.md", "Error E4021 means the model file is missing from the local library"),
    "c": ("rag.md", "Dense retrieval embeds the query and finds chunks with similar meaning"),
    "d": ("rag.md", "Hybrid search merges dense and sparse rankings with reciprocal rank fusion"),
    "e": ("code.py", "def normalize_whitespace(text): collapse runs of whitespace"),
}


async def load(
    store: QdrantStore, embedder: FakeEmbedder, encoder: Bm25SparseEncoder
) -> dict[str, str]:
    """Insert DOCS; returns a map from doc key to point id."""
    import uuid

    records, ids = [], {}
    for i, (key, (source, text)) in enumerate(DOCS.items()):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
        ids[key] = pid
        records.append(
            ChunkRecord(
                id=pid,
                text=text,
                dense=(await embedder.embed_documents([text]))[0],
                sparse=encoder.encode_document(text),
                payload={
                    "source": source,
                    "chunk_index": i,
                    "doc_type": "code" if source.endswith(".py") else "markdown",
                    "file_hash": f"hash-{source}",
                },
            )
        )
    await store.upsert(records)
    return ids


async def test_ensure_collection_is_idempotent(store: QdrantStore) -> None:
    await store.ensure_collection(dim=64)  # fixture already created it
    assert await store.count() == 0


async def test_dimension_mismatch_raises(store: QdrantStore) -> None:
    with pytest.raises(EmbeddingDimensionError, match="--reset"):
        await store.ensure_collection(dim=128)


async def test_reset_recreates_empty_collection(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    assert await store.count() == 5
    await store.reset_collection(dim=128)
    assert await store.count() == 0
    await store.ensure_collection(dim=128)  # now the new dimension is the valid one


async def test_upsert_count_hash_and_delete(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    assert await store.count() == 5
    assert await store.count("ops.md") == 2
    assert await store.get_source_hash("ops.md") == "hash-ops.md"
    assert await store.get_source_hash("missing.md") is None

    await store.delete_source("ops.md")
    assert await store.count() == 3
    assert await store.get_source_hash("ops.md") is None
    assert await store.source_counts() == {"rag.md": 2, "code.py": 1}


async def test_upsert_same_ids_does_not_duplicate(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    await load(store, embedder, sparse_encoder)
    assert await store.count() == 5


async def test_sparse_finds_exact_identifier(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    ids = await load(store, embedder, sparse_encoder)
    hits = await store.search(
        dense=None, sparse=sparse_encoder.encode_query("E4021"), k=3, prefetch_limit=20
    )
    assert hits[0].id == ids["b"]
    assert hits[0].source == "ops.md"
    assert hits[0].metadata["doc_type"] == "markdown"
    assert "text" not in hits[0].metadata


async def test_hybrid_ranks_exact_term_match_first(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    ids = await load(store, embedder, sparse_encoder)
    query = "what does error E4021 mean"
    hits = await store.search(
        dense=await embedder.embed_query(query),
        sparse=sparse_encoder.encode_query(query),
        k=3,
        prefetch_limit=20,
    )
    assert hits[0].id == ids["b"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


async def test_dense_only_returns_k_results(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    hits = await store.search(
        dense=await embedder.embed_query("dense retrieval similar meaning"),
        sparse=None,
        k=2,
        prefetch_limit=20,
    )
    assert len(hits) == 2
    assert hits[0].text.startswith("Dense retrieval")


async def test_empty_sparse_query_falls_back_to_dense(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    hits = await store.search(
        dense=await embedder.embed_query("dense retrieval"),
        sparse=SparseVec([], []),
        k=2,
        prefetch_limit=20,
    )
    assert len(hits) == 2


async def test_filters_restrict_results(
    store: QdrantStore, embedder: FakeEmbedder, sparse_encoder: Bm25SparseEncoder
) -> None:
    await load(store, embedder, sparse_encoder)
    query = "search retrieval model memory"
    kwargs = {
        "dense": await embedder.embed_query(query),
        "sparse": sparse_encoder.encode_query(query),
        "k": 10,
        "prefetch_limit": 20,
    }
    by_source = await store.search(**kwargs, flt=SearchFilter(sources=["rag.md"]))  # type: ignore[arg-type]
    assert {h.source for h in by_source} == {"rag.md"}
    by_type = await store.search(**kwargs, flt=SearchFilter(doc_types=["code"]))  # type: ignore[arg-type]
    assert {h.source for h in by_type} == {"code.py"}


async def test_search_requires_a_query_vector(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="needs a dense vector"):
        await store.search(dense=None, sparse=None, k=3, prefetch_limit=10)


async def test_unreachable_server_raises_vector_store_error(settings: Settings) -> None:
    from agentic_rag.core.exceptions import VectorStoreError

    dead = QdrantStore(
        settings.model_copy(update={"qdrant_url": "http://127.0.0.1:1", "qdrant_timeout_s": 1}),
        client=AsyncQdrantClient(url="http://127.0.0.1:1", timeout=1),
    )
    assert await dead.health_check() is False
    with pytest.raises(VectorStoreError, match="docker compose up"):
        await dead.ensure_collection(dim=64)
    await dead.aclose()


async def test_hybrid_surfaces_both_dense_only_and_sparse_only_winners(store: QdrantStore) -> None:
    """The core claim of hybrid search, with hand-built vectors so no embedder is involved.

    `semantic` is the dense nearest neighbour but shares no query term; `lexical` contains the
    rare query term but its dense vector is orthogonal to the query. Each retriever alone misses
    one of them at k=2; fused, both appear.
    """
    import uuid

    from agentic_rag.embeddings import SparseVec as SV

    def basis(i: int) -> list[float]:
        v = [0.0] * 64
        v[i] = 1.0
        return v

    enc = Bm25SparseEncoder()
    rows = {
        "semantic": (basis(0), "a paraphrase of the idea with different wording"),
        "lexical": (basis(5), "the identifier zzq7 appears in this passage"),
        "filler1": (basis(1), "unrelated filler about cooking pasta"),
        "filler2": (basis(2), "unrelated filler about gardening tomatoes"),
        "filler3": (basis(3), "unrelated filler about sailing boats"),
    }
    ids = {k: str(uuid.uuid5(uuid.NAMESPACE_URL, k)) for k in rows}
    await store.upsert(
        [
            ChunkRecord(
                id=ids[k],
                text=text,
                dense=dense,
                sparse=enc.encode_document(text),
                payload={"source": "s.md", "chunk_index": i, "doc_type": "markdown"},
            )
            for i, (k, (dense, text)) in enumerate(rows.items())
        ]
    )
    query_dense = basis(0)
    query_sparse: SV = enc.encode_query("zzq7")

    async def top(dense: list[float] | None, sparse: SV | None) -> list[str]:
        hits = await store.search(dense=dense, sparse=sparse, k=2, prefetch_limit=20)
        return [h.id for h in hits]

    dense_only = await top(query_dense, None)
    sparse_only = await top(None, query_sparse)
    hybrid = await top(query_dense, query_sparse)

    assert dense_only[0] == ids["semantic"]
    assert ids["lexical"] not in dense_only
    assert sparse_only == [ids["lexical"]]
    assert set(hybrid) == {ids["semantic"], ids["lexical"]}
