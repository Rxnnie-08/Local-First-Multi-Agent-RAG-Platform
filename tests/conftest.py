from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag.config import Settings
from agentic_rag.embeddings import Bm25SparseEncoder
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.vectorstore import QdrantStore
from fakes import FakeEmbedder


@pytest.fixture
def settings() -> Settings:
    """Isolated settings: ignores .env and real env vars, and retries without sleeping."""
    return Settings(
        _env_file=None,
        ollama_base_url="http://ollama.test",
        ollama_model="test-model",
        ollama_max_attempts=3,
        ollama_retry_initial_wait_s=0.0,
        ollama_retry_max_wait_s=0.0,
        structured_max_repairs=2,
        chunk_max_tokens=64,
        chunk_overlap_tokens=8,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def sparse_encoder() -> Bm25SparseEncoder:
    return Bm25SparseEncoder()


@pytest.fixture
async def store(settings: Settings) -> AsyncIterator[QdrantStore]:
    """Qdrant's embedded local mode: same client API, no server needed."""
    qstore = QdrantStore(settings, client=AsyncQdrantClient(":memory:"))
    await qstore.ensure_collection(dim=64)
    yield qstore
    await qstore.aclose()


@pytest.fixture
def pipeline(
    store: QdrantStore,
    embedder: FakeEmbedder,
    sparse_encoder: Bm25SparseEncoder,
    settings: Settings,
) -> IngestionPipeline:
    return IngestionPipeline(store, embedder, sparse_encoder, settings)


@pytest.fixture
def retriever(
    store: QdrantStore,
    embedder: FakeEmbedder,
    sparse_encoder: Bm25SparseEncoder,
    settings: Settings,
) -> HybridRetriever:
    return HybridRetriever(store, embedder, sparse_encoder, settings)
