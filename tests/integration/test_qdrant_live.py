"""End-to-end against real Qdrant + real Ollama. Run with: pytest -m integration

Uses a throwaway collection, which is deleted afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentic_rag.config import Settings
from agentic_rag.services import Services, open_services

pytestmark = pytest.mark.integration

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture
async def live() -> AsyncIterator[Services]:
    settings = Settings(qdrant_collection=f"test_{uuid.uuid4().hex[:8]}")
    async with open_services(settings) as svc:
        if not await svc.store.health_check():
            pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}")
        if not await svc.client.health_check():
            pytest.skip(f"Ollama not reachable at {settings.ollama_base_url}")
        if not await svc.client.has_model(settings.embedding_model):
            pytest.skip(f"Embedding model {settings.embedding_model} not pulled")
        try:
            yield svc
        finally:
            await svc.store.drop_collection()


async def test_live_ingest_and_hybrid_query(live: Services) -> None:
    report = await live.pipeline.ingest_path(SAMPLES)
    assert report.failed == 0
    assert report.ingested == 3

    exact = await live.retriever.retrieve("Error E-4021", k=3, mode="hybrid")
    assert exact.hits[0].source == "samples/ollama_ops.md"

    semantic = await live.retriever.retrieve(
        "how do I stop the model from unloading so quickly", k=3, mode="hybrid"
    )
    assert semantic.hits[0].source == "samples/ollama_ops.md"
