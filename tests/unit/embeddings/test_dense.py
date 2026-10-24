from __future__ import annotations

import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import EmbeddingDimensionError
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.llm import OllamaClient
from fakes import FakeOllama, embed_response


def make(
    settings: Settings, fake: FakeOllama, batch_size: int | None = None
) -> tuple[OllamaClient, OllamaEmbedder]:
    client = OllamaClient(settings, transport=fake.transport)
    return client, OllamaEmbedder(client, dim=3, batch_size=batch_size)


async def test_documents_are_batched_and_prefixed(settings: Settings) -> None:
    fake = FakeOllama([embed_response(2, 3), embed_response(2, 3), embed_response(1, 3)])
    client, embedder = make(settings, fake, batch_size=2)
    async with client:
        vectors = await embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert len(fake.requests) == 3
    first = fake.json_body(0)
    assert first["model"] == "nomic-embed-text"
    assert first["input"] == ["search_document: a", "search_document: b"]
    assert fake.json_body(2)["input"] == ["search_document: e"]


async def test_query_uses_query_prefix(settings: Settings) -> None:
    fake = FakeOllama([embed_response(1, 3)])
    client, embedder = make(settings, fake)
    async with client:
        vec = await embedder.embed_query("what is rrf?")
    assert len(vec) == 3
    assert fake.json_body()["input"] == ["search_query: what is rrf?"]


async def test_empty_input_makes_no_request(settings: Settings) -> None:
    fake = FakeOllama([])
    client, embedder = make(settings, fake)
    async with client:
        assert await embedder.embed_documents([]) == []
    assert fake.requests == []


async def test_dimension_mismatch_is_reported(settings: Settings) -> None:
    fake = FakeOllama([embed_response(1, 5)])
    client, embedder = make(settings, fake)
    async with client:
        with pytest.raises(EmbeddingDimensionError, match="EMBEDDING_DIM"):
            await embedder.embed_query("x")
