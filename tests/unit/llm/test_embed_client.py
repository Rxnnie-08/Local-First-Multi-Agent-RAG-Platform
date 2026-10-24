from __future__ import annotations

import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import ModelNotFoundError, OllamaResponseError
from agentic_rag.llm import OllamaClient
from fakes import FakeOllama, embed_response, error_response


async def test_embed_returns_vectors_and_metrics(settings: Settings) -> None:
    fake = FakeOllama([embed_response(2, 4)])
    async with OllamaClient(settings, transport=fake.transport) as client:
        result = await client.embed(["a", "b"])

    assert len(result.vectors) == 2
    assert len(result.vectors[0]) == 4
    assert result.metrics.prompt_tokens == 12
    body = fake.json_body()
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["a", "b"]
    assert fake.requests[0].url.path == "/api/embed"


async def test_embed_empty_makes_no_request(settings: Settings) -> None:
    fake = FakeOllama([])
    async with OllamaClient(settings, transport=fake.transport) as client:
        result = await client.embed([])
    assert result.vectors == []
    assert fake.requests == []


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [[0.1, 0.2]]},  # wrong count for 2 inputs
        {"embeddings": "nope"},
        {"embeddings": [[0.1], []]},  # empty row
        {"embeddings": [[0.1], ["x"]]},  # non-numeric
        {"embeddings": [[0.1], [True]]},  # bool is not a number here
        {},
    ],
)
async def test_embed_rejects_malformed_responses(
    settings: Settings, body: dict[str, object]
) -> None:
    fake = FakeOllama([httpx.Response(200, json=body)])
    async with OllamaClient(settings, transport=fake.transport) as client:
        with pytest.raises(OllamaResponseError):
            await client.embed(["a", "b"])


async def test_embed_missing_model_has_pull_hint(settings: Settings) -> None:
    fake = FakeOllama(
        [error_response(404, 'model "nomic-embed-text" not found, try pulling it first')]
    )
    async with OllamaClient(settings, transport=fake.transport) as client:
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            await client.embed(["a"])
