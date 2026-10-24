"""Live tests against a real Ollama server. Run with: pytest -m integration"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.llm import ChatMessage, OllamaClient, StructuredExtractor

pytestmark = pytest.mark.integration


class Capital(BaseModel):
    country: str
    capital: str


@pytest.fixture
async def live_client() -> AsyncIterator[OllamaClient]:
    settings = Settings()
    async with OllamaClient(settings) as client:
        if not await client.health_check():
            pytest.skip(f"Ollama not reachable at {settings.ollama_base_url}")
        if not await client.has_model(settings.ollama_model):
            pytest.skip(f"Model {settings.ollama_model} not pulled")
        yield client


async def test_live_chat(live_client: OllamaClient) -> None:
    result = await live_client.chat(
        [ChatMessage(role="user", content="Reply with the single word: ready")], max_tokens=16
    )
    assert result.content.strip()
    assert result.metrics.completion_tokens > 0


async def test_live_structured_extraction(live_client: OllamaClient) -> None:
    extractor = StructuredExtractor(live_client)
    result = await extractor.extract(Capital, "Paris is the capital of France.")
    assert result.data.capital.lower() == "paris"
