"""End-to-end check of Phase 1 against your real Ollama. Run: python scripts/smoke_test.py"""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel, Field

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import LLMError
from agentic_rag.core.logging import configure_logging
from agentic_rag.llm import ChatMessage, LLMMetrics, OllamaClient, StructuredExtractor


class Person(BaseModel):
    name: str
    age: int
    skills: list[str] = Field(description="Technical skills only; exclude degrees and majors")


def describe(label: str, m: LLMMetrics) -> str:
    tps = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "n/a"
    return (
        f"{label}: {m.total_tokens} tokens "
        f"({m.prompt_tokens} in / {m.completion_tokens} out), "
        f"{tps} tok/s, load {m.load_duration_s:.2f}s, wall {m.wall_latency_s:.2f}s, "
        f"{m.calls} call(s)"
    )


async def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    async with OllamaClient(settings) as client:
        print(f"Ollama: {settings.ollama_base_url}  model: {settings.ollama_model}")

        if not await client.health_check():
            print("FAIL: Ollama is not reachable. Is it running? (try: ollama list)")
            return 1
        print("OK   health check")

        try:
            await client.ensure_model(settings.ollama_model)
            print("OK   model is pulled")

            plain = await client.chat(
                [ChatMessage(role="user", content="Reply with the single word: ready")],
                max_tokens=16,
            )
            print(f"OK   chat -> {plain.content.strip()!r}")
            print("     " + describe("chat", plain.metrics))

            extractor = StructuredExtractor(client)
            result = await extractor.extract(
                Person,
                "Priya is 21 and studies CSE-AIML. She knows Python, PyTorch and SQL.",
            )
            print(f"OK   structured -> {result.data.model_dump()}  (attempts: {result.attempts})")
            print("     " + describe("structured", result.metrics))
        except LLMError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1

    print("\nPhase 1 smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
