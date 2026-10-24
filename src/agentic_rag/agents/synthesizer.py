"""Final answer generation: grounded in retrieved passages, with numbered citations."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agentic_rag.agents.models import Citation, Observation, Passage
from agentic_rag.agents.prompts import (
    DIRECT_SYSTEM,
    GROUNDED_SYSTEM,
    NO_RESULTS_SYSTEM,
    WEAK_NOTE,
    format_context,
)
from agentic_rag.config import Settings
from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.chunker import CHARS_PER_TOKEN, estimate_tokens
from agentic_rag.llm import ChatMessage, LLMMetrics, OllamaClient

log = get_logger(__name__)

AnswerMode = Literal["grounded", "no_results", "direct"]
_CITE_RE = re.compile(r"\[(\d+)\]")
_PASSAGE_OVERHEAD_TOKENS = 12  # "[n] (source | location)" header per passage


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    citations: list[Citation]
    metrics: LLMMetrics
    mode: AnswerMode
    detail: dict[str, Any] = field(default_factory=dict)


class Synthesizer:
    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def select_passages(self, observations: Sequence[Observation]) -> tuple[list[Passage], int]:
        """Take passages in order until the context budget is spent. Returns (chosen, dropped).

        The budget exists because Ollama silently drops the start of a prompt that exceeds
        num_ctx, which would remove the instructions rather than the overflow.
        """
        budget = self._settings.answer_context_tokens
        chosen: list[Passage] = []
        used = 0
        candidates: list[Passage] = []
        seen: set[tuple[str, str, str]] = set()
        for passage in (p for o in observations for p in o.passages):
            key = (passage.source, passage.location, passage.text)
            if key not in seen:  # a re-worded retry often returns the same chunks
                seen.add(key)
                candidates.append(passage)
        for index, passage in enumerate(candidates):
            cost = estimate_tokens(passage.text) + _PASSAGE_OVERHEAD_TOKENS
            if used + cost <= budget:
                chosen.append(passage)
                used += cost
                continue
            room_chars = int((budget - used - _PASSAGE_OVERHEAD_TOKENS) * CHARS_PER_TOKEN)
            if room_chars >= 300:  # a clipped passage is useful; a few words are not
                clipped = passage.text[:room_chars].rstrip() + " …"
                chosen.append(passage.model_copy(update={"text": clipped}))
                index += 1
            return chosen, len(candidates) - index
        return chosen, 0

    async def answer(
        self,
        *,
        question: str,
        history: Sequence[ChatMessage],
        observations: Sequence[Observation],
        on_token: Callable[[str], None] | None = None,
    ) -> SynthesisResult:
        passages, dropped = self.select_passages(observations)
        if passages:
            mode: AnswerMode = "grounded"
            weak = not any(o.ok and o.confident for o in observations)
            system = GROUNDED_SYSTEM.format(weak_note=WEAK_NOTE if weak else "")
            user = f"Context passages:\n\n{format_context(passages)}\n\nQuestion: {question}"
        elif observations:
            mode = "no_results"
            system = NO_RESULTS_SYSTEM
            user = f"Question: {question}"
        else:
            mode = "direct"
            system = DIRECT_SYSTEM
            user = f"Question: {question}"

        messages = [
            ChatMessage(role="system", content=system),
            *history,
            ChatMessage(role="user", content=user),
        ]
        result = await self._client.chat(
            messages,
            max_tokens=self._settings.answer_max_tokens,
            on_token=on_token,
        )

        answer = result.content.strip()
        detail: dict[str, Any] = {
            "mode": mode,
            "passages_used": len(passages),
            "passages_dropped": dropped,
            "finish_reason": result.finish_reason,
        }
        if result.finish_reason == "length":
            detail["truncated"] = True
            log.warning("answer_truncated", max_tokens=self._settings.answer_max_tokens)
        limit = self._settings.ollama_num_ctx
        if result.metrics.prompt_tokens + self._settings.answer_max_tokens > limit:
            detail["near_context_limit"] = True
            log.warning(
                "prompt_near_context_limit",
                prompt_tokens=result.metrics.prompt_tokens,
                num_ctx=limit,
            )

        cited = {int(n) for n in _CITE_RE.findall(answer)}
        citations = [
            Citation(
                n=n,
                source=p.source,
                location=p.location,
                score=p.score,
                snippet=" ".join(p.text.split())[:200],
                cited=n in cited,
            )
            for n, p in enumerate(passages, start=1)
        ]
        return SynthesisResult(answer, citations, result.metrics, mode, detail)
