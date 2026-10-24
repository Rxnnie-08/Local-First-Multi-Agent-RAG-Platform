"""Second-stage re-ranking: a cross-encoder rescored the top candidates from hybrid search.

Hybrid search is a fast, recall-oriented first stage. A cross-encoder reads the query and each
passage together, which is far more precise but too slow to run over the whole collection, so it
only reorders the shortlist. Its 0-1 relevance score is also the only usable confidence signal in
this system: RRF scores are rank-based and say nothing about absolute relevance.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import RerankerError
from agentic_rag.core.logging import get_logger
from agentic_rag.vectorstore.models import SearchHit

log = get_logger(__name__)


class RankedHit(BaseModel):
    hit: SearchHit
    score: float | None = None  # reranker relevance in [0, 1]; None when not reranked


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def warmup(self) -> None: ...

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], *, top_n: int
    ) -> list[RankedHit]: ...


class NoopReranker:
    """Keeps the first-stage order. Scores are None, so callers cannot judge confidence."""

    name = "none"

    async def warmup(self) -> None:
        return None

    async def rerank(self, query: str, hits: Sequence[SearchHit], *, top_n: int) -> list[RankedHit]:
        return [RankedHit(hit=h) for h in hits[:top_n]]


def _scoring_text(hit: SearchHit) -> str:
    heading = hit.metadata.get("heading_path")
    return f"{heading}\n{hit.text}" if heading else hit.text


class FlashRankReranker:
    """FlashRank ONNX cross-encoder. The first use downloads the model (CC-BY-SA) from
    huggingface.co into `cache_dir`; after that it runs fully offline on CPU."""

    name = "flashrank"

    def __init__(self, *, model_name: str, cache_dir: str, max_length: int = 512) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._max_length = max_length
        self._ranker: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._ranker is None:
                try:
                    from flashrank import Ranker  # lazy: importing pulls in onnxruntime

                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    self._ranker = Ranker(
                        model_name=self._model_name,
                        cache_dir=str(self._cache_dir),
                        max_length=self._max_length,
                        log_level="WARNING",  # Ranker calls logging.basicConfig on the root logger
                    )
                except Exception as exc:  # download, zip or onnx failures all look different
                    raise RerankerError(
                        f"Could not load FlashRank model {self._model_name!r}. The first run "
                        f"downloads it from huggingface.co into {self._cache_dir}; check your "
                        f"connection, or set RERANKER=none. Cause: {type(exc).__name__}: {exc}"
                    ) from exc
                log.info("reranker_loaded", model=self._model_name)
            return self._ranker

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def rerank(self, query: str, hits: Sequence[SearchHit], *, top_n: int) -> list[RankedHit]:
        if not hits:
            return []
        ranker = await asyncio.to_thread(self._load)
        passages = [{"id": i, "text": _scoring_text(h)} for i, h in enumerate(hits)]

        def score() -> list[dict[str, Any]]:
            from flashrank import RerankRequest

            result: list[dict[str, Any]] = ranker.rerank(
                RerankRequest(query=query, passages=passages)
            )
            return result

        try:
            results = await asyncio.to_thread(score)
        except Exception as exc:
            raise RerankerError(f"FlashRank scoring failed: {type(exc).__name__}: {exc}") from exc
        ranked = [RankedHit(hit=hits[int(r["id"])], score=float(r["score"])) for r in results]
        return ranked[:top_n]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "none":
        return NoopReranker()
    return FlashRankReranker(model_name=settings.rerank_model, cache_dir=settings.rerank_cache_dir)
