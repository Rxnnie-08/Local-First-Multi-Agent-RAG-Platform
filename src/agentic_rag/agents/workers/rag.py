"""RAG specialist: hybrid retrieval over the ingested documents, then cross-encoder re-ranking."""

from __future__ import annotations

import time

from agentic_rag.agents.models import Observation, Passage
from agentic_rag.agents.workers.base import WorkerTask
from agentic_rag.config import Settings
from agentic_rag.core.exceptions import RerankerError
from agentic_rag.core.logging import get_logger
from agentic_rag.observability import SpanKind, document_attributes, set_attributes, span
from agentic_rag.retrieval import HybridRetriever, RankedHit, Reranker

log = get_logger(__name__)


def _location(hit: RankedHit) -> str:
    meta = hit.hit.metadata
    if meta.get("heading_path"):
        return str(meta["heading_path"])
    if meta.get("page"):
        return f"page {meta['page']}"
    return ""


class RagWorker:
    name = "rag"
    description = (
        "Searches the user's ingested documents (PDFs, notes, source code) and returns the most "
        "relevant passages. Use for any question the user's own material might answer."
    )

    def __init__(self, retriever: HybridRetriever, reranker: Reranker, settings: Settings) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._settings = settings

    async def run(self, task: WorkerTask) -> Observation:
        s = self._settings
        started = time.perf_counter()
        retrieval = await self._retriever.retrieve(
            task.query, k=s.retrieval_candidates, mode="hybrid"
        )
        detail: dict[str, object] = {
            "candidates": len(retrieval.hits),
            "retrieval_s": round(time.perf_counter() - started, 3),
            "reranker": self._reranker.name,
        }
        if not retrieval.hits:
            return Observation(
                worker=self.name,
                query=task.query,
                ok=False,
                confident=False,
                summary="No documents matched (the collection may be empty).",
                detail=detail,
            )

        rerank_started = time.perf_counter()
        with span(
            "rerank",
            SpanKind.RERANKER,
            input_value=task.query,
            attributes={
                "reranker.model_name": self._reranker.name,
                "reranker.top_k": s.rerank_top_n,
                "reranker.input_count": len(retrieval.hits),
            },
        ) as rsp:
            try:
                ranked = await self._reranker.rerank(
                    task.query, retrieval.hits, top_n=s.rerank_top_n
                )
            except RerankerError as exc:
                log.warning("rerank_failed_using_retrieval_order", error=str(exc))
                ranked = [RankedHit(hit=h) for h in retrieval.hits[: s.rerank_top_n]]
                detail["rerank_error"] = str(exc)
                set_attributes(rsp, {"reranker.error": str(exc)})
            set_attributes(
                rsp,
                document_attributes(
                    "reranker.output_documents",
                    [
                        {
                            "id": r.hit.id,
                            "score": r.score,
                            "content": r.hit.text,
                            "metadata": {"source": r.hit.source},
                        }
                        for r in ranked
                    ],
                ),
            )
        detail["rerank_s"] = round(time.perf_counter() - rerank_started, 3)

        scores = [r.score for r in ranked if r.score is not None]
        top_score = max(scores) if scores else None
        detail["top_score"] = top_score
        # Unscored hits (no reranker) are kept; scored hits below the floor are noise.
        kept = [r for r in ranked if r.score is None or r.score >= s.rerank_min_passage_score]
        detail["dropped_below_floor"] = len(ranked) - len(kept)
        passages = [
            Passage(
                text=r.hit.text,
                source=r.hit.source,
                location=_location(r),
                score=r.score,
            )
            for r in kept
        ]
        if not passages:
            return Observation(
                worker=self.name,
                query=task.query,
                ok=True,
                confident=False,
                summary=(
                    f"Search ran but nothing relevant was found (best relevance {top_score:.4f})."
                    if top_score is not None
                    else "Search ran but nothing relevant was found."
                ),
                detail=detail,
            )
        # Without a reranker there is no absolute relevance signal, so retrieval is taken at face
        # value; the grounded answer prompt still refuses when the passages do not answer.
        confident = True if top_score is None else top_score >= s.rerank_min_score

        sources = sorted({p.source for p in passages})
        score_text = f", top relevance {top_score:.2f}" if top_score is not None else ""
        summary = f"{len(passages)} passages from {', '.join(sources)}{score_text}."
        if not confident:
            summary += " Relevance is low; these may not answer the question."
        return Observation(
            worker=self.name,
            query=task.query,
            ok=True,
            confident=confident,
            summary=summary,
            passages=passages,
            detail=detail,
        )
