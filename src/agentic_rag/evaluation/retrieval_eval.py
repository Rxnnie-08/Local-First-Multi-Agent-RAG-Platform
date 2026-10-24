"""Retrieval benchmark: dense vs sparse vs hybrid vs hybrid + rerank, without any LLM calls.

Answers two questions with data instead of intuition: does hybrid search actually beat dense on
this corpus, and where should RERANK_MIN_SCORE sit.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.evaluation.dataset import EvalCase
from agentic_rag.evaluation.scoring import mean, normalize
from agentic_rag.retrieval import HybridRetriever, Reranker
from agentic_rag.vectorstore.models import SearchHit

MODES = ("dense", "sparse", "hybrid", "hybrid+rerank")


class CaseRanks(BaseModel):
    id: str
    category: str
    query: str
    ranks: dict[str, int | None]  # mode -> 1-based rank of the first chunk holding the evidence
    top_score: float | None  # best reranker score for this query (None without a reranker)


class ModeMetrics(BaseModel):
    mode: str
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    mean_latency_s: float


class Calibration(BaseModel):
    """How well the reranker's top score separates answerable from unanswerable questions."""

    n_positive: int
    n_negative: int
    min_positive: float
    max_negative: float
    separable: bool
    recommended: float
    best_accuracy: float
    current: float
    current_accuracy: float


class RetrievalReport(BaseModel):
    created: str
    collection: str
    reranker: str
    k: int
    modes: list[ModeMetrics]
    by_category_hit_at_5: dict[str, dict[str, float]]  # category -> mode -> hit@5
    cases: list[CaseRanks]
    calibration: Calibration | None


def is_hit(case: EvalCase, hit: SearchHit) -> bool:
    return (
        case.evidence is not None
        and Path(hit.source).name in case.sources
        and normalize(case.evidence) in normalize(hit.text)
    )


def first_hit_rank(case: EvalCase, hits: Sequence[SearchHit]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_hit(case, hit):
            return rank
    return None


def _metrics(mode: str, ranks: Sequence[int | None], latencies: Sequence[float]) -> ModeMetrics:
    n = len(ranks)

    def at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0

    mrr = mean([1.0 / r if r else 0.0 for r in ranks])
    return ModeMetrics(
        mode=mode,
        n=n,
        hit_at_1=at(1),
        hit_at_3=at(3),
        hit_at_5=at(5),
        mrr=mrr,
        mean_latency_s=mean(latencies),
    )


def calibrate(
    positives: Sequence[float], negatives: Sequence[float], current: float
) -> Calibration | None:
    if not positives or not negatives:
        return None

    def accuracy(t: float) -> float:
        right = sum(1 for s in positives if s >= t) + sum(1 for s in negatives if s < t)
        return right / (len(positives) + len(negatives))

    lo, hi = max(negatives), min(positives)
    separable = lo < hi
    if separable:
        # Geometric mean: scores span orders of magnitude (1e-5 .. 0.99), so the arithmetic
        # midpoint would sit far too close to the positives.
        recommended = math.sqrt(lo * hi) if lo > 0 else hi / 2
    else:
        scores = sorted({*positives, *negatives})
        candidates = [(a + b) / 2 for a, b in itertools.pairwise(scores)] or [current]
        # Ties go to the HIGHER threshold: a false "confident" lets irrelevant passages produce a
        # confident wrong answer, while a false "not confident" only costs an extra search.
        recommended = max(candidates, key=lambda t: (accuracy(t), t))
    return Calibration(
        n_positive=len(positives),
        n_negative=len(negatives),
        min_positive=hi,
        max_negative=lo,
        separable=separable,
        recommended=recommended,
        best_accuracy=accuracy(recommended),
        current=current,
        current_accuracy=accuracy(current),
    )


async def run_retrieval_eval(
    retriever: HybridRetriever,
    reranker: Reranker,
    cases: Sequence[EvalCase],
    settings: Settings,
    *,
    collection: str,
    k: int = 10,
) -> RetrievalReport:
    scored = [c for c in cases if c.is_retrieval_case]
    negatives_cases = [c for c in cases if c.abstain]
    ranks_by_mode: dict[str, list[int | None]] = defaultdict(list)
    latency_by_mode: dict[str, list[float]] = defaultdict(list)
    rows: list[CaseRanks] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    async def run_case(case: EvalCase) -> tuple[dict[str, int | None], float | None]:
        query = case.retrieval_query
        ranks: dict[str, int | None] = {}
        for mode in ("dense", "sparse", "hybrid"):
            started = time.perf_counter()
            result = await retriever.retrieve(query, k=k, mode=mode)
            latency_by_mode[mode].append(time.perf_counter() - started)
            ranks[mode] = first_hit_rank(case, result.hits)
        started = time.perf_counter()
        candidates = await retriever.retrieve(query, k=settings.retrieval_candidates, mode="hybrid")
        ranked = await reranker.rerank(query, candidates.hits, top_n=k)
        latency_by_mode["hybrid+rerank"].append(time.perf_counter() - started)
        ranks["hybrid+rerank"] = first_hit_rank(case, [r.hit for r in ranked])
        scores = [r.score for r in ranked if r.score is not None]
        return ranks, (max(scores) if scores else None)

    for case in scored:
        ranks, top = await run_case(case)
        for mode, rank in ranks.items():
            ranks_by_mode[mode].append(rank)
        if top is not None:
            positive_scores.append(top)
        rows.append(
            CaseRanks(
                id=case.id,
                category=case.category,
                query=case.retrieval_query,
                ranks=ranks,
                top_score=top,
            )
        )
    for case in negatives_cases:
        ranks, top = await run_case(case)
        if top is not None:
            negative_scores.append(top)
        rows.append(
            CaseRanks(
                id=case.id,
                category=case.category,
                query=case.retrieval_query,
                ranks={},
                top_score=top,
            )
        )

    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({c.category for c in scored}):
        subset = [r for r in rows if r.category == category and r.ranks]
        by_category[category] = {
            mode: sum(1 for r in subset if (rk := r.ranks.get(mode)) is not None and rk <= 5)
            / len(subset)
            for mode in MODES
        }
    return RetrievalReport(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        collection=collection,
        reranker=reranker.name,
        k=k,
        modes=[_metrics(m, ranks_by_mode[m], latency_by_mode[m]) for m in MODES],
        by_category_hit_at_5=by_category,
        cases=rows,
        calibration=calibrate(positive_scores, negative_scores, settings.rerank_min_score),
    )


def format_retrieval_report(report: RetrievalReport) -> str:
    lines = [
        f"Retrieval benchmark ({report.cases[0].id if report.cases else '-'}..; "
        f"reranker: {report.reranker}; k={report.k})",
        "",
        f"{'mode':<15}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}{'latency':>10}",
    ]
    for m in report.modes:
        lines.append(
            f"{m.mode:<15}{m.n:>4}{m.hit_at_1:>8.2f}{m.hit_at_3:>8.2f}{m.hit_at_5:>8.2f}"
            f"{m.mrr:>8.3f}{m.mean_latency_s * 1000:>8.0f}ms"
        )
    lines += ["", "hit@5 by question type:", f"{'':<16}" + "".join(f"{m:>15}" for m in MODES)]
    for category, per_mode in report.by_category_hit_at_5.items():
        lines.append(f"{category:<16}" + "".join(f"{per_mode[m]:>15.2f}" for m in MODES))
    c = report.calibration
    if c is not None:
        lines += [
            "",
            "Reranker threshold (is the top rerank score high enough to answer?):",
            f"  answerable questions:   lowest top score {c.min_positive:.4f}  (n={c.n_positive})",
            f"  unanswerable questions: highest top score {c.max_negative:.4f}  (n={c.n_negative})",
            f"  separable: {'yes' if c.separable else 'NO'};  suggested RERANK_MIN_SCORE "
            f"= {c.recommended:.3f} (accuracy {c.best_accuracy:.0%})",
            f"  current {c.current:.2f} scores {c.current_accuracy:.0%}",
        ]
    misses = [r for r in report.cases if r.ranks and r.ranks.get("hybrid+rerank") is None]
    if misses:
        lines += ["", "Not found in the top results after reranking:"]
        lines += [f"  {r.id}: {r.query}" for r in misses]
    return "\n".join(lines)
