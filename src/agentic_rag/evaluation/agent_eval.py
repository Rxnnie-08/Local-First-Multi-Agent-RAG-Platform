"""End-to-end evaluation of the agent: routing, grounding, abstention, speed and token usage."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentic_rag.agents.models import AgentResponse
from agentic_rag.config import Settings
from agentic_rag.evaluation.dataset import EvalCase, HistoryTurn
from agentic_rag.evaluation.scoring import (
    answer_matches,
    first_step,
    is_abstention,
    mean,
    percentile,
)


class Asker(Protocol):
    async def ask(self, question: str, *, thread_id: str = "default") -> AgentResponse: ...

    async def seed_history(self, thread_id: str, turns: Sequence[tuple[str, str]]) -> None: ...


class CaseResult(BaseModel):
    id: str
    category: str
    question: str
    answer: str = ""
    route: list[str] = Field(default_factory=list)
    expected_first: str
    first: str = ""
    route_ok: bool = False
    # None means "not applicable to this case"
    source_hit: bool | None = None  # an expected source file was among the cited passages
    answer_ok: bool | None = None  # required strings appear in the answer
    abstain_ok: bool | None = None  # the system said it could not find the answer
    # Answerable document question where the system refused instead of answering.
    false_refusal: bool | None = None
    web_ok: bool | None = None  # a web answer cited at least one URL
    passed: bool = False
    error: str | None = None
    latency_s: float = 0.0
    supervisor_s: float = 0.0
    worker_s: float = 0.0
    synthesize_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float | None = None


def evaluate_case(case: EvalCase, response: AgentResponse, *, latency_s: float) -> CaseResult:
    """Score one response. Pure, so the scoring rules are unit-testable without a model."""
    first = first_step(response.route)
    result = CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=response.answer,
        route=response.route,
        expected_first=case.expect_first,
        first=first,
        route_ok=first == case.expect_first,
        latency_s=latency_s,
        llm_calls=response.metrics.llm_calls,
        prompt_tokens=response.metrics.prompt_tokens,
        completion_tokens=response.metrics.completion_tokens,
        tokens_per_second=response.metrics.tokens_per_second,
    )
    for step in response.trace:
        if step.node == "supervisor":
            result.supervisor_s += step.latency_s
        elif step.node == "synthesize":
            result.synthesize_s += step.latency_s
        else:
            result.worker_s += step.latency_s

    if case.sources:
        result.source_hit = any(Path(c.source).name in case.sources for c in response.citations)
    result.answer_ok = answer_matches(
        response.answer, case.answer_contains, case.answer_min_fraction
    )
    if case.abstain:
        result.abstain_ok = is_abstention(response.answer)
    if case.evidence is not None and not case.abstain and case.expect_first == "rag":
        result.false_refusal = is_abstention(response.answer) and result.answer_ok is not True
    if case.category == "web":
        result.web_ok = bool(response.answer.strip()) and any(
            re.match(r"https?://", c.source) for c in response.citations
        )
    if case.category == "direct":
        result.answer_ok = bool(response.answer.strip())
    checks = [result.route_ok, result.answer_ok, result.abstain_ok, result.web_ok]
    result.passed = all(c is not False for c in checks)
    return result


class CategorySummary(BaseModel):
    n: int
    passed: int
    route_ok: int
    answer_ok: int | None  # of the cases that have an answer check
    answer_checked: int


class Summary(BaseModel):
    n: int
    passed: int
    pass_rate: float
    routing_accuracy: float
    source_hit_rate: float | None
    abstention_rate: float | None
    false_refusal_rate: float | None = None
    answer_accuracy: float | None
    errors: int
    latency_mean_s: float
    latency_median_s: float
    latency_p95_s: float
    mean_tokens_per_second: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    by_category: dict[str, CategorySummary]


def _rate(flags: Sequence[bool | None]) -> float | None:
    checked = [f for f in flags if f is not None]
    return sum(checked) / len(checked) if checked else None


def summarize(results: Sequence[CaseResult]) -> Summary:
    by_cat: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    categories: dict[str, CategorySummary] = {}
    for name, rs in sorted(by_cat.items()):
        answered = [r.answer_ok for r in rs if r.answer_ok is not None]
        categories[name] = CategorySummary(
            n=len(rs),
            passed=sum(r.passed for r in rs),
            route_ok=sum(r.route_ok for r in rs),
            answer_ok=sum(answered) if answered else None,
            answer_checked=len(answered),
        )
    latencies = [r.latency_s for r in results]
    tps = [r.tokens_per_second for r in results if r.tokens_per_second]
    n = len(results)
    return Summary(
        n=n,
        passed=sum(r.passed for r in results),
        pass_rate=sum(r.passed for r in results) / n if n else 0.0,
        routing_accuracy=sum(r.route_ok for r in results) / n if n else 0.0,
        source_hit_rate=_rate([r.source_hit for r in results]),
        abstention_rate=_rate([r.abstain_ok for r in results]),
        false_refusal_rate=_rate([r.false_refusal for r in results]),
        answer_accuracy=_rate([r.answer_ok for r in results if r.category != "direct"]),
        errors=sum(r.error is not None for r in results),
        latency_mean_s=mean(latencies),
        latency_median_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        mean_tokens_per_second=mean(tps) if tps else None,
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        by_category=categories,
    )


class AgentReport(BaseModel):
    created: str
    model: str
    config: dict[str, Any]
    summary: Summary
    results: list[CaseResult]


def config_snapshot(settings: Settings) -> dict[str, Any]:
    keys = (
        "ollama_model",
        "ollama_num_ctx",
        "ollama_temperature",
        "embedding_model",
        "chunk_max_tokens",
        "chunk_overlap_tokens",
        "reranker",
        "rerank_model",
        "rerank_min_score",
        "rerank_min_passage_score",
        "rerank_top_n",
        "retrieval_candidates",
        "graph_max_steps",
        "answer_max_tokens",
        "answer_context_tokens",
    )
    return {k: getattr(settings, k) for k in keys}


async def run_agent_eval(
    agent: Asker,
    cases: Sequence[EvalCase],
    settings: Settings,
    *,
    progress: Callable[[int, int, CaseResult], None] | None = None,
) -> AgentReport:
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        thread = f"eval-{case.id}"
        started = time.perf_counter()
        try:
            if case.history:
                await agent.seed_history(thread, [(t.question, t.answer) for t in case.history])
            response = await agent.ask(case.question, thread_id=thread)
            result = evaluate_case(case, response, latency_s=time.perf_counter() - started)
        except Exception as exc:  # one broken case must not abort a 10-minute run
            result = CaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                expected_first=case.expect_first,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started,
            )
        results.append(result)
        if progress:
            progress(index, len(cases), result)
    return AgentReport(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=settings.ollama_model,
        config=config_snapshot(settings),
        summary=summarize(results),
        results=results,
    )


def format_agent_report(report: AgentReport) -> str:
    s = report.summary

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    lines = [
        f"Agent evaluation: model {report.model}, {s.n} cases",
        "",
        f"  passed (route and answer both right): {s.passed}/{s.n} ({s.pass_rate:.0%})",
        f"  routing accuracy:                     {pct(s.routing_accuracy)}",
        f"  answer accuracy (where checkable):    {pct(s.answer_accuracy)}",
        f"  source hit rate (rag questions):      {pct(s.source_hit_rate)}",
        f"  abstention (unanswerable questions):  {pct(s.abstention_rate)}",
        f"  false refusals (answerable, refused): {pct(s.false_refusal_rate)}",
        f"  errors: {s.errors}",
        "",
        f"  latency  mean {s.latency_mean_s:.1f}s  median {s.latency_median_s:.1f}s  "
        f"p95 {s.latency_p95_s:.1f}s",
        f"  speed    {s.mean_tokens_per_second or 0:.1f} tok/s mean;  tokens "
        f"{s.total_prompt_tokens} prompt + {s.total_completion_tokens} completion",
        "",
        f"{'category':<16}{'n':>4}{'passed':>8}{'routed':>8}{'answer':>10}",
    ]
    for name, c in s.by_category.items():
        answer = f"{c.answer_ok}/{c.answer_checked}" if c.answer_checked else "-"
        lines.append(f"{name:<16}{c.n:>4}{c.passed:>8}{c.route_ok:>8}{answer:>10}")
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines += ["", "Failures:"]
        for r in failed:
            why = (
                [f"error {r.error}"]
                if r.error
                else [
                    *(
                        [f"routed {r.first} (expected {r.expected_first})"]
                        if not r.route_ok
                        else []
                    ),
                    *(["wrong answer"] if r.answer_ok is False else []),
                    *(["did not abstain"] if r.abstain_ok is False else []),
                    *(["no web citation"] if r.web_ok is False else []),
                ]
            )
            lines.append(f"  {r.id} [{r.category}] {r.question!r}: {', '.join(why)}")
            lines.append(f"      answer: {' '.join(r.answer.split())[:140]!r}")
    return "\n".join(lines)


def save_report(report: AgentReport | BaseModel, directory: Path, label: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}_{safe}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


__all__ = [
    "AgentReport",
    "Asker",
    "CaseResult",
    "HistoryTurn",
    "Summary",
    "evaluate_case",
    "format_agent_report",
    "run_agent_eval",
    "save_report",
    "summarize",
]
