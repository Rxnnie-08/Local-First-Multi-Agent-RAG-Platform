from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from agentic_rag.agents.models import AgentResponse, Citation, RunMetrics, StepRecord
from agentic_rag.config import Settings
from agentic_rag.evaluation.agent_eval import (
    evaluate_case,
    format_agent_report,
    run_agent_eval,
    save_report,
    summarize,
)
from agentic_rag.evaluation.dataset import EvalCase, load_cases
from agentic_rag.evaluation.retrieval_eval import (
    calibrate,
    first_hit_rank,
    format_retrieval_report,
    run_retrieval_eval,
)
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.vectorstore.models import SearchHit
from fakes import FakeReranker

ROOT = Path(__file__).resolve().parents[3]
CASES = load_cases(ROOT / "eval" / "cases.jsonl")


def case(**kw: object) -> EvalCase:
    base: dict[str, object] = {
        "id": "c",
        "category": "rag_fact",
        "question": "q",
        "expect_first": "rag",
    }
    return EvalCase.model_validate(base | kw)


def response(
    answer: str,
    route: Sequence[str],
    citations: Sequence[Citation] = (),
    trace: Sequence[StepRecord] = (),
) -> AgentResponse:
    return AgentResponse(
        thread_id="t",
        question="q",
        answer=answer,
        route=list(route),
        citations=list(citations),
        metrics=RunMetrics(llm_calls=2, prompt_tokens=100, completion_tokens=20),
        trace=list(trace),
    )


def cite(source: str) -> Citation:
    return Citation(n=1, source=source, snippet="s", cited=True)


# --- retrieval ------------------------------------------------------------------------------


def hit(text: str, source: str = "corpus/orbit_handbook.md") -> SearchHit:
    return SearchHit(id="1", score=1.0, text=text, source=source, chunk_index=0)


def test_first_hit_rank_needs_the_right_file_and_the_evidence() -> None:
    c = case(evidence="port 8421", sources=["orbit_handbook.md"])
    hits = [
        hit("unrelated"),
        hit("The server uses  PORT 8421.", "corpus/orbit_api.md"),  # right text, wrong file
        hit("The server uses  PORT 8421."),  # whitespace and case are normalised
    ]
    assert first_hit_rank(c, hits) == 3
    assert first_hit_rank(c, hits[:2]) is None


def test_calibrate_separable_picks_the_geometric_mean() -> None:
    cal = calibrate([0.99, 0.86, 0.90], [0.00002, 0.0001], current=0.25)
    assert cal is not None
    assert cal.separable
    assert cal.recommended == pytest.approx((0.0001 * 0.86) ** 0.5)
    assert cal.best_accuracy == 1.0
    assert cal.current_accuracy == 1.0


def test_calibrate_overlap_reports_it_and_picks_best_accuracy() -> None:
    cal = calibrate([0.9, 0.2, 0.8], [0.3, 0.01], current=0.25)
    assert cal is not None
    assert not cal.separable
    assert cal.best_accuracy >= cal.current_accuracy
    assert 0.2 < cal.recommended < 0.9


def test_calibrate_needs_both_classes() -> None:
    assert calibrate([0.9], [], current=0.25) is None
    assert calibrate([], [0.1], current=0.25) is None


async def test_retrieval_benchmark_on_the_real_corpus(
    pipeline: IngestionPipeline, retriever: HybridRetriever, settings: Settings
) -> None:
    report = await pipeline.ingest_path(ROOT / "eval" / "corpus")
    assert report.failed == 0

    result = await run_retrieval_eval(
        retriever, FakeReranker(0.9), CASES, settings, collection="test"
    )
    by_mode = {m.mode: m for m in result.modes}

    assert set(by_mode) == {"dense", "sparse", "hybrid", "hybrid+rerank"}
    n_scored = len([c for c in CASES if c.is_retrieval_case])
    assert all(m.n == n_scored for m in result.modes)
    # With a lexical encoder and a bag-of-words fake embedder this corpus is easy: if hybrid
    # cannot find most facts the plumbing (not the models) is broken.
    assert by_mode["hybrid"].hit_at_5 >= 0.7
    assert by_mode["sparse"].hit_at_5 >= 0.5
    assert 0.0 <= by_mode["hybrid"].mrr <= 1.0
    # ORB-2210 is an exact identifier: sparse must retrieve it
    l01 = next(r for r in result.cases if r.id == "l01")
    assert l01.ranks["sparse"] is not None
    assert result.calibration is None or result.calibration.n_negative == 4
    assert "hybrid+rerank" in format_retrieval_report(result)


# --- agent evaluation -----------------------------------------------------------------------


def test_correct_rag_answer_passes() -> None:
    c = case(sources=["orbit_handbook.md"], answer_contains=["8421"])
    r = evaluate_case(
        c,
        response(
            "It listens on 8421 [1].",
            ["supervisor", "rag", "supervisor", "synthesize"],
            [cite("corpus/orbit_handbook.md")],
        ),
        latency_s=3.0,
    )
    assert r.passed
    assert r.route_ok
    assert r.answer_ok
    assert r.source_hit
    assert r.first == "rag"


def test_wrong_route_fails_even_if_the_answer_is_right() -> None:
    c = case(answer_contains=["8421"])
    r = evaluate_case(c, response("8421", ["supervisor", "web", "synthesize"]), latency_s=1)
    assert not r.passed
    assert not r.route_ok
    assert r.answer_ok


def test_wrong_answer_fails_with_the_right_route() -> None:
    c = case(answer_contains=["8421"])
    r = evaluate_case(c, response("It is 80.", ["supervisor", "rag", "synthesize"]), latency_s=1)
    assert r.route_ok
    assert r.answer_ok is False
    assert not r.passed


def test_abstain_case_needs_an_abstention() -> None:
    c = case(abstain=True)
    good = evaluate_case(
        c,
        response("I found no relevant information.", ["supervisor", "rag", "synthesize"]),
        latency_s=1,
    )
    bad = evaluate_case(
        c, response("The CEO is Jane Doe.", ["supervisor", "rag", "synthesize"]), latency_s=1
    )
    assert good.passed
    assert good.abstain_ok
    assert not bad.passed
    assert bad.abstain_ok is False


def test_web_case_needs_a_url_citation() -> None:
    c = case(category="web", expect_first="web", live=True)
    route_ = ["supervisor", "web", "synthesize"]
    ok = evaluate_case(c, response("3.15", route_, [cite("https://python.org")]), latency_s=1)
    no_url = evaluate_case(c, response("3.15", route_, [cite("python sandbox")]), latency_s=1)
    assert ok.passed
    assert not no_url.passed


def test_direct_case_needs_a_nonempty_answer_and_no_worker() -> None:
    c = case(category="direct", expect_first="answer")
    assert evaluate_case(c, response("Hello!", ["supervisor", "synthesize"]), latency_s=1).passed
    assert not evaluate_case(c, response("  ", ["supervisor", "synthesize"]), latency_s=1).passed
    assert not evaluate_case(
        c, response("Hello!", ["supervisor", "rag", "synthesize"]), latency_s=1
    ).passed


def test_latency_is_split_by_node() -> None:
    trace = [
        StepRecord(node="supervisor", latency_s=2.0),
        StepRecord(node="rag", latency_s=0.5),
        StepRecord(node="supervisor", latency_s=0.0),
        StepRecord(node="synthesize", latency_s=4.0),
    ]
    c = case(category="direct", expect_first="answer")
    r = evaluate_case(c, response("x", [t.node for t in trace], trace=trace), latency_s=7)
    assert (r.supervisor_s, r.worker_s, r.synthesize_s) == (2.0, 0.5, 4.0)


class ScriptedAgent:
    """Answers each case from a dict; records seeding calls; can raise for one case."""

    def __init__(self, answers: dict[str, AgentResponse], explode_on: str | None = None) -> None:
        self._answers = answers
        self._explode_on = explode_on
        self.seeded: dict[str, Sequence[tuple[str, str]]] = {}
        self.threads: list[str] = []

    async def seed_history(self, thread_id: str, turns: Sequence[tuple[str, str]]) -> None:
        self.seeded[thread_id] = turns

    async def ask(self, question: str, *, thread_id: str = "default") -> AgentResponse:
        self.threads.append(thread_id)
        if question == self._explode_on:
            raise RuntimeError("model fell over")
        return self._answers[question]


async def test_run_agent_eval_scores_seeds_history_and_survives_errors(
    settings: Settings,
) -> None:
    cases = [
        case(id="a", question="qa", answer_contains=["yes"]),
        case(
            id="b",
            question="qb",
            history=[{"question": "earlier", "answer": "before"}],
            answer_contains=["yes"],
        ),
        case(id="c", question="qc", answer_contains=["yes"]),
    ]
    ok = ["supervisor", "rag", "synthesize"]
    agent = ScriptedAgent(
        {"qa": response("yes", ok), "qb": response("no", ok), "qc": response("yes", ok)},
        explode_on="qc",
    )
    seen: list[str] = []
    report = await run_agent_eval(
        agent, cases, settings, progress=lambda i, n, r: seen.append(f"{i}/{n}:{r.id}")
    )

    assert [r.passed for r in report.results] == [True, False, False]
    assert report.results[2].error is not None
    assert "fell over" in report.results[2].error
    assert agent.seeded == {"eval-b": [("earlier", "before")]}
    assert agent.threads == ["eval-a", "eval-b", "eval-c"]  # one thread per case: no leakage
    assert seen == ["1/3:a", "2/3:b", "3/3:c"]
    assert report.summary.errors == 1
    assert report.model == settings.ollama_model

    text = format_agent_report(report)
    assert "Failures:" in text
    assert "model fell over" in text
    path = save_report(report, Path(ROOT / "eval" / "reports" / "_unit"), "test/model:1")
    try:
        assert path.exists()
        assert path.name.endswith("_test-model-1.json")
    finally:
        path.unlink()
        path.parent.rmdir()


def test_summary_rates_and_categories() -> None:
    ok = ["supervisor", "rag", "synthesize"]
    results = [
        evaluate_case(case(id="1", answer_contains=["a"]), response("a", ok), latency_s=2.0),
        evaluate_case(case(id="2", answer_contains=["a"]), response("b", ok), latency_s=4.0),
        evaluate_case(
            case(id="3", category="abstain", abstain=True), response("no relevant", ok), latency_s=6
        ),
    ]
    s = summarize(results)
    assert (s.n, s.passed) == (3, 2)
    assert s.routing_accuracy == 1.0
    assert s.answer_accuracy == 0.5
    assert s.abstention_rate == 1.0
    assert s.latency_mean_s == pytest.approx(4.0)
    assert s.by_category["rag_fact"].answer_ok == 1
    assert s.by_category["rag_fact"].answer_checked == 2
    assert summarize([]).pass_rate == 0.0
