from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_rag.agents import AgentApp, Observation, RagWorker, WorkerTask
from agentic_rag.config import Settings
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.llm import OllamaClient
from agentic_rag.retrieval import HybridRetriever
from fakes import FakeOllama, FakeReranker, chat_response, route

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


@pytest.fixture
async def ingested(pipeline: IngestionPipeline) -> None:
    await pipeline.ingest_path(SAMPLES)


class CrashingWorker:
    name = "crash"
    description = "Always fails."

    async def run(self, task: WorkerTask) -> Observation:
        raise RuntimeError("tool exploded")


@pytest.fixture
async def make_app(settings: Settings, retriever: HybridRetriever) -> AsyncIterator[Any]:
    clients: list[OllamaClient] = []

    def build(
        script: list[httpx.Response | Exception],
        *,
        reranker: FakeReranker | None = None,
        extra_workers: list[Any] | None = None,
        cfg: Settings | None = None,
    ) -> tuple[AgentApp, FakeOllama, FakeReranker]:
        s = cfg or settings
        fake = FakeOllama(script)
        client = OllamaClient(s, transport=fake.transport)
        clients.append(client)
        rr = reranker or FakeReranker(0.9)
        workers: list[Any] = [RagWorker(retriever, rr, s), *(extra_workers or [])]
        return AgentApp(client=client, workers=workers, settings=s), fake, rr

    yield build
    for client in clients:
        await client.aclose()


async def test_rag_fast_path_uses_two_llm_calls(make_app: Any, ingested: None) -> None:
    app, fake, _ = make_app(
        [route("rag", "default keep alive"), chat_response("It is 5 minutes [1].")]
    )
    resp = await app.ask("What is the default keep alive?")

    assert resp.route == ["supervisor", "rag", "supervisor", "synthesize"]
    assert len(fake.requests) == 2  # second supervisor decision came from policy, not the LLM
    assert resp.answer == "It is 5 minutes [1]."
    assert resp.metrics.llm_calls == 2
    assert resp.citations
    assert resp.citations[0].cited is True
    assert all(c.cited is False for c in resp.citations[1:])
    sources = [s.detail.get("source") for s in resp.trace if s.node == "supervisor"]
    assert sources == ["llm", "policy"]


async def test_direct_route_skips_workers(make_app: Any) -> None:
    app, fake, rr = make_app([route("answer"), chat_response("Hello!")])
    resp = await app.ask("hi there")

    assert resp.route == ["supervisor", "synthesize"]
    assert resp.citations == []
    assert rr.calls == 0
    assert resp.trace[-1].detail["mode"] == "direct"
    assert len(fake.requests) == 2


async def test_low_confidence_triggers_second_llm_decision(make_app: Any, ingested: None) -> None:
    app, fake, rr = make_app(
        [
            route("rag", "keep alive"),
            route("rag", "ollama keep_alive default duration"),  # reworded retry
            route("answer"),
            chat_response("Probably 5 minutes [1]."),
        ],
        reranker=FakeReranker(0.1),
    )
    resp = await app.ask("keep alive?")

    assert resp.route == ["supervisor", "rag", "supervisor", "rag", "supervisor", "synthesize"]
    assert rr.calls == 2
    assert len(fake.requests) == 4  # 3 routing decisions (none by policy) + the answer
    assert [s.detail["source"] for s in resp.trace if s.node == "supervisor"] == ["llm"] * 3
    assert resp.trace[-1].detail["mode"] == "grounded"


async def test_repeated_query_forces_answer(make_app: Any, ingested: None) -> None:
    app, _, rr = make_app(
        [route("rag", "keep alive?"), route("rag", "Keep Alive?"), chat_response("Unsure.")],
        reranker=FakeReranker(0.1),
    )
    resp = await app.ask("keep alive?")

    assert rr.calls == 1
    assert resp.route[-1] == "synthesize"
    assert resp.trace[2].detail["source"] == "policy"
    assert "already ran" in resp.trace[2].detail["reason"]


async def test_step_limit_forces_answer(make_app: Any, ingested: None, settings: Settings) -> None:
    cfg = settings.model_copy(update={"graph_max_steps": 2})
    app, fake, _ = make_app(
        [
            route("rag", "a one"),
            route("rag", "a two"),
            chat_response("Best effort."),
        ],
        reranker=FakeReranker(0.1),
        cfg=cfg,
    )
    resp = await app.ask("anything")

    assert resp.route == ["supervisor", "rag", "supervisor", "rag", "supervisor", "synthesize"]
    assert len(fake.requests) == 3
    assert resp.trace[4].detail["reason"] == "step limit reached"


async def test_supervisor_failure_falls_back_to_rag(make_app: Any, ingested: None) -> None:
    bad = chat_response("not json at all")
    app, fake, _ = make_app([bad, bad, chat_response("Fallback answer [1].")])
    resp = await app.ask("what is a chunk?")

    assert resp.route == ["supervisor", "rag", "supervisor", "synthesize"]
    assert resp.trace[0].detail["source"] == "fallback"
    assert resp.answer == "Fallback answer [1]."
    assert len(fake.requests) == 3  # initial + 1 repair, then answer


async def test_no_results_mode_on_empty_collection(make_app: Any) -> None:
    app, fake, _ = make_app(
        [route("rag", "anything"), route("answer"), chat_response("Nothing found.")]
    )
    resp = await app.ask("anything?")

    assert resp.trace[-1].detail["mode"] == "no_results"
    assert resp.citations == []
    system = fake.json_body(-1)["messages"][0]["content"]
    assert "found nothing relevant" in system


async def test_rerank_failure_degrades_to_retrieval_order(make_app: Any, ingested: None) -> None:
    app, _, _ = make_app(
        [route("rag", "chunk overlap"), chat_response("Answer [1].")],
        reranker=FakeReranker(fail=True),
    )
    resp = await app.ask("chunk overlap?")

    rag_step = next(s for s in resp.trace if s.node == "rag")
    assert "rerank_error" in rag_step.detail
    assert resp.citations  # still answered from retrieval order
    assert resp.route == ["supervisor", "rag", "supervisor", "synthesize"]


async def test_unscored_hits_count_as_confident(make_app: Any, ingested: None) -> None:
    app, fake, _ = make_app(
        [route("rag", "chunk overlap"), chat_response("Answer [1].")],
        reranker=FakeReranker(None),
    )
    resp = await app.ask("chunk overlap?")
    assert len(fake.requests) == 2
    assert resp.citations[0].score is None


async def test_history_persists_per_thread_and_is_isolated(make_app: Any) -> None:
    app, fake, _ = make_app(
        [
            route("answer"),
            chat_response("Nice to meet you, Ana."),
            route("answer"),
            chat_response("Your name is Ana."),
            route("answer"),
            chat_response("I do not know."),
        ]
    )
    await app.ask("My name is Ana.", thread_id="t1")
    await app.ask("What is my name?", thread_id="t1")
    await app.ask("What is my name?", thread_id="t2")

    second_supervisor_prompt = fake.json_body(2)["messages"][-1]["content"]
    assert "My name is Ana." in second_supervisor_prompt
    second_answer_messages = fake.json_body(3)["messages"]
    assert any("My name is Ana." in m["content"] for m in second_answer_messages)
    other_thread_messages = fake.json_body(5)["messages"]
    assert not any("Ana" in m["content"] for m in other_thread_messages[:-1])


async def test_per_turn_state_resets_but_trace_is_per_turn(make_app: Any) -> None:
    app, _, _ = make_app(
        [route("answer"), chat_response("one"), route("answer"), chat_response("two")]
    )
    first = await app.ask("q1", thread_id="t")
    second = await app.ask("q2", thread_id="t")
    assert first.route == second.route == ["supervisor", "synthesize"]
    assert second.answer == "two"


async def test_worker_exception_is_contained(make_app: Any) -> None:
    app, _, _ = make_app(
        [route("crash", "do it"), route("answer"), chat_response("Sorry, the tool failed.")],
        extra_workers=[CrashingWorker()],
    )
    resp = await app.ask("run the crash tool")

    crash = next(s for s in resp.trace if s.node == "crash")
    assert crash.detail["ok"] is False
    assert "RuntimeError" in crash.detail["error"]
    assert resp.answer == "Sorry, the tool failed."


async def test_decision_schema_lists_registered_workers_in_order(make_app: Any) -> None:
    app, fake, _ = make_app([route("answer"), chat_response("hi")])
    await app.ask("hello")

    schema = fake.json_body(0)["format"]
    assert schema["properties"]["next"]["enum"] == ["rag", "answer"]
    assert next(iter(schema["properties"])) == "reason"


async def test_context_budget_trims_passages(
    make_app: Any, ingested: None, settings: Settings
) -> None:
    cfg = settings.model_copy(update={"answer_context_tokens": 300, "rerank_top_n": 5})
    app, fake, _ = make_app(
        [route("rag", "ollama chunk embedding"), chat_response("Answer [1].")], cfg=cfg
    )
    resp = await app.ask("ollama chunk embedding?")

    answer_step = resp.trace[-1]
    assert answer_step.detail["passages_dropped"] >= 1
    prompt = fake.json_body(-1)["messages"][-1]["content"]
    assert len(prompt) < 300 * 4 + 400


async def test_truncated_answer_is_flagged(make_app: Any) -> None:
    app, _, _ = make_app([route("answer"), chat_response("cut off", done_reason="length")])
    resp = await app.ask("hello")
    assert resp.trace[-1].detail["truncated"] is True


async def test_empty_question_rejected(make_app: Any) -> None:
    app, _, _ = make_app([])
    with pytest.raises(ValueError, match="empty"):
        await app.ask("   ")


def test_reserved_and_duplicate_worker_names_rejected(
    settings: Settings, retriever: HybridRetriever
) -> None:
    client = OllamaClient(settings, transport=FakeOllama([]).transport)
    rag = RagWorker(retriever, FakeReranker(), settings)
    with pytest.raises(ValueError, match="unique"):
        AgentApp(client=client, workers=[rag, rag], settings=settings)
    with pytest.raises(ValueError, match="at least one"):
        AgentApp(client=client, workers=[], settings=settings)


async def test_irrelevant_passages_are_dropped_and_answer_is_no_results(
    make_app: Any, ingested: None
) -> None:
    app, fake, _ = make_app(
        [
            route("rag", "refund policy"),
            chat_response("Nothing about refunds."),
        ],
        reranker=FakeReranker(0.00001),
    )
    resp = await app.ask("What is the refund policy?")

    rag_steps = [s for s in resp.trace if s.node == "rag"]
    assert len(rag_steps) == 1  # no pointless reworded retry
    assert len(fake.requests) == 2  # one routing call + the answer
    assert rag_steps[0].detail["passages"] == 0
    assert rag_steps[0].detail["dropped_below_floor"] == 5
    assert resp.trace[2].detail["source"] == "policy"
    assert resp.trace[-1].detail["mode"] == "no_results"
    assert resp.citations == []
    assert "found nothing relevant" in fake.json_body(-1)["messages"][0]["content"]


async def test_drifted_query_is_replaced_by_the_question(make_app: Any, ingested: None) -> None:
    app, _, _ = make_app(
        [route("rag", "default keep alive"), chat_response("Not in the documents.")],
        reranker=FakeReranker(0.9),
    )
    resp = await app.ask("What is the capital of France?")

    rag_step = next(s for s in resp.trace if s.node == "rag")
    assert rag_step.detail["query"] == "What is the capital of France?"
    assert "query replaced" in resp.trace[0].detail["reason"]


async def test_rewrite_is_kept_for_follow_up_questions(make_app: Any, ingested: None) -> None:
    app, _, _ = make_app(
        [route("rag", "how to change the default keep alive"), chat_response("Use it [1].")],
    )
    resp = await app.ask("How do I change it?")
    rag_step = next(s for s in resp.trace if s.node == "rag")
    assert rag_step.detail["query"] == "how to change the default keep alive"


async def test_duplicate_passages_across_retries_are_merged(
    make_app: Any, ingested: None, settings: Settings
) -> None:
    app, fake, _ = make_app(
        [
            route("rag", "chunk size"),
            route("rag", "chunking size tokens"),
            route("answer"),
            chat_response("Answer."),
        ],
        reranker=FakeReranker(0.1),
    )
    resp = await app.ask("chunk size?")
    answer_step = resp.trace[-1]
    unique = len(resp.citations)
    assert answer_step.detail["passages_used"] == unique
    prompt = fake.json_body(-1)["messages"][-1]["content"]
    assert prompt.count("[1] (") == 1
    assert unique <= 5 + 5


class ProbeWorker:
    name = "probe"
    description = "Another source."

    async def run(self, task: WorkerTask) -> Observation:
        return Observation(
            worker="probe", query=task.query, ok=True, confident=True, summary="found"
        )


async def test_empty_search_consults_llm_when_another_worker_is_untried(
    make_app: Any, ingested: None
) -> None:
    app, fake, _ = make_app(
        [route("rag", "refund policy"), route("probe", "refund policy"), chat_response("Done.")],
        reranker=FakeReranker(0.00001),
        extra_workers=[ProbeWorker()],
    )
    resp = await app.ask("What is the refund policy?")

    assert resp.route == ["supervisor", "rag", "supervisor", "probe", "supervisor", "synthesize"]
    assert len(fake.requests) == 3


async def test_first_question_is_searched_verbatim_not_rewritten(
    make_app: Any, ingested: None
) -> None:
    app, _, _ = make_app([route("rag", "keep alive default"), chat_response("Five minutes [1].")])
    resp = await app.ask("How long does the model stay loaded by default?")
    rag_step = next(s for s in resp.trace if s.node == "rag")
    assert rag_step.detail["query"] == "How long does the model stay loaded by default?"


# --- streaming a turn ---------------------------------------------------------------------------


async def test_ask_stream_yields_steps_then_tokens_then_the_final_response(
    make_app: Any, ingested: None
) -> None:
    from agentic_rag.agents import FinalEvent, StepEvent, TokenEvent
    from fakes import stream_response

    app, fake, _ = make_app(
        [route("rag", "default keep alive"), stream_response(["It is ", "5 minutes ", "[1]."])]
    )
    events = [e async for e in app.ask_stream("What is the default keep alive?")]

    kinds = [e.type for e in events]
    assert kinds[-1] == "final"
    assert kinds.count("final") == 1
    assert kinds.index("token") > kinds.index("step")  # steps arrive before the answer starts
    steps = [e.node for e in events if isinstance(e, StepEvent)]
    assert steps == ["supervisor", "rag", "supervisor", "synthesize"]
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert text == "It is 5 minutes [1]." == final.response.answer
    assert final.response.route == steps
    assert fake.json_body(0)["stream"] is False  # routing is not streamed
    assert fake.json_body(1)["stream"] is True  # the answer is


async def test_ask_does_not_stream_the_answer(make_app: Any, ingested: None) -> None:
    app, fake, _ = make_app([route("rag", "keep alive"), chat_response("Five minutes [1].")])
    await app.ask("keep alive?")
    assert fake.json_body(-1)["stream"] is False


async def test_ask_stream_reports_failures_as_an_error_event(make_app: Any, ingested: None) -> None:
    from agentic_rag.agents import ErrorEvent
    from fakes import error_response

    app, _, _ = make_app([route("rag", "keep alive"), error_response(500, "model crashed")])
    events = [e async for e in app.ask_stream("keep alive?")]

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].error_type == "OllamaServerError"
    assert "final" not in [e.type for e in events]


async def test_ask_stream_rejects_an_empty_question(make_app: Any) -> None:
    app, _, _ = make_app([])
    with pytest.raises(ValueError, match="empty"):
        _ = [e async for e in app.ask_stream("   ")]


async def test_closing_the_stream_early_cancels_the_turn_cleanly(
    make_app: Any, ingested: None
) -> None:
    from fakes import stream_response

    app, _, _ = make_app([route("rag", "keep alive"), stream_response(["a", "b", "c"])])
    stream = app.ask_stream("keep alive?")
    first = await stream.__anext__()
    assert first.type == "step"
    await stream.aclose()  # a disconnecting client must not leave the task running or raise


async def test_reset_thread_forgets_the_conversation(make_app: Any, ingested: None) -> None:
    app, fake, _ = make_app(
        [
            route("rag", "keep alive"),
            chat_response("Five minutes [1]."),
            route("answer"),
            chat_response("Fresh start."),
        ]
    )
    await app.ask("keep alive?", thread_id="t")
    await app.reset_thread("t")
    await app.ask("and then?", thread_id="t")
    prompt = fake.json_body(2)["messages"][-1]["content"]
    assert "keep alive?" not in prompt  # the earlier turn is gone from the supervisor's history


async def test_a_turn_is_stopped_when_its_consumer_is_cancelled_the_way_a_web_server_does(
    make_app: Any,
) -> None:
    """Starlette cancels a disconnected client's request with an anyio scope, which keeps
    re-raising inside the cancelled task. The turn (and its worker) must still be stopped."""
    import anyio

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowWorker:
        name = "slow"
        description = "Takes a while."

        async def run(self, task: WorkerTask) -> Observation:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return Observation(worker="slow", query="q", ok=True, confident=True, summary="done")

    app, _, _ = make_app([route("slow", "q")], extra_workers=[SlowWorker()])

    async def consume() -> None:
        async for _ in app.ask_stream("hello there"):
            pass

    with anyio.CancelScope() as scope:
        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 5)
        scope.cancel()
        await task
    await asyncio.wait_for(cancelled.wait(), 5)
