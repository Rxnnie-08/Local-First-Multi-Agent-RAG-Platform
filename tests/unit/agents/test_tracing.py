from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentic_rag.agents import AgentApp, RagWorker
from agentic_rag.config import Settings
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.llm import OllamaClient
from agentic_rag.observability import (
    configure_tracer,
    phoenix_reachable,
    setup_tracing,
    shutdown_tracing,
    span,
)
from agentic_rag.observability.tracing import SpanKind
from agentic_rag.retrieval import HybridRetriever
from fakes import FakeOllama, FakeReranker, chat_response, route

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    configure_tracer(provider)
    yield exp
    configure_tracer(None)


@pytest.fixture
async def app(
    settings: Settings, retriever: HybridRetriever, pipeline: IngestionPipeline
) -> AsyncIterator[Any]:
    await pipeline.ingest_path(SAMPLES)
    clients: list[OllamaClient] = []

    def build(script: list[Any]) -> AgentApp:
        client = OllamaClient(settings, transport=FakeOllama(script).transport)
        clients.append(client)
        return AgentApp(
            client=client,
            workers=[RagWorker(retriever, FakeReranker(0.9), settings)],
            settings=settings,
        )

    yield build
    for c in clients:
        await c.aclose()


def by_name(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def attr(s: ReadableSpan, key: str) -> Any:
    assert s.attributes is not None
    return s.attributes.get(key)


def parent_name(s: ReadableSpan, spans: list[ReadableSpan]) -> str | None:
    if s.parent is None:
        return None
    return next(p.name for p in spans if p.context.span_id == s.parent.span_id)


async def test_a_turn_produces_a_nested_span_tree(app: Any, exporter: InMemorySpanExporter) -> None:
    agent = app([route("rag", "default keep alive"), chat_response("Five minutes [1].")])
    resp = await agent.ask("What is the default keep alive?", thread_id="t-42")
    spans = list(exporter.get_finished_spans())

    (turn,) = by_name(spans, "agent.turn")
    assert turn.parent is None
    assert attr(turn, "openinference.span.kind") == "AGENT"
    assert attr(turn, "session.id") == "t-42"
    assert attr(turn, "input.value") == "What is the default keep alive?"
    assert attr(turn, "output.value") == "Five minutes [1]."
    assert attr(turn, "agent.route") == "supervisor -> rag -> supervisor -> synthesize"
    assert attr(turn, "llm.token_count.total") == resp.metrics.total_tokens

    assert parent_name(by_name(spans, "supervisor")[0], spans) == "agent.turn"
    assert parent_name(by_name(spans, "worker.rag")[0], spans) == "agent.turn"
    assert parent_name(by_name(spans, "retrieve")[0], spans) == "worker.rag"
    assert parent_name(by_name(spans, "rerank")[0], spans) == "worker.rag"
    assert parent_name(by_name(spans, "synthesize")[0], spans) == "agent.turn"
    # LLM calls nest under the node that made them
    llm_parents = {parent_name(s, spans) for s in by_name(spans, "ollama.chat")}
    assert llm_parents == {"supervisor", "synthesize"}


async def test_llm_spans_carry_model_tokens_and_messages(
    app: Any, exporter: InMemorySpanExporter
) -> None:
    agent = app([route("answer"), chat_response("Hello!")])
    await agent.ask("hi")
    spans = list(exporter.get_finished_spans())

    llm = by_name(spans, "ollama.chat")[-1]
    assert attr(llm, "openinference.span.kind") == "LLM"
    assert attr(llm, "llm.model_name") == "test-model"
    assert attr(llm, "llm.token_count.prompt") == 10
    assert attr(llm, "llm.token_count.completion") == 20
    assert attr(llm, "llm.output_messages.0.message.content") == "Hello!"
    assert attr(llm, "llm.input_messages.0.message.role") == "system"
    assert attr(llm, "llm.tokens_per_second") == pytest.approx(20.0)


async def test_retriever_span_records_documents(app: Any, exporter: InMemorySpanExporter) -> None:
    agent = app([route("rag", "keep alive"), chat_response("x")])
    await agent.ask("keep alive?")

    retrieve = by_name(list(exporter.get_finished_spans()), "retrieve")[0]
    assert attr(retrieve, "openinference.span.kind") == "RETRIEVER"
    assert attr(retrieve, "retrieval.documents.0.document.content")
    assert attr(retrieve, "retrieval.documents.0.document.score") is not None


async def test_capture_content_false_hides_text_but_keeps_numbers(
    app: Any, exporter: InMemorySpanExporter
) -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_tracer(provider, capture_content=False)
    agent = app([route("answer"), chat_response("secret answer")])
    await agent.ask("secret question")

    for s in exporter.get_finished_spans():
        assert s.attributes is not None
        text = " ".join(str(v) for v in s.attributes.values())
        assert "secret" not in text
    llm = by_name(list(exporter.get_finished_spans()), "ollama.chat")[-1]
    assert attr(llm, "llm.token_count.prompt") == 10


async def test_untraced_app_still_works(app: Any) -> None:
    configure_tracer(None)
    resp = await app([route("answer"), chat_response("ok")]).ask("hi")
    assert resp.answer == "ok"


def test_exceptions_are_recorded_and_reraised(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), span("boom", SpanKind.CHAIN):
        raise RuntimeError("bad")
    (s,) = exporter.get_finished_spans()
    assert s.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in s.events)


def test_setup_tracing_is_disabled_by_default(settings: Settings) -> None:
    assert setup_tracing(settings) is None


def test_setup_tracing_installs_a_provider_and_shutdown_resets(settings: Settings) -> None:
    enabled = settings.model_copy(update={"tracing_enabled": True, "tracing_project": "unit-test"})
    provider = setup_tracing(enabled)
    try:
        assert provider is not None
        assert provider.resource.attributes["openinference.project.name"] == "unit-test"
    finally:
        configure_tracer(None)  # skip the network flush: no Phoenix is running here


def test_shutdown_without_setup_is_safe() -> None:
    configure_tracer(None)
    shutdown_tracing()


async def test_phoenix_reachable_is_false_for_a_closed_port() -> None:
    assert not await phoenix_reachable("http://127.0.0.1:9/v1/traces", timeout_s=1.0)
