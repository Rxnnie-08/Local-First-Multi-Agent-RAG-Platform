"""Composition root: builds and tears down the object graph in one place."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

from agentic_rag.agents import AgentApp, CodeWorker, RagWorker, WebWorker, Worker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.exceptions import RerankerError, SandboxError
from agentic_rag.embeddings import Bm25SparseEncoder, OllamaEmbedder
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.llm import OllamaClient
from agentic_rag.observability import setup_tracing, shutdown_tracing
from agentic_rag.retrieval import HybridRetriever, NoopReranker, Reranker, build_reranker
from agentic_rag.sandbox import CodeRunner, DockerSandbox
from agentic_rag.vectorstore import QdrantStore
from agentic_rag.websearch import SearchProvider, build_search_provider


def make_agent(
    *,
    settings: Settings,
    client: OllamaClient,
    retriever: HybridRetriever,
    reranker: Reranker,
    runner: CodeRunner | None,
    search: SearchProvider | None,
) -> AgentApp:
    """Register workers. Order matters: the first is the routing fallback, so RAG stays first.
    A worker whose dependency is None (disabled or unavailable) is simply not registered, which
    also removes it from the supervisor's route enum."""
    workers: list[Worker] = [RagWorker(retriever, reranker, settings)]
    if search is not None:
        workers.append(WebWorker(search, settings))
    if runner is not None:
        workers.append(CodeWorker(client, runner, settings))
    return AgentApp(client=client, workers=workers, settings=settings)


@dataclass(frozen=True)
class Services:
    settings: Settings
    client: OllamaClient
    embedder: OllamaEmbedder
    store: QdrantStore
    sparse: Bm25SparseEncoder
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    reranker: Reranker
    runner: CodeRunner | None
    search: SearchProvider | None
    agent: AgentApp


@asynccontextmanager
async def open_services(settings: Settings | None = None) -> AsyncIterator[Services]:
    s = settings or get_settings()
    setup_tracing(s)  # no-op unless TRACING_ENABLED=true
    client = OllamaClient(s)
    store = QdrantStore(s)
    embedder = OllamaEmbedder(client)
    sparse = Bm25SparseEncoder(k1=s.sparse_k1, b=s.sparse_b, avg_doc_len=s.sparse_avg_doc_len)
    retriever = HybridRetriever(store, embedder, sparse, s)
    reranker = build_reranker(s)
    runner: CodeRunner | None = DockerSandbox(s) if s.code_worker_enabled else None
    search = build_search_provider(s)
    agent = make_agent(
        settings=s,
        client=client,
        retriever=retriever,
        reranker=reranker,
        runner=runner,
        search=search,
    )
    try:
        yield Services(
            settings=s,
            client=client,
            embedder=embedder,
            store=store,
            sparse=sparse,
            pipeline=IngestionPipeline(store, embedder, sparse, s),
            retriever=retriever,
            reranker=reranker,
            runner=runner,
            search=search,
            agent=agent,
        )
    finally:
        await client.aclose()
        await store.aclose()
        shutdown_tracing()  # flush the last turn before exit


async def prepare(svc: Services, warn: Callable[[str], None] = print) -> Services:
    """Warm up the optional components, and drop the ones that cannot work right now, with a
    warning, rather than failing on the first question that needs them."""
    reranker = svc.reranker
    try:
        await reranker.warmup()
    except RerankerError as exc:
        warn(f"WARNING: {exc}\nContinuing without re-ranking (answers will be less precise).")
        reranker = NoopReranker()

    runner = svc.runner
    if runner is not None:
        try:
            await runner.check()
        except SandboxError as exc:
            warn(f"WARNING: code worker disabled. {exc}")
            runner = None

    if reranker is svc.reranker and runner is svc.runner:
        return svc
    agent = make_agent(
        settings=svc.settings,
        client=svc.client,
        retriever=svc.retriever,
        reranker=reranker,
        runner=runner,
        search=svc.search,
    )
    return replace(svc, reranker=reranker, runner=runner, agent=agent)
