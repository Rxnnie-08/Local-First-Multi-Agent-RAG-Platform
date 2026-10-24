from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.agents import AgentApp, RagWorker
from agentic_rag.api import create_app
from agentic_rag.api.app import Runtime
from agentic_rag.config import Settings
from agentic_rag.embeddings import Bm25SparseEncoder, OllamaEmbedder
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.llm import OllamaClient
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.services import Services
from agentic_rag.vectorstore import QdrantStore
from fakes import (
    FakeEmbedder,
    FakeOllama,
    FakeReranker,
    chat_response,
    error_response,
    route,
    stream_response,
)

SAMPLES = Path(__file__).resolve().parents[3] / "samples"
KEY = "s3cret-key"

Build = Callable[..., Awaitable[tuple[httpx.AsyncClient, FastAPI, FakeOllama]]]


@pytest.fixture
async def make_client(
    settings: Settings,
    store: QdrantStore,
    embedder: FakeEmbedder,
    sparse_encoder: Bm25SparseEncoder,
    retriever: HybridRetriever,
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> AsyncIterator[Build]:
    """An HTTP client talking straight to the ASGI app, with a scripted fake Ollama behind it."""
    opened: list[Any] = []

    async def build(
        script: list[httpx.Response | Exception],
        **overrides: Any,
    ) -> tuple[httpx.AsyncClient, FastAPI, FakeOllama]:
        cfg = settings.model_copy(update={"api_upload_dir": tmp_path / "uploads", **overrides})
        fake = FakeOllama(script)
        llm = OllamaClient(cfg, transport=fake.transport)
        reranker = FakeReranker(0.9)
        agent = AgentApp(client=llm, workers=[RagWorker(retriever, reranker, cfg)], settings=cfg)
        services = Services(
            settings=cfg,
            client=llm,
            embedder=cast("OllamaEmbedder", embedder),
            store=store,
            sparse=sparse_encoder,
            pipeline=pipeline,
            retriever=retriever,
            reranker=reranker,
            runner=None,
            search=None,
            agent=agent,
        )
        app = create_app(services=services)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api.test")
        opened.append((lifespan, http, llm))
        return http, app, fake

    yield build
    for lifespan, http, llm in opened:
        await http.aclose()
        await lifespan.__aexit__(None, None, None)
        await llm.aclose()


@pytest.fixture
async def ingested(pipeline: IngestionPipeline) -> None:
    await pipeline.ingest_path(SAMPLES)


def sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


# --- health -------------------------------------------------------------------------------------


async def test_health_needs_nothing(make_client: Build) -> None:
    http, _, fake = await make_client([])
    resp = await http.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert fake.requests == []


async def test_ready_reports_each_dependency(make_client: Build) -> None:
    tags = httpx.Response(200, json={"models": [{"name": "test-model:latest"}]})
    http, _, _ = await make_client([tags, tags])
    resp = await http.get("/ready")
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ready"
    assert body["ollama"] is True
    assert body["model_available"] is True
    assert body["qdrant"] is True
    assert body["workers"] == ["rag"]


async def test_ready_is_503_when_ollama_is_down(make_client: Build) -> None:
    http, _, _ = await make_client([httpx.ConnectError("refused")] * 3)
    resp = await http.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"
    assert resp.json()["ollama"] is False


# --- ask ------------------------------------------------------------------------------------------


async def test_ask_returns_answer_citations_and_a_thread_id(
    make_client: Build, ingested: None
) -> None:
    http, _, _ = await make_client(
        [route("rag", "default keep alive"), chat_response("It is 5 minutes [1].")]
    )
    resp = await http.post("/v1/ask", json={"question": "What is the default keep alive?"})
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == "It is 5 minutes [1]."
    assert body["route"] == ["supervisor", "rag", "supervisor", "synthesize"]
    assert body["citations"][0]["cited"] is True
    assert len(body["thread_id"]) == 32
    assert body["trace"] is None
    assert body["metrics"]["llm_calls"] == 2


async def test_ask_can_include_the_trace(make_client: Build, ingested: None) -> None:
    http, _, _ = await make_client([route("rag", "keep alive"), chat_response("Yes [1].")])
    resp = await http.post("/v1/ask?include_trace=true", json={"question": "keep alive?"})
    assert [s["node"] for s in resp.json()["trace"]][-1] == "synthesize"


async def test_the_same_thread_id_continues_the_conversation(
    make_client: Build, ingested: None
) -> None:
    http, _, fake = await make_client(
        [
            route("rag", "keep alive"),
            chat_response("Five minutes [1]."),
            route("answer"),
            chat_response("Follow-up."),
        ]
    )
    first = await http.post("/v1/ask", json={"question": "keep alive?", "thread_id": "abc"})
    await http.post("/v1/ask", json={"question": "and then?", "thread_id": "abc"})
    assert first.json()["thread_id"] == "abc"
    assert "keep alive?" in fake.json_body(2)["messages"][-1]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 4001},
        {"question": "hi", "thread_id": "has space"},
        {"question": "hi", "thread_id": "../etc"},
        {"question": "hi", "unexpected": 1},
        {},
    ],
)
async def test_bad_requests_are_422_and_never_reach_the_model(
    make_client: Build, payload: dict[str, Any]
) -> None:
    http, _, fake = await make_client([])
    resp = await http.post("/v1/ask", json=payload)
    assert resp.status_code == 422
    assert fake.requests == []


async def test_a_whitespace_question_is_422(make_client: Build) -> None:
    http, _, _ = await make_client([])
    assert (await http.post("/v1/ask", json={"question": "   "})).status_code == 422


async def test_a_busy_thread_is_409(make_client: Build) -> None:
    http, app, fake = await make_client([])
    runtime: Runtime = app.state.runtime
    async with runtime.thread_lock("busy"):
        resp = await http.post("/v1/ask", json={"question": "hi", "thread_id": "busy"})
    assert resp.status_code == 409
    assert fake.requests == []


async def test_the_slot_and_lock_are_released_after_a_failed_turn(
    make_client: Build, ingested: None
) -> None:
    http, app, _ = await make_client(
        [route("rag", "keep alive"), error_response(500, "boom")]
        + [error_response(500, "boom")] * 2
        + [route("answer"), chat_response("Recovered.")]
    )
    failed = await http.post("/v1/ask", json={"question": "keep alive?", "thread_id": "t"})
    assert failed.status_code == 502
    assert failed.json()["error"] == "OllamaServerError"

    ok = await http.post("/v1/ask", json={"question": "hello", "thread_id": "t"})
    assert ok.status_code == 200
    runtime: Runtime = app.state.runtime
    assert not runtime.thread_lock("t").locked()


async def test_unreachable_ollama_is_503(make_client: Build, ingested: None) -> None:
    http, _, _ = await make_client(
        [route("rag", "keep alive")] + [httpx.ConnectError("refused")] * 3
    )
    resp = await http.post("/v1/ask", json={"question": "keep alive?"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "OllamaConnectionError"


# --- auth -----------------------------------------------------------------------------------------


async def test_api_key_is_required_when_configured(make_client: Build, ingested: None) -> None:
    from pydantic import SecretStr

    http, _, _ = await make_client(
        [route("rag", "keep alive"), chat_response("Yes [1].")], api_key=SecretStr(KEY)
    )
    body = {"question": "keep alive?"}
    assert (await http.post("/v1/ask", json=body)).status_code == 401
    assert (await http.post("/v1/ask", json=body, headers={"X-API-Key": "nope"})).status_code == 401
    assert (await http.get("/health")).status_code == 200  # liveness stays open
    ok = await http.post("/v1/ask", json=body, headers={"X-API-Key": KEY})
    assert ok.status_code == 200


# --- streaming ------------------------------------------------------------------------------------


async def test_stream_sends_start_steps_tokens_then_final(
    make_client: Build, ingested: None
) -> None:
    http, app, _ = await make_client(
        [route("rag", "keep alive"), stream_response(["Five ", "minutes ", "[1]."])]
    )
    resp = await http.post("/v1/ask/stream", json={"question": "keep alive?", "thread_id": "s1"})
    events = sse_events(resp.text)
    names = [name for name, _ in events]

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert names[0] == "start"
    assert events[0][1] == {"thread_id": "s1"}
    assert names[-1] == "final"
    assert names.index("token") > names.index("step")
    assert "".join(d["text"] for n, d in events if n == "token") == "Five minutes [1]."
    final = events[-1][1]["response"]
    assert final["answer"] == "Five minutes [1]."
    assert final["trace"] == []  # omitted unless requested
    runtime: Runtime = app.state.runtime
    assert not runtime.thread_lock("s1").locked()  # released when the stream ended


async def test_stream_failure_is_an_error_event_and_releases_the_thread(
    make_client: Build, ingested: None
) -> None:
    http, app, _ = await make_client([route("rag", "keep alive"), error_response(500, "boom")])
    resp = await http.post("/v1/ask/stream", json={"question": "keep alive?", "thread_id": "s2"})
    events = sse_events(resp.text)

    assert events[-1][0] == "error"
    assert events[-1][1]["error_type"] == "OllamaServerError"
    assert "final" not in [n for n, _ in events]
    runtime: Runtime = app.state.runtime
    assert not runtime.thread_lock("s2").locked()


async def test_stream_on_a_busy_thread_is_a_real_409(make_client: Build) -> None:
    http, app, _ = await make_client([])
    runtime: Runtime = app.state.runtime
    async with runtime.thread_lock("busy"):
        resp = await http.post("/v1/ask/stream", json={"question": "hi", "thread_id": "busy"})
    assert resp.status_code == 409


# --- sessions -------------------------------------------------------------------------------------


async def test_delete_session_forgets_history(make_client: Build, ingested: None) -> None:
    http, _, fake = await make_client(
        [
            route("rag", "keep alive"),
            chat_response("Five minutes [1]."),
            route("answer"),
            chat_response("Fresh."),
        ]
    )
    await http.post("/v1/ask", json={"question": "keep alive?", "thread_id": "d"})
    assert (await http.delete("/v1/sessions/d")).status_code == 204
    await http.post("/v1/ask", json={"question": "and then?", "thread_id": "d"})
    assert "keep alive?" not in fake.json_body(2)["messages"][-1]["content"]


async def test_delete_session_rejects_a_bad_id_and_a_busy_thread(make_client: Build) -> None:
    http, app, _ = await make_client([])
    assert (await http.delete("/v1/sessions/bad%20id")).status_code == 422
    runtime: Runtime = app.state.runtime
    async with runtime.thread_lock("busy"):
        assert (await http.delete("/v1/sessions/busy")).status_code == 409


# --- documents ------------------------------------------------------------------------------------


def files(*items: tuple[str, bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, data, "application/octet-stream")) for name, data in items]


async def test_upload_ingests_then_skips_unchanged_and_lists(make_client: Build) -> None:
    http, _, _ = await make_client([])
    note = b"# Notes\n\nThe quarterly report is due on the 5th of every month.\n"

    first = await http.post("/v1/documents", files=files(("notes.md", note)))
    again = await http.post("/v1/documents", files=files(("notes.md", note)))
    listing = await http.get("/v1/documents")

    assert first.status_code == 200
    assert first.json()["ingested"] == 1
    assert first.json()["chunks"] >= 1
    assert again.json()["skipped"] == 1
    assert again.json()["ingested"] == 0
    assert [d["source"] for d in listing.json()] == ["notes.md"]


async def test_upload_rejects_unsupported_types(make_client: Build, tmp_path: Path) -> None:
    http, _, _ = await make_client([])
    resp = await http.post("/v1/documents", files=files(("run.exe", b"MZ")))
    assert resp.status_code == 415
    assert not (tmp_path / "uploads" / "run.exe").exists()


async def test_upload_cannot_escape_the_upload_directory(
    make_client: Build, tmp_path: Path
) -> None:
    http, _, _ = await make_client([])
    resp = await http.post(
        "/v1/documents", files=files(("../../evil.md", b"# Evil\n\nsome text here\n"))
    )
    assert resp.status_code == 200
    assert (tmp_path / "uploads" / "evil.md").exists()
    assert not (tmp_path / "evil.md").exists()
    assert not list(tmp_path.parent.glob("evil.md"))


async def test_upload_over_the_size_limit_is_413_and_leaves_no_file(
    make_client: Build, tmp_path: Path
) -> None:
    http, _, _ = await make_client([], api_max_upload_mb=1)
    big = b"a" * (1024 * 1024 + 1)
    resp = await http.post("/v1/documents", files=files(("big.md", big)))
    assert resp.status_code == 413
    assert not (tmp_path / "uploads" / "big.md").exists()


async def test_upload_rejects_duplicate_names_in_one_request(make_client: Build) -> None:
    http, _, _ = await make_client([])
    resp = await http.post("/v1/documents", files=files(("a.md", b"# A"), ("a.md", b"# B")))
    assert resp.status_code == 422


async def test_uploads_require_the_api_key_too(make_client: Build) -> None:
    from pydantic import SecretStr

    http, _, _ = await make_client([], api_key=SecretStr(KEY))
    assert (await http.post("/v1/documents", files=files(("a.md", b"# A")))).status_code == 401
    assert (await http.get("/v1/documents")).status_code == 401


async def test_concurrent_turns_are_limited_by_the_slot_semaphore(
    make_client: Build, ingested: None
) -> None:
    http, app, _ = await make_client(
        [
            route("rag", "keep alive"),
            chat_response("A [1]."),
            route("rag", "chunk size"),
            chat_response("B [1]."),
        ]
    )
    runtime: Runtime = app.state.runtime
    assert runtime.turn_slots._value == 1  # default: one turn at a time
    a, b = await asyncio.gather(
        http.post("/v1/ask", json={"question": "keep alive?", "thread_id": "a"}),
        http.post("/v1/ask", json={"question": "chunk size?", "thread_id": "b"}),
    )
    assert a.status_code == b.status_code == 200
    assert runtime.turn_slots._value == 1


# --- chat page ------------------------------------------------------------------------------------


async def test_chat_page_is_served_with_a_strict_content_security_policy(
    make_client: Build,
) -> None:
    http, _, fake = await make_client([])
    resp = await http.get("/")
    csp = resp.headers["content-security-policy"]

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<title>Agentic RAG chat</title>" in resp.text
    assert "script-src 'self'" in csp
    assert "'unsafe-inline'" not in csp
    assert "frame-ancestors 'none'" in csp
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "<script>" not in resp.text  # no inline script: the policy would block it
    assert fake.requests == []


async def test_chat_assets_are_served_and_the_page_needs_no_api_key(make_client: Build) -> None:
    from pydantic import SecretStr

    http, _, _ = await make_client([], api_key=SecretStr(KEY))
    assert (await http.get("/")).status_code == 200  # the page holds no data; calls need the key
    js = await http.get("/static/chat.js")
    css = await http.get("/static/chat.css")
    assert js.status_code == 200
    assert "/v1/ask/stream" in js.text
    assert css.status_code == 200
    assert "#f4eff7" in css.text
    assert (await http.get("/static/../app.py")).status_code in (404, 400)


def test_the_chat_script_never_builds_markup_from_text() -> None:
    from agentic_rag.api.app import STATIC_DIR

    js = (STATIC_DIR / "chat.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js  # model and document text must only ever become text nodes
    assert "eval(" not in js
    assert "insertAdjacentHTML" not in js
