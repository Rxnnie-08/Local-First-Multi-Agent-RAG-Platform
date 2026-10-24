"""FastAPI application.

    GET    /health              liveness (no dependencies)
    GET    /ready               Ollama, the model and Qdrant are reachable
    POST   /v1/ask              one turn, JSON
    POST   /v1/ask/stream       one turn, Server-Sent Events: step* token* final | error
    POST   /v1/documents        upload files to ingest (multipart)
    GET    /v1/documents        what is ingested
    DELETE /v1/sessions/{id}    forget a conversation

There is deliberately no endpoint that ingests a path on the server: anyone who can reach the API
could read any file the process can. Uploads go to one fixed directory with sanitized names.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import secrets
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from agentic_rag.agents import AgentEvent, ErrorEvent, FinalEvent
from agentic_rag.api.schemas import (
    THREAD_ID_PATTERN,
    AskRequest,
    AskResponse,
    ErrorResponse,
    Health,
    IngestResponse,
    Readiness,
    SourceCount,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.exceptions import (
    AgenticRAGError,
    OllamaConnectionError,
    OllamaTimeoutError,
)
from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.parsers import classify
from agentic_rag.services import Services, open_services, prepare

log = get_logger(__name__)

_CHUNK = 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"
# The chat page loads only its own script and stylesheet and talks only to this server.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


class Runtime:
    """What the request handlers share. Built once at startup."""

    def __init__(self, services: Services) -> None:
        self.services = services
        settings = services.settings
        self.turn_slots = asyncio.Semaphore(settings.api_max_concurrent_turns)
        self.ingest_lock = asyncio.Lock()
        self._thread_locks: dict[str, asyncio.Lock] = {}

    def thread_lock(self, thread_id: str) -> asyncio.Lock:
        return self._thread_locks.setdefault(thread_id, asyncio.Lock())

    def release_thread(self, thread_id: str) -> None:
        lock = self._thread_locks.get(thread_id)
        if lock is not None and not lock.locked():
            del self._thread_locks[thread_id]  # do not grow without bound


class _StreamingResponse(StreamingResponse):
    """A StreamingResponse that always runs `on_close` when it ends, however it ends.

    Starlette does not close the body generator when a client disconnects mid-stream, so without
    this the turn would keep running (and the thread stay locked) until garbage collection.
    """

    def __init__(
        self, content: AsyncIterator[str], *, on_close: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        super().__init__(content, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


class Lease:
    """A thread lock plus a turn slot. `release` is idempotent, because a streamed response can
    be torn down by its generator finishing, by the client disconnecting, or both."""

    def __init__(self, runtime: Runtime, thread_id: str) -> None:
        self._runtime = runtime
        self._thread_id = thread_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._runtime.turn_slots.release()
        self._runtime.thread_lock(self._thread_id).release()
        self._runtime.release_thread(self._thread_id)


def get_runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


def _require_key(
    runtime: RuntimeDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = runtime.services.settings.api_key
    if expected is None:
        return
    supplied = (x_api_key or "").encode()
    if not secrets.compare_digest(supplied, expected.get_secret_value().encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


def _status_for(exc: AgenticRAGError) -> int:
    if isinstance(exc, OllamaConnectionError):
        return 503
    if isinstance(exc, OllamaTimeoutError):
        return 504
    return 502


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _with_heartbeat(
    stream: AsyncIterator[AgentEvent], interval: float, deadline: float
) -> AsyncGenerator[AgentEvent | None, None]:
    """Yield the stream's events, and None whenever `interval` seconds pass without one, so the
    connection is not closed as idle during the long routing and retrieval steps. Raises
    TimeoutError at the deadline."""
    pending: asyncio.Task[AgentEvent] | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if pending is None:
                pending = asyncio.ensure_future(anext(stream))
            done, _ = await asyncio.wait({pending}, timeout=min(interval, remaining))
            if not done:
                yield None
                continue
            task, pending = pending, None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    """Build the app. With no `services` it opens its own (Ollama, Qdrant, workers) at startup and
    closes them at shutdown; tests pass prebuilt ones."""
    cfg = settings or (services.settings if services else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.runtime = Runtime(services)
            yield
            return
        async with open_services(cfg) as svc:
            svc = await prepare(svc, warn=lambda msg: log.warning("startup_warning", message=msg))
            app.state.runtime = Runtime(svc)
            yield

    app = FastAPI(title="Agentic RAG", version="0.6.0", lifespan=lifespan)

    origins = [o.strip() for o in cfg.api_cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    @app.exception_handler(AgenticRAGError)
    async def _backend_error(_: Request, exc: AgenticRAGError) -> JSONResponse:
        log.warning("request_failed", error_type=type(exc).__name__, error=str(exc))
        body = ErrorResponse(error=type(exc).__name__, detail=str(exc))
        return JSONResponse(status_code=_status_for(exc), content=body.model_dump())

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "chat.html", headers=_PAGE_HEADERS)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        return Health()

    @app.get("/ready", response_model=Readiness)
    async def ready(runtime: RuntimeDep, response: Response) -> Readiness:
        svc = runtime.services
        ollama = await svc.client.health_check()
        model_ok = ollama and await svc.client.has_model(svc.settings.ollama_model)
        qdrant = await svc.store.health_check()
        documents: int | None = None
        if qdrant:
            with contextlib.suppress(AgenticRAGError):
                documents = await svc.store.count()
        ok = ollama and model_ok and qdrant
        if not ok:
            response.status_code = 503
        return Readiness(
            status="ready" if ok else "not_ready",
            ollama=ollama,
            model=svc.settings.ollama_model,
            model_available=model_ok,
            qdrant=qdrant,
            documents=documents,
            workers=svc.agent.worker_names,
            reranker=svc.reranker.name,
        )

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(_require_key)])

    async def _acquire(runtime: Runtime, thread_id: str) -> Lease:
        """Take the thread's lock (fail fast if a turn is already running on it) and a turn slot."""
        lock = runtime.thread_lock(thread_id)
        if lock.locked():
            raise HTTPException(
                status_code=409, detail=f"A turn is already running on thread {thread_id!r}."
            )
        await lock.acquire()
        try:
            await runtime.turn_slots.acquire()
        except BaseException:
            lock.release()
            runtime.release_thread(thread_id)
            raise
        return Lease(runtime, thread_id)

    @v1.post("/ask", response_model=AskResponse, responses={409: {"model": ErrorResponse}})
    async def ask(
        body: AskRequest,
        runtime: RuntimeDep,
        include_trace: Annotated[bool, Query()] = False,
    ) -> AskResponse:
        thread_id = body.thread_id or uuid.uuid4().hex
        lease = await _acquire(runtime, thread_id)
        try:
            result = await asyncio.wait_for(
                runtime.services.agent.ask(body.question, thread_id=thread_id),
                timeout=runtime.services.settings.api_turn_timeout_s,
            )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="The turn timed out.") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            lease.release()
        return AskResponse(
            thread_id=result.thread_id,
            answer=result.answer,
            route=result.route,
            citations=result.citations,
            metrics=result.metrics,
            trace=result.trace if include_trace else None,
        )

    @v1.post("/ask/stream", responses={409: {"model": ErrorResponse}})
    async def ask_stream(
        request: Request,
        body: AskRequest,
        runtime: RuntimeDep,
        include_trace: Annotated[bool, Query()] = False,
    ) -> StreamingResponse:
        settings = runtime.services.settings
        thread_id = body.thread_id or uuid.uuid4().hex
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        # Acquire before the response starts, so "busy" is a real HTTP 409 and not a stream error.
        lease = await _acquire(runtime, thread_id)

        async def events() -> AsyncGenerator[str, None]:
            stream = runtime.services.agent.ask_stream(question, thread_id=thread_id)
            deadline = time.monotonic() + settings.api_turn_timeout_s
            tick = min(settings.api_heartbeat_s, 1.0)  # how often a vanished client is looked for
            last_sent = time.monotonic()
            try:
                yield _sse("start", json.dumps({"thread_id": thread_id}))
                # aclosing: leaving this loop early must stop the turn, not leave it running
                async with contextlib.aclosing(_with_heartbeat(stream, tick, deadline)) as beats:
                    async for event in beats:
                        if event is None:
                            if await request.is_disconnected():
                                return  # stops the turn through the cleanup below
                            if time.monotonic() - last_sent >= settings.api_heartbeat_s:
                                last_sent = time.monotonic()
                                yield ": keep-alive\n\n"
                            continue
                        if isinstance(event, FinalEvent) and not include_trace:
                            trimmed = event.response.model_copy(update={"trace": []})
                            event = FinalEvent(response=trimmed)
                        yield _sse(event.type, event.model_dump_json())
            except TimeoutError:
                err = ErrorEvent(error_type="TimeoutError", message="The turn timed out.")
                yield _sse("error", err.model_dump_json())
            finally:
                lease.release()

        chunks = events()

        async def close() -> None:
            with contextlib.suppress(Exception):
                await chunks.aclose()  # runs events()' cleanup if the client left mid-stream
            lease.release()

        return _StreamingResponse(
            chunks,
            on_close=close,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @v1.delete("/sessions/{thread_id}", status_code=204)
    async def delete_session(thread_id: str, runtime: RuntimeDep) -> Response:
        if not _valid_thread_id(thread_id):
            raise HTTPException(status_code=422, detail="Invalid thread id.")
        if runtime.thread_lock(thread_id).locked():
            raise HTTPException(status_code=409, detail="A turn is running on this thread.")
        await runtime.services.agent.reset_thread(thread_id)
        runtime.release_thread(thread_id)
        return Response(status_code=204)

    @v1.get("/documents", response_model=list[SourceCount])
    async def list_documents(runtime: RuntimeDep) -> list[SourceCount]:
        counts = await runtime.services.store.source_counts()
        return [SourceCount(source=s, chunks=n) for s, n in sorted(counts.items())]

    @v1.post("/documents", response_model=IngestResponse)
    async def upload_documents(
        request: Request, runtime: RuntimeDep, files: Annotated[list[UploadFile], File()]
    ) -> IngestResponse:
        settings = runtime.services.settings
        limit = settings.api_max_upload_mb * 1024 * 1024
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > 5 * limit:  # cheap guard; the per-file cap below
            raise HTTPException(status_code=413, detail="The upload is too large.")  # is exact
        names = [_safe_name(f.filename) for f in files]
        bad = [n for n in names if classify(Path(n)) is None]
        if bad:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {', '.join(bad)}")
        if len(set(names)) != len(names):
            raise HTTPException(status_code=422, detail="Two uploaded files have the same name.")

        upload_dir = settings.api_upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        results = []
        async with runtime.ingest_lock:  # one at a time: they share the files and the embedder
            saved: list[Path] = []
            for upload, name in zip(files, names, strict=True):
                target = upload_dir / name
                written = 0
                with target.open("wb") as out:
                    while chunk := await upload.read(_CHUNK):
                        written += len(chunk)
                        if written > limit:
                            out.close()
                            target.unlink(missing_ok=True)
                            raise HTTPException(
                                status_code=413,
                                detail=f"{name} exceeds the {settings.api_max_upload_mb} MB limit.",
                            )
                        out.write(chunk)
                saved.append(target)
            for path in saved:
                report = await runtime.services.pipeline.ingest_path(path)
                results.extend(report.files)
        return IngestResponse(
            ingested=sum(1 for f in results if f.status == "ingested"),
            skipped=sum(1 for f in results if f.status == "skipped_unchanged"),
            failed=sum(1 for f in results if f.status == "failed"),
            chunks=sum(f.chunks for f in results),
            files=results,
        )

    app.include_router(v1)
    return app


def _valid_thread_id(value: str) -> bool:
    return re.fullmatch(THREAD_ID_PATTERN, value) is not None


def _safe_name(filename: str | None) -> str:
    """The file name only: no directories, no leading dots, no odd characters."""
    name = Path((filename or "").replace("\\", "/")).name.strip()
    cleaned = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).lstrip(".")
    if not cleaned:
        raise HTTPException(status_code=422, detail="An uploaded file has no usable name.")
    return cleaned
