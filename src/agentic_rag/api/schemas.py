"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.agents.models import Citation, RunMetrics, StepRecord
from agentic_rag.ingestion.pipeline import FileResult

THREAD_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    # Turns that share a thread_id share conversation history. Omit it to start a new one; the
    # response tells you the id to send next time.
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)


class AskResponse(BaseModel):
    thread_id: str
    answer: str
    route: list[str]
    citations: list[Citation]
    metrics: RunMetrics
    trace: list[StepRecord] | None = None  # only with ?include_trace=true


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    ollama: bool
    model: str
    model_available: bool
    qdrant: bool
    documents: int | None = None
    workers: list[str]
    reranker: str


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    failed: int
    chunks: int
    files: list[FileResult]


class SourceCount(BaseModel):
    source: str
    chunks: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
