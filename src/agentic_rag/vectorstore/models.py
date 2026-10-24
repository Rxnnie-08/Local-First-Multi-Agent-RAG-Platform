from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.embeddings.sparse import SparseVec


class ChunkRecord(BaseModel):
    """One chunk ready to be written: id, vectors and payload."""

    id: str
    text: str
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilter(BaseModel):
    """Restrict a search to certain files and/or document types."""

    sources: list[str] | None = None
    doc_types: list[str] | None = None
