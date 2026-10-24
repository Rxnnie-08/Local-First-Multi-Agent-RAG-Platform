"""Async ingestion: discover -> hash -> (skip if unchanged) -> parse -> chunk -> embed -> upsert.

Idempotent: chunk ids derive from (source, file hash, index), unchanged files are skipped by hash,
and a changed file has its old chunks deleted only AFTER its new embeddings are ready, so a failed
embed never leaves a half-ingested file. If the process dies between delete and upsert, the source
has no stored hash, so the next run re-ingests it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, computed_field

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import ParseError
from agentic_rag.core.logging import get_logger
from agentic_rag.embeddings.dense import Embedder
from agentic_rag.embeddings.sparse import Bm25SparseEncoder
from agentic_rag.ingestion.chunker import build_embedding_text, chunk_document
from agentic_rag.ingestion.parsers import discover_files, parse_file, sha256_of, source_name
from agentic_rag.vectorstore.models import ChunkRecord
from agentic_rag.vectorstore.qdrant_store import QdrantStore

log = get_logger(__name__)

FileStatus = Literal["ingested", "skipped_unchanged", "failed"]


class FileResult(BaseModel):
    source: str
    status: FileStatus
    chunks: int = 0
    error: str | None = None
    duration_s: float = 0.0


class IngestReport(BaseModel):
    files: list[FileResult]
    duration_s: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ingested(self) -> int:
        return sum(1 for f in self.files if f.status == "ingested")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def skipped(self) -> int:
        return sum(1 for f in self.files if f.status == "skipped_unchanged")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return sum(1 for f in self.files if f.status == "failed")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunks(self) -> int:
        return sum(f.chunks for f in self.files)


def _resolve_existing(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.exists():
        raise ParseError(f"Path does not exist: {root}")
    return root


class IngestionPipeline:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        sparse_encoder: Bm25SparseEncoder,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._settings = settings
        self._max_bytes = int(settings.ingest_max_file_mb * 1_048_576)

    async def ingest_path(self, path: Path, *, force: bool = False) -> IngestReport:
        root = await asyncio.to_thread(_resolve_existing, path)

        started = time.perf_counter()
        await self._store.ensure_collection(dim=self._embedder.dim)
        files = await asyncio.to_thread(discover_files, root)
        log.info("ingest_start", root=str(root), files=len(files), force=force)

        semaphore = asyncio.Semaphore(self._settings.ingest_concurrency)

        async def run(file: Path) -> FileResult:
            async with semaphore:
                return await self._ingest_file(root, file, force=force)

        results = await asyncio.gather(*(run(f) for f in files))
        report = IngestReport(files=list(results), duration_s=time.perf_counter() - started)
        log.info(
            "ingest_done",
            ingested=report.ingested,
            skipped=report.skipped,
            failed=report.failed,
            chunks=report.chunks,
            duration_s=round(report.duration_s, 2),
        )
        return report

    async def _ingest_file(self, root: Path, file: Path, *, force: bool) -> FileResult:
        source = source_name(root, file)
        started = time.perf_counter()

        def result(status: FileStatus, chunks: int = 0, error: str | None = None) -> FileResult:
            return FileResult(
                source=source,
                status=status,
                chunks=chunks,
                error=error,
                duration_s=time.perf_counter() - started,
            )

        try:
            file_hash = await asyncio.to_thread(sha256_of, file, self._max_bytes)
            if not force and await self._store.get_source_hash(source) == file_hash:
                return result("skipped_unchanged")

            doc = await asyncio.to_thread(
                parse_file, file, source=source, file_hash=file_hash, max_bytes=self._max_bytes
            )
            chunks = chunk_document(
                doc,
                max_tokens=self._settings.chunk_max_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
            )
            if not chunks:
                raise ParseError(f"{file.name} produced no chunks")

            embed_texts = [build_embedding_text(source, c) for c in chunks]
            dense = await self._embedder.embed_documents(embed_texts)
            ingested_at = datetime.now(timezone.utc).isoformat()
            records = [
                ChunkRecord(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{file_hash}:{c.index}")),
                    text=c.text,
                    dense=dense[i],
                    sparse=self._sparse.encode_document(embed_texts[i]),
                    payload=self._payload(source, file_hash, c.index, c.token_estimate, c.metadata)
                    | {"ingested_at": ingested_at},
                )
                for i, c in enumerate(chunks)
            ]
            await self._store.delete_source(source)
            await self._store.upsert(records)
            return result("ingested", chunks=len(records))
        except Exception as exc:  # isolate failures: one bad file must not sink the batch
            log.warning("ingest_file_failed", source=source, error_type=type(exc).__name__)
            return result("failed", error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _payload(
        source: str,
        file_hash: str,
        index: int,
        token_estimate: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "source": source,
            "chunk_index": index,
            "file_hash": file_hash,
            "token_estimate": token_estimate,
            **metadata,
        }
