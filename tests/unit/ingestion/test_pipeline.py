from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import ParseError
from agentic_rag.embeddings import Bm25SparseEncoder
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.vectorstore import QdrantStore
from fakes import ExplodingEmbedder, FakeEmbedder, make_pdf

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


async def test_ingest_samples_directory(pipeline: IngestionPipeline, store: QdrantStore) -> None:
    report = await pipeline.ingest_path(SAMPLES)

    assert report.failed == 0
    assert report.ingested == 3
    assert report.chunks > 3
    assert await store.count() == report.chunks
    counts = await store.source_counts()
    assert set(counts) == {
        "samples/rag_basics.md",
        "samples/ollama_ops.md",
        "samples/text_utils.py",
    }


async def test_reingest_is_a_noop(pipeline: IngestionPipeline, store: QdrantStore) -> None:
    first = await pipeline.ingest_path(SAMPLES)
    second = await pipeline.ingest_path(SAMPLES)

    assert second.skipped == 3
    assert second.ingested == 0
    assert await store.count() == first.chunks


async def test_force_reingests_without_duplicating(
    pipeline: IngestionPipeline, store: QdrantStore
) -> None:
    first = await pipeline.ingest_path(SAMPLES)
    forced = await pipeline.ingest_path(SAMPLES, force=True)

    assert forced.ingested == 3
    assert await store.count() == first.chunks


async def test_changed_file_replaces_old_chunks(
    pipeline: IngestionPipeline, store: QdrantStore, retriever: HybridRetriever, tmp_path: Path
) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("# Notes\n\n" + "Original zebra content about stripes. " * 40)
    await pipeline.ingest_path(tmp_path)
    old_count = await store.count()
    assert old_count > 1

    doc.write_text("# Notes\n\nTiny replacement about quokkas.")
    report = await pipeline.ingest_path(tmp_path)

    assert report.ingested == 1
    assert await store.count() == 1  # stale chunks are gone, not accumulated
    hits = (await retriever.retrieve("quokkas", mode="sparse")).hits
    assert hits
    assert "quokkas" in hits[0].text
    assert not (await retriever.retrieve("zebra", mode="sparse")).hits


async def test_one_bad_file_does_not_sink_the_batch(
    pipeline: IngestionPipeline, store: QdrantStore, tmp_path: Path
) -> None:
    (tmp_path / "good.md").write_text("# Good\n\nThis file is fine and has content.")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf")
    (tmp_path / "binary.txt").write_bytes(b"abc\x00def")

    report = await pipeline.ingest_path(tmp_path)

    by_source = {f.source.split("/")[-1]: f for f in report.files}
    assert by_source["good.md"].status == "ingested"
    assert by_source["broken.pdf"].status == "failed"
    assert "PDF" in (by_source["broken.pdf"].error or "")
    assert by_source["binary.txt"].status == "failed"
    assert report.failed == 2
    assert await store.count() == by_source["good.md"].chunks


async def test_pdf_pages_are_ingested_with_page_metadata(
    pipeline: IngestionPipeline, retriever: HybridRetriever, tmp_path: Path
) -> None:
    (tmp_path / "paper.pdf").write_bytes(make_pdf(["Intro to attention", "Quokkas of Rottnest"]))
    report = await pipeline.ingest_path(tmp_path)
    assert report.ingested == 1

    hit = (await retriever.retrieve("Rottnest", mode="sparse")).hits[0]
    assert hit.metadata["page"] == 2
    assert hit.metadata["doc_type"] == "pdf"


async def test_failed_embed_keeps_previous_version(
    store: QdrantStore,
    sparse_encoder: Bm25SparseEncoder,
    settings: Settings,
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "keep.md"
    doc.write_text("# Keep\n\nOriginal text that must survive.")
    await pipeline.ingest_path(tmp_path)
    original_hash = await store.get_source_hash(f"{tmp_path.name}/keep.md")
    assert original_hash

    doc.write_text("# Keep\n\nNew text that cannot be embedded.")
    broken = IngestionPipeline(store, ExplodingEmbedder(), sparse_encoder, settings)
    report = await broken.ingest_path(tmp_path)

    assert report.failed == 1
    assert "embedder is down" in (report.files[0].error or "")
    assert await store.count() == 1
    assert await store.get_source_hash(f"{tmp_path.name}/keep.md") == original_hash


async def test_missing_path_raises(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="does not exist"):
        await pipeline.ingest_path(tmp_path / "nope")


async def test_single_file_ingest_uses_filename_as_source(
    pipeline: IngestionPipeline, store: QdrantStore
) -> None:
    await pipeline.ingest_path(SAMPLES / "ollama_ops.md")
    assert set(await store.source_counts()) == {"ollama_ops.md"}


async def test_hybrid_finds_identifier_query(
    pipeline: IngestionPipeline, retriever: HybridRetriever, embedder: FakeEmbedder
) -> None:
    await pipeline.ingest_path(SAMPLES)
    result = await retriever.retrieve("Error E-4021", k=3, mode="hybrid")
    assert result.hits[0].source == "samples/ollama_ops.md"
    assert "E-4021" in result.hits[0].text
    assert result.embed_s >= 0
    assert result.search_s >= 0
