from __future__ import annotations

import itertools

from agentic_rag.ingestion.chunker import (
    _PROSE_SEPS,
    CHARS_PER_TOKEN,
    Chunk,
    _recursive,
    build_embedding_text,
    chunk_document,
    estimate_tokens,
    split_text,
)
from agentic_rag.ingestion.parsers import DocType, ParsedDocument, Section


def make_doc(
    text: str, doc_type: DocType = "text", *, metadata: dict[str, object] | None = None
) -> ParsedDocument:
    return ParsedDocument(
        source="f.txt",
        doc_type=doc_type,
        file_hash="h",
        sections=[Section(text=text)],
        metadata=metadata or {},
    )


SENTENCES = " ".join(f"Sentence number {i:03d} talks about topic {i % 7}." for i in range(120))


def test_recursive_split_is_lossless_and_bounded() -> None:
    text = "Para one is here.\n\nPara two follows. It has two sentences.\n" + "word " * 200
    pieces = _recursive(text, 80, _PROSE_SEPS)
    assert "".join(pieces) == text
    assert all(len(p) <= 80 for p in pieces)


def test_hard_split_when_no_separator_exists() -> None:
    pieces = _recursive("a" * 1000, 100, _PROSE_SEPS)
    assert "".join(pieces) == "a" * 1000
    assert all(len(p) <= 100 for p in pieces)


def test_chunks_respect_size_limit() -> None:
    limit = int(64 * CHARS_PER_TOKEN)
    chunks = split_text(SENTENCES, "text", limit, overlap=int(8 * CHARS_PER_TOKEN))
    assert len(chunks) > 3
    assert all(len(c) <= limit for c in chunks)


def test_overlap_carries_whole_sentences_into_next_chunk() -> None:
    limit, overlap = 300, 80
    chunks = split_text(SENTENCES, "text", limit, overlap)
    for prev, nxt in itertools.pairwise(chunks):
        first_sentence = nxt.split(".")[0] + "."
        assert first_sentence in prev


def test_no_overlap_when_disabled() -> None:
    chunks = split_text(SENTENCES, "text", 300, 0)
    seen: set[str] = set()
    for chunk in chunks:
        sentences = {s.strip() + "." for s in chunk.split(".") if s.strip()}
        assert not (sentences & seen)
        seen |= sentences


def test_every_sentence_survives_chunking() -> None:
    chunks = split_text(SENTENCES, "text", 300, 80)
    joined = " ".join(chunks)
    for i in range(120):
        assert f"Sentence number {i:03d}" in joined


def test_markdown_fenced_block_stays_atomic() -> None:
    fence = "```python\ndef f():\n    x = 1\n\n    return x\n```"
    text = "Intro paragraph. " * 12 + "\n\n" + fence + "\n\n" + "Outro paragraph. " * 12
    chunks = split_text(text, "markdown", 260, 40)
    holders = [c for c in chunks if "def f():" in c]
    assert holders
    assert all(fence in c for c in holders)


def test_code_prefers_definition_boundaries() -> None:
    funcs = [f"def func_{i}(x):\n    return x + {i}\n" for i in range(12)]
    chunks = split_text("\n".join(funcs), "code", 90, 0)
    assert len(chunks) > 1
    assert all(c.startswith("def ") for c in chunks)


def test_chunk_document_metadata_and_indexes() -> None:
    doc = ParsedDocument(
        source="notes.pdf",
        doc_type="pdf",
        file_hash="h",
        sections=[
            Section(text="First page text. " * 30, metadata={"page": 1}),
            Section(text="Second page text. " * 30, metadata={"page": 2}),
        ],
        metadata={"title": "T"},
    )
    chunks = chunk_document(doc, max_tokens=64, overlap_tokens=8)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["doc_type"] == "pdf" and c.metadata["title"] == "T" for c in chunks)
    # chunks never span a section boundary
    assert all(("First" in c.text) != ("Second" in c.text) for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    doc = ParsedDocument(source="e.txt", doc_type="text", file_hash="h", sections=[])
    assert chunk_document(doc, max_tokens=64, overlap_tokens=8) == []


def test_token_estimate() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 35) == 10


def test_embedding_text_includes_location_header() -> None:
    chunk = Chunk(
        index=0,
        text="It defaults to 5m.",
        token_estimate=5,
        metadata={"heading_path": "Ollama > Keep alive", "page": 3},
    )
    text = build_embedding_text("docs/ops.md", chunk)
    assert text == "docs/ops.md | Ollama > Keep alive | page 3\nIt defaults to 5m."
