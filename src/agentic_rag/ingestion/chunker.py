"""Structure-aware chunking.

Splitting is lossless and hierarchical: try the coarsest boundary first (blank lines, or top-level
`def`/`class` in code), fall back to lines, sentences, words, and finally a hard character cut.
Pieces are then packed greedily up to the size limit, carrying a tail of whole pieces into the next
chunk as overlap. Fenced code blocks in Markdown are kept atomic when they fit.

Sizes use a character heuristic (`CHARS_PER_TOKEN`) instead of a tokenizer, which keeps ingestion
dependency-free. It is deliberately conservative so chunks stay under the embedding model's context.
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.ingestion.parsers import DocType, ParsedDocument

CHARS_PER_TOKEN = 3.5

_DEF_BOUNDARY = (
    r"(?m)^(?=(?:export |pub |async |static |public |private )*"
    r"(?:def|class|function|func|fn|interface|struct|impl|enum)\b)"
)
_PROSE_SEPS = (r"(?<=\n\n)", r"(?<=\n)", r"(?<=[.!?])(?=\s)", r"(?<=\s)")
_CODE_SEPS = (_DEF_BOUNDARY, r"(?<=\n\n)", r"(?<=\n)", r"(?<=\s)")
_FENCE_SEPS = (r"(?<=\n)", r"(?<=\s)")
_FENCE_SPLIT_RE = re.compile(r"(?ms)(^(?:```|~~~).*?^(?:```|~~~)[^\n]*(?:\n|$))")


class Chunk(BaseModel):
    index: int
    text: str
    token_estimate: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def _split(text: str, pattern: str) -> list[str]:
    return [part for part in re.split(pattern, text) if part]


def _recursive(text: str, limit: int, seps: tuple[str, ...]) -> list[str]:
    """Break `text` into pieces of at most `limit` chars; ''.join(pieces) == text."""
    if len(text) <= limit:
        return [text]
    for i, sep in enumerate(seps):
        parts = _split(text, sep)
        if len(parts) > 1:
            pieces: list[str] = []
            for part in parts:
                pieces.extend(_recursive(part, limit, seps[i + 1 :]))
            return pieces
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _pack(pieces: list[str], limit: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    fresh = 0  # pieces added since the last emit (i.e. not just carried-over overlap)
    for piece in pieces:
        if current and current_len + len(piece) > limit:
            if fresh:
                chunks.append("".join(current))
            while current and (current_len > overlap or current_len + len(piece) > limit):
                current_len -= len(current.pop(0))
            fresh = 0
        current.append(piece)
        current_len += len(piece)
        fresh += 1
    if fresh and current:
        chunks.append("".join(current))
    return chunks


def split_text(text: str, kind: DocType, limit: int, overlap: int) -> list[str]:
    if kind == "code":
        pieces = _recursive(text, limit, _CODE_SEPS)
    elif kind == "markdown":
        pieces = []
        for i, segment in enumerate(_FENCE_SPLIT_RE.split(text)):
            if not segment:
                continue
            seps = _FENCE_SEPS if i % 2 == 1 else _PROSE_SEPS  # odd indices are fenced blocks
            pieces.extend(_recursive(segment, limit, seps))
    else:
        pieces = _recursive(text, limit, _PROSE_SEPS)
    return [c.strip() for c in _pack(pieces, limit, overlap) if c.strip()]


def chunk_document(doc: ParsedDocument, *, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    limit = int(max_tokens * CHARS_PER_TOKEN)
    overlap = int(overlap_tokens * CHARS_PER_TOKEN)
    base: dict[str, Any] = {"doc_type": doc.doc_type, **doc.metadata}
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, doc.doc_type, limit, overlap):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    token_estimate=estimate_tokens(text),
                    metadata={**base, **section.metadata},
                )
            )
    return chunks


def build_embedding_text(source: str, chunk: Chunk) -> str:
    """Text actually embedded: the chunk prefixed with where it came from.

    The location header (file, heading path, page) lets a chunk that says only "it defaults to 5m"
    still embed near the question "what is the default keep-alive?". The raw chunk text is what gets
    stored and shown.
    """
    header = [source]
    if chunk.metadata.get("heading_path"):
        header.append(str(chunk.metadata["heading_path"]))
    if chunk.metadata.get("page"):
        header.append(f"page {chunk.metadata['page']}")
    return f"{' | '.join(header)}\n{chunk.text}"
