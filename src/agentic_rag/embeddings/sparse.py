"""Deterministic BM25-style sparse encoder (no model download, no vocabulary file).

Tokens are hashed into Qdrant's u32 index space. Documents get BM25 term-frequency weights;
queries get weight 1 per unique term. Qdrant's `Modifier.IDF` on the sparse vector config applies
the inverse-document-frequency half of BM25 at search time using live collection statistics, so
the encoder itself stays stateless.

Hashing uses crc32, not Python's `hash()`, which is randomized per process.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from typing import NamedTuple

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_INDEX_MASK = 0x7FFFFFFF

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


class SparseVec(NamedTuple):
    indices: list[int]
    values: list[float]

    @property
    def is_empty(self) -> bool:
        return not self.indices


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. camelCase / snake_case identifiers also yield their parts, so a
    query for 'user name' can match `getUserName` and `user_name` in code."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):  # '_' is not in the class, so snake_case splits here
        lower = word.lower()
        parts = [p.lower() for p in _CAMEL_RE.findall(word)]
        if len(parts) > 1:
            tokens.append(lower)
            tokens.extend(parts)
        else:
            tokens.append(lower)
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & _INDEX_MASK


class Bm25SparseEncoder:
    def __init__(self, *, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 120.0) -> None:
        self._k1 = k1
        self._b = b
        self._avg_doc_len = avg_doc_len

    def encode_document(self, text: str) -> SparseVec:
        tokens = tokenize(text)
        if not tokens:
            return SparseVec([], [])
        doc_len = len(tokens)
        norm = self._k1 * (1.0 - self._b + self._b * doc_len / self._avg_doc_len)
        counts = Counter(_index(t) for t in tokens)
        indices = sorted(counts)
        values = [counts[i] * (self._k1 + 1.0) / (counts[i] + norm) for i in indices]
        return SparseVec(indices, values)

    def encode_query(self, text: str) -> SparseVec:
        indices = sorted({_index(t) for t in tokenize(text)})
        return SparseVec(indices, [1.0] * len(indices))
