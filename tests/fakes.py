"""Test doubles: a scripted fake Ollama server built on httpx.MockTransport."""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence
from typing import Any

import httpx

from agentic_rag.embeddings.sparse import tokenize


class FakeOllama:
    """Replays a scripted sequence of responses (or exceptions) and records requests."""

    def __init__(self, script: list[httpx.Response | Exception]) -> None:
        self._script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"Unexpected extra request: {request.method} {request.url}")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def json_body(self, index: int = -1) -> dict[str, Any]:
        body = json.loads(self.requests[index].content)
        assert isinstance(body, dict)
        return body


def chat_response(
    content: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    eval_duration_s: float = 1.0,
    done_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": done_reason,
            "total_duration": int(2e9),
            "load_duration": int(0.5e9),
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": int(0.25e9),
            "eval_count": completion_tokens,
            "eval_duration": int(eval_duration_s * 1e9),
        },
    )


def stream_response(
    pieces: Sequence[str],
    *,
    prompt_tokens: int = 10,
    done: bool = True,
    extra_lines: Sequence[dict[str, Any]] = (),
) -> httpx.Response:
    """An Ollama /api/chat reply with stream=true: one JSON object per line, the last with done."""
    lines: list[dict[str, Any]] = [
        {"model": "test-model", "message": {"role": "assistant", "content": p}, "done": False}
        for p in pieces
    ]
    lines += list(extra_lines)
    if done:
        lines.append(
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": int(2e9),
                "load_duration": int(0.5e9),
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": int(0.25e9),
                "eval_count": len(pieces),
                "eval_duration": int(1e9),
            }
        )
    return httpx.Response(
        200,
        content="\n".join(json.dumps(line) for line in lines).encode(),
        headers={"content-type": "application/x-ndjson"},
    )


def error_response(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": message})


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: similar wording gives similar vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.document_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in tokenize(text):
            vec[zlib.crc32(token.encode()) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ExplodingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedder is down")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def embed_response(count: int, dim: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "nomic-embed-text",
            "embeddings": [[0.1 * (i + 1)] * dim for i in range(count)],
            "total_duration": int(1e9),
            "load_duration": int(0.1e9),
            "prompt_eval_count": 6 * count,
        },
    )


class FakeReranker:
    """Returns hits in the given order with a fixed score for every hit (or None)."""

    name = "fake"

    def __init__(self, score: float | None = 0.9, *, fail: bool = False) -> None:
        self._score = score
        self._fail = fail
        self.calls = 0

    async def warmup(self) -> None:
        return None

    async def rerank(self, query: str, hits: Sequence[Any], *, top_n: int) -> list[Any]:
        from agentic_rag.core.exceptions import RerankerError
        from agentic_rag.retrieval import RankedHit

        self.calls += 1
        if self._fail:
            raise RerankerError("fake reranker failure")
        return [RankedHit(hit=h, score=self._score) for h in hits[:top_n]]


class _FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Stands in for asyncio.subprocess.Process. `hang=True` never finishes until kill()."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        *,
        hang: bool = False,
    ) -> None:
        import asyncio

        self.stdin = _FakeStdin()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self._out, self._err, self._code = stdout, stderr, returncode
        self._done = asyncio.Event()
        self.killed = False
        self.returncode: int | None = None
        if not hang:
            self._finish()

    def _finish(self) -> None:
        self.stdout.feed_data(self._out)
        self.stdout.feed_eof()
        self.stderr.feed_data(self._err)
        self.stderr.feed_eof()
        self.returncode = self._code
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        await self.wait()
        return self._out, self._err

    def kill(self) -> None:
        self.killed = True
        self._code = -9
        if not self._done.is_set():
            self._finish()


class FakeSpawner:
    """Process factory: hands out queued FakeProcess objects and records every command line."""

    def __init__(self, *processes: FakeProcess, missing: bool = False) -> None:
        self._queue = list(processes)
        self._missing = missing
        self.calls: list[tuple[str, ...]] = []
        self.spawned: list[FakeProcess] = []

    async def __call__(self, *args: str, **kwargs: Any) -> FakeProcess:
        self.calls.append(tuple(args))
        if self._missing:
            raise FileNotFoundError("docker")
        proc = self._queue.pop(0) if self._queue else FakeProcess()
        self.spawned.append(proc)
        return proc


def route(next_: str, query: str = "", reason: str = "because") -> httpx.Response:
    """A scripted supervisor routing decision."""
    return chat_response(json.dumps({"reason": reason, "next": next_, "query": query}))
