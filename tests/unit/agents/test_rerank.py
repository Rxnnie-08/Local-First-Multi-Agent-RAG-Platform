from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import RerankerError
from agentic_rag.retrieval import FlashRankReranker, NoopReranker, build_reranker
from agentic_rag.vectorstore.models import SearchHit


def hits(n: int = 3) -> list[SearchHit]:
    return [
        SearchHit(
            id=str(i),
            score=1.0 / (i + 1),
            text=f"text {i}",
            source="a.md",
            chunk_index=i,
            metadata={"heading_path": f"H{i}"},
        )
        for i in range(n)
    ]


class FakeRanker:
    loads = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).loads += 1
        self.kwargs = kwargs

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        # Reverse order with descending scores so we can see that reordering happened.
        ids = [p["id"] for p in request.passages][::-1]
        return [{"id": i, "text": "", "score": 0.9 - 0.1 * n} for n, i in enumerate(ids)]


class FakeRequest:
    def __init__(self, query: str, passages: list[dict[str, Any]]) -> None:
        self.query = query
        self.passages = passages


@pytest.fixture
def fake_flashrank(monkeypatch: pytest.MonkeyPatch) -> type[FakeRanker]:
    FakeRanker.loads = 0
    mod = types.ModuleType("flashrank")
    mod.Ranker = FakeRanker  # type: ignore[attr-defined]
    mod.RerankRequest = FakeRequest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashrank", mod)
    return FakeRanker


async def test_noop_keeps_order_and_truncates() -> None:
    out = await NoopReranker().rerank("q", hits(5), top_n=2)
    assert [r.hit.id for r in out] == ["0", "1"]
    assert all(r.score is None for r in out)


async def test_flashrank_reorders_and_scores(
    fake_flashrank: type[FakeRanker], tmp_path: Any
) -> None:
    rr = FlashRankReranker(model_name="m", cache_dir=str(tmp_path / "cache"))
    out = await rr.rerank("q", hits(3), top_n=2)

    assert [r.hit.id for r in out] == ["2", "1"]
    assert out[0].score == pytest.approx(0.9)
    assert (tmp_path / "cache").is_dir()


async def test_flashrank_loads_once_and_passes_quiet_logging(
    fake_flashrank: type[FakeRanker], tmp_path: Any
) -> None:
    rr = FlashRankReranker(model_name="m", cache_dir=str(tmp_path))
    await rr.warmup()
    await rr.rerank("q", hits(), top_n=3)
    await rr.rerank("q", hits(), top_n=3)
    assert fake_flashrank.loads == 1


async def test_flashrank_scores_heading_with_text(
    monkeypatch: pytest.MonkeyPatch, fake_flashrank: type[FakeRanker], tmp_path: Any
) -> None:
    seen: list[str] = []
    original = FakeRanker.rerank

    def spy(self: FakeRanker, request: Any) -> list[dict[str, Any]]:
        seen.extend(p["text"] for p in request.passages)
        return original(self, request)

    monkeypatch.setattr(FakeRanker, "rerank", spy)
    await FlashRankReranker(model_name="m", cache_dir=str(tmp_path)).rerank("q", hits(1), top_n=1)
    assert seen == ["H0\ntext 0"]


async def test_flashrank_empty_hits_does_not_load(
    fake_flashrank: type[FakeRanker], tmp_path: Any
) -> None:
    out = await FlashRankReranker(model_name="m", cache_dir=str(tmp_path)).rerank("q", [], top_n=3)
    assert out == []
    assert fake_flashrank.loads == 0


async def test_flashrank_load_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    class Boom:
        def __init__(self, **kwargs: Any) -> None:
            raise OSError("no network")

    mod = types.ModuleType("flashrank")
    mod.Ranker = Boom  # type: ignore[attr-defined]
    mod.RerankRequest = FakeRequest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashrank", mod)

    rr = FlashRankReranker(model_name="m", cache_dir=str(tmp_path))
    with pytest.raises(RerankerError, match="RERANKER=none"):
        await rr.warmup()
    with pytest.raises(RerankerError):
        await rr.rerank("q", hits(), top_n=2)


async def test_flashrank_scoring_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, fake_flashrank: type[FakeRanker], tmp_path: Any
) -> None:
    def explode(self: FakeRanker, request: Any) -> list[dict[str, Any]]:
        raise ValueError("bad tensor")

    monkeypatch.setattr(FakeRanker, "rerank", explode)
    rr = FlashRankReranker(model_name="m", cache_dir=str(tmp_path))
    with pytest.raises(RerankerError, match="scoring failed"):
        await rr.rerank("q", hits(), top_n=2)


def test_build_reranker_respects_setting(settings: Settings) -> None:
    assert isinstance(
        build_reranker(settings.model_copy(update={"reranker": "none"})), NoopReranker
    )
    assert isinstance(build_reranker(settings), FlashRankReranker)
