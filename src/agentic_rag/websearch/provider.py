"""Web search behind a small interface, so the backend can be swapped (e.g. for a self-hosted
SearXNG container) without touching the agent code."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import WebSearchError
from agentic_rag.core.logging import get_logger

log = get_logger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


class DuckDuckGoProvider:
    """DuckDuckGo via the `ddgs` package. No API key, but it is an unofficial scraper: it can be
    rate-limited or break when DuckDuckGo changes. Failures surface as WebSearchError."""

    name = "duckduckgo"

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    def _search_sync(self, query: str, max_results: int) -> list[dict[str, Any]]:
        from ddgs import DDGS  # lazy: the import is slow and the package is optional

        rows: list[dict[str, Any]] = DDGS(timeout=int(self._timeout_s)).text(
            query, max_results=max_results
        )
        return rows

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        try:
            # ddgs has its own timeout; the outer one covers a stuck thread.
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, query, max_results),
                timeout=self._timeout_s + 5,
            )
        except TimeoutError as exc:
            raise WebSearchError(f"web search timed out after {self._timeout_s:.0f}s") from exc
        except Exception as exc:  # ddgs raises its own hierarchy; "no results" is one of them
            if type(exc).__name__ == "DDGSException" and "no results" in str(exc).lower():
                return []
            raise WebSearchError(f"web search failed: {type(exc).__name__}: {exc}") from exc

        results: list[SearchResult] = []
        for row in rows:
            url = str(row.get("href") or row.get("url") or "")
            title = str(row.get("title") or "").strip()
            snippet = " ".join(str(row.get("body") or row.get("snippet") or "").split())
            if url and (title or snippet):
                results.append(SearchResult(title=title or url, url=url, snippet=snippet))
        return results[:max_results]


def build_search_provider(settings: Settings) -> SearchProvider | None:
    if not settings.web_search_enabled:
        return None
    return DuckDuckGoProvider(timeout_s=settings.web_timeout_s)
