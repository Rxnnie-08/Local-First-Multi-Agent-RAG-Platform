"""Web worker: public-web search for what the user's documents cannot answer.

Only result titles and snippets are used; pages are not fetched. That keeps it fast on CPU and
avoids pulling arbitrary page text into the prompt. Snippets are still untrusted input: the answer
prompt tells the model to treat passages as data, never as instructions.
"""

from __future__ import annotations

from agentic_rag.agents.models import Observation, Passage
from agentic_rag.agents.workers.base import WorkerTask
from agentic_rag.config import Settings
from agentic_rag.core.exceptions import WebSearchError
from agentic_rag.core.logging import get_logger
from agentic_rag.observability import SpanKind, set_attributes, set_output, span
from agentic_rag.websearch import SearchProvider

log = get_logger(__name__)


class WebWorker:
    name = "web"
    description = (
        "Searches the public web for current or general information the user's own documents "
        "are unlikely to contain: news, prices, latest versions, public facts, third-party "
        "documentation. The query is sent to a search engine, so keep it free of private details."
    )

    def __init__(self, provider: SearchProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    async def run(self, task: WorkerTask) -> Observation:
        detail: dict[str, object] = {"provider": self._provider.name}
        try:
            with span(
                "web.search",
                SpanKind.TOOL,
                input_value=task.query,
                attributes={"tool.name": "web_search", "web.provider": self._provider.name},
            ) as sp:
                results = await self._provider.search(
                    task.query, max_results=self._settings.web_max_results
                )
                set_output(sp, [r.url for r in results])
                set_attributes(sp, {"web.results": len(results)})
        except WebSearchError as exc:
            log.warning("web_search_failed", error=str(exc))
            return Observation(
                worker=self.name,
                query=task.query,
                ok=False,
                confident=False,
                summary="The web search failed.",
                error=str(exc),
                detail=detail,
            )
        detail["results"] = len(results)
        if not results:
            return Observation(
                worker=self.name,
                query=task.query,
                ok=True,
                confident=False,
                summary="The web search returned no results.",
                detail=detail,
            )
        passages = [
            Passage(text=f"{r.title}\n{r.snippet}".strip(), source=r.url, location="web")
            for r in results
        ]
        return Observation(
            worker=self.name,
            query=task.query,
            ok=True,
            confident=True,  # a search engine ranks by relevance but gives no absolute score
            summary=f"{len(passages)} web results.",
            passages=passages,
            detail=detail,
        )
