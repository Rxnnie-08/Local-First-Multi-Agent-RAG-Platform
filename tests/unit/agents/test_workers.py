from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.agents import AgentApp, CodeWorker, RagWorker, WebWorker, WorkerTask
from agentic_rag.agents.workers.code import extract_code
from agentic_rag.config import Settings
from agentic_rag.core.exceptions import SandboxError, WebSearchError
from agentic_rag.ingestion import IngestionPipeline
from agentic_rag.llm import OllamaClient
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.sandbox import SandboxResult
from agentic_rag.services import make_agent
from agentic_rag.websearch import DuckDuckGoProvider, SearchResult, build_search_provider
from fakes import FakeOllama, FakeReranker, chat_response, error_response, route

SAMPLES = Path(__file__).resolve().parents[3] / "samples"

TASK = WorkerTask(query="compute 17% of 2340", question="What is 17% of 2340?")


def result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    *,
    timed_out: bool = False,
    truncated: bool = False,
) -> SandboxResult:
    return SandboxResult(stdout, stderr, exit_code, timed_out, truncated, 0.1)


class FakeRunner:
    def __init__(self, *results: SandboxResult | Exception) -> None:
        self._results = list(results)
        self.codes: list[str] = []

    async def check(self) -> None:
        return None

    async def run(self, code: str) -> SandboxResult:
        self.codes.append(code)
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(code: str) -> Any:
    return chat_response(f"Here you go:\n```python\n{code}\n```")


@pytest.fixture
async def client(settings: Settings) -> Any:
    clients: list[OllamaClient] = []

    def make(fake: FakeOllama) -> OllamaClient:
        c = OllamaClient(settings, transport=fake.transport)
        clients.append(c)
        return c

    yield make
    for c in clients:
        await c.aclose()


# --- extract_code ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("Sure!\n```py\nx = 1\nprint(x)\n```\nDone.", "x = 1\nprint(x)"),
        ("```\nprint(2)\n```", "print(2)"),
        ("```python\nprint(3)\n", "print(3)"),  # cut off by the token limit
        ("print(4)", "print(4)"),
        ("   ", ""),
    ],
)
def test_extract_code(text: str, expected: str) -> None:
    assert extract_code(text) == expected


# --- CodeWorker -----------------------------------------------------------------------------


async def test_code_worker_success(settings: Settings, client: Any) -> None:
    fake = FakeOllama([reply("print(2340*17/100)")])
    runner = FakeRunner(result("397.8\n"))
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)

    assert obs.ok
    assert obs.confident
    assert runner.codes == ["print(2340*17/100)"]
    assert "397.8" in obs.passages[0].text
    assert "print(2340*17/100)" in obs.passages[0].text
    assert obs.passages[0].source == "python sandbox"
    assert obs.metrics is not None
    assert obs.metrics.calls == 1
    system = fake.json_body(0)["messages"][0]["content"]
    assert "standard library" in system


async def test_code_worker_repairs_once_with_the_error_message(
    settings: Settings, client: Any
) -> None:
    fake = FakeOllama([reply("print(1/0)"), reply("print(1)")])
    runner = FakeRunner(
        result(stderr="Traceback...\nZeroDivisionError: division by zero", exit_code=1),
        result("1\n"),
    )
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)

    assert obs.ok
    assert obs.detail["attempts"] == 2
    assert obs.metrics is not None
    assert obs.metrics.calls == 2
    repair_prompt = fake.json_body(1)["messages"][-1]["content"]
    assert "ZeroDivisionError" in repair_prompt


async def test_code_worker_gives_up_after_max_repairs(settings: Settings, client: Any) -> None:
    fake = FakeOllama([reply("bad"), reply("still bad")])
    fail = result(stderr="NameError: bad", exit_code=1)
    runner = FakeRunner(fail, fail)
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)

    assert not obs.ok
    assert not obs.confident
    assert obs.error is not None
    assert "NameError" in obs.error
    assert obs.passages == []
    assert len(runner.codes) == 2


async def test_code_worker_does_not_repair_a_timeout(settings: Settings, client: Any) -> None:
    fake = FakeOllama([reply("while True: pass")])
    runner = FakeRunner(result(exit_code=None, timed_out=True))
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)

    assert not obs.ok
    assert obs.error is not None
    assert "did not finish" in obs.error
    assert len(fake.requests) == 1


async def test_code_worker_without_output_is_not_confident(settings: Settings, client: Any) -> None:
    fake = FakeOllama([reply("x = 1")])
    obs = await CodeWorker(client(fake), FakeRunner(result("")), settings).run(TASK)
    assert obs.ok
    assert not obs.confident


async def test_code_worker_reports_unavailable_sandbox(settings: Settings, client: Any) -> None:
    fake = FakeOllama([reply("print(1)")])
    runner = FakeRunner(SandboxError("Docker is not usable"))
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)

    assert not obs.ok
    assert obs.error is not None
    assert "Docker is not usable" in obs.error


async def test_code_worker_contains_llm_failure(settings: Settings, client: Any) -> None:
    fake = FakeOllama([error_response(400, "bad request")])
    obs = await CodeWorker(client(fake), FakeRunner(), settings).run(TASK)
    assert not obs.ok
    assert obs.error is not None
    assert "OllamaRequestError" in obs.error


async def test_code_worker_with_empty_reply(settings: Settings, client: Any) -> None:
    fake = FakeOllama([chat_response("   ")])
    runner = FakeRunner()
    obs = await CodeWorker(client(fake), runner, settings).run(TASK)
    assert not obs.ok
    assert runner.codes == []


# --- WebWorker ------------------------------------------------------------------------------


class FakeProvider:
    name = "fake"

    def __init__(self, results: list[SearchResult] | Exception) -> None:
        self._results = results
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.queries.append((query, max_results))
        if isinstance(self._results, Exception):
            raise self._results
        return self._results


async def test_web_worker_turns_results_into_cited_passages(settings: Settings) -> None:
    provider = FakeProvider(
        [
            SearchResult(title="Python 3.13", url="https://python.org/a", snippet="Latest release"),
            SearchResult(title="Docs", url="https://docs.python.org/b", snippet="Reference"),
        ]
    )
    obs = await WebWorker(provider, settings).run(WorkerTask("latest python", "q"))

    assert obs.ok
    assert obs.confident
    assert [p.source for p in obs.passages] == ["https://python.org/a", "https://docs.python.org/b"]
    assert obs.passages[0].text == "Python 3.13\nLatest release"
    assert provider.queries == [("latest python", settings.web_max_results)]


async def test_web_worker_empty_results(settings: Settings) -> None:
    obs = await WebWorker(FakeProvider([]), settings).run(WorkerTask("zzz", "q"))
    assert obs.ok
    assert not obs.confident
    assert obs.passages == []


async def test_web_worker_failure_is_an_observation(settings: Settings) -> None:
    obs = await WebWorker(FakeProvider(WebSearchError("rate limited")), settings).run(
        WorkerTask("x", "q")
    )
    assert not obs.ok
    assert obs.error == "rate limited"


# --- DuckDuckGoProvider ---------------------------------------------------------------------


def install_fake_ddgs(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> None:
    class DDGS:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            if isinstance(behaviour, Exception):
                raise behaviour
            return list(behaviour)

    mod = types.ModuleType("ddgs")
    mod.DDGS = DDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", mod)


async def test_duckduckgo_maps_and_cleans_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_ddgs(
        monkeypatch,
        [
            {"title": " T1 ", "href": "https://a.example", "body": "line one\n  line two"},
            {"title": "", "href": "", "body": "no url, dropped"},
            {"title": "T3", "href": "https://c.example", "body": ""},
        ],
    )
    out = await DuckDuckGoProvider().search("q", max_results=5)

    assert [r.url for r in out] == ["https://a.example", "https://c.example"]
    assert out[0].snippet == "line one line two"
    assert out[0].title == "T1"


async def test_duckduckgo_no_results_is_empty_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DDGSException(Exception):
        pass

    install_fake_ddgs(monkeypatch, DDGSException("No results found."))
    assert await DuckDuckGoProvider().search("q", max_results=5) == []


async def test_duckduckgo_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_ddgs(monkeypatch, RuntimeError("202 Ratelimit"))
    with pytest.raises(WebSearchError, match="Ratelimit"):
        await DuckDuckGoProvider().search("q", max_results=5)


def test_search_provider_can_be_disabled(settings: Settings) -> None:
    assert build_search_provider(settings.model_copy(update={"web_search_enabled": False})) is None
    assert build_search_provider(settings) is not None


# --- wiring: all three workers in one graph ---------------------------------------------------


@pytest.fixture
async def all_workers(settings: Settings, retriever: HybridRetriever, client: Any) -> Any:
    def build(
        script: list[Any],
        *,
        runner: FakeRunner | None,
        provider: FakeProvider | None,
        reranker: FakeReranker | None = None,
    ) -> tuple[AgentApp, FakeOllama]:
        fake = FakeOllama(script)
        app = make_agent(
            settings=settings,
            client=client(fake),
            retriever=retriever,
            reranker=reranker or FakeReranker(0.9),
            runner=runner,
            search=provider,
        )
        return app, fake

    return build


async def test_supervisor_can_route_to_python(all_workers: Any) -> None:
    app, fake = all_workers(
        [
            route("python", "compute 17% of 2340"),
            reply("print(2340*0.17)"),
            chat_response("It is 397.8 [1]."),
        ],
        runner=FakeRunner(result("397.8\n")),
        provider=FakeProvider([]),
    )
    resp = await app.ask("What is 17% of 2340?")

    assert resp.route == ["supervisor", "python", "supervisor", "synthesize"]
    assert resp.answer == "It is 397.8 [1]."
    assert resp.citations[0].source == "python sandbox"
    assert resp.metrics.llm_calls == 3  # routing + code generation + answer
    assert fake.json_body(0)["format"]["properties"]["next"]["enum"] == [
        "rag",
        "web",
        "python",
        "answer",
    ]


async def test_empty_search_falls_through_to_web(
    all_workers: Any, pipeline: IngestionPipeline
) -> None:
    await pipeline.ingest_path(SAMPLES)
    provider = FakeProvider(
        [SearchResult(title="Paris", url="https://x.example", snippet="Capital")]
    )
    app, _ = all_workers(
        [
            route("rag", "capital of France"),
            route("web", "capital of France"),
            chat_response("Paris [1]."),
        ],
        runner=None,
        provider=provider,
        reranker=FakeReranker(0.00001),
    )
    resp = await app.ask("What is the capital of France?")

    assert resp.route == ["supervisor", "rag", "supervisor", "web", "supervisor", "synthesize"]
    assert resp.citations[0].source == "https://x.example"
    assert provider.queries


async def test_disabled_workers_are_not_routable(all_workers: Any) -> None:
    app, fake = all_workers([route("answer"), chat_response("hi")], runner=None, provider=None)
    await app.ask("hello")
    assert fake.json_body(0)["format"]["properties"]["next"]["enum"] == ["rag", "answer"]


async def test_failed_code_run_does_not_crash_the_turn(all_workers: Any) -> None:
    app, _ = all_workers(
        [
            route("python", "compute"),
            reply("boom"),
            reply("boom again"),
            route("answer"),
            chat_response("I could not run that."),
        ],
        runner=FakeRunner(
            result(stderr="NameError", exit_code=1), result(stderr="NameError", exit_code=1)
        ),
        provider=None,
    )
    resp = await app.ask("compute something")

    assert resp.answer == "I could not run that."
    python_step = next(s for s in resp.trace if s.node == "python")
    assert python_step.detail["ok"] is False
    assert python_step.llm is not None
    assert python_step.llm.calls == 2


def test_rag_worker_is_still_exported() -> None:
    assert RagWorker.name == "rag"
