"""The LangGraph state machine.

    begin_turn -> supervisor -+-> <worker> -> supervisor   (cycle, bounded by graph_max_steps)
                              +-> synthesize -> finalize -> END

The supervisor's `next` is an enum of the registered workers plus "answer", and the conditional
edges are built from the same registry, so adding a worker is registering it in the list passed to
`AgentApp`. Conversation memory lives in the checkpointer, keyed by thread id.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from openinference.semconv.trace import SpanAttributes

from agentic_rag.agents.models import (
    AgentEvent,
    AgentResponse,
    Citation,
    ErrorEvent,
    FinalEvent,
    Observation,
    RunMetrics,
    StepEvent,
    StepRecord,
    TokenEvent,
)
from agentic_rag.agents.state import AgentState
from agentic_rag.agents.supervisor import ANSWER, Supervisor
from agentic_rag.agents.synthesizer import Synthesizer
from agentic_rag.agents.workers.base import Worker, WorkerTask
from agentic_rag.config import Settings
from agentic_rag.core.logging import get_logger
from agentic_rag.llm import ChatMessage, OllamaClient, StructuredExtractor
from agentic_rag.observability import SpanKind, set_attributes, set_output, span

log = get_logger(__name__)

SYNTHESIZE = "synthesize"
NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _dump(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = model.model_dump(mode="json")
    return result


class AgentApp:
    def __init__(
        self,
        *,
        client: OllamaClient,
        workers: Sequence[Worker],
        settings: Settings,
        checkpointer: Any = None,
    ) -> None:
        if not workers:
            raise ValueError("AgentApp needs at least one worker")
        names = [w.name for w in workers]
        if len(set(names)) != len(names) or ANSWER in names or SYNTHESIZE in names:
            raise ValueError(f"Worker names must be unique and not reserved: {names}")

        self._settings = settings
        self._workers = {w.name: w for w in workers}
        self._supervisor = Supervisor(
            StructuredExtractor(client, max_repairs=settings.supervisor_max_repairs),
            self._workers,
            settings,
        )
        self._synthesizer = Synthesizer(client, settings)
        self._graph = self._build(checkpointer or MemorySaver())
        self._reapers: set[asyncio.Future[None]] = set()  # keeps cleanup tasks alive

    # -- public API ---------------------------------------------------------------

    @property
    def worker_names(self) -> list[str]:
        return list(self._workers)

    def _config(self, thread_id: str) -> Any:
        return {
            "configurable": {"thread_id": thread_id},
            # Each supervisor loop is ~3 graph steps; leave generous headroom over graph_max_steps.
            "recursion_limit": 6 * self._settings.graph_max_steps + 10,
        }

    @staticmethod
    def _clean(question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        return question

    async def ask(self, question: str, *, thread_id: str = "default") -> AgentResponse:
        """Run one turn. Turns on the same thread share conversation history.

        Do not run two turns concurrently on the same thread: they would interleave history.
        """
        question = self._clean(question)
        started = time.perf_counter()
        with span(
            "agent.turn",
            SpanKind.AGENT,
            input_value=question,
            attributes={SpanAttributes.SESSION_ID: thread_id},
        ) as turn:
            final = await self._graph.ainvoke({"question": question}, self._config(thread_id))
            return self._finish_turn(turn, final, question, thread_id, started)

    async def ask_stream(
        self, question: str, *, thread_id: str = "default"
    ) -> AsyncIterator[AgentEvent]:
        """Like `ask`, but yields a StepEvent as each node finishes, a TokenEvent per piece of the
        answer as it is generated, then a FinalEvent with the same AgentResponse `ask` returns.
        Failures are reported as an ErrorEvent instead of being raised.

        The turn runs in its own task and feeds a queue. That keeps its tracing span inside one
        task, and lets a client that disconnects cancel the work by closing this generator.
        """
        question = self._clean(question)
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def produce() -> None:
            started = time.perf_counter()
            try:
                with span(
                    "agent.turn",
                    SpanKind.AGENT,
                    input_value=question,
                    attributes={SpanAttributes.SESSION_ID: thread_id},
                ) as turn:
                    config = self._config(thread_id)
                    config["configurable"]["stream_tokens"] = True
                    async for mode, data in self._graph.astream(
                        {"question": question}, config, stream_mode=["updates", "custom"]
                    ):
                        if mode == "custom":
                            await queue.put(TokenEvent(text=str(data["token"])))
                        else:
                            for update in data.values():
                                if isinstance(update, dict) and update.get("trace"):
                                    step = StepRecord.model_validate(update["trace"][-1])
                                    await queue.put(
                                        StepEvent(
                                            node=step.node,
                                            latency_s=step.latency_s,
                                            detail=step.detail,
                                        )
                                    )
                    snapshot = await self._graph.aget_state(config)
                    response = self._finish_turn(
                        turn, snapshot.values, question, thread_id, started
                    )
                await queue.put(FinalEvent(response=response))
            except Exception as exc:  # reported to the consumer; the task must not die silently
                log.warning("turn_failed", error_type=type(exc).__name__, error=str(exc))
                await queue.put(ErrorEvent(error_type=type(exc).__name__, message=str(exc)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            # Stop the turn in a separate task. The caller may itself be mid-cancellation (a web
            # server cancelling a disconnected client's request), and in that state awaiting here
            # raises at once, which would abandon the turn half-stopped.
            reaper = asyncio.ensure_future(self._reap(task))
            self._reapers.add(reaper)
            reaper.add_done_callback(self._reapers.discard)

    @staticmethod
    async def _reap(task: asyncio.Task[None]) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _finish_turn(
        self, turn: Any, final: dict[str, Any], question: str, thread_id: str, started: float
    ) -> AgentResponse:
        total = time.perf_counter() - started
        trace = [StepRecord.model_validate(t) for t in final.get("trace", [])]
        response = AgentResponse(
            thread_id=thread_id,
            question=question,
            answer=final.get("answer", ""),
            route=[t.node for t in trace],
            citations=[Citation.model_validate(c) for c in final.get("citations", [])],
            metrics=RunMetrics.from_trace(trace, total),
            trace=trace,
        )
        m = response.metrics
        set_output(turn, response.answer)
        set_attributes(
            turn,
            {
                "agent.route": " -> ".join(response.route),
                "agent.llm_calls": m.llm_calls,
                "agent.citations": len(response.citations),
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: m.prompt_tokens,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: m.completion_tokens,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: m.total_tokens,
                "llm.tokens_per_second": m.tokens_per_second,
                "agent.total_latency_s": round(total, 3),
            },
        )
        return response

    async def reset_thread(self, thread_id: str) -> None:
        """Forget a conversation."""
        await self._graph.checkpointer.adelete_thread(thread_id)

    async def seed_history(self, thread_id: str, turns: Sequence[tuple[str, str]]) -> None:
        """Pre-load conversation history for a thread without running the model. Used by the
        evaluation harness so follow-up questions have context deterministically."""
        history: list[dict[str, str]] = []
        for question, answer in turns:
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        config: Any = {"configurable": {"thread_id": thread_id}}
        await self._graph.aupdate_state(config, {"history": history}, as_node="finalize")

    # -- graph construction ---------------------------------------------------------

    def _build(self, checkpointer: Any) -> Any:
        # langgraph's generics reject plain async callables, so the builder is typed loosely
        graph: Any = StateGraph(AgentState)
        graph.add_node("begin_turn", self._begin_turn)
        graph.add_node("supervisor", self._supervise)
        graph.add_node(SYNTHESIZE, self._synthesize)
        graph.add_node("finalize", self._finalize)
        for name in self._workers:
            graph.add_node(name, self._worker_node(name))
            graph.add_edge(name, "supervisor")  # the cycle: every worker reports back

        graph.add_edge(START, "begin_turn")
        graph.add_edge("begin_turn", "supervisor")
        routes = {name: name for name in self._workers} | {ANSWER: SYNTHESIZE}
        graph.add_conditional_edges("supervisor", lambda s: s["next"], routes)
        graph.add_edge(SYNTHESIZE, "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)

    # -- nodes ----------------------------------------------------------------------

    async def _begin_turn(self, state: AgentState) -> dict[str, Any]:
        return {
            "steps": 0,
            "next": "",
            "query": "",
            "observations": [],
            "trace": [],
            "answer": "",
            "citations": [],
            # `history` is deliberately not reset: it is the conversation memory.
        }

    async def _supervise(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = state.get("steps", 0)
        with span("supervisor", SpanKind.CHAIN, input_value=state["question"]) as sp:
            decision = await self._supervisor.decide(
                question=state["question"],
                history=self._history(state),
                observations=self._observations(state),
                steps=steps,
            )
            set_output(sp, decision.next)
            set_attributes(
                sp,
                {
                    "supervisor.next": decision.next,
                    "supervisor.query": decision.query,
                    "supervisor.reason": decision.reason,
                    "supervisor.source": decision.source,
                    "supervisor.step": steps,
                },
            )
        record = StepRecord(
            node="supervisor",
            latency_s=time.perf_counter() - started,
            llm=decision.metrics,
            detail={
                "next": decision.next,
                "query": decision.query,
                "reason": decision.reason,
                "source": decision.source,
            },
        )
        return {
            "next": decision.next,
            "query": decision.query,
            "steps": steps + 1,
            "trace": [*state.get("trace", []), _dump(record)],
        }

    def _worker_node(self, name: str) -> NodeFn:
        worker = self._workers[name]

        async def run(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            query = state.get("query") or state["question"]
            with span(f"worker.{name}", SpanKind.CHAIN, input_value=query) as sp:
                try:
                    observation = await worker.run(
                        WorkerTask(query=query, question=state["question"])
                    )
                except Exception as exc:  # a worker bug or tool crash must not kill the turn
                    log.warning("worker_crashed", worker=name, error_type=type(exc).__name__)
                    observation = Observation(
                        worker=name,
                        query=query,
                        ok=False,
                        confident=False,
                        summary=f"The {name} worker failed.",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                set_output(sp, observation.summary)
                set_attributes(
                    sp,
                    {
                        "worker.name": name,
                        "worker.ok": observation.ok,
                        "worker.confident": observation.confident,
                        "worker.passages": len(observation.passages),
                        "worker.error": observation.error,
                    },
                )
            record = StepRecord(
                node=name,
                latency_s=time.perf_counter() - started,
                llm=observation.metrics,
                detail={
                    "query": query,
                    "ok": observation.ok,
                    "confident": observation.confident,
                    "passages": len(observation.passages),
                    "summary": observation.summary,
                    **({"error": observation.error} if observation.error else {}),
                    **observation.detail,
                },
            )
            return {
                "observations": [*state.get("observations", []), _dump(observation)],
                "trace": [*state.get("trace", []), _dump(record)],
            }

        return run

    async def _synthesize(
        self, state: AgentState, writer: StreamWriter, config: RunnableConfig
    ) -> dict[str, Any]:
        # Tokens are only streamed when ask_stream asked for it: a streamed call cannot be retried.
        streaming = bool(config.get("configurable", {}).get("stream_tokens"))
        started = time.perf_counter()
        with span("synthesize", SpanKind.CHAIN, input_value=state["question"]) as sp:
            result = await self._synthesizer.answer(
                question=state["question"],
                history=self._history(state),
                observations=self._observations(state),
                on_token=(lambda piece: writer({"token": piece})) if streaming else None,
            )
            set_output(sp, result.answer)
            set_attributes(
                sp,
                {
                    "synthesize.mode": result.mode,
                    "synthesize.passages_used": result.detail.get("passages_used"),
                    "synthesize.passages_dropped": result.detail.get("passages_dropped"),
                    "synthesize.truncated": result.detail.get("truncated"),
                },
            )
        record = StepRecord(
            node=SYNTHESIZE,
            latency_s=time.perf_counter() - started,
            llm=result.metrics,
            detail=result.detail,
        )
        return {
            "answer": result.answer,
            "citations": [_dump(c) for c in result.citations],
            "trace": [*state.get("trace", []), _dump(record)],
        }

    async def _finalize(self, state: AgentState) -> dict[str, Any]:
        history = [
            *state.get("history", []),
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state["answer"]},
        ]
        keep = 2 * max(self._settings.history_turns, 1) * 4  # retain more than is shown
        return {"history": history[-keep:]}

    # -- state helpers ----------------------------------------------------------------

    def _history(self, state: AgentState) -> list[ChatMessage]:
        window = 2 * self._settings.history_turns
        if window == 0:
            return []
        return [ChatMessage.model_validate(m) for m in state.get("history", [])[-window:]]

    @staticmethod
    def _observations(state: AgentState) -> list[Observation]:
        return [Observation.model_validate(o) for o in state.get("observations", [])]
