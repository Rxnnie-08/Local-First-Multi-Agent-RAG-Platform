"""Typed models shared by the agent graph.

LangGraph checkpoints serialize state, and pydantic objects inside checkpointed state trigger a
"deserializing unregistered type" warning that is slated to become an error. So the graph state
holds plain dicts (`model_dump(mode="json")`) and these models are the validated view of them at
each node boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, create_model

from agentic_rag.llm.schemas import LLMMetrics


class Passage(BaseModel):
    """One piece of evidence a worker hands to the synthesizer."""

    text: str
    source: str  # file path, URL, or tool name
    location: str = ""  # heading path, page, line range ...
    score: float | None = None


class Observation(BaseModel):
    """What a worker found. `summary` is what the supervisor sees; `passages` go to the answer."""

    worker: str
    query: str
    ok: bool
    confident: bool
    summary: str
    passages: list[Passage] = Field(default_factory=list)
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    metrics: LLMMetrics | None = None  # set by workers that call the LLM themselves


class SupervisorDecisionBase(BaseModel):
    # Field order matters: the model emits JSON keys in schema order, so `reason` comes first and
    # acts as a one-line chain of thought before the choice.
    reason: str = Field(description="At most 15 words explaining the choice.")
    next: str
    query: str = Field(
        default="",
        description="Self-contained query or task for the chosen worker. Empty for 'answer'.",
    )


def make_decision_model(options: Sequence[str]) -> type[SupervisorDecisionBase]:
    """Decision schema whose `next` is an enum of exactly the registered workers plus 'answer',
    so the model cannot route to a worker that does not exist."""
    literal = Literal[tuple(options)]  # type: ignore[valid-type]
    model = create_model(
        "SupervisorDecision",
        __base__=SupervisorDecisionBase,
        next=(literal, ...),
    )
    return cast("type[SupervisorDecisionBase]", model)


class Citation(BaseModel):
    n: int
    source: str
    location: str = ""
    score: float | None = None
    snippet: str
    cited: bool  # True if the answer text actually references [n]


class StepRecord(BaseModel):
    node: str
    latency_s: float
    llm: LLMMetrics | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None
    llm_latency_s: float = 0.0
    total_latency_s: float = 0.0

    @classmethod
    def from_trace(cls, trace: Sequence[StepRecord], total_latency_s: float) -> RunMetrics:
        merged: LLMMetrics | None = None
        for step in trace:
            if step.llm is not None:
                merged = step.llm if merged is None else merged.merge(step.llm)
        if merged is None:
            return cls(total_latency_s=total_latency_s)
        return cls(
            llm_calls=merged.calls,
            prompt_tokens=merged.prompt_tokens,
            completion_tokens=merged.completion_tokens,
            total_tokens=merged.total_tokens,
            tokens_per_second=merged.tokens_per_second,
            llm_latency_s=merged.wall_latency_s,
            total_latency_s=total_latency_s,
        )


class AgentResponse(BaseModel):
    thread_id: str
    question: str
    answer: str
    route: list[str]  # nodes visited in order, e.g. supervisor, rag, supervisor, synthesize
    citations: list[Citation]
    metrics: RunMetrics
    trace: list[StepRecord]


# --- events emitted while a turn is running (AgentApp.ask_stream) -------------------------------


class StepEvent(BaseModel):
    """A graph node finished: the supervisor decided, or a worker reported back."""

    type: Literal["step"] = "step"
    node: str
    latency_s: float
    detail: dict[str, Any] = Field(default_factory=dict)


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class FinalEvent(BaseModel):
    type: Literal["final"] = "final"
    response: AgentResponse


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error_type: str
    message: str


AgentEvent = StepEvent | TokenEvent | FinalEvent | ErrorEvent
