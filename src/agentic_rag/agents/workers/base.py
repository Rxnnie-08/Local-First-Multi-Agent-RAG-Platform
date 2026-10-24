from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_rag.agents.models import Observation


@dataclass(frozen=True)
class WorkerTask:
    query: str  # self-contained task written by the supervisor
    question: str  # the user's original question, for context


class Worker(Protocol):
    """A capability the supervisor can route to.

    Contract: `run` must not raise for expected failures (empty results, tool errors); return an
    Observation with ok=False and an `error`. The graph also contains unexpected exceptions, but a
    worker knows its own failure modes best.
    """

    name: str
    description: str  # shown to the supervisor's LLM; say when to use this worker

    async def run(self, task: WorkerTask) -> Observation: ...
