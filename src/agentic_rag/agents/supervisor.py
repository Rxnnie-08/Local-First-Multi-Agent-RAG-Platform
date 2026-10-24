"""Supervisor: decides the next step. Rules first; the LLM only when judgement is needed.

On CPU each LLM call costs ~10 s, so the common case (retrieval came back confident) is decided by
a rule rather than a second model call. The LLM is consulted for the first routing decision and
whenever the last worker's result is not clearly good enough.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from agentic_rag.agents.models import Observation, make_decision_model
from agentic_rag.agents.prompts import (
    SUPERVISOR_SYSTEM,
    SUPERVISOR_USER,
    format_history,
    format_observations,
)
from agentic_rag.agents.workers.base import Worker
from agentic_rag.config import Settings
from agentic_rag.core.exceptions import LLMError
from agentic_rag.core.logging import get_logger
from agentic_rag.embeddings.sparse import tokenize
from agentic_rag.llm import ChatMessage, LLMMetrics, StructuredExtractor

log = get_logger(__name__)

ANSWER = "answer"
# Words that mean a question leans on the conversation, so the model's rewrite may legitimately
# share no words with it.
_REFERENCE_WORDS = frozenset(
    [
        "it",
        "its",
        "they",
        "them",
        "their",
        "those",
        "these",
        "that",
        "this",
        "one",
        "ones",
        "former",
        "latter",
        "above",
        "previous",
        "same",
        "also",
        "more",
        "else",
        "another",
        "other",
    ]
)


def _drifted(question: str, query: str) -> bool:
    """True when the rewritten query has nothing in common with a self-contained question.

    Small models sometimes copy the previous turn's query instead of the current question (seen
    with llama3.1: "capital of France" became "default keep alive"). A rewrite is only needed to
    resolve references, so when the question has none and the query shares no content word with
    it, the rewrite is untrustworthy.
    """
    q_words = set(question.lower().replace("?", " ").split())
    if q_words & _REFERENCE_WORDS:
        return False
    q_tokens = set(tokenize(question))
    return len(q_tokens) >= 2 and not (q_tokens & set(tokenize(query)))


DecisionSource = Literal["llm", "policy", "fallback"]


@dataclass(frozen=True)
class Decision:
    next: str
    query: str
    reason: str
    source: DecisionSource
    metrics: LLMMetrics | None = None


class Supervisor:
    def __init__(
        self,
        extractor: StructuredExtractor,
        workers: Mapping[str, Worker],
        settings: Settings,
    ) -> None:
        self._extractor = extractor
        self._workers = workers
        self._settings = settings
        self._decision_model = make_decision_model([*workers, ANSWER])
        self._system = SUPERVISOR_SYSTEM.format(
            worker_lines="\n".join(f"- {w.name}: {w.description}" for w in workers.values())
        )

    def _policy(self, observations: Sequence[Observation], steps: int) -> Decision | None:
        if steps >= self._settings.graph_max_steps:
            return Decision(ANSWER, "", "step limit reached", "policy")
        if not observations:
            return None
        last = observations[-1]
        if last.ok and last.confident:
            return Decision(ANSWER, "", "latest result is confident", "policy")
        if last.ok and not last.passages and {o.worker for o in observations} >= set(self._workers):
            # Every passage fell below the relevance floor and there is nobody else to ask. In
            # practice a reworded query scores just as low, so a retry costs ~7 s of routing LLM
            # calls for nothing. (A weak but non-empty result still gets one reworded retry.)
            return Decision(
                ANSWER, "", "nothing relevant found and no other worker to try", "policy"
            )
        return None

    def _fallback(self, question: str, observations: Sequence[Observation]) -> Decision:
        """Used when the routing LLM call fails: search once with the raw question, then answer."""
        if not observations and self._workers:
            first = next(iter(self._workers))
            return Decision(first, question, "routing model failed; searching", "fallback")
        return Decision(ANSWER, "", "routing model failed; answering", "fallback")

    async def decide(
        self,
        *,
        question: str,
        history: Sequence[ChatMessage],
        observations: Sequence[Observation],
        steps: int,
    ) -> Decision:
        if (policy := self._policy(observations, steps)) is not None:
            return policy

        user_prompt = SUPERVISOR_USER.format(
            history=format_history(history),
            question=question,
            observations=format_observations(observations),
            steps=steps,
            max_steps=self._settings.graph_max_steps,
        )
        try:
            result = await self._extractor.extract(
                self._decision_model,
                user_prompt,
                system=self._system,
                max_tokens=self._settings.supervisor_max_tokens,
            )
        except LLMError as exc:
            log.warning("supervisor_llm_failed", error_type=type(exc).__name__, error=str(exc))
            return self._fallback(question, observations)

        data = result.data
        if data.next == ANSWER:
            return Decision(ANSWER, "", data.reason, "llm", result.metrics)

        query = data.query.strip() or question
        reason = data.reason
        if _drifted(question, query):
            log.warning("supervisor_query_drift", question=question, query=query)
            query = question
            reason += " [query replaced: it did not match the question]"
        elif (
            not history
            and not observations
            and data.next == next(iter(self._workers), None)
            and not set(question.lower().replace("?", " ").split()) & _REFERENCE_WORDS
        ):
            # A first question has nothing to resolve, so the rewrite can only lose information
            # (seen: "How long can a worker hold a job before the system takes it back?" became
            # "job duration before system takes it back", which dropped the matching terms).
            query = question
        repeated = any(
            o.worker == data.next and o.query.strip().lower() == query.lower() for o in observations
        )
        if repeated:
            return Decision(
                ANSWER, "", f"already ran {data.next} with this query", "policy", result.metrics
            )
        return Decision(data.next, query, reason, "llm", result.metrics)
