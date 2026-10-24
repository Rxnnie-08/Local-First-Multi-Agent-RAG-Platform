from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Graph state. Values are plain JSON-compatible data (see models.py for why).

    `history` persists across turns on a thread; everything else is reset by `begin_turn`.
    Fields are replaced, not appended, by each node, which is safe while nodes run one at a time.
    Parallel worker fan-out would need reducer-annotated fields instead.
    """

    question: str
    history: list[dict[str, str]]
    steps: int  # supervisor invocations this turn
    next: str  # supervisor's routing choice: a worker name or "answer"
    query: str  # task for the chosen worker
    observations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
