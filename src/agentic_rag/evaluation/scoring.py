"""Deterministic scoring helpers. No LLM judge: string checks are repeatable, free and fast, and
they cannot be fooled by a fluent wrong answer the way a small judge model can."""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
_WS = re.compile(r"\s+")
_ABSTAIN = re.compile(
    r"could not find|couldn't find|can't find|cannot find|unable to find|no relevant|"
    r"no information|not (?:found|mentioned|specified|covered|in the documents)|"
    r"don't know|do not know|doesn't (?:say|mention|specify)|does not (?:say|mention|specify)|"
    # Asking which one is meant also declines to fabricate an answer.
    r"is ambiguous|which (?:one|orbit) (?:is|do you)|please (?:clarify|specify)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase, drop thousands separators (479,001,600 == 479001600), collapse whitespace."""
    return _WS.sub(" ", _DIGIT_COMMA.sub("", text.lower())).strip()


def answer_matches(answer: str, required: Sequence[str], min_fraction: float = 1.0) -> bool | None:
    """True when at least `min_fraction` of the required entries appear (default: all of them).
    An entry "a|b" is satisfied by either alternative. None when nothing is required."""
    if not required:
        return None
    haystack = normalize(answer)
    found = sum(any(normalize(alt) in haystack for alt in entry.split("|")) for entry in required)
    return found / len(required) >= min_fraction


def is_abstention(answer: str) -> bool:
    return _ABSTAIN.search(answer) is not None


def first_step(route: Sequence[str]) -> str:
    """The first decision that was not 'supervisor': a worker name, or 'answer' if it went
    straight to the synthesizer."""
    for node in route:
        if node == "supervisor":
            continue
        return "answer" if node == "synthesize" else node
    return "answer"


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0
