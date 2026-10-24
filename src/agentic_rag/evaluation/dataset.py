"""The labelled evaluation set.

The cases are written against a corpus of a *fictional* product (eval/corpus), so a correct answer
cannot come from the model's pretraining. That is what makes the numbers measure retrieval and
routing instead of memorisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_rag.evaluation.scoring import normalize

Category = Literal[
    "rag_fact",
    "rag_lexical",
    "rag_paraphrase",
    "rag_followup",
    "abstain",
    "python",
    "direct",
    "web",
]
FirstStep = Literal["rag", "web", "python", "answer"]


class HistoryTurn(BaseModel):
    question: str
    answer: str


class EvalCase(BaseModel):
    id: str
    category: Category
    question: str
    history: list[HistoryTurn] = Field(default_factory=list)  # seeded, not run through the LLM
    # What a good supervisor would rewrite a follow-up into. Used only by the retrieval benchmark,
    # which has no supervisor.
    standalone: str | None = None
    expect_first: FirstStep
    evidence: str | None = None  # phrase that must appear in a retrieved chunk
    sources: list[str] = Field(default_factory=list)  # acceptable source file names
    # Each entry must appear in the answer; "a|b" means either. Empty = not checked.
    answer_contains: list[str] = Field(default_factory=list)
    # Share of answer_contains entries that must appear. 1.0 = all; 0.5 = at least half.
    answer_min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    abstain: bool = False  # the documents do not contain the answer; the right move is to say so
    live: bool = False  # depends on the live internet, so results change over time

    @property
    def retrieval_query(self) -> str:
        return self.standalone or self.question

    @property
    def is_retrieval_case(self) -> bool:
        return self.evidence is not None


def load_cases(path: Path) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids in " + str(path))
    return cases


_ROUTE_ALIASES = {
    "code_sandbox": "python",
    "code": "python",
    "sandbox": "python",
    "direct": "answer",
}
_CATEGORY_FOR = {"rag": "rag_fact", "python": "python", "web": "web", "answer": "direct"}


def load_simple_json(
    path: Path, corpus_dir: Path | None = None, *, min_keyword_fraction: float = 0.5
) -> list[EvalCase]:
    """Load the lightweight format: a JSON list of
    {id, query, expected_route, expected_answer_keywords, ground_truth_context}.

    `expected_route` is the first step the supervisor should take (rag, web, python or answer;
    "code_sandbox" is accepted as an alias for python). When a corpus directory is given, the file
    that contains `ground_truth_context` is found automatically, which enables the source check.
    """
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    texts = (
        {p.name: normalize(_read_source(p)) for p in corpus_dir.iterdir() if p.is_file()}
        if corpus_dir
        else {}
    )
    cases: list[EvalCase] = []
    for row in rows:
        route = _ROUTE_ALIASES.get(str(row["expected_route"]).lower(), str(row["expected_route"]))
        context = str(row.get("ground_truth_context") or "")
        sources = (
            [name for name, text in texts.items() if normalize(context) in text]
            if route == "rag" and context
            else []
        )
        cases.append(
            EvalCase.model_validate(
                {
                    "id": row["id"],
                    "category": _CATEGORY_FOR.get(route, "rag_fact"),
                    "question": row["query"],
                    "expect_first": route,
                    "evidence": context if sources else None,
                    "sources": sources,
                    "answer_contains": row.get("expected_answer_keywords", []),
                    "answer_min_fraction": min_keyword_fraction,
                }
            )
        )
    return cases


def load_any(path: Path, corpus_dir: Path | None = None) -> list[EvalCase]:
    """.jsonl = the full case format; .json = the lightweight list format."""
    return load_simple_json(path, corpus_dir) if path.suffix == ".json" else load_cases(path)


def _read_source(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def check_dataset(cases: list[EvalCase], corpus_dir: Path) -> list[str]:
    """Problems that would make the benchmark meaningless: a case whose evidence is not in any
    of its listed sources would score as a retrieval failure no system could avoid."""
    problems: list[str] = []
    texts = {p.name: normalize(_read_source(p)) for p in corpus_dir.iterdir() if p.is_file()}
    for case in cases:
        if case.evidence is not None:
            missing = [s for s in case.sources if s not in texts]
            if missing or not case.sources:
                problems.append(f"{case.id}: unknown or empty sources {case.sources}")
                continue
            if not any(normalize(case.evidence) in texts[s] for s in case.sources):
                problems.append(f"{case.id}: evidence {case.evidence!r} not found in its sources")
        if case.expect_first == "rag" and not case.abstain and case.evidence is None:
            problems.append(f"{case.id}: rag case without evidence")
        if case.abstain and (case.evidence or case.answer_contains):
            problems.append(f"{case.id}: abstain case must not have evidence or expected answer")
    return problems


__all__ = [
    "EvalCase",
    "HistoryTurn",
    "check_dataset",
    "load_any",
    "load_cases",
    "load_simple_json",
]
