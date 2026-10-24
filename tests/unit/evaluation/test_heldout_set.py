from __future__ import annotations

from pathlib import Path

from agentic_rag.evaluation.dataset import check_dataset, load_cases

HELDOUT = Path(__file__).resolve().parents[3] / "eval" / "heldout"
DEV = Path(__file__).resolve().parents[3] / "eval"


def test_heldout_cases_are_consistent_with_their_corpus() -> None:
    cases = load_cases(HELDOUT / "cases.jsonl")
    assert check_dataset(cases, HELDOUT / "corpus") == []
    assert sum(c.abstain for c in cases) >= 10  # enough to make an abstention rate mean something


def test_heldout_set_shares_nothing_with_the_development_set() -> None:
    # Follow-ups like "Can I change it?" are generic by design; their history is what differs.
    held = {c.question.lower() for c in load_cases(HELDOUT / "cases.jsonl") if not c.history}
    dev = {c.question.lower() for c in load_cases(DEV / "cases.jsonl") if not c.history}
    assert not held & dev
    corpus = " ".join(p.read_text(encoding="utf-8").lower() for p in (DEV / "corpus").glob("*.md"))
    assert "kestrel" not in corpus
