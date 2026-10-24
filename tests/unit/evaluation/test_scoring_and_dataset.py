from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.evaluation.dataset import EvalCase, check_dataset, load_cases
from agentic_rag.evaluation.scoring import (
    answer_matches,
    first_step,
    is_abstention,
    normalize,
    percentile,
)

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "eval" / "cases.jsonl"
CORPUS = ROOT / "eval" / "corpus"


def test_normalize_drops_thousands_separators_but_keeps_decimals() -> None:
    assert normalize("479,001,600  items") == "479001600 items"
    assert normalize("1,5") == "15"  # a limitation, documented by this test
    assert normalize("A  B\n C") == "a b c"


@pytest.mark.parametrize(
    ("answer", "required", "expected"),
    [
        ("It is 397.8.", ["397.8"], True),
        ("The result is 479,001,600", ["479001600"], True),
        ("Seven attempts", ["7|seven"], True),
        ("Eight attempts", ["7|seven"], False),
        ("The port is 8421 (orbit.toml)", ["8421", "orbit.toml"], True),
        ("The port is 8421", ["8421", "orbit.toml"], False),
        ("anything", [], None),
    ],
)
def test_answer_matches(answer: str, required: list[str], expected: bool | None) -> None:
    assert answer_matches(answer, required) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I found no relevant information in the user's documents.", True),
        ("Could not find it in the documents.", True),
        ("The documents do not mention a CEO.", False),  # "do not mention" is not matched
        ("I don't know.", True),
        ("The CEO is Jane Doe.", False),
    ],
)
def test_is_abstention(text: str, expected: bool) -> None:
    assert is_abstention(text) is expected


def test_first_step() -> None:
    assert first_step(["supervisor", "rag", "supervisor", "synthesize"]) == "rag"
    assert first_step(["supervisor", "synthesize"]) == "answer"
    assert first_step(["supervisor", "python", "supervisor", "synthesize"]) == "python"
    assert first_step([]) == "answer"


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0


# --- the real dataset ---------------------------------------------------------------------------


def test_shipped_dataset_loads_and_is_consistent_with_the_corpus() -> None:
    cases = load_cases(CASES)
    assert check_dataset(cases, CORPUS) == []
    assert len(cases) >= 30


def test_shipped_dataset_covers_every_category() -> None:
    categories = {c.category for c in load_cases(CASES)}
    assert categories == {
        "rag_fact",
        "rag_lexical",
        "rag_paraphrase",
        "rag_followup",
        "abstain",
        "python",
        "direct",
        "web",
    }


def test_followups_have_history_and_a_standalone_rewrite() -> None:
    followups = [c for c in load_cases(CASES) if c.category == "rag_followup"]
    assert followups
    for c in followups:
        assert c.history
        assert c.standalone


def test_check_dataset_catches_evidence_that_is_not_in_the_sources() -> None:
    bad = EvalCase(
        id="x",
        category="rag_fact",
        question="q",
        expect_first="rag",
        evidence="this sentence is nowhere",
        sources=["orbit_handbook.md"],
    )
    ghost = bad.model_copy(update={"id": "y", "evidence": "port 8421", "sources": ["nope.md"]})
    problems = check_dataset([bad, ghost], CORPUS)
    assert any("x:" in p and "not found" in p for p in problems)
    assert any("y:" in p and "unknown" in p for p in problems)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = '{"id":"a","category":"direct","question":"hi","expect_first":"answer"}\n'
    path = tmp_path / "cases.jsonl"
    path.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


# --- the lightweight JSON format -----------------------------------------------------------------


def test_min_fraction_allows_partial_keyword_matches() -> None:
    keywords = ["24-month", "replacement", "thermal degradation", "30%"]
    assert answer_matches("A 24-month replacement warranty.", keywords) is False
    assert answer_matches("A 24-month replacement warranty.", keywords, 0.5) is True
    assert answer_matches("A warranty.", keywords, 0.5) is False


def test_simple_json_format_loads_and_finds_the_source_file(tmp_path: Path) -> None:
    from agentic_rag.evaluation.dataset import load_any

    path = tmp_path / "set.json"
    path.write_text(
        """[
      {"id": "a", "query": "Nexus-9 battery warranty?", "expected_route": "rag",
       "ground_truth_context": "Nexus-9 battery carries a limited 24-month replacement warranty",
       "expected_answer_keywords": ["24-month", "replacement"]},
      {"id": "b", "query": "20th Fibonacci?", "expected_route": "code_sandbox",
       "expected_answer_keywords": ["6765"]},
      {"id": "c", "query": "Hello", "expected_route": "answer"}
    ]""",
        encoding="utf-8",
    )
    a, b, c = load_any(path, CORPUS)

    assert a.expect_first == "rag"
    assert a.sources == ["nexus9_datasheet.md"]
    assert a.evidence is not None
    assert a.answer_min_fraction == 0.5
    assert b.expect_first == "python"  # alias for the worker's real name
    assert b.category == "python"
    assert c.category == "direct"
    assert check_dataset([a, b, c], CORPUS) == []


def test_users_evaluation_set_is_valid_against_the_corpus() -> None:
    from agentic_rag.evaluation.dataset import load_any

    cases = load_any(ROOT / "tests" / "evaluation_set.json", CORPUS)
    assert [c.expect_first for c in cases] == ["rag", "python"]
    assert cases[0].sources == ["nexus9_datasheet.md"]
    assert check_dataset(cases, CORPUS) == []


def test_asking_for_clarification_counts_as_not_answering() -> None:
    assert is_abstention(
        "The question is ambiguous because there are multiple Orbits. Which one is meant?"
    )
    assert not is_abstention("Patrick Woods is the CEO of Orbit.")
