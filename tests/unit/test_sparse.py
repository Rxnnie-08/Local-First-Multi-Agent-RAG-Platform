from __future__ import annotations

from agentic_rag.embeddings import Bm25SparseEncoder, tokenize


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("The Quick brown fox is on the mat") == ["quick", "brown", "fox", "mat"]


def test_tokenize_splits_camel_and_snake_case_but_keeps_whole_identifier() -> None:
    tokens = tokenize("getUserName user_name")
    assert {"getusername", "get", "user", "name"} <= set(tokens)
    assert tokens.count("user") == 2  # once from getUserName, once from user_name


def test_tokenize_keeps_error_code_digits() -> None:
    assert "4021" in tokenize("Error E-4021 means the model is missing")


def test_encoding_is_deterministic_and_sorted() -> None:
    enc = Bm25SparseEncoder()
    a = enc.encode_document("hybrid search merges rankings")
    b = enc.encode_document("hybrid search merges rankings")
    assert a == b
    assert a.indices == sorted(a.indices)
    assert len(a.indices) == len(a.values) == 4


def test_term_frequency_saturates() -> None:
    enc = Bm25SparseEncoder(k1=1.2, b=0.0)  # b=0 isolates the tf term
    one = enc.encode_document("alpha beta")
    many = enc.encode_document("alpha " * 50 + "beta")
    idx_alpha = enc.encode_query("alpha").indices[0]
    w_one = dict(zip(one.indices, one.values, strict=True))[idx_alpha]
    w_many = dict(zip(many.indices, many.values, strict=True))[idx_alpha]
    assert w_many > w_one
    assert w_many < 1.2 + 1.0  # bounded above by k1 + 1


def test_query_weights_are_one_per_unique_term() -> None:
    q = Bm25SparseEncoder().encode_query("keep keep alive")
    assert len(q.indices) == 2
    assert q.values == [1.0, 1.0]


def test_empty_after_stopword_removal() -> None:
    enc = Bm25SparseEncoder()
    assert enc.encode_query("the of and").is_empty
    assert enc.encode_document("").is_empty
