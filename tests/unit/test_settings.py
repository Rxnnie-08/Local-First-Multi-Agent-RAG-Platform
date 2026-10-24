from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_rag.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.ollama_model == "llama3.1:8b"
    assert s.ollama_num_ctx == 4096
    assert s.ollama_max_attempts == 3


def test_env_vars_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "mistral:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/")
    s = Settings(_env_file=None)
    assert s.ollama_model == "mistral:7b"
    assert s.ollama_base_url == "http://host.docker.internal:11434"  # trailing slash stripped


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_num_ctx=10)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")


def test_context_budget_must_fit_in_num_ctx() -> None:
    with pytest.raises(ValueError, match="num_ctx"):
        Settings(
            _env_file=None, ollama_num_ctx=2048, answer_context_tokens=1800, answer_max_tokens=512
        )


def test_blank_api_key_means_no_key() -> None:
    assert Settings(_env_file=None, api_key="").api_key is None
    assert Settings(_env_file=None, api_key="  ").api_key is None
    key = Settings(_env_file=None, api_key="abc").api_key
    assert key is not None
    assert key.get_secret_value() == "abc"
    assert "abc" not in repr(Settings(_env_file=None, api_key="abc"))  # never printed
