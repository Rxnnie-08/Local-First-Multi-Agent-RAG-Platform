"""Environment-driven configuration. Every field maps to an upper-case env var."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Ollama connection -------------------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_num_ctx: int = Field(default=4096, ge=512)
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_keep_alive: str = "30m"

    # --- Resilience --------------------------------------------------------
    ollama_timeout_s: float = Field(default=180.0, gt=0)
    ollama_connect_timeout_s: float = Field(default=5.0, gt=0)
    ollama_max_attempts: int = Field(default=3, ge=1, le=10)
    ollama_retry_initial_wait_s: float = Field(default=0.5, ge=0)
    ollama_retry_max_wait_s: float = Field(default=8.0, ge=0)

    # --- Structured output -------------------------------------------------
    ollama_use_schema_format: bool = True
    structured_max_repairs: int = Field(default=2, ge=0, le=5)

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = Field(default=768, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    # nomic-embed-text is trained with task prefixes; omitting them measurably hurts retrieval.
    embedding_doc_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "documents"
    qdrant_timeout_s: int = Field(default=30, ge=1)

    # --- Chunking / ingestion ----------------------------------------------
    chunk_max_tokens: int = Field(default=400, ge=32)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    ingest_concurrency: int = Field(default=2, ge=1, le=16)
    ingest_max_file_mb: float = Field(default=5.0, gt=0)

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    # Each retriever (dense, sparse) fetches k * multiplier candidates before RRF fusion.
    retrieval_prefetch_multiplier: int = Field(default=4, ge=1, le=20)
    sparse_k1: float = Field(default=1.2, gt=0)
    sparse_b: float = Field(default=0.75, ge=0, le=1)
    sparse_avg_doc_len: float = Field(default=120.0, gt=0)

    # --- Reranking ---------------------------------------------------------
    reranker: Literal["flashrank", "none"] = "flashrank"
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    # Relative paths resolve against the working directory. FlashRank's own default is /tmp,
    # which does not exist on Windows.
    rerank_cache_dir: str = "data/models/flashrank"
    retrieval_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    # Reranker relevance (0-1) at or above which retrieval counts as "confident". Uncalibrated.
    rerank_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    # Passages scoring below this are dropped before answering (they only add prompt tokens, which
    # on CPU means seconds). Observed: relevant 0.9+, irrelevant ~0.0001.
    rerank_min_passage_score: float = Field(default=0.001, ge=0.0, le=1.0)

    # --- Agent graph -------------------------------------------------------
    graph_max_steps: int = Field(default=4, ge=1, le=10)
    history_turns: int = Field(default=4, ge=0, le=20)
    supervisor_max_tokens: int = Field(default=160, ge=32)
    supervisor_max_repairs: int = Field(default=1, ge=0, le=3)
    answer_max_tokens: int = Field(default=512, ge=32)
    answer_context_tokens: int = Field(default=1800, ge=200)

    # --- Tools (Phase 4) ---------------------------------------------------
    code_worker_enabled: bool = True
    code_max_tokens: int = Field(default=400, ge=64)
    code_max_repairs: int = Field(default=1, ge=0, le=2)
    sandbox_image: str = "python:3.12-slim"
    # Wall-clock limit for the whole `docker run`, including ~1 s of container start-up.
    sandbox_timeout_s: float = Field(default=15.0, gt=0)
    sandbox_memory_mb: int = Field(default=256, ge=32)
    sandbox_cpus: float = Field(default=1.0, gt=0)
    sandbox_pids: int = Field(default=64, ge=8)
    sandbox_max_output_chars: int = Field(default=3000, ge=200)
    # The query text is sent to DuckDuckGo, so this is the one feature that leaves the machine.
    web_search_enabled: bool = True
    web_max_results: int = Field(default=5, ge=1, le=10)
    web_timeout_s: float = Field(default=10.0, gt=0)

    # --- Tracing (Phase 5) -------------------------------------------------
    tracing_enabled: bool = False
    # Phoenix accepts OTLP/HTTP on its UI port (docker compose up -d phoenix).
    tracing_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    tracing_project: str = "agentic-rag"
    # Record prompts, answers and passage text in spans. Phoenix is local, but turn this off if the
    # traces will ever be shared.
    tracing_capture_content: bool = True

    # --- HTTP API (Phase 6) ------------------------------------------------
    # Loopback by default. Binding anything else requires API_KEY (scripts/serve.py enforces it).
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # If set, every /v1 request must send it in the X-API-Key header.
    api_key: SecretStr | None = None
    # Comma-separated browser origins allowed to call the API; empty disables CORS.
    api_cors_origins: str = ""
    api_upload_dir: Path = Path("data/uploads")
    api_max_upload_mb: int = Field(default=20, ge=1, le=200)
    # Wall-clock limit for one turn. CPU inference is slow, so this is generous.
    api_turn_timeout_s: float = Field(default=300.0, gt=0)
    # Turns running at once. One CPU-bound model gains nothing from parallel requests.
    api_max_concurrent_turns: int = Field(default=1, ge=1, le=8)
    # Also how quickly a vanished streaming client is noticed (the keep-alive write fails).
    api_heartbeat_s: float = Field(default=5.0, gt=0)

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_api_key_means_none(cls, value: object) -> object:
        # `API_KEY=` in a .env file arrives as "", which must not count as a key.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        # Ollama silently drops the START of a prompt that exceeds num_ctx, so fail loudly instead.
        # 1200 tokens of headroom covers the system prompt, history and the question.
        needed = self.answer_context_tokens + self.answer_max_tokens + 1200
        if needed > self.ollama_num_ctx:
            raise ValueError(
                f"ANSWER_CONTEXT_TOKENS + ANSWER_MAX_TOKENS + 1200 = {needed} exceeds "
                f"OLLAMA_NUM_CTX ({self.ollama_num_ctx}). Raise OLLAMA_NUM_CTX or lower the others."
            )
        return self

    @field_validator("ollama_base_url", "qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        level = value.upper()
        if not isinstance(getattr(logging, level, None), int):
            raise ValueError(f"unknown log level: {value!r}")
        return level


@lru_cache
def get_settings() -> Settings:
    return Settings()
