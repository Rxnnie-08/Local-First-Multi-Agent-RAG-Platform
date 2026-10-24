"""Tracing entry point (compatibility wrapper).

The tracing implementation lives in `agentic_rag.observability`. It instruments this project's own
steps (turn, supervisor, workers, retrieval, reranking, LLM calls, sandbox, web search) with
OpenInference attributes. An earlier version of this file auto-instrumented LangChain, which has two
problems here: it needs the extra package `openinference-instrumentation-langchain`, and it would
only see LangGraph's own nodes, not the Ollama calls, retrieval, reranker or sandbox. Using both
would also record every graph node twice.

Normal use does not need this file: set TRACING_ENABLED=true and the app configures tracing itself.
Call `setup_tracing()` here only from a script that does not go through `open_services`.
"""

from __future__ import annotations

import os

from opentelemetry.sdk.trace import TracerProvider

from agentic_rag.config import get_settings
from agentic_rag.observability import setup_tracing as _setup_tracing

_TRACES_PATH = "/v1/traces"


def _collector_endpoint(default: str) -> str:
    """PHOENIX_COLLECTOR_ENDPOINT may be Phoenix's base URL (http://127.0.0.1:6006) or the full
    trace endpoint; accept both."""
    raw = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip().rstrip("/")
    if not raw:
        return default
    return raw if raw.endswith(_TRACES_PATH) else raw + _TRACES_PATH


def setup_tracing(project_name: str = "enterprise-local-rag") -> TracerProvider | None:
    """Send traces to Arize Phoenix. Returns the provider, or None if tracing is not configured.

    Start Phoenix with `docker compose up -d phoenix` and open http://127.0.0.1:6006.
    """
    settings = get_settings()
    enabled = settings.model_copy(
        update={
            "tracing_enabled": True,
            "tracing_project": project_name,
            "tracing_endpoint": _collector_endpoint(settings.tracing_endpoint),
        }
    )
    return _setup_tracing(enabled)
