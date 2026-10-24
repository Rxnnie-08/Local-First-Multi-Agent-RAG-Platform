"""OpenTelemetry tracing with OpenInference attributes, viewable in Phoenix.

Design:
- The instrumentation is always present in the code but costs nothing until `setup_tracing`
  installs a real tracer: the default tracer is OpenTelemetry's no-op one.
- Spans are created in real time with `span(...)`, so they nest through contextvars: a turn contains
  its graph nodes, which contain their LLM / retrieval / tool calls.
- Tracing must never break the app. Exporter failures are logged by OpenTelemetry and swallowed;
  attribute helpers are defensive.
- Prompts and answers are recorded in full by default (Phoenix runs on your machine), and can be
  switched off with TRACING_CAPTURE_CONTENT=false.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer

from agentic_rag.config import Settings
from agentic_rag.core.logging import get_logger

log = get_logger(__name__)

SpanKind = OpenInferenceSpanKindValues
_MAX_VALUE_CHARS = 8000
_PROJECT_ATTR = "openinference.project.name"  # how Phoenix groups traces into projects


@dataclass
class _State:
    tracer: Tracer = field(default_factory=lambda: trace.NoOpTracer())
    capture_content: bool = True
    provider: TracerProvider | None = None


_state = _State()


def configure_tracer(provider: TracerProvider | None, *, capture_content: bool = True) -> None:
    """Install a tracer provider (or None to go back to no-op). Tests use this directly."""
    _state.provider = provider
    _state.capture_content = capture_content
    _state.tracer = (
        provider.get_tracer("agentic_rag") if provider is not None else trace.NoOpTracer()
    )


def setup_tracing(settings: Settings) -> TracerProvider | None:
    """Export spans to Phoenix over OTLP/HTTP. Returns None when tracing is disabled."""
    if not settings.tracing_enabled:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "agentic-rag", _PROJECT_ATTR: settings.tracing_project}
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.tracing_endpoint, timeout=5))
    )
    configure_tracer(provider, capture_content=settings.tracing_capture_content)
    log.info(
        "tracing_enabled", endpoint=settings.tracing_endpoint, project=settings.tracing_project
    )
    return provider


def shutdown_tracing() -> None:
    """Flush pending spans. Call before the process exits, or the last turn may be lost."""
    provider = _state.provider
    if provider is not None:
        provider.force_flush(timeout_millis=5000)
        provider.shutdown()
    configure_tracer(None)


async def phoenix_reachable(endpoint: str, *, timeout_s: float = 2.0) -> bool:
    """Any HTTP answer from the Phoenix port counts; only a connection failure does not."""
    base = endpoint.split("/v1/", 1)[0]
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            await client.get(base)
    except httpx.HTTPError:
        return False
    return True


def _clip(value: str) -> str:
    return value if len(value) <= _MAX_VALUE_CHARS else value[:_MAX_VALUE_CHARS] + "…[clipped]"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


@contextmanager
def span(
    name: str,
    kind: OpenInferenceSpanKindValues,
    *,
    input_value: Any = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Open a span as the current one. Exceptions are recorded on it and re-raised."""
    with _state.tracer.start_as_current_span(name) as sp:
        sp.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)
        if input_value is not None:
            set_input(sp, input_value)
        if attributes:
            set_attributes(sp, attributes)
        yield sp


def set_attributes(sp: Span, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            sp.set_attribute(
                key, value if isinstance(value, str | bool | int | float) else _to_text(value)
            )


def set_input(sp: Span, value: Any) -> None:
    if _state.capture_content:
        sp.set_attribute(SpanAttributes.INPUT_VALUE, _clip(_to_text(value)))


def set_output(sp: Span, value: Any) -> None:
    if _state.capture_content:
        sp.set_attribute(SpanAttributes.OUTPUT_VALUE, _clip(_to_text(value)))


def message_attributes(prefix: str, messages: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Flatten (role, content) pairs into OpenInference's indexed attribute names."""
    out: dict[str, Any] = {}
    for i, (role, content) in enumerate(messages):
        out[f"{prefix}.{i}.message.role"] = role
        if _state.capture_content:
            out[f"{prefix}.{i}.message.content"] = _clip(content)
    return out


def document_attributes(prefix: str, documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten documents ({id, content, score, metadata}) into indexed attribute names."""
    out: dict[str, Any] = {}
    for i, doc in enumerate(documents):
        base = f"{prefix}.{i}.document"
        if doc.get("id") is not None:
            out[f"{base}.id"] = str(doc["id"])
        if doc.get("score") is not None:
            out[f"{base}.score"] = float(doc["score"])
        if _state.capture_content and doc.get("content") is not None:
            out[f"{base}.content"] = _clip(str(doc["content"]))
        if doc.get("metadata"):
            out[f"{base}.metadata"] = _to_text(doc["metadata"])
    return out
