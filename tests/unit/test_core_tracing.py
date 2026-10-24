from __future__ import annotations

import pytest

from agentic_rag.core import tracing
from agentic_rag.observability import configure_tracer


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("", "http://default/v1/traces"),
        ("http://127.0.0.1:6006", "http://127.0.0.1:6006/v1/traces"),
        ("http://127.0.0.1:6006/", "http://127.0.0.1:6006/v1/traces"),
        ("http://host:6006/v1/traces", "http://host:6006/v1/traces"),
    ],
)
def test_collector_endpoint_accepts_base_url_or_full_url(
    monkeypatch: pytest.MonkeyPatch, env: str, expected: str
) -> None:
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", env)
    assert tracing._collector_endpoint("http://default/v1/traces") == expected


def test_setup_tracing_wrapper_enables_the_project_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006")
    provider = tracing.setup_tracing("my-project")
    try:
        assert provider is not None
        assert provider.resource.attributes["openinference.project.name"] == "my-project"
    finally:
        configure_tracer(None)
