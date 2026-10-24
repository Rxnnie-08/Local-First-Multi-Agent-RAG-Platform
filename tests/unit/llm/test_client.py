from __future__ import annotations

import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaResponseError,
    OllamaServerError,
    OllamaTimeoutError,
)
from agentic_rag.llm import ChatMessage, OllamaClient
from fakes import FakeOllama, chat_response, error_response

HELLO = [ChatMessage(role="user", content="hi")]


def make_client(settings: Settings, fake: FakeOllama) -> OllamaClient:
    return OllamaClient(settings, transport=fake.transport)


async def test_chat_returns_content_and_metrics(settings: Settings) -> None:
    fake = FakeOllama(
        [chat_response("hello", prompt_tokens=7, completion_tokens=40, eval_duration_s=2.0)]
    )
    async with make_client(settings, fake) as client:
        result = await client.chat(HELLO)

    assert result.content == "hello"
    assert result.finish_reason == "stop"
    m = result.metrics
    assert (m.prompt_tokens, m.completion_tokens, m.total_tokens) == (7, 40, 47)
    assert m.tokens_per_second == pytest.approx(20.0)
    assert m.load_duration_s == pytest.approx(0.5)
    assert m.wall_latency_s >= 0


async def test_chat_payload_uses_settings_and_overrides(settings: Settings) -> None:
    fake = FakeOllama([chat_response("ok")])
    async with make_client(settings, fake) as client:
        await client.chat(HELLO, temperature=0.7, max_tokens=64, response_format="json")

    body = fake.json_body()
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["options"] == {"temperature": 0.7, "num_ctx": 4096, "num_predict": 64}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_format_omitted_when_not_requested(settings: Settings) -> None:
    fake = FakeOllama([chat_response("ok")])
    async with make_client(settings, fake) as client:
        await client.chat(HELLO)
    assert "format" not in fake.json_body()


async def test_retries_on_5xx_then_succeeds(settings: Settings) -> None:
    fake = FakeOllama(
        [error_response(503, "busy"), error_response(500, "oops"), chat_response("ok")]
    )
    async with make_client(settings, fake) as client:
        result = await client.chat(HELLO)
    assert result.content == "ok"
    assert len(fake.requests) == 3


async def test_gives_up_after_max_attempts(settings: Settings) -> None:
    fake = FakeOllama([error_response(500, "boom")] * 3)
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaServerError, match="boom"):
            await client.chat(HELLO)
    assert len(fake.requests) == 3


async def test_connection_errors_are_retried_then_raised(settings: Settings) -> None:
    fake = FakeOllama([httpx.ConnectError("refused")] * 3)
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama"):
            await client.chat(HELLO)
    assert len(fake.requests) == 3


async def test_connection_error_recovers(settings: Settings) -> None:
    fake = FakeOllama([httpx.ConnectError("refused"), chat_response("back")])
    async with make_client(settings, fake) as client:
        result = await client.chat(HELLO)
    assert result.content == "back"


async def test_read_timeout_is_not_retried(settings: Settings) -> None:
    fake = FakeOllama([httpx.ReadTimeout("slow")])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaTimeoutError):
            await client.chat(HELLO)
    assert len(fake.requests) == 1


async def test_missing_model_is_not_retried(settings: Settings) -> None:
    fake = FakeOllama([error_response(404, 'model "nope" not found, try pulling it first')])
    async with make_client(settings, fake) as client:
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            await client.chat(HELLO, model="nope")
    assert len(fake.requests) == 1


async def test_other_4xx_is_not_retried(settings: Settings) -> None:
    fake = FakeOllama([error_response(400, "invalid format")])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaRequestError, match="invalid format"):
            await client.chat(HELLO)
    assert len(fake.requests) == 1


async def test_non_json_body_raises_response_error(settings: Settings) -> None:
    fake = FakeOllama([httpx.Response(200, text="<html>not json</html>")])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaResponseError):
            await client.chat(HELLO)


async def test_missing_content_raises_response_error(settings: Settings) -> None:
    fake = FakeOllama([httpx.Response(200, json={"done": True})])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaResponseError, match=r"message\.content"):
            await client.chat(HELLO)


async def test_health_check_true_and_false(settings: Settings) -> None:
    ok = FakeOllama([httpx.Response(200, json={"models": []})])
    async with make_client(settings, ok) as client:
        assert await client.health_check() is True

    down = FakeOllama([httpx.ConnectError("refused")])
    async with make_client(settings, down) as client:
        assert await client.health_check() is False
    assert len(down.requests) == 1  # health checks never retry


async def test_list_models_and_has_model(settings: Settings) -> None:
    tags = {"models": [{"name": "llama3.1:8b"}, {"name": "llama3:latest"}]}
    fake = FakeOllama([httpx.Response(200, json=tags)] * 3)
    async with make_client(settings, fake) as client:
        assert await client.list_models() == ["llama3.1:8b", "llama3:latest"]
        assert await client.has_model("llama3") is True  # bare name == :latest
        assert await client.has_model("mistral:7b") is False


async def test_ensure_model_raises_with_pull_hint(settings: Settings) -> None:
    fake = FakeOllama([httpx.Response(200, json={"models": []})])
    async with make_client(settings, fake) as client:
        with pytest.raises(ModelNotFoundError, match="ollama pull mistral:7b"):
            await client.ensure_model("mistral:7b")


# --- streaming ------------------------------------------------------------------------------


async def test_stream_passes_pieces_to_the_callback_and_returns_the_full_reply(
    settings: Settings,
) -> None:
    from fakes import stream_response

    fake = FakeOllama([stream_response(["Hel", "lo", "!"], prompt_tokens=7)])
    seen: list[str] = []
    async with make_client(settings, fake) as client:
        result = await client.chat(HELLO, on_token=seen.append)

    assert seen == ["Hel", "lo", "!"]
    assert result.content == "Hello!"
    assert result.metrics.prompt_tokens == 7
    assert result.metrics.completion_tokens == 3
    assert result.finish_reason == "stop"
    assert fake.json_body()["stream"] is True


async def test_stream_is_off_without_a_callback(settings: Settings) -> None:
    fake = FakeOllama([chat_response("hi")])
    async with make_client(settings, fake) as client:
        await client.chat(HELLO)
    assert fake.json_body()["stream"] is False


async def test_stream_http_error_is_mapped_and_not_retried(settings: Settings) -> None:
    fake = FakeOllama([error_response(500, "boom"), chat_response("never used")])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaServerError, match="boom"):
            await client.chat(HELLO, on_token=lambda _: None)
    assert len(fake.requests) == 1  # a retry after partial output would repeat it


async def test_stream_unknown_model_is_reported_clearly(settings: Settings) -> None:
    fake = FakeOllama([error_response(404, "model 'x' not found")])
    async with make_client(settings, fake) as client:
        with pytest.raises(ModelNotFoundError):
            await client.chat(HELLO, on_token=lambda _: None)


async def test_stream_that_ends_without_done_is_an_error(settings: Settings) -> None:
    from fakes import stream_response

    fake = FakeOllama([stream_response(["partial"], done=False)])
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaResponseError, match="before it reported completion"):
            await client.chat(HELLO, on_token=lambda _: None)


async def test_stream_error_line_is_raised(settings: Settings) -> None:
    from fakes import stream_response

    fake = FakeOllama(
        [stream_response(["a"], done=False, extra_lines=[{"error": "out of memory"}])]
    )
    async with make_client(settings, fake) as client:
        with pytest.raises(OllamaServerError, match="out of memory"):
            await client.chat(HELLO, on_token=lambda _: None)
