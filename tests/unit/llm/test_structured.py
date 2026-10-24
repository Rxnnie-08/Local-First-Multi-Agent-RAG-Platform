from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import StructuredOutputError
from agentic_rag.llm import OllamaClient, StructuredExtractor
from agentic_rag.llm.structured import parse_json_object
from fakes import FakeOllama, chat_response


class Person(BaseModel):
    name: str
    age: int
    skills: list[str]


GOOD = json.dumps({"name": "Priya", "age": 21, "skills": ["Python", "SQL"]})
MISSING_AGE = json.dumps({"name": "Priya", "skills": []})


def make_extractor(
    settings: Settings, fake: FakeOllama
) -> tuple[OllamaClient, StructuredExtractor]:
    client = OllamaClient(settings, transport=fake.transport)
    return client, StructuredExtractor(client)


async def test_valid_output_first_try(settings: Settings) -> None:
    fake = FakeOllama([chat_response(GOOD)])
    client, extractor = make_extractor(settings, fake)
    async with client:
        result = await extractor.extract(Person, "Priya is 21 and knows Python and SQL.")

    assert result.data == Person(name="Priya", age=21, skills=["Python", "SQL"])
    assert result.attempts == 1
    assert result.metrics.calls == 1


async def test_schema_is_sent_as_format_and_in_system_prompt(settings: Settings) -> None:
    fake = FakeOllama([chat_response(GOOD)])
    client, extractor = make_extractor(settings, fake)
    async with client:
        await extractor.extract(Person, "text", system="Be terse.")

    body = fake.json_body()
    assert body["format"] == Person.model_json_schema()
    system = body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("Be terse.")
    assert '"skills"' in system["content"]


async def test_plain_json_mode_when_schema_format_disabled() -> None:
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://ollama.test",
        ollama_use_schema_format=False,
        ollama_retry_max_wait_s=0.0,
    )
    fake = FakeOllama([chat_response(GOOD)])
    client, extractor = make_extractor(settings, fake)
    async with client:
        await extractor.extract(Person, "text")
    assert fake.json_body()["format"] == "json"


async def test_repair_loop_feeds_errors_back_and_sums_metrics(settings: Settings) -> None:
    fake = FakeOllama(
        [
            chat_response(MISSING_AGE, prompt_tokens=100, completion_tokens=10),
            chat_response(GOOD, prompt_tokens=150, completion_tokens=12),
        ]
    )
    client, extractor = make_extractor(settings, fake)
    async with client:
        result = await extractor.extract(Person, "Priya is 21.")

    assert result.attempts == 2
    assert result.data.age == 21
    assert result.metrics.calls == 2
    assert result.metrics.prompt_tokens == 250
    assert result.metrics.completion_tokens == 22

    second_request = fake.json_body(1)["messages"]
    assert [m["role"] for m in second_request] == ["system", "user", "assistant", "user"]
    assert second_request[2]["content"] == MISSING_AGE
    assert "age: Field required" in second_request[3]["content"]


async def test_raises_after_exhausting_repairs(settings: Settings) -> None:
    fake = FakeOllama([chat_response("not json at all")] * 3)  # 1 try + 2 repairs
    client, extractor = make_extractor(settings, fake)
    async with client:
        with pytest.raises(StructuredOutputError) as info:
            await extractor.extract(Person, "text")

    err = info.value
    assert err.attempts == 3
    assert err.schema_name == "Person"
    assert err.raw_output == "not json at all"
    assert "not valid JSON" in err.last_error
    assert len(fake.requests) == 3


async def test_zero_repairs_means_single_attempt(settings: Settings) -> None:
    fake = FakeOllama([chat_response(MISSING_AGE)])
    client = OllamaClient(settings, transport=fake.transport)
    extractor = StructuredExtractor(client, max_repairs=0)
    async with client:
        with pytest.raises(StructuredOutputError) as info:
            await extractor.extract(Person, "text")
    assert info.value.attempts == 1


@pytest.mark.parametrize(
    "raw",
    [
        GOOD,
        f"```json\n{GOOD}\n```",
        f"```\n{GOOD}\n```",
        f"Sure! Here is the JSON:\n{GOOD}\nHope that helps.",
        f"  \n{GOOD}\n  ",
    ],
)
def test_parse_json_object_tolerates_wrapping(raw: str) -> None:
    assert Person.model_validate(parse_json_object(raw)).name == "Priya"


def test_parse_json_object_raises_on_garbage() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("no braces here")
