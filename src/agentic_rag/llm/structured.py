"""Pydantic-validated JSON extraction with a self-repair loop.

Flow: ask the model for JSON (constrained by the schema when the server supports it), parse and
validate. If that fails, feed the concrete validation errors back to the model and retry, up to
`structured_max_repairs` times. Metrics from every attempt are summed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from agentic_rag.core.exceptions import StructuredOutputError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.client import OllamaClient, ResponseFormat
from agentic_rag.llm.prompts import REPAIR_PROMPT, STRUCTURED_SYSTEM_PROMPT
from agentic_rag.llm.schemas import ChatMessage, LLMMetrics

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_json_object(raw: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def summarize_error(exc: json.JSONDecodeError | ValidationError) -> str:
    """Compact, model-readable error text. Deliberately omits the offending input values."""
    if isinstance(exc, json.JSONDecodeError):
        return f"- Output is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})."
    lines = []
    for err in exc.errors():
        path = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {path}: {err['msg']}")
    return "\n".join(lines)


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    data: T
    metrics: LLMMetrics
    attempts: int
    raw_output: str


class StructuredExtractor:
    def __init__(self, client: OllamaClient, *, max_repairs: int | None = None) -> None:
        self._client = client
        self._max_repairs = (
            client.settings.structured_max_repairs if max_repairs is None else max_repairs
        )

    async def extract(
        self,
        schema: type[T],
        user_prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StructuredResult[T]:
        schema_dict = schema.model_json_schema()
        response_format: ResponseFormat = (
            schema_dict if self._client.settings.ollama_use_schema_format else "json"
        )
        system_prompt = STRUCTURED_SYSTEM_PROMPT.format(schema=json.dumps(schema_dict, indent=2))
        if system:
            system_prompt = f"{system}\n\n{system_prompt}"

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        metrics: LLMMetrics | None = None
        raw = ""
        last_error = ""
        total_attempts = self._max_repairs + 1

        for attempt in range(1, total_attempts + 1):
            result = await self._client.chat(
                messages,
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            metrics = result.metrics if metrics is None else metrics.merge(result.metrics)
            raw = result.content

            try:
                data = schema.model_validate(parse_json_object(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = summarize_error(exc)
                log.warning(
                    "structured_validation_failed",
                    schema=schema.__name__,
                    attempt=attempt,
                    of=total_attempts,
                    error=last_error,
                )
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=raw),
                    ChatMessage(role="user", content=REPAIR_PROMPT.format(errors=last_error)),
                ]
                continue

            return StructuredResult(data=data, metrics=metrics, attempts=attempt, raw_output=raw)

        raise StructuredOutputError(
            f"{schema.__name__}: no schema-valid output after {total_attempts} attempt(s). "
            f"Last error: {last_error}",
            schema_name=schema.__name__,
            attempts=total_attempts,
            last_error=last_error,
            raw_output=raw,
        )
