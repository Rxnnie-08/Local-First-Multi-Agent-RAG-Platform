"""Request/response models and per-call performance metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


def _seconds(value: object) -> float:
    """Ollama reports durations in nanoseconds."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value) / 1e9


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


class LLMMetrics(BaseModel):
    """Token and latency accounting for one call, or several merged calls."""

    model: str
    calls: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_s: float = 0.0
    load_duration_s: float = 0.0
    prompt_eval_duration_s: float = 0.0
    eval_duration_s: float = 0.0
    # End-to-end wall clock as seen by the caller, including any retries.
    wall_latency_s: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed (completion tokens / eval time). None if not measurable."""
        if self.eval_duration_s <= 0:
            return None
        return self.completion_tokens / self.eval_duration_s

    @classmethod
    def from_ollama(cls, model: str, body: Mapping[str, Any], wall_latency_s: float) -> LLMMetrics:
        return cls(
            model=model,
            prompt_tokens=_count(body.get("prompt_eval_count")),
            completion_tokens=_count(body.get("eval_count")),
            total_duration_s=_seconds(body.get("total_duration")),
            load_duration_s=_seconds(body.get("load_duration")),
            prompt_eval_duration_s=_seconds(body.get("prompt_eval_duration")),
            eval_duration_s=_seconds(body.get("eval_duration")),
            wall_latency_s=wall_latency_s,
        )

    def merge(self, other: LLMMetrics) -> LLMMetrics:
        """Sum two metrics objects (e.g. across repair attempts)."""
        return LLMMetrics(
            model=self.model,
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_duration_s=self.total_duration_s + other.total_duration_s,
            load_duration_s=self.load_duration_s + other.load_duration_s,
            prompt_eval_duration_s=self.prompt_eval_duration_s + other.prompt_eval_duration_s,
            eval_duration_s=self.eval_duration_s + other.eval_duration_s,
            wall_latency_s=self.wall_latency_s + other.wall_latency_s,
        )


class EmbedResult(BaseModel):
    vectors: list[list[float]]
    metrics: LLMMetrics


class ChatResult(BaseModel):
    content: str
    metrics: LLMMetrics
    finish_reason: str | None = None
