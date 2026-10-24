"""Code worker: the LLM writes a short Python script, a Docker sandbox runs it.

For arithmetic, counting, date maths and data wrangling, running code is far more reliable than
asking an 8B model to compute in its head. The script is untrusted; see `sandbox.runner`.
"""

from __future__ import annotations

import re
import time

from agentic_rag.agents.models import Observation, Passage
from agentic_rag.agents.workers.base import WorkerTask
from agentic_rag.config import Settings
from agentic_rag.core.exceptions import LLMError, SandboxError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm import ChatMessage, LLMMetrics, OllamaClient
from agentic_rag.observability import SpanKind, set_attributes, set_output, span
from agentic_rag.sandbox import CodeRunner, SandboxResult

log = get_logger(__name__)

CODE_SYSTEM = """\
You write one short, self-contained Python 3 script that solves the task.
- Use only the standard library. There is no network, no internet and no file access.
- Print the final result with print(). Print only what is needed.
- Reply with the script inside a single ```python code block and nothing else."""

REPAIR_USER = """\
The script failed:
{error}

Fix it. Reply with the complete corrected script in a single ```python code block."""

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*)", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the script out of a model reply. Falls back to an unterminated fence (a reply cut off
    by the token limit) and finally to the whole reply."""
    if match := _FENCE_RE.search(text):
        return match.group(1).strip()
    if match := _OPEN_FENCE_RE.search(text):
        return match.group(1).strip()
    return text.strip()


def _failure_text(result: SandboxResult, timeout_s: float) -> str:
    if result.timed_out:
        return f"The script did not finish within {timeout_s:.0f} seconds."
    tail = result.stderr.strip() or f"exit code {result.exit_code} with no error output"
    return tail[-600:]


class CodeWorker:
    name = "python"
    description = (
        "Writes and runs a short Python script (standard library only, no network, strict time "
        "limit) in an isolated sandbox. Use for arithmetic, percentages, unit or date "
        "calculations, counting, sorting or transforming data given in the question. Not for "
        "questions about the user's documents."
    )

    def __init__(self, client: OllamaClient, runner: CodeRunner, settings: Settings) -> None:
        self._client = client
        self._runner = runner
        self._settings = settings

    async def run(self, task: WorkerTask) -> Observation:
        s = self._settings
        started = time.perf_counter()
        messages = [
            ChatMessage(role="system", content=CODE_SYSTEM),
            ChatMessage(
                role="user", content=f"Task: {task.query}\nOriginal question: {task.question}"
            ),
        ]
        metrics: LLMMetrics | None = None
        code = ""
        result: SandboxResult | None = None
        attempts = 0

        def fail(summary: str, error: str) -> Observation:
            return Observation(
                worker=self.name,
                query=task.query,
                ok=False,
                confident=False,
                summary=summary,
                error=error,
                detail={"attempts": attempts, "total_s": round(time.perf_counter() - started, 2)},
                metrics=metrics,
            )

        for attempt in range(s.code_max_repairs + 1):
            attempts = attempt + 1
            try:
                reply = await self._client.chat(messages, max_tokens=s.code_max_tokens)
            except LLMError as exc:
                return fail("Could not generate a script.", f"{type(exc).__name__}: {exc}")
            metrics = reply.metrics if metrics is None else metrics.merge(reply.metrics)
            code = extract_code(reply.content)
            if not code:
                return fail("The model returned no code.", "empty reply")

            try:
                with span(
                    "sandbox.run",
                    SpanKind.TOOL,
                    input_value=code,
                    attributes={"tool.name": "python_sandbox", "sandbox.attempt": attempts},
                ) as sp:
                    result = await self._runner.run(code)
                    set_output(sp, result.stdout or result.stderr)
                    set_attributes(
                        sp,
                        {
                            "sandbox.exit_code": result.exit_code,
                            "sandbox.timed_out": result.timed_out,
                            "sandbox.duration_s": round(result.duration_s, 3),
                        },
                    )
            except SandboxError as exc:
                return fail("The code sandbox is unavailable.", str(exc))
            if result.ok:
                break
            log.info("code_attempt_failed", attempt=attempts, timed_out=result.timed_out)
            if result.timed_out or attempt == s.code_max_repairs:
                break
            messages += [
                ChatMessage(role="assistant", content=reply.content),
                ChatMessage(
                    role="user",
                    content=REPAIR_USER.format(error=_failure_text(result, s.sandbox_timeout_s)),
                ),
            ]

        assert result is not None  # the loop runs at least once
        detail = {
            "attempts": attempts,
            "exit_code": result.exit_code,
            "sandbox_s": round(result.duration_s, 2),
            "truncated": result.truncated,
            "code": code,
            "total_s": round(time.perf_counter() - started, 2),
        }
        if not result.ok:
            error = _failure_text(result, s.sandbox_timeout_s)
            return Observation(
                worker=self.name,
                query=task.query,
                ok=False,
                confident=False,
                summary="The script failed.",
                error=error,
                detail=detail,
                metrics=metrics,
            )

        output = result.stdout.strip()
        text = f"Python script:\n```python\n{code}\n```\nOutput:\n{output or '(no output)'}"
        if result.truncated:
            text += "\n(output truncated)"
        return Observation(
            worker=self.name,
            query=task.query,
            ok=True,
            confident=bool(output),
            summary=f"Ran a script; output: {output[:120]}"
            if output
            else "Ran a script with no output.",
            passages=[Passage(text=text, source="python sandbox", location="stdout")],
            detail=detail,
            metrics=metrics,
        )
