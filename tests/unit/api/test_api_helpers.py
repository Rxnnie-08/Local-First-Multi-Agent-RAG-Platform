from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from fastapi import HTTPException

from agentic_rag.agents import TokenEvent
from agentic_rag.api.app import Lease, Runtime, _safe_name, _with_heartbeat


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("notes.md", "notes.md"),
        ("../../etc/passwd.md", "passwd.md"),
        ("..\\..\\win.md", "win.md"),
        (".hidden.md", "hidden.md"),
        ("my report (final).md", "my report _final_.md"),
        ("/abs/path/file.txt", "file.txt"),
    ],
)
def test_safe_name_keeps_only_a_plain_file_name(raw: str, expected: str) -> None:
    assert _safe_name(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "..", "/"])
def test_safe_name_rejects_names_with_nothing_usable(raw: str | None) -> None:
    with pytest.raises(HTTPException) as exc:
        _safe_name(raw)
    assert exc.value.status_code == 422


async def test_lease_release_is_idempotent() -> None:
    class FakeServices:
        class settings:
            api_max_concurrent_turns = 1

    runtime = Runtime(FakeServices())  # type: ignore[arg-type]
    lock = runtime.thread_lock("t")
    await lock.acquire()
    await runtime.turn_slots.acquire()
    lease = Lease(runtime, "t")

    lease.release()
    lease.release()  # the generator's finally and the background task may both call it

    assert runtime.turn_slots._value == 1  # not 2
    assert "t" not in runtime._thread_locks


async def _slow(delays: list[float]) -> AsyncIterator[TokenEvent]:
    for i, delay in enumerate(delays):
        await asyncio.sleep(delay)
        yield TokenEvent(text=str(i))


async def test_heartbeat_fills_the_silence_between_events() -> None:
    out = [e async for e in _with_heartbeat(_slow([0.12]), 0.03, time.monotonic() + 5)]
    last = out[-1]
    assert last is not None
    assert last.type == "token"
    assert out.count(None) >= 2


async def test_heartbeat_passes_events_through_in_order_and_ends() -> None:
    out = [e async for e in _with_heartbeat(_slow([0.0, 0.0, 0.0]), 1.0, time.monotonic() + 5)]
    assert [e.text for e in out if e is not None and isinstance(e, TokenEvent)] == ["0", "1", "2"]


async def test_heartbeat_raises_at_the_deadline_and_stops_the_source() -> None:
    closed = asyncio.Event()

    async def source() -> AsyncIterator[TokenEvent]:
        try:
            await asyncio.sleep(10)
            yield TokenEvent(text="never")
        finally:
            closed.set()

    stream = source()
    with pytest.raises(TimeoutError):
        async for _ in _with_heartbeat(stream, 0.02, time.monotonic() + 0.1):
            pass
    await asyncio.sleep(0)
    assert closed.is_set()
