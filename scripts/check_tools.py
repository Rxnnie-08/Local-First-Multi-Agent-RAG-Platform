"""Check the Phase 4 tools on their own, without the LLM.

python scripts/check_tools.py

Runs a tiny script in the Docker sandbox, then proves the sandbox really is isolated (no network,
read-only filesystem, memory limit, time limit), then tries one DuckDuckGo search.
"""

from __future__ import annotations

import asyncio
import sys

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.sandbox import DockerSandbox
from agentic_rag.websearch import build_search_provider

NETWORK = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
    "    print('NETWORK REACHABLE')\n"
    "except OSError as e:\n"
    "    print('blocked:', type(e).__name__)\n"
)
WRITE = (
    "try:\n"
    "    open('/etc/pwned', 'w').write('x')\n"
    "    print('ROOT FS WRITABLE')\n"
    "except OSError as e:\n"
    "    print('blocked:', type(e).__name__)\n"
)
MEMORY = "x = bytearray(2 * 1024**3)\nprint('ALLOCATED 2 GB')\n"
FOREVER = "while True:\n    pass\n"


async def main() -> int:
    settings = get_settings()
    configure_logging("WARNING")
    failed = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failed
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    box = DockerSandbox(settings)
    try:
        await box.check()
    except AgenticRAGError as exc:
        print(f"[FAIL] sandbox unavailable: {exc}")
        return 1

    r = await box.run("print(sum(range(10)))")
    report("runs code", r.ok and r.stdout.strip() == "45", f"{r.duration_s:.1f}s")
    r = await box.run(NETWORK)
    report(
        "no network",
        r.ok and "blocked" in r.stdout and "REACHABLE" not in r.stdout,
        r.stdout.strip(),
    )
    r = await box.run(WRITE)
    report("read-only filesystem", "blocked" in r.stdout, r.stdout.strip())
    r = await box.run(MEMORY)
    report("memory limit", not r.ok and "ALLOCATED" not in r.stdout, f"exit {r.exit_code}")
    r = await box.run("print(1/0)")
    report("errors are reported", not r.ok and "ZeroDivisionError" in r.stderr)
    r = await box.run(FOREVER)
    report("time limit", r.timed_out, f"after {r.duration_s:.0f}s")

    provider = build_search_provider(settings)
    if provider is None:
        print("[SKIP] web search disabled (WEB_SEARCH_ENABLED=false)")
    else:
        try:
            results = await provider.search("python programming language", max_results=3)
            report("web search", len(results) > 0, f"{len(results)} results via {provider.name}")
        except AgenticRAGError as exc:
            report("web search", False, str(exc)[:200])
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main()))
