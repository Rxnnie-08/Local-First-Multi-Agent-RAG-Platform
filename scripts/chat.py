"""Multi-turn chat over your ingested documents.

python scripts/chat.py                       # interactive
python scripts/chat.py --once "what is keep_alive?" --trace

Commands:  /trace  toggle the per-step trace   /new  start a fresh conversation   /exit  quit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from agentic_rag.agents import AgentResponse
from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.observability import phoenix_reachable
from agentic_rag.services import open_services, prepare


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--once", metavar="QUESTION", help="ask one question and exit")
    p.add_argument("--trace", action="store_true", help="show the per-step trace")
    return p.parse_args()


def show(response: AgentResponse, *, trace: bool) -> None:
    print(f"\n{response.answer}\n")
    cited = [c for c in response.citations if c.cited]
    if cited:
        print("Sources:")
        for c in cited:
            where = f" [{c.location}]" if c.location else ""
            score = f"  (relevance {c.score:.2f})" if c.score is not None else ""
            print(f"  [{c.n}] {c.source}{where}{score}")
    elif response.citations:
        print(f"(no citations: {len(response.citations)} retrieved passages were not used)")
    m = response.metrics
    tps = f"{m.tokens_per_second:.1f}" if m.tokens_per_second is not None else "n/a"
    print(
        f"route: {' -> '.join(response.route)} | llm calls {m.llm_calls} | "
        f"tokens {m.prompt_tokens}+{m.completion_tokens} | {tps} tok/s | "
        f"{m.total_latency_s:.1f}s total"
    )
    if trace:
        for step in response.trace:
            llm = f" llm {step.llm.wall_latency_s:.1f}s" if step.llm else ""
            print(f"  - {step.node:<10} {step.latency_s:6.2f}s{llm}  {step.detail}")


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging("WARNING" if settings.log_level == "INFO" else settings.log_level)

    async with open_services(settings) as svc:
        try:
            if await svc.store.count() == 0:
                print("The collection is empty. Run: python scripts/ingest.py samples")
                return 1
            svc = await prepare(svc)
            if settings.tracing_enabled:
                if await phoenix_reachable(settings.tracing_endpoint):
                    print(f"Tracing to Phoenix: {settings.tracing_endpoint.split('/v1/')[0]}")
                else:
                    print(
                        "WARNING: TRACING_ENABLED=true but Phoenix is not reachable at "
                        f"{settings.tracing_endpoint}. Start it with: docker compose up -d phoenix"
                    )
            thread = uuid.uuid4().hex[:8]
            trace = args.trace

            if args.once:
                show(await svc.agent.ask(args.once, thread_id=thread), trace=trace)
                return 0

            print("Ask about your documents. /trace toggles tracing, /new resets, /exit quits.")
            while True:
                try:
                    line = await asyncio.to_thread(input, "\nyou> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                line = line.strip()
                if not line:
                    continue
                if line in {"/exit", "/quit"}:
                    return 0
                if line == "/trace":
                    trace = not trace
                    print(f"trace {'on' if trace else 'off'}")
                    continue
                if line == "/new":
                    thread = uuid.uuid4().hex[:8]
                    print("started a new conversation")
                    continue
                try:
                    show(await svc.agent.ask(line, thread_id=thread), trace=trace)
                except AgenticRAGError as exc:
                    print(f"FAIL: {type(exc).__name__}: {exc}")
        except AgenticRAGError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main()))
