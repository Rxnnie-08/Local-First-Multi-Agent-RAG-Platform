"""Query the collection.

python scripts/query.py "what is the default keep alive?"
python scripts/query.py "error E-4021" --mode compare     # dense vs sparse vs hybrid
python scripts/query.py "chunk overlap" -k 3 --source samples/rag_basics.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.retrieval import RetrievalMode, RetrievalResult
from agentic_rag.services import open_services
from agentic_rag.vectorstore import SearchFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("query")
    p.add_argument(
        "-k", type=int, default=None, help="number of results (default: RETRIEVAL_TOP_K)"
    )
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse", "compare"], default="hybrid")
    p.add_argument("--source", action="append", help="restrict to a source (repeatable)")
    return p.parse_args()


def show(result: RetrievalResult) -> None:
    print(
        f"\n=== {result.mode.upper()}  embed {result.embed_s * 1000:.0f} ms, "
        f"search {result.search_s * 1000:.0f} ms ==="
    )
    if not result.hits:
        print("  (no results)")
    for rank, hit in enumerate(result.hits, start=1):
        where = hit.metadata.get("heading_path") or (
            f"page {hit.metadata['page']}" if hit.metadata.get("page") else ""
        )
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank:>2}. {hit.score:.4f}  {hit.source}" + (f"  [{where}]" if where else ""))
        print(f"      {snippet}")


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging("WARNING" if settings.log_level == "INFO" else settings.log_level)
    flt = SearchFilter(sources=args.source) if args.source else None
    modes: list[RetrievalMode] = (
        ["dense", "sparse", "hybrid"] if args.mode == "compare" else [args.mode]
    )

    async with open_services(settings) as svc:
        try:
            if await svc.store.count() == 0:
                print("The collection is empty. Run: python scripts/ingest.py <path>")
                return 1
            for mode in modes:
                show(await svc.retriever.retrieve(args.query, k=args.k, mode=mode, flt=flt))
        except AgenticRAGError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    sys.exit(asyncio.run(main()))
