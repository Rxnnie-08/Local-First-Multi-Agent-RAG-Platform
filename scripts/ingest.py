"""Ingest files or folders into Qdrant.

python scripts/ingest.py samples
python scripts/ingest.py C:\\path\\to\\notes C:\\path\\to\\paper.pdf --force
python scripts/ingest.py samples --reset      # drop and recreate the collection first
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.services import open_services


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="files or directories to ingest")
    p.add_argument("--force", action="store_true", help="re-ingest even if the file is unchanged")
    p.add_argument("--reset", action="store_true", help="drop and recreate the collection first")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging("WARNING" if settings.log_level == "INFO" else settings.log_level)

    async with open_services(settings) as svc:
        try:
            if not await svc.store.health_check():
                print(f"FAIL: Qdrant is not reachable at {settings.qdrant_url}.")
                print("      Start it with: docker compose up -d")
                return 1
            if not await svc.client.health_check():
                print(f"FAIL: Ollama is not reachable at {settings.ollama_base_url}.")
                return 1
            await svc.client.ensure_model(settings.embedding_model)

            if args.reset:
                await svc.store.reset_collection(dim=svc.embedder.dim)
                print(f"Collection '{svc.store.collection}' reset.")

            exit_code = 0
            for path in args.paths:
                report = await svc.pipeline.ingest_path(path, force=args.force)
                print(f"\n{path}")
                for f in report.files:
                    detail = f"{f.chunks} chunks" if f.status == "ingested" else (f.error or "")
                    print(f"  {f.status:<18} {f.source}  {detail}  ({f.duration_s:.1f}s)")
                print(
                    f"  -> {report.ingested} ingested, {report.skipped} unchanged, "
                    f"{report.failed} failed, {report.chunks} chunks in {report.duration_s:.1f}s"
                )
                if report.failed:
                    exit_code = 1
            print(f"\nTotal chunks in collection: {await svc.store.count()}")
            return exit_code
        except AgenticRAGError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    sys.exit(asyncio.run(main()))
