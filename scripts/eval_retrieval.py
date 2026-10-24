"""Retrieval benchmark on the Orbit evaluation corpus (no LLM calls, about a minute).

python scripts/eval_retrieval.py

Ingests eval/corpus into its OWN Qdrant collection (your documents are untouched), then compares
dense, sparse, hybrid and hybrid + rerank, and suggests a value for RERANK_MIN_SCORE.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.evaluation.agent_eval import save_report
from agentic_rag.evaluation.dataset import check_dataset, load_cases
from agentic_rag.evaluation.retrieval_eval import format_retrieval_report, run_retrieval_eval
from agentic_rag.services import open_services, prepare

ROOT = Path(__file__).resolve().parents[1]
EVAL_COLLECTION = "agentic_rag_eval"


async def main() -> int:
    settings = get_settings().model_copy(
        update={"qdrant_collection": EVAL_COLLECTION, "tracing_enabled": False}
    )
    configure_logging("WARNING")
    cases = load_cases(ROOT / "eval" / "cases.jsonl")
    if problems := check_dataset(cases, ROOT / "eval" / "corpus"):
        print("The evaluation set is inconsistent:\n  " + "\n  ".join(problems))
        return 1

    async with open_services(settings) as svc:
        try:
            svc = await prepare(svc)
            report = await svc.pipeline.ingest_path(ROOT / "eval" / "corpus")
            print(
                f"Corpus: {report.ingested} ingested, {report.skipped} unchanged, "
                f"{await svc.store.count()} chunks in '{EVAL_COLLECTION}'\n"
            )
            result = await run_retrieval_eval(
                svc.retriever, svc.reranker, cases, settings, collection=EVAL_COLLECTION
            )
        except AgenticRAGError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1
    print(format_retrieval_report(result))
    path = save_report(result, ROOT / "eval" / "reports", f"retrieval_{settings.embedding_model}")
    print(f"\nSaved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main()))
