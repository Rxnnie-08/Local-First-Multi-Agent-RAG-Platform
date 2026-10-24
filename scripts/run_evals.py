"""Evaluate the live agent: routing, answers, sources, abstention, latency and tokens/second.

python scripts/run_evals.py                          # tests/evaluation_set.json
python scripts/run_evals.py --set eval/cases.jsonl   # the full 39-case set (about 15 min on CPU)
python scripts/run_evals.py --set eval/cases.jsonl --skip-live --category python --limit 5
python scripts/run_evals.py --model mistral:7b --trace   # another model; traces go to Phoenix

Accepts two formats. A .json file is a list of
  {id, query, expected_route, expected_answer_keywords, ground_truth_context}
where expected_route is the first step the supervisor should take (rag, web, python or answer;
"code_sandbox" is accepted for python) and at least half the keywords must appear in the answer.
A .jsonl file uses the richer format described in src/agentic_rag/evaluation/dataset.py.

By default it ingests eval/corpus into a separate collection, so your own documents are untouched.
Use --no-ingest --collection documents to evaluate against your own collection instead.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import AgenticRAGError
from agentic_rag.core.logging import configure_logging
from agentic_rag.evaluation.agent_eval import (
    CaseResult,
    format_agent_report,
    run_agent_eval,
    save_report,
)
from agentic_rag.evaluation.dataset import check_dataset, load_any
from agentic_rag.services import open_services, prepare

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval" / "corpus"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--set", default="tests/evaluation_set.json", help="evaluation file (.json/.jsonl)"
    )
    p.add_argument("--model", help="override OLLAMA_MODEL for this run")
    p.add_argument("--category", action="append", help="only these categories (repeatable)")
    p.add_argument("--skip-live", action="store_true", help="skip cases that need the internet")
    p.add_argument("--limit", type=int, help="run only the first N selected cases")
    p.add_argument("--trace", action="store_true", help="send traces to Phoenix (project: eval)")
    p.add_argument("--label", help="report file label (default: the model name)")
    p.add_argument("--collection", default="agentic_rag_eval", help="Qdrant collection to query")
    p.add_argument("--no-ingest", action="store_true", help="do not ingest eval/corpus first")
    return p.parse_args()


def show(done: int, total: int, r: CaseResult) -> None:
    mark = "PASS" if r.passed else "FAIL"
    print(f"[Q{done}/{total} {mark}: {r.question}]")
    print(f" -> Expected: {r.expected_first} | Actual route: {' -> '.join(r.route) or 'error'}")
    detail = r.error or " ".join(r.answer.split())[:120]
    print(f" -> {r.latency_s:.1f}s | {detail}\n")


async def main() -> int:
    args = parse_args()
    update: dict[str, object] = {
        "qdrant_collection": args.collection,
        "tracing_enabled": args.trace,
        "tracing_project": "agentic-rag-eval",
    }
    if args.model:
        update["ollama_model"] = args.model
    settings = get_settings().model_copy(update=update)
    configure_logging("WARNING")

    path = Path(args.set)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"Evaluation set not found at {path}.")
        return 1
    cases = load_any(path, CORPUS)
    if path.suffix == ".jsonl" and (problems := check_dataset(cases, CORPUS)):
        print("The evaluation set is inconsistent:\n  " + "\n  ".join(problems))
        return 1
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.skip_live:
        cases = [c for c in cases if not c.live]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.")
        return 1

    async with open_services(settings) as svc:
        try:
            svc = await prepare(svc)
            if not await svc.client.has_model(settings.ollama_model):
                print(
                    f"Model {settings.ollama_model!r} is not pulled. Run: ollama pull "
                    f"{settings.ollama_model}"
                )
                return 1
            if not args.no_ingest:
                await svc.pipeline.ingest_path(CORPUS)
            print(
                f"Model {settings.ollama_model}; {len(cases)} cases; collection "
                f"'{settings.qdrant_collection}'; workers: {', '.join(svc.agent.worker_names)}\n"
            )
            report = await run_agent_eval(svc.agent, cases, settings, progress=show)
        except AgenticRAGError as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1
    print("--- Live Evaluation Results Summary ---")
    print(format_agent_report(report))
    saved = save_report(report, ROOT / "eval" / "reports", args.label or settings.ollama_model)
    print(f"\nSaved {saved.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main()))
