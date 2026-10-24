"""Run the HTTP API.

python scripts/serve.py                      # http://127.0.0.1:8000  (docs at /docs)
python scripts/serve.py --port 9000
API_KEY=change-me python scripts/serve.py --host 0.0.0.0   # reachable from other machines

Qdrant and Ollama must be running (docker compose up -d qdrant; ollama serve). The API runs on the
host, not in a container, because the code sandbox needs your local Docker.

Binding anything other than loopback without API_KEY is refused: the API can upload files and run
generated code, so it must not be open to a network.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from agentic_rag.api import create_app
from agentic_rag.config import get_settings
from agentic_rag.core.logging import configure_logging

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", help="bind address (default: API_HOST, 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default: API_PORT, 8000)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    if host not in LOOPBACK and settings.api_key is None:
        print(
            f"Refusing to listen on {host} without an API key. Set API_KEY in .env or the "
            "environment, or bind to 127.0.0.1."
        )
        return 1
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        log_level="warning" if settings.log_level == "INFO" else settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
