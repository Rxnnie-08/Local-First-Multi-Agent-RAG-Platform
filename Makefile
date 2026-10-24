.PHONY: serve eval eval-full eval-retrieval tools sandbox-image chat install test integration lint format typecheck smoke up down ingest-samples

install:
	pip install -r requirements-dev.txt
	pip install -e . --no-deps

test:
	pytest --cov=agentic_rag --cov-report=term-missing

integration:
	pytest -m integration

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

tools:
	python scripts/check_tools.py

sandbox-image:
	docker pull python:3.12-slim

chat:
	python scripts/chat.py

smoke:
	python scripts/smoke_test.py

up:
	docker compose up -d

down:
	docker compose down

ingest-samples:
	python scripts/ingest.py samples

eval-retrieval:
	python scripts/eval_retrieval.py

eval-full:
	python scripts/run_evals.py --set eval/cases.jsonl --skip-live

eval:
	python scripts/run_evals.py

serve:
	python scripts/serve.py
