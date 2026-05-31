.PHONY: install dev lint test run clean docker

install:
	uv sync

dev:
	uv sync --dev

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

run:
	uv run python -m src.main

docker:
	docker build -t flyclaw .

clean:
	rm -rf build/ dist/ *.egg-info data/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
