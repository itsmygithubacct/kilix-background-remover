UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help setup lock-check corpus-check test lint format-check typecheck build check clean

help:
	@printf '%s\n' \
		'kilix-background-remover' \
		'' \
		'  make setup         Sync the exact locked development environment' \
		'  make corpus-check  Verify the deterministic owned corpus' \
		'  make check         Run lock, corpus, test, lint, format and type gates'

setup:
	$(UV) sync --locked --all-groups

lock-check:
	$(UV) lock --check

corpus-check:
	$(UV) run --frozen pytest tests/test_corpus.py

test:
	$(UV) run --frozen pytest

lint:
	$(UV) run --frozen ruff check src tests tools

format-check:
	$(UV) run --frozen ruff format --check src tests tools

typecheck:
	$(UV) run --frozen mypy

build:
	$(UV) build --no-build-isolation

check:
	$(MAKE) lock-check
	$(MAKE) corpus-check test lint format-check typecheck build

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
