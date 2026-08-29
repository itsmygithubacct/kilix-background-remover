UV ?= uv
UV_RELEASE_VERSION := uv 0.12.5 (x86_64-unknown-linux-gnu)
UV_RELEASE_SHA256 := b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46

.DEFAULT_GOAL := help

.PHONY: help toolchain-check setup lock-check corpus-check test lint format-check typecheck build check clean

help:
	@printf '%s\n' \
		'kilix-background-remover' \
		'' \
		'  make toolchain-check Verify the release-pinned uv executable' \
		'  make setup         Sync the exact locked development environment' \
		'  make corpus-check  Verify the deterministic owned corpus' \
		'  make check         Run lock, corpus, test, lint, format and type gates'

toolchain-check:
	@uv_path="$$(command -v "$(UV)")"; \
	if [ -z "$$uv_path" ]; then \
		printf '%s\n' 'toolchain refusal: configured uv executable was not found' >&2; \
		exit 1; \
	fi; \
	actual_version="$$("$$uv_path" --version)"; \
	if [ "$$actual_version" != "$(UV_RELEASE_VERSION)" ]; then \
		printf 'toolchain refusal: uv version mismatch: expected %s; got %s\n' \
			'$(UV_RELEASE_VERSION)' "$$actual_version" >&2; \
		exit 1; \
	fi; \
	actual_sha256="$$(sha256sum "$$uv_path" | cut -d ' ' -f 1)"; \
	if [ "$$actual_sha256" != "$(UV_RELEASE_SHA256)" ]; then \
		printf 'toolchain refusal: uv digest mismatch: expected %s; got %s\n' \
			'$(UV_RELEASE_SHA256)' "$$actual_sha256" >&2; \
		exit 1; \
	fi; \
	printf 'release uv: PASS version=%s sha256=%s\n' "$$actual_version" "$$actual_sha256"

setup: toolchain-check
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

check: toolchain-check
	$(MAKE) lock-check
	$(MAKE) corpus-check test lint format-check typecheck build

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
