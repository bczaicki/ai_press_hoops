# Common tasks. `make help` lists them.
EXTRAS = --extra cv --extra vlm --extra decide --extra court

.PHONY: help setup web api dev serve share test test-live lint doctor samples smoke

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install Python + web dependencies
	uv sync $(EXTRAS) --group dev
	npm --prefix web install --no-audit --no-fund

web:  ## build the web UI into web/dist
	npm --prefix web run build

api:  ## run the API with autoreload on :8000
	uv run uvicorn hoops.server:create_app --factory --reload --port 8000

dev:  ## API + Vite dev server with hot reload (UI on :5173)
	@trap 'kill 0' INT TERM; \
	uv run uvicorn hoops.server:create_app --factory --reload --port 8000 & \
	npm --prefix web run dev & \
	wait

serve:  ## production-style local server on :8000 (builds the UI if needed)
	uv run hoops serve

share:  ## public link for a friend (needs HOOPS_APP_PASSWORD in .env)
	uv run hoops serve --share

test:  ## fast tests
	uv run pytest -q -m "not slow"

test-live:  ## tests that call paid APIs (a few cents)
	HOOPS_LIVE=1 uv run pytest -q -m "not slow"

lint:  ## ruff lint + format check
	uv run ruff check hoops tests scripts && uv run ruff format --check hoops tests scripts

doctor:  ## check ffmpeg, MPS, keys
	uv run hoops doctor

samples:  ## download CC BY sample clips into data/samples
	uv run python scripts/fetch_samples.py

smoke:  ## tiny live check of each configured AI provider
	uv run hoops smoke
