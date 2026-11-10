# Reproducible pipeline. `make all` runs everything from scratch.
# Uses SQLite by default; for MySQL run `make mysql` and export BIOTK_DB_URL (see README).
PYTHON ?= python
SEEDS ?= 0 1 2

.PHONY: all install data features baselines train evaluate report serve test lint mysql clean

all: data baselines train evaluate report

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m biotoolkit.data --config configs/data.yaml

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

mysql:
	docker compose up -d --wait mysql

clean:
	rm -rf data artifacts
