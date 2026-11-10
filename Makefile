# Reproducible pipeline. `make all` runs everything from scratch.
# Uses SQLite by default; for MySQL run `make mysql` and export BIOTK_DB_URL (see README).
PYTHON ?= python
SEEDS ?= 0 1 2

.PHONY: all install data features baselines train evaluate report serve docker-build test lint mysql clean

all: data baselines train evaluate report

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m biotoolkit.data --config configs/data.yaml

features:
	$(PYTHON) -m biotoolkit.features

# Baselines and neural models run in separate processes: LightGBM and PyTorch
# ship different OpenMP runtimes that cannot share a process on macOS.
baselines: features
	$(PYTHON) -m biotoolkit.train --seeds $(SEEDS) --config configs/baselines/*.yaml

train:
	$(PYTHON) -m biotoolkit.train --seeds $(SEEDS) --config configs/lstm.yaml configs/cnn.yaml configs/cnn_lstm.yaml configs/ablations/*.yaml
	$(PYTHON) -m biotoolkit.cli export-best --experiment lstm --out artifacts/models/lstm_best.pt

evaluate:
	$(PYTHON) -m biotoolkit.evaluate

report:
	$(PYTHON) -m biotoolkit.interpret
	$(PYTHON) -m biotoolkit.evaluate --readme

serve:
	$(PYTHON) -m uvicorn biotoolkit.serve:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t biotoolkit .

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

mysql:
	docker compose up -d --wait mysql

clean:
	rm -rf data artifacts
