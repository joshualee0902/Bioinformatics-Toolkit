"""Config-driven training entry point for baselines and PyTorch sequence models.

Run with:  python -m biotoolkit.train --config configs/lstm.yaml --seeds 0 1 2

Each (config, seed) pair becomes one row in the `experiments` table, with its
metrics in `metrics`, test/val predictions in artifacts/predictions/, and
model files in artifacts/models/<run_id>.(pt|joblib).

Baseline and neural configs must be run in separate commands (see
models/neural.py for why); the Makefile does this.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import yaml

from biotoolkit import db, evaluate, features

log = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/models")
BASELINE_MODELS = {"majority", "gc_lr", "kmer_lr", "gbt"}
NEURAL_MODELS = {"lstm", "cnn", "cnn_lstm"}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_baseline(cfg: dict, seed: int, X_train, y_train, X_val, y_val):
    from biotoolkit.models import baselines

    random.seed(seed)
    np.random.seed(seed)
    estimator = baselines.build(cfg["model"], cfg.get("params", {}), seed)
    return baselines.fit(estimator, cfg["model"], X_train, y_train, X_val, y_val)


def run_baseline(cfg: dict, seed: int, run_id: str, splits: dict, X: dict) -> dict:
    """Fit one baseline seed; save the model and its feature importances. Returns probabilities."""
    import joblib

    from biotoolkit.models import baselines

    y = {name: df["label"].to_numpy() for name, df in splits.items()}
    estimator = train_baseline(cfg, seed, X["train"], y["train"], X["val"], y["val"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODEL_DIR / f"{run_id}.joblib")
    names = [n for fs in cfg["features"] for n in features.feature_names(fs)]
    np.savez(MODEL_DIR / f"{run_id}_importance.npz", names=names,
             values=baselines.importances(estimator, cfg["model"]))
    return {name: baselines.predict_proba(estimator, X[name]) for name in ("val", "test")}


def run_neural(cfg: dict, seed: int, run_id: str, splits: dict, engine) -> dict:
    """Train one neural seed, logging a learning curve; save weights. Returns probabilities."""
    from biotoolkit.models import neural

    def log_epoch(epoch, stats):
        db.log_metrics(engine, run_id, "epoch", stats, epoch=epoch)

    model, _ = neural.train_neural(cfg, seed, splits["train"], splits["val"], log_epoch)
    neural.save_model(model, cfg, MODEL_DIR / f"{run_id}.pt")
    return {name: neural.predict_sequences(model, cfg, splits[name]["sequence"].tolist())
            for name in ("val", "test")}


def run_experiment(cfg: dict, seeds: list[int], engine) -> None:
    """Train one config for every seed and record everything in the database."""
    db.delete_experiment(engine, cfg["name"])
    splits = {name: db.load_sequences(engine, name) for name in ("train", "val", "test")}
    is_neural = cfg["model"] in NEURAL_MODELS
    if not is_neural:
        X = {name: features.load(cfg["features"], df["id"].to_numpy())
             for name, df in splits.items()}

    for seed in seeds:
        log.info("%s seed=%d", cfg["name"], seed)
        run_id = db.start_run(engine, cfg["name"], cfg["model"], cfg, seed)
        if is_neural:
            probs = run_neural(cfg, seed, run_id, splits, engine)
        else:
            probs = run_baseline(cfg, seed, run_id, splits, X)
        results = evaluate.record_evaluation(
            engine,
            run_id,
            {name: (splits[name]["id"].to_numpy(), splits[name]["label"].to_numpy(), probs[name])
             for name in ("val", "test")},
        )
        log.info("  val auroc %.4f  test auroc %.4f",
                 results["val"]["auroc"], results["test"]["auroc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config")
    parser.add_argument("--config", nargs="+", required=True, help="one or more YAML configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    configs = [load_config(path) for path in args.config]
    kinds = {cfg["model"] in NEURAL_MODELS for cfg in configs}
    if len(kinds) > 1:
        parser.error("run baseline and neural configs in separate commands (see models/neural.py)")
    engine = db.get_engine()
    for cfg in configs:
        run_experiment(cfg, args.seeds, engine)


if __name__ == "__main__":
    main()
