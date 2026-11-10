"""Baseline smoke-test helper. Must not import torch (see biotoolkit/models/neural.py)."""

from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from biotoolkit import features
from biotoolkit.models import baselines
from biotoolkit.train import train_baseline

SAMPLE = Path(__file__).parent / "data" / "sample_500.tsv"


def load_sample():
    """400 train / 100 validation sequences; the file is sorted by label, so shuffle it."""
    df = pd.read_csv(SAMPLE, sep="\t").sample(frac=1, random_state=0)
    return df.iloc[:400], df.iloc[400:]


def train_baseline_on_sample(model: str) -> None:
    train, val = load_sample()
    sets = {"majority": ["gc"], "gc_lr": ["gc"], "kmer_lr": ["kmer3", "kmer4"],
            "gbt": ["kmer3", "toolkit"]}[model]

    def X(d):
        return sp.hstack([sp.csr_matrix(features.compute(s, d["sequence"].tolist()))
                          for s in sets], format="csr")

    cfg = {"model": model, "params": {"n_estimators": 20} if model == "gbt" else {}}
    est = train_baseline(cfg, 0, X(train), train["label"], X(val), val["label"])
    prob = baselines.predict_proba(est, X(val))
    assert prob.shape == (len(val),) and ((prob >= 0) & (prob <= 1)).all()
    assert len(baselines.importances(est, model)) == X(val).shape[1]


