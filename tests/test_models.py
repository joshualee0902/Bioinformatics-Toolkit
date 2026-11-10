"""Smoke tests: every model trains for one epoch on 500 real sequences."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from biotoolkit import features
from biotoolkit.evaluate import bootstrap_auroc_ci, compute_metrics
from biotoolkit.models.neural import (
    load_model,
    one_hot,
    predict_neural,
    reverse_complement_batch,
    save_model,
    train_neural,
)
from biotoolkit.sequence import reverse_complement
from biotoolkit.train import load_config
from tests.baseline_check import load_sample, train_baseline_on_sample


@pytest.fixture(scope="module")
def sample():
    return load_sample()


def test_one_hot_and_reverse_complement():
    x = one_hot(["ACGTN"], length=6)
    assert x.shape == (1, 6, 4)
    assert x[0, :4].argmax(dim=1).tolist() == [0, 1, 2, 3]
    assert x[0, 4:].sum() == 0  # N and padding are all zeros
    rc = reverse_complement_batch(one_hot(["AACGT"], 5))
    assert torch.equal(rc, one_hot([reverse_complement("AACGT")], 5))


def test_feature_shapes():
    seqs = ["ACGTACGTAC" * 20, "ATGAAATAG" * 22]
    for k in features.KMER_SIZES:
        matrix = features.kmer_matrix(seqs, k)
        assert matrix.shape == (2, 4**k)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-5)
    toolkit = features.toolkit_matrix(seqs)
    assert toolkit.shape == (2, len(features.toolkit_feature_names()))


def test_kmer_matrix_counts_and_skips_n():
    matrix = features.kmer_matrix(["AACNAA"], 2).toarray()[0]
    names = features.kmer_names(2)
    assert matrix[names.index("AA")] == pytest.approx(2 / 3)  # windows: AA, AC, AA
    assert matrix[names.index("AC")] == pytest.approx(1 / 3)


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1] * 50)
    p = np.array([0.1, 0.4, 0.35, 0.8] * 50)
    m = compute_metrics(y, p)
    assert m["auroc"] == pytest.approx(0.75)
    low, high = bootstrap_auroc_ci(y, p, n_boot=200)
    assert low <= m["auroc"] <= high


@pytest.mark.parametrize("model", ["majority", "gc_lr", "kmer_lr"])
def test_baselines_train(model):
    train_baseline_on_sample(model)


def test_gbt_trains_in_separate_process():
    # LightGBM and torch ship different OpenMP runtimes that crash in one process on macOS
    # (see models/neural.py). The pipeline runs them separately, so this test does too.
    code = "from tests.baseline_check import train_baseline_on_sample as f; f('gbt')"
    root = Path(__file__).parent.parent
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("config", ["configs/lstm.yaml", "configs/cnn.yaml",
                                    "configs/cnn_lstm.yaml", "configs/ablations/lstm_rc.yaml"])
def test_neural_models_train_one_epoch(sample, config, tmp_path):
    train, val = sample
    cfg = load_config(Path(__file__).parent.parent / config)
    cfg["train"].update(epochs=1, device="cpu")
    model, history = train_neural(cfg, 0, train, val)
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"])

    save_model(model, cfg, tmp_path / "m.pt")
    loaded, _ = load_model(tmp_path / "m.pt")
    x = one_hot(val["sequence"].tolist(), 200)
    np.testing.assert_allclose(predict_neural(model, x), predict_neural(loaded, x), rtol=1e-5)
