"""PyTorch side of training: one-hot encoding, the training loop, and model files.

Kept separate from the baselines on purpose: LightGBM and PyTorch ship
different OpenMP runtimes that crash when used in the same process on macOS,
so a process imports either this module or models/baselines.py, never both.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from biotoolkit.features import encode_int
from biotoolkit.models.cnn import CNNClassifier, CNNLSTMClassifier
from biotoolkit.models.lstm import BiLSTMClassifier

log = logging.getLogger(__name__)

NEURAL_MODELS = {"lstm": BiLSTMClassifier, "cnn": CNNClassifier, "cnn_lstm": CNNLSTMClassifier}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


# ---------- one-hot encoding ----------


def one_hot(seqs: list[str], length: int) -> torch.Tensor:
    """(n, length, 4) float tensor in A, C, G, T order. N and padding are all zeros.

    Sequences longer than `length` are truncated (first `length` bases kept).
    """
    codes = encode_int([s[:length].ljust(length, "N") for s in seqs])
    x = np.zeros((len(seqs), length, 4), dtype=np.float32)
    rows, cols = np.nonzero(codes >= 0)
    x[rows, cols, codes[rows, cols]] = 1.0
    return torch.from_numpy(x)


def reverse_complement_batch(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of one-hot DNA: reverse positions, and swap A<->T, C<->G.

    With channels in A, C, G, T order, complementing is just reversing the channels.
    """
    return x.flip(dims=[1, 2])


# ---------- neural models ----------


def pick_device(name: str = "auto") -> torch.device:
    """"auto" picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_neural(cfg: dict) -> nn.Module:
    return NEURAL_MODELS[cfg["model"]](**cfg.get("params", {}))


@torch.no_grad()
def predict_neural(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    probs = [
        torch.sigmoid(model(x[i : i + batch_size].to(device))).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(probs).numpy()


def train_neural(cfg: dict, seed: int, train_df, val_df, on_epoch=None):
    """Train with AdamW, gradient clipping, and early stopping on validation AUROC.

    Returns the model restored to its best epoch, and a list of per-epoch stats.
    """
    set_seed(seed)
    tcfg = cfg["train"]
    length = cfg.get("data", {}).get("max_length", 200)
    x_train = one_hot(train_df["sequence"].tolist(), length)
    y_train = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32)
    x_val = one_hot(val_df["sequence"].tolist(), length)
    y_val = val_df["label"].to_numpy()

    device = pick_device(tcfg.get("device", "auto"))
    model = build_neural(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    best_auroc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        for i in range(0, len(order), tcfg["batch_size"]):
            idx = order[i : i + tcfg["batch_size"]]
            xb, yb = x_train[idx], y_train[idx]
            if tcfg.get("rc_augment", False):
                flip = torch.rand(len(xb), generator=generator) < 0.5
                xb = torch.where(flip[:, None, None], reverse_complement_batch(xb), xb)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(idx)

        val_prob = predict_neural(model, x_val)
        stats = {
            "train_loss": total_loss / len(x_train),
            "val_loss": log_loss(y_val, val_prob),
            "val_auroc": roc_auc_score(y_val, val_prob),
        }
        history.append(stats)
        if on_epoch:
            on_epoch(epoch, stats)
        log.info("  epoch %2d  loss %.4f  val_auroc %.4f  (%.0fs)",
                 epoch, stats["train_loss"], stats["val_auroc"], time.time() - start)

        if stats["val_auroc"] > best_auroc:
            best_auroc, bad_epochs = stats["val_auroc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model.cpu(), history


def save_model(model: nn.Module, cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_neural(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["config"]


def predict_sequences(model: nn.Module, cfg: dict, seqs: list[str]) -> np.ndarray:
    """Probability of "coding" for raw sequences (already validated, uppercase ACGTN).

    Sequences longer than the model's training length are truncated; shorter
    ones are padded with N (all-zero one-hot rows).
    """
    return predict_neural(model, one_hot(seqs, cfg["data"]["max_length"]))

