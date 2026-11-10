"""Interpretability: gradient saliency, a period-3 (codon) test, and baseline importances.

Run with:  python -m biotoolkit.interpret

Question: coding DNA has a 3-base rhythm because it is read in codons. If a
model really learned codon structure, the saliency along a coding sequence
should repeat every 3 positions, which shows up as a peak at frequency 1/3 in
its Fourier spectrum. We measure that peak and compare it against a null where
the saliency positions are shuffled (which destroys any periodicity).
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from biotoolkit import db  # noqa: E402
from biotoolkit.evaluate import REPORT_DIR  # noqa: E402
from biotoolkit.models.neural import load_model, one_hot  # noqa: E402
from biotoolkit.sequence import STOP_CODONS  # noqa: E402
from biotoolkit.train import MODEL_DIR  # noqa: E402

log = logging.getLogger(__name__)

N_SEQUENCES = 2000
N_PERMUTATIONS = 200
ALPHA = 0.01


def saliency(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Input x gradient at the observed base: shape (n, length).

    Positive values push the prediction toward "coding".
    """
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].clone().requires_grad_(True)
        model(xb).sum().backward()
        out.append((xb.grad * xb).sum(dim=2).detach())
    return torch.cat(out).numpy()


def power_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean normalized power spectrum across rows. Returns (frequencies, power)."""
    centered = signal - signal.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power /= power[:, 1:].sum(axis=1, keepdims=True) + 1e-12
    return np.fft.rfftfreq(signal.shape[1]), power.mean(axis=0)


def period3_score(signal: np.ndarray) -> float:
    """Power at frequency 1/3 divided by the median power of all other non-zero frequencies."""
    freqs, power = power_spectrum(signal)
    p3 = np.argsort(np.abs(freqs - 1 / 3))[:1]
    others = np.setdiff1d(np.arange(1, len(freqs)), p3)
    return float(power[p3].max() / np.median(power[others]))


def permutation_test(signal: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray, float]:
    """Observed period-3 score, null scores from shuffled positions, and the p-value."""
    rng = np.random.default_rng(seed)
    observed = period3_score(signal)
    null = np.array([period3_score(rng.permuted(signal, axis=1)) for _ in range(N_PERMUTATIONS)])
    p_value = (1 + (null >= observed).sum()) / (1 + N_PERMUTATIONS)
    return observed, null, float(p_value)


def analyse_model(engine, experiment: str, test: pd.DataFrame) -> dict | None:
    run_id = db.best_run(engine, experiment)
    if run_id is None:
        return None
    model, cfg = load_model(MODEL_DIR / f"{run_id}.pt")
    length = cfg["data"]["max_length"]
    x = one_hot(test["sequence"].tolist(), length)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).numpy()
    sal = saliency(model, x)

    y = test["label"].to_numpy()
    groups = {
        "coding (true positives)": (y == 1) & (prob >= 0.5),
        "non-coding (true negatives)": (y == 0) & (prob < 0.5),
    }
    result = {"experiment": experiment, "run_id": run_id, "saliency": sal, "groups": {}}
    for name, mask in groups.items():
        observed, null, p = permutation_test(sal[mask])
        result["groups"][name] = {
            "n": int(mask.sum()), "score": observed, "null_mean": float(null.mean()),
            "null_95": float(np.quantile(null, 0.95)), "p": p,
            "spectrum": power_spectrum(sal[mask]),
        }
    # Reference: the raw input itself (is there a period-3 signal in the DNA?).
    gc = (x[:, :, 1] + x[:, :, 2]).numpy()
    result["input_gc_score"] = {name: period3_score(gc[mask]) for name, mask in groups.items()}
    return result


def plot_spectra(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4), squeeze=False)
    for ax, r in zip(axes[0], results, strict=True):
        for name, g in r["groups"].items():
            freqs, power = g["spectrum"]
            ax.plot(freqs[1:], power[1:], label=f"{name} (n={g['n']})")
        ax.axvline(1 / 3, color="k", ls=":", lw=1)
        ax.text(1 / 3, ax.get_ylim()[1] * 0.95, " period 3", fontsize=8, va="top")
        ax.set(xlabel="Frequency (cycles per base)", ylabel="Mean normalized power",
               title=f"Saliency spectrum: {r['experiment']}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "periodicity_spectrum.png", dpi=150)
    plt.close(fig)


def plot_examples(r: dict, test: pd.DataFrame, n: int = 3) -> None:
    colors = {"A": "tab:green", "C": "tab:blue", "G": "tab:orange", "T": "tab:red", "N": "grey"}
    coding = np.flatnonzero(test["label"].to_numpy() == 1)[:n]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    for ax, i in zip(np.atleast_1d(axes), coding, strict=True):
        seq = test["sequence"].iloc[i]
        values = r["saliency"][i][: len(seq)]
        ax.bar(range(len(values)), values, color=[colors[b] for b in seq[: len(values)]], width=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("saliency")
        ax.set_title(f"Coding test sequence id {test['id'].iloc[i]}", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("Position (bp); bar colour = base (A green, C blue, "
                                       "G orange, T red)")
    fig.suptitle(f"Input x gradient saliency ({r['experiment']}, best seed)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "saliency_examples.png", dpi=150)
    plt.close(fig)


def baseline_importances(engine, experiment: str, top: int = 15) -> pd.DataFrame | None:
    run_id = db.best_run(engine, experiment)
    path = MODEL_DIR / f"{run_id}_importance.npz" if run_id else None
    if path is None or not path.exists():
        return None
    data = np.load(path)
    df = pd.DataFrame({"feature": data["names"], "importance": data["values"]})
    df["abs"] = df["importance"].abs()
    return df.sort_values("abs", ascending=False).head(top).drop(columns="abs")


def describe_feature(name: str) -> str:
    """Format a feature name and flag it if it involves a stop codon."""
    if name.startswith("codon_"):  # e.g. codon_TAA_frame1
        has_stop = name.split("_")[1] in STOP_CODONS
    else:  # a k-mer such as "CTAAG", or a named toolkit feature
        has_stop = "stop" in name or any(c in name for c in STOP_CODONS)
    return f"`{name}`" + (" (involves a stop codon)" if has_stop else "")


def write_report(results: list[dict], importances: dict) -> None:
    lines = ["### Period-3 (codon) signal in saliency", ""]
    lines += ["| Model | Group | Sequences | Period-3 score | Shuffled null mean (95th pct) | "
              "p-value |", "|---|---|---|---|---|---|"]
    for r in results:
        for name, g in r["groups"].items():
            lines.append(f"| `{r['experiment']}` | {name} | {g['n']:,} | {g['score']:.2f} | "
                         f"{g['null_mean']:.2f} ({g['null_95']:.2f}) | {g['p']:.4f} |")
    lines += ["", "Period-3 score = power at frequency 1/3 divided by the median power at the "
              f"other frequencies (1 means no peak). The null shuffles positions within each "
              f"saliency map {N_PERMUTATIONS} times. Sequences: a seeded random sample of "
              f"{N_SEQUENCES:,} test sequences.", ""]

    for r in results:
        cod = r["groups"]["coding (true positives)"]
        non = r["groups"]["non-coding (true negatives)"]
        found = cod["p"] < ALPHA and cod["score"] > non["score"]
        verdict = "**A period-3 signal was found**" if found else "**No clear period-3 signal**"
        lines.append(
            f"- `{r['experiment']}`: {verdict} in saliency on correctly classified coding "
            f"sequences (score {cod['score']:.2f} vs. {non['score']:.2f} on non-coding, "
            f"p = {cod['p']:.4f}, threshold {ALPHA}). For reference, the GC indicator of the "
            f"input DNA itself scores {r['input_gc_score']['coding (true positives)']:.2f} on "
            f"coding and {r['input_gc_score']['non-coding (true negatives)']:.2f} on non-coding "
            "sequences."
        )

    for experiment, df in importances.items():
        if df is None:
            continue
        kind = "coefficient (positive = coding)" if experiment == "kmer_lr" else "total gain"
        lines += ["", f"### Top features: `{experiment}` ({kind})", "",
                  "| Rank | Feature | Importance |", "|---|---|---|"]
        lines += [f"| {i} | {describe_feature(f)} | {v:.4g} |"
                  for i, (f, v) in enumerate(zip(df["feature"], df["importance"], strict=True), 1)]
    (REPORT_DIR / "interpretation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Saliency and periodicity analysis")
    parser.add_argument("--models", nargs="+", default=["lstm", "cnn"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    engine = db.get_engine()
    test = db.load_sequences(engine, "test").sample(n=N_SEQUENCES, random_state=0)
    test = test.reset_index(drop=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results = [r for m in args.models if (r := analyse_model(engine, m, test)) is not None]
    for r in results:
        for name, g in r["groups"].items():
            log.info("%s %-28s score %.2f  p %.4f", r["experiment"], name, g["score"], g["p"])
    if results:
        plot_spectra(results)
        plot_examples(results[0], test)
    importances = {e: baseline_importances(engine, e) for e in ("kmer_lr", "gbt")}
    write_report(results, importances)
    log.info("Wrote %s", REPORT_DIR / "interpretation.md")


if __name__ == "__main__":
    main()
