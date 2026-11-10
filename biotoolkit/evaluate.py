"""Metrics, bootstrap confidence intervals, results table, and plots.

Run with:  python -m biotoolkit.evaluate
Everything here reads from the database (and saved prediction files), so the
results table and figures are always regenerated from recorded runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sqlalchemy import text  # noqa: E402

log = logging.getLogger(__name__)

PREDICTION_DIR = Path("artifacts/predictions")
REPORT_DIR = Path("reports")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_auroc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """95% percentile bootstrap confidence interval for AUROC (resampling test sequences)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # a resample with only one class has no AUROC
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def save_predictions(run_id: str, split: str, ids, y_true, y_prob) -> None:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PREDICTION_DIR / f"{run_id}_{split}.npz", ids=ids, y_true=y_true, y_prob=y_prob)


def load_predictions(run_id: str, split: str):
    data = np.load(PREDICTION_DIR / f"{run_id}_{split}.npz")
    return data["ids"], data["y_true"], data["y_prob"]


def record_evaluation(engine, run_id: str, split_data: dict, n_boot: int = 1000) -> dict:
    """Compute and store metrics for each split; add a bootstrap CI on test AUROC.

    split_data maps split name -> (ids, y_true, y_prob).
    """
    from biotoolkit import db

    results = {}
    for split, (ids, y_true, y_prob) in split_data.items():
        metrics = compute_metrics(y_true, y_prob)
        if split == "test":
            low, high = bootstrap_auroc_ci(y_true, y_prob, n_boot=n_boot)
            metrics.update({"auroc_ci_low": low, "auroc_ci_high": high})
        db.log_metrics(engine, run_id, split, metrics)
        save_predictions(run_id, split, ids, y_true, y_prob)
        results[split] = metrics
    return results


# ---------- results table (generated from SQL) ----------

MAIN_EXPERIMENTS = ["majority", "gc_lr", "kmer_lr", "gbt", "lstm", "cnn", "cnn_lstm"]

RESULTS_SQL = """
SELECT e.experiment, e.model, m.split, m.metric,
       COUNT(*) AS n, AVG(m.value) AS mean, AVG(m.value * m.value) AS mean_sq
FROM metrics m
JOIN experiments e ON e.run_id = m.run_id
WHERE m.epoch IS NULL
GROUP BY e.experiment, e.model, m.split, m.metric
"""

DATA_SQL = "SELECT step, split, count FROM pipeline_log ORDER BY id"


def results_frame(engine) -> pd.DataFrame:
    """One row per experiment with mean and sample std of every metric across seeds."""
    with engine.connect() as conn:
        df = pd.read_sql_query(text(RESULTS_SQL), conn)
    # sample std from the SQL aggregates: var = n/(n-1) * (E[x^2] - E[x]^2)
    var = (df["mean_sq"] - df["mean"] ** 2).clip(lower=0) * df["n"] / (df["n"] - 1).clip(lower=1)
    df["std"] = np.sqrt(var)
    df["key"] = df["split"] + "_" + df["metric"]
    wide = df.pivot_table(index=["experiment", "model"], columns="key", values=["mean", "std"])
    wide.columns = [f"{key}_{stat}" for stat, key in wide.columns]
    seeds = df.groupby("experiment")["n"].max().rename("seeds")
    return wide.reset_index().join(seeds, on="experiment")


def _fmt(row, key: str) -> str:
    return f"{row[f'{key}_mean']:.4f} ± {row[f'{key}_std']:.4f}"


def results_markdown(results: pd.DataFrame, experiments: list[str]) -> str:
    lines = [
        "| Experiment | Model | Seeds | Val AUROC | Test AUROC | Test AUROC 95% CI | "
        "Test AUPRC | Test accuracy | Test F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_name = results.set_index("experiment")
    for name in experiments:
        if name not in by_name.index:
            continue
        r = by_name.loc[name]
        ci = f"[{r['test_auroc_ci_low_mean']:.4f}, {r['test_auroc_ci_high_mean']:.4f}]"
        lines.append(
            f"| `{name}` | {r['model']} | {int(r['seeds'])} | {_fmt(r, 'val_auroc')} | "
            f"**{_fmt(r, 'test_auroc')}** | {ci} | {_fmt(r, 'test_auprc')} | "
            f"{_fmt(r, 'test_accuracy')} | {_fmt(r, 'test_f1')} |"
        )
    return "\n".join(lines)


def data_markdown(engine) -> str:
    with engine.connect() as conn:
        df = pd.read_sql_query(text(DATA_SQL), conn)
    lines = ["| Step | Split | Sequences |", "|---|---|---|"]
    lines += [f"| {r.step} | {r.split} | {r.count:,} |" for r in df.itertuples()]
    return "\n".join(lines)


def write_tables(engine) -> pd.DataFrame:
    results = results_frame(engine)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "results.csv", index=False)
    ablations = sorted(set(results["experiment"]) - set(MAIN_EXPERIMENTS))
    text_out = (
        "## Main results (test set = official benchmark test split)\n\n"
        + results_markdown(results, MAIN_EXPERIMENTS)
        + "\n\n## Ablations\n\n"
        + results_markdown(results, ablations)
        + "\n\nMean ± sample std across seeds. The 95% CI is the percentile bootstrap "
        "(1,000 resamples of the test set) on test AUROC, averaged across seeds.\n"
    )
    (REPORT_DIR / "results.md").write_text(text_out)
    (REPORT_DIR / "data_summary.md").write_text(data_markdown(engine) + "\n")
    return results


# ---------- plots ----------


def runs(engine, experiment: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(
            text("SELECT run_id, seed FROM experiments WHERE experiment = :e ORDER BY seed"),
            conn,
            params={"e": experiment},
        )


def plot_roc(engine, experiments: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in experiments:
        r = runs(engine, name)
        if r.empty:
            continue
        _, y, p = load_predictions(r.run_id.iloc[0], "test")  # first seed
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Test ROC curves (seed 0)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_learning_curves(engine, experiments: list[str]) -> None:
    query = text("""
        SELECT e.experiment, m.epoch, m.metric, AVG(m.value) AS value
        FROM metrics m JOIN experiments e ON e.run_id = m.run_id
        WHERE m.epoch IS NOT NULL
        GROUP BY e.experiment, m.epoch, m.metric
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in experiments:
        d = df[df.experiment == name]
        for ax, metric in zip(axes, ["train_loss", "val_auroc"], strict=True):
            m = d[d.metric == metric].sort_values("epoch")
            ax.plot(m.epoch, m.value, marker="o", ms=3, label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Training loss (mean over seeds)")
    axes[1].set(xlabel="Epoch", ylabel="Validation AUROC", title="Validation AUROC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_ablation(results: pd.DataFrame, series: dict, xlabel: str, title: str, out: str) -> None:
    """series maps a label -> list of (x value, experiment name)."""
    by_name = results.set_index("experiment")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, points in series.items():
        points = [(x, e) for x, e in points if e in by_name.index]
        if not points:
            continue
        xs = [x for x, _ in points]
        mean = [by_name.loc[e, "test_auroc_mean"] for _, e in points]
        std = [by_name.loc[e, "test_auroc_std"] for _, e in points]
        ax.errorbar(xs, mean, yerr=std, marker="o", capsize=3, label=label)
    ax.set(xlabel=xlabel, ylabel="Test AUROC (mean ± std)", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / out, dpi=150)
    plt.close(fig)


def write_plots(engine, results: pd.DataFrame) -> None:
    plot_roc(engine, ["gc_lr", "kmer_lr", "gbt", "lstm", "cnn", "cnn_lstm"])
    plot_learning_curves(engine, ["lstm", "cnn", "cnn_lstm"])
    plot_ablation(
        results,
        {m: [(50, f"{m}_len50"), (100, f"{m}_len100"), (200, m)] for m in ("lstm", "cnn")},
        "Sequence length used (bp)", "Performance vs. sequence length", "length_ablation.png",
    )
    plot_ablation(
        results,
        {"k-mer logistic regression": [(k, f"kmer_lr_k{k}") for k in (3, 4, 5, 6)]},
        "k", "k-mer size ablation", "kmer_ablation.png",
    )


# ---------- README ----------


def inject(readme: str, marker: str, content: str) -> str:
    """Replace the text between <!-- MARKER:START --> and <!-- MARKER:END -->."""
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if start not in readme:
        return readme
    head, rest = readme.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{content.strip()}\n{end}{tail}"


def update_readme(path: Path = Path("README.md")) -> None:
    """Copy the generated report files into the README's marked sections."""
    readme = path.read_text()
    for marker, file in [("DATA", "data_summary.md"), ("RESULTS", "results.md"),
                         ("INTERPRETATION", "interpretation.md")]:
        if (REPORT_DIR / file).exists():
            readme = inject(readme, marker, (REPORT_DIR / file).read_text())
    path.write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results table and plots from the DB")
    parser.add_argument("--readme", action="store_true", help="only refresh README sections")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.readme:
        update_readme()
        return

    from biotoolkit import db

    engine = db.get_engine()
    results = write_tables(engine)
    write_plots(engine, results)
    log.info("Wrote tables and plots to %s/", REPORT_DIR)


if __name__ == "__main__":
    main()
