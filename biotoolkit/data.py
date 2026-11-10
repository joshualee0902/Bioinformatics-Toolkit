"""Data pipeline: download, validate, de-duplicate, split, and load into the database.

Run with:  python -m biotoolkit.data --config configs/data.yaml

Dataset: `demo_coding_vs_intergenomic_seqs` from the genomic-benchmarks package
(human coding sequences vs. intergenic sequences, 200 bp each). The files are
one-sequence-per-.txt (not FASTA), so we read them with the package's own
download helper instead of Biopython.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from biotoolkit import db
from biotoolkit.sequence import gc_fraction, reverse_complement

log = logging.getLogger(__name__)

CLASS_LABELS = {"coding_seqs": 1, "intergenomic_seqs": 0}


def download(dataset: str, version: int) -> Path:
    """Download the benchmark (cached in ~/.genomic_benchmarks) and return its folder."""
    from genomic_benchmarks.loc2seq import download_dataset

    return Path(download_dataset(dataset, version=version))


def read_raw(root: Path) -> pd.DataFrame:
    """Read every sequence file into a DataFrame with columns sequence, label, split, source."""
    rows = []
    for split in ("train", "test"):
        for class_name, label in CLASS_LABELS.items():
            folder = root / split / class_name
            files = sorted(folder.glob("*.txt"), key=lambda p: int(p.stem))
            if not files:
                raise FileNotFoundError(f"No sequence files found in {folder}")
            for path in files:
                rows.append(
                    {
                        "sequence": path.read_text().strip(),
                        "label": label,
                        "split": split,
                        "source": f"{split}/{class_name}/{path.name}",
                    }
                )
    return pd.DataFrame(rows)


class StepLog:
    """Collects (step, split, count) rows and logs them as they happen."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, step: str, df: pd.DataFrame) -> None:
        for split, count in df["split"].value_counts().sort_index().items():
            self.rows.append({"step": step, "split": split, "count": int(count)})
        log.info("%-34s %s", step, df["split"].value_counts().sort_index().to_dict())

    def record_count(self, step: str, split: str, count: int) -> None:
        self.rows.append({"step": step, "split": split, "count": int(count)})
        log.info("%-34s %s: %d", step, split, count)


def clean(df: pd.DataFrame, cfg: dict, steps: StepLog) -> pd.DataFrame:
    """Uppercase, drop invalid characters, apply the N rule and the length range.

    N rule: a sequence is kept only if its fraction of N bases is at most
    cfg["max_n_fraction"]. Kept N bases become all-zero vectors in one-hot
    encoding and are skipped by k-mer counting.
    """
    df = df.copy()
    df["sequence"] = df["sequence"].str.upper()
    steps.record("raw", df)

    valid_chars = df["sequence"].str.fullmatch(r"[ACGTN]+")
    df = df[valid_chars]
    steps.record("only_ACGTN_characters", df)

    n_fraction = df["sequence"].str.count("N") / df["sequence"].str.len()
    df = df[n_fraction <= cfg["max_n_fraction"]]
    steps.record("n_fraction_within_limit", df)

    length = df["sequence"].str.len()
    df = df[(length >= cfg["min_length"]) & (length <= cfg["max_length"])]
    steps.record("length_in_range", df)
    return df


def canonical(seq: str) -> str:
    """A sequence and its reverse complement share the same canonical form."""
    return min(seq, reverse_complement(seq))


def deduplicate(df: pd.DataFrame, steps: StepLog) -> pd.DataFrame:
    """Remove exact and reverse-complement duplicates.

    The official test set is never changed. Instead, any training sequence whose
    canonical form also appears in test is removed (cross-split leakage). Then
    duplicates inside train are collapsed to one copy, and canonical forms that
    appear with both labels are dropped entirely because they are ambiguous.
    Validation is carved from train afterwards, so train/val cannot overlap.
    """
    df = df.copy()
    df["canonical"] = df["sequence"].map(canonical)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    test_dups = int(test["canonical"].duplicated().sum())
    steps.record_count("dup_within_test_kept_untouched", "test", test_dups)

    leaked = train["canonical"].isin(set(test["canonical"]))
    steps.record_count("removed_train_overlapping_test", "train", int(leaked.sum()))
    train = train[~leaked]

    labels_per_key = train.groupby("canonical")["label"].nunique()
    conflicting = train["canonical"].isin(labels_per_key[labels_per_key > 1].index)
    steps.record_count("removed_train_conflicting_labels", "train", int(conflicting.sum()))
    train = train[~conflicting]

    within = train["canonical"].duplicated(keep="first")
    steps.record_count("removed_train_duplicates", "train", int(within.sum()))
    train = train[~within]

    out = pd.concat([train, test]).drop(columns="canonical")
    steps.record("after_dedup", out)
    return out


def split_validation(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Carve a stratified validation set from train with a fixed seed."""
    train = df[df["split"] == "train"]
    _, val_idx = train_test_split(
        train.index, test_size=val_fraction, random_state=seed, stratify=train["label"]
    )
    df = df.copy()
    df.loc[val_idx, "split"] = "val"
    return df


def build(cfg: dict, db_url: str | None = None) -> pd.DataFrame:
    """Run the whole pipeline and write the result into a freshly reset database."""
    steps = StepLog()
    root = download(cfg["dataset"], cfg["version"])
    df = read_raw(root)
    df = clean(df, cfg, steps)
    df = deduplicate(df, steps)
    df = split_validation(df, cfg["val_fraction"], cfg["seed"])
    steps.record("final", df)

    # Stable order: test, train, val, each in original file order.
    df = df.sort_values(["split", "label", "source"], kind="stable").reset_index(drop=True)
    df["length"] = df["sequence"].str.len()
    df["gc_content"] = df["sequence"].map(gc_fraction)

    engine = db.get_engine(db_url)
    db.reset_database(engine)
    db.bulk_insert(
        engine,
        db.Sequence,
        df[["sequence", "label", "split", "length", "gc_content", "source"]].to_dict("records"),
    )
    db.bulk_insert(engine, db.PipelineLog, steps.rows)
    log.info("Wrote %d sequences to %s", len(df), engine.url.render_as_string(hide_password=True))
    return df


def export_sample(path: Path, n: int = 500, seed: int = 0, db_url: str | None = None) -> None:
    """Write a small, class-balanced sample of real training sequences for smoke tests."""
    train = db.load_sequences(db.get_engine(db_url), split="train")
    sample = train.groupby("label").sample(n=n // 2, random_state=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample[["sequence", "label"]].to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--export-sample", type=Path, help="write a 500-sequence test fixture")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.export_sample:
        export_sample(args.export_sample)
        return
    with open(args.config) as f:
        build(yaml.safe_load(f))


if __name__ == "__main__":
    main()
