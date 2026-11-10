"""Feature engineering: k-mer frequencies plus features built from the original toolkit.

Run with:  python -m biotoolkit.features

Feature matrices are saved under data/features/ (row order = sequence id order)
and the `features` table records which file and row belongs to each sequence.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biotoolkit import db
from biotoolkit.sequence import (
    CODONS,
    STOP,
    all_proteins_from_orfs,
    gc_content_subsec,
    gc_fraction,
    gen_reading_frames,
)

log = logging.getLogger(__name__)

FEATURE_DIR = Path("data/features")
KMER_SIZES = (3, 4, 5, 6)
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
GC_WINDOW = 20


# ---------- k-mers ----------


def kmer_names(k: int) -> list[str]:
    """All 4**k k-mers in the same order as the k-mer matrix columns."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def encode_int(seqs: list[str]) -> np.ndarray:
    """Encode sequences as an (n, max_len) int array: A=0 C=1 G=2 T=3, N/padding=-1."""
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), -1, dtype=np.int8)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = [BASE_INDEX.get(b, -1) for b in seq]
    return out


def kmer_matrix(seqs: list[str], k: int) -> sp.csr_matrix:
    """Normalized k-mer counts, shape (n, 4**k). Windows containing N are skipped.

    Each k-mer is turned into a base-4 number, e.g. "ACG" -> 0*16 + 1*4 + 2 = 6.
    """
    codes = encode_int(seqs)
    n, length = codes.shape
    n_windows = length - k + 1
    if n_windows < 1:
        return sp.csr_matrix((n, 4**k), dtype=np.float32)

    index = np.zeros((n, n_windows), dtype=np.int64)
    valid = np.ones((n, n_windows), dtype=bool)
    for offset in range(k):
        window = codes[:, offset : offset + n_windows]
        index = index * 4 + np.maximum(window, 0)
        valid &= window >= 0

    rows = np.repeat(np.arange(n), n_windows)[valid.ravel()]
    cols = index.ravel()[valid.ravel()]
    counts = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, 4**k)
    ).tocsr()
    totals = np.maximum(valid.sum(axis=1), 1).astype(np.float32)
    return sp.diags(1.0 / totals) @ counts


# ---------- toolkit features ----------


def toolkit_feature_names() -> list[str]:
    names = ["gc_fraction", "gc_window_variance", "longest_orf_aa", "longest_stop_free_aa"]
    names += [f"stop_density_frame{f}" for f in range(6)]
    names += [f"codon_{codon}_frame{f}" for f in range(3) for codon in CODONS]
    return names


def _longest_stop_free_run(aa_seq: list[str]) -> int:
    best = current = 0
    for aa in aa_seq:
        current = 0 if aa == STOP else current + 1
        best = max(best, current)
    return best


def toolkit_features(seq: str) -> np.ndarray:
    """Hand-engineered features computed with the original toolkit functions.

    - GC content and the variance of GC across 20 bp windows
    - longest ORF (ATG ... stop protein, in amino acids) across all six frames
    - longest stretch without a stop codon across all six frames
    - stop-codon density in each of the six frames
    - codon usage (64 frequencies) in each of the three forward frames
    """
    windows = np.array(gc_content_subsec(seq, k=GC_WINDOW), dtype=np.float32) / 100
    proteins = all_proteins_from_orfs(seq, ordered=True)
    frames = gen_reading_frames(seq)

    stop_density = [rf.count(STOP) / max(len(rf), 1) for rf in frames]
    codon_usage = []
    for frame in range(3):
        codons = [seq[i : i + 3] for i in range(frame, len(seq) - 2, 3)]
        counts = Counter(codons)
        total = max(len(codons), 1)
        codon_usage += [counts.get(c, 0) / total for c in CODONS]

    return np.array(
        [
            gc_fraction(seq),
            windows.var() if len(windows) else 0.0,
            len(proteins[0]) if proteins else 0,
            max(_longest_stop_free_run(rf) for rf in frames),
            *stop_density,
            *codon_usage,
        ],
        dtype=np.float32,
    )


def toolkit_matrix(seqs: list[str], processes: int | None = None) -> np.ndarray:
    if len(seqs) < 2000:
        return np.stack([toolkit_features(s) for s in seqs])
    with Pool(processes) as pool:
        return np.stack(pool.map(toolkit_features, seqs, chunksize=500))


# ---------- feature sets on disk ----------


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "gc":
        return ["gc_fraction"]
    if feature_set.startswith("kmer"):
        return kmer_names(int(feature_set.removeprefix("kmer")))
    if feature_set == "toolkit":
        return toolkit_feature_names()
    raise ValueError(f"Unknown feature set: {feature_set}")


def compute(feature_set: str, seqs: list[str]):
    """Compute one feature set for a list of sequences (sparse for k-mers, dense otherwise)."""
    if feature_set == "gc":
        return np.array([[gc_fraction(s)] for s in seqs], dtype=np.float32)
    if feature_set.startswith("kmer"):
        return kmer_matrix(seqs, int(feature_set.removeprefix("kmer")))
    if feature_set == "toolkit":
        return toolkit_matrix(seqs)
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_all(engine, feature_sets: list[str]) -> None:
    """Compute every feature set for every sequence, save to disk, and index in the DB."""
    from sqlalchemy import delete

    seqs = db.load_sequences(engine)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in feature_sets:
        matrix = compute(name, seqs["sequence"].tolist())
        path = FEATURE_DIR / f"{name}.npz"
        sp.save_npz(path, sp.csr_matrix(matrix))
        np.save(FEATURE_DIR / f"{name}_ids.npy", seqs["id"].to_numpy())

        with engine.begin() as conn:
            conn.execute(delete(db.Feature).where(db.Feature.feature_set == name))
        rows = [
            {"sequence_id": int(sid), "feature_set": name, "path": str(path), "row_index": i}
            for i, sid in enumerate(seqs["id"])
        ]
        db.bulk_insert(engine, db.Feature, rows, chunk_size=20000)
        log.info("feature set %-8s shape=%s -> %s", name, matrix.shape, path)


def load(feature_sets: list[str], ids: np.ndarray) -> sp.csr_matrix:
    """Load and horizontally stack feature sets for the given sequence ids (in that order)."""
    blocks = []
    for name in feature_sets:
        # "gc" is the first column of the toolkit feature set.
        file_name = "toolkit" if name == "gc" else name
        matrix = sp.load_npz(FEATURE_DIR / f"{file_name}.npz")
        if name == "gc":
            matrix = matrix[:, :1]
        all_ids = np.load(FEATURE_DIR / f"{file_name}_ids.npy")
        position = pd.Series(np.arange(len(all_ids)), index=all_ids)
        blocks.append(matrix[position.loc[ids].to_numpy()])
    return sp.hstack(blocks, format="csr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument("--sets", nargs="+", default=[f"kmer{k}" for k in KMER_SIZES] + ["toolkit"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_all(db.get_engine(), args.sets)


if __name__ == "__main__":
    main()
