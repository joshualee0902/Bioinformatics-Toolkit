"""Command-line interface.

    biotoolkit predict --model artifacts/models/lstm_best.pt --fasta examples/sample.fasta
    biotoolkit export-best --experiment lstm --out artifacts/models/lstm_best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys

from Bio import SeqIO


def predict(args: argparse.Namespace) -> int:
    from biotoolkit.models.neural import load_model, predict_sequences
    from biotoolkit.sequence import validate_seq

    ids, seqs = [], []
    for record in SeqIO.parse(args.fasta, "fasta"):
        seq = validate_seq(str(record.seq), allow_n=True)
        if seq is False:
            print(f"error: record {record.id!r} contains characters other than A, C, G, T, N",
                  file=sys.stderr)
            return 2
        ids.append(record.id)
        seqs.append(seq)
    if not seqs:
        print(f"error: no FASTA records found in {args.fasta}", file=sys.stderr)
        return 2

    model, cfg = load_model(args.model)
    probs = predict_sequences(model, cfg, seqs)
    print("id\tlength\tprob_coding")
    for record_id, seq, p in zip(ids, seqs, probs, strict=True):
        print(f"{record_id}\t{len(seq)}\t{p:.6f}")
    return 0


def export_best(args: argparse.Namespace) -> int:
    """Copy the best seed (by validation AUROC) of an experiment to a stable path."""
    from biotoolkit import db
    from biotoolkit.train import MODEL_DIR

    run_id = db.best_run(db.get_engine(), args.experiment)
    if run_id is None:
        print(f"error: no runs found for experiment {args.experiment!r}", file=sys.stderr)
        return 1
    shutil.copy(MODEL_DIR / f"{run_id}.pt", args.out)
    print(f"{args.experiment} run {run_id} -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biotoolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="per-sequence coding probabilities for a FASTA file")
    p.add_argument("--model", required=True, help="path to a .pt model file")
    p.add_argument("--fasta", required=True, help="path to a FASTA file")
    p.set_defaults(func=predict)

    e = sub.add_parser("export-best", help="copy an experiment's best model to a fixed path")
    e.add_argument("--experiment", default="lstm")
    e.add_argument("--out", default="artifacts/models/lstm_best.pt")
    e.set_defaults(func=export_best)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
