import pandas as pd

from biotoolkit import db
from biotoolkit.data import StepLog, canonical, clean, deduplicate, split_validation
from biotoolkit.sequence import reverse_complement


def _df(rows):
    return pd.DataFrame(rows, columns=["sequence", "label", "split", "source"])


def test_canonical_matches_reverse_complement():
    seq = "AACGT"
    assert canonical(seq) == canonical(reverse_complement(seq))


def test_clean_applies_rules():
    df = _df(
        [
            ("acgtacgtac", 1, "train", "a"),  # lowercase -> kept
            ("ACGTXCGTAC", 1, "train", "b"),  # invalid character
            ("NNNNNCGTAC", 1, "train", "c"),  # too many N
            ("ACGT", 1, "train", "d"),  # too short
        ]
    )
    cfg = {"min_length": 8, "max_length": 12, "max_n_fraction": 0.1}
    out = clean(df, cfg, StepLog())
    assert out["sequence"].tolist() == ["ACGTACGTAC"]


def test_deduplicate_removes_leakage_and_keeps_test_untouched():
    rc = reverse_complement("AAAACCCC")
    df = _df(
        [
            ("AAAACCCC", 1, "test", "t1"),
            ("AAAACCCC", 1, "test", "t2"),  # duplicate inside test: kept
            (rc, 1, "train", "r1"),  # reverse complement of a test sequence: leaked
            ("GGGGTTTA", 0, "train", "r2"),
            ("GGGGTTTA", 0, "train", "r3"),  # exact duplicate inside train
            ("ACACACAC", 0, "train", "r4"),
            ("ACACACAC", 1, "train", "r5"),  # same sequence, both labels: ambiguous
        ]
    )
    steps = StepLog()
    out = deduplicate(df, steps)
    assert sorted(out[out.split == "test"]["source"]) == ["t1", "t2"]
    assert out[out.split == "train"]["source"].tolist() == ["r2"]
    counts = {r["step"]: r["count"] for r in steps.rows if r["split"] in ("train", "test")}
    assert counts["removed_train_overlapping_test"] == 1
    assert counts["removed_train_duplicates"] == 1
    assert counts["removed_train_conflicting_labels"] == 2


def test_split_validation_is_stratified_and_seeded():
    df = _df([(f"S{i}", i % 2, "train", str(i)) for i in range(100)])
    a = split_validation(df, 0.2, seed=1)
    b = split_validation(df, 0.2, seed=1)
    assert (a["split"] == b["split"]).all()
    val = a[a.split == "val"]
    assert len(val) == 20 and val["label"].sum() == 10


def test_database_round_trip(tmp_path):
    engine = db.get_engine(f"sqlite:///{tmp_path}/t.db")
    db.bulk_insert(
        engine,
        db.Sequence,
        [{"sequence": "ACGT", "label": 1, "split": "train", "length": 4, "gc_content": 0.5,
          "source": "x"}],
    )
    run_id = db.start_run(engine, "exp", "model", {"a": 1}, seed=0)
    db.log_metrics(engine, run_id, "test", {"auroc": 0.9})
    assert db.load_sequences(engine, "train")["sequence"].tolist() == ["ACGT"]
    db.delete_experiment(engine, "exp")
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM metrics").scalar() == 0
