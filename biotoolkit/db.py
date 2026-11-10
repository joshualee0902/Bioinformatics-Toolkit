"""Database schema and helpers (SQLAlchemy).

Works with MySQL 8 (via docker-compose) or SQLite as a zero-setup fallback.
Set BIOTK_DB_URL to choose, for example:
    sqlite:///data/biotoolkit.db                                   (default)
    mysql+pymysql://biotk:biotk@127.0.0.1:3306/biotoolkit
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_DB_URL = "sqlite:///data/biotoolkit.db"


class Base(DeclarativeBase):
    pass


class Sequence(Base):
    __tablename__ = "sequences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sequence: Mapped[str] = mapped_column(String(512))
    label: Mapped[int] = mapped_column(Integer)  # 1 = coding, 0 = non-coding (intergenic)
    split: Mapped[str] = mapped_column(String(8), index=True)  # train / val / test
    length: Mapped[int] = mapped_column(Integer)
    gc_content: Mapped[float] = mapped_column(Float)  # fraction between 0 and 1
    source: Mapped[str] = mapped_column(String(128))  # original file in the benchmark


class Feature(Base):
    """Points a sequence at its row in a feature matrix stored on disk."""

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sequence_id: Mapped[int] = mapped_column(ForeignKey("sequences.id"), index=True)
    feature_set: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(String(256))
    row_index: Mapped[int] = mapped_column(Integer)


class Experiment(Base):
    __tablename__ = "experiments"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)  # config name
    model: Mapped[str] = mapped_column(String(32))
    config: Mapped[str] = mapped_column(Text)  # JSON
    seed: Mapped[int] = mapped_column(Integer)
    git_commit: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("experiments.run_id"), index=True)
    split: Mapped[str] = mapped_column(String(8))
    metric: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)  # set for learning curves


class PipelineLog(Base):
    """Row counts at each data-pipeline step, so the README can quote them from the DB."""

    __tablename__ = "pipeline_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    step: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(8))
    count: Mapped[int] = mapped_column(Integer)


def get_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("BIOTK_DB_URL", DEFAULT_DB_URL)
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def reset_database(engine: Engine) -> None:
    """Drop and recreate every table. Used by `make data` to rebuild from scratch."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def bulk_insert(engine: Engine, table, rows: list[dict], chunk_size: int = 5000) -> None:
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            conn.execute(insert(table), rows[i : i + chunk_size])


def load_sequences(engine: Engine, split: str | None = None) -> pd.DataFrame:
    """Read sequences (optionally one split) into a DataFrame, ordered by id."""
    query = select(
        Sequence.id, Sequence.sequence, Sequence.label, Sequence.split, Sequence.length
    ).order_by(Sequence.id)
    if split:
        query = query.where(Sequence.split == split)
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def start_run(engine: Engine, experiment: str, model: str, config: dict, seed: int) -> str:
    """Record a new experiment run and return its run_id."""
    run_id = str(uuid.uuid4())
    bulk_insert(
        engine,
        Experiment,
        [
            {
                "run_id": run_id,
                "experiment": experiment,
                "model": model,
                "config": json.dumps(config, sort_keys=True),
                "seed": seed,
                "git_commit": git_commit(),
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        ],
    )
    return run_id


def log_metrics(
    engine: Engine, run_id: str, split: str, metrics: dict[str, float], epoch: int | None = None
) -> None:
    rows = [
        {"run_id": run_id, "split": split, "metric": k, "value": float(v), "epoch": epoch}
        for k, v in metrics.items()
    ]
    bulk_insert(engine, Metric, rows)


def delete_experiment(engine: Engine, experiment: str) -> None:
    """Remove earlier runs of an experiment so re-running it does not double count."""
    with engine.begin() as conn:
        run_ids = select(Experiment.run_id).where(Experiment.experiment == experiment)
        conn.execute(delete(Metric).where(Metric.run_id.in_(run_ids)))
        conn.execute(delete(Experiment).where(Experiment.experiment == experiment))


def best_run(engine: Engine, experiment: str) -> str | None:
    """run_id of the experiment's seed with the highest validation AUROC."""
    query = (
        select(Experiment.run_id)
        .join(Metric, Metric.run_id == Experiment.run_id)
        .where(
            Experiment.experiment == experiment,
            Metric.split == "val",
            Metric.metric == "auroc",
            Metric.epoch.is_(None),
        )
        .order_by(Metric.value.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        return conn.execute(query).scalar()
