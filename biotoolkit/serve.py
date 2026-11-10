"""FastAPI inference service.

Run with:  uvicorn biotoolkit.serve:app   (or `make serve`)
The model file comes from the MODEL_PATH environment variable
(default artifacts/models/lstm_best.pt, written by `make train`).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from biotoolkit.models.neural import load_model, predict_sequences
from biotoolkit.sequence import validate_seq

DEFAULT_MODEL_PATH = "artifacts/models/lstm_best.pt"
MAX_SEQUENCES = 1000
MAX_LENGTH = 10_000


class PredictRequest(BaseModel):
    sequences: list[str] = Field(min_length=1, max_length=MAX_SEQUENCES)


class PredictResponse(BaseModel):
    model: str
    probabilities: list[float]  # probability that each sequence is protein-coding


def validate_batch(sequences: list[str]) -> list[str]:
    """Uppercase and check every sequence; raise a 422 naming the first bad one."""
    clean = []
    for i, seq in enumerate(sequences):
        if len(seq) > MAX_LENGTH:
            raise HTTPException(422, f"sequences[{i}] is longer than {MAX_LENGTH} bases")
        valid = validate_seq(seq, allow_n=True)
        if valid is False:
            raise HTTPException(
                422, f"sequences[{i}] must be a non-empty string of A, C, G, T or N"
            )
        clean.append(valid)
    return clean


def create_app(model_path: str | None = None) -> FastAPI:
    model_path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["model"], state["config"] = load_model(model_path)
        yield

    app = FastAPI(title="biotoolkit coding-sequence classifier", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": os.path.basename(model_path)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        seqs = validate_batch(request.sequences)
        probs = predict_sequences(state["model"], state["config"], seqs)
        return PredictResponse(model=os.path.basename(model_path),
                               probabilities=[round(float(p), 6) for p in probs])

    return app


app = create_app()
