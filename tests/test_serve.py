from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from biotoolkit import cli
from biotoolkit.models.neural import save_model, train_neural
from biotoolkit.serve import create_app
from biotoolkit.train import load_config

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    df = pd.read_csv(ROOT / "tests/data/sample_500.tsv", sep="\t").sample(frac=1, random_state=0)
    cfg = load_config(ROOT / "configs/lstm.yaml")
    cfg["train"].update(epochs=1, device="cpu")
    model, _ = train_neural(cfg, 0, df.iloc[:400], df.iloc[400:])
    path = tmp_path_factory.mktemp("model") / "m.pt"
    save_model(model, cfg, path)
    return path


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(str(model_path))) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probabilities(client):
    r = client.post("/predict", json={"sequences": ["ACGT" * 50, "atgnnnccc"]})
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    assert len(probs) == 2 and all(0 <= p <= 1 for p in probs)


@pytest.mark.parametrize("body", [{"sequences": ["ACGTX"]}, {"sequences": [""]},
                                  {"sequences": []}, {"wrong": 1}])
def test_predict_rejects_invalid_input(client, body):
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_cli_predict(model_path, tmp_path, capsys):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">a\nACGTACGT\n>b\nTTTTGGGG\n")
    assert cli.main(["predict", "--model", str(model_path), "--fasta", str(fasta)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "id\tlength\tprob_coding" and len(lines) == 3


def test_cli_rejects_invalid_fasta(model_path, tmp_path):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">a\nACGZ\n")
    assert cli.main(["predict", "--model", str(model_path), "--fasta", str(fasta)]) == 2
