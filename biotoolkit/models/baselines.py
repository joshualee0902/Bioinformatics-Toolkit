"""Baseline classifiers (scikit-learn / LightGBM).

- majority:  always predicts the training majority class (sanity floor)
- gc_lr:     logistic regression on GC content alone (sanity floor)
- kmer_lr:   logistic regression on normalized k-mer frequencies
- gbt:       LightGBM gradient-boosted trees on k-mers + toolkit features
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def build(model: str, params: dict, seed: int):
    if model == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model == "gc_lr":
        scaler = StandardScaler(with_mean=False)  # inputs are sparse; centering is not needed
        return make_pipeline(scaler, LogisticRegression(random_state=seed, **params))
    if model == "kmer_lr":
        # MaxAbsScaler keeps the k-mer matrix sparse.
        return make_pipeline(MaxAbsScaler(), LogisticRegression(random_state=seed, **params))
    if model == "gbt":
        import lightgbm as lgb

        return lgb.LGBMClassifier(random_state=seed, verbose=-1, **params)
    raise ValueError(f"Unknown baseline model: {model}")


def fit(estimator, model: str, X_train, y_train, X_val, y_val, early_stopping_rounds: int = 50):
    """Fit a baseline. LightGBM uses the validation set for early stopping."""
    if model == "gbt":
        import lightgbm as lgb

        estimator.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        estimator.fit(X_train, y_train)
    return estimator


def predict_proba(estimator, X) -> np.ndarray:
    """Probability of the coding class (label 1)."""
    return estimator.predict_proba(X)[:, 1]


def importances(estimator, model: str) -> np.ndarray:
    """Per-feature importance: coefficient for logistic regression, total gain for GBT."""
    if model in ("gc_lr", "kmer_lr"):
        return estimator[-1].coef_[0]  # signed: positive pushes toward "coding"
    if model == "gbt":
        return estimator.booster_.feature_importance(importance_type="gain")
    return np.zeros(estimator.n_features_in_)
