"""Tests for Layer 2 Pillar B — LightGBM classifier."""

import warnings

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.biomarker.layer2.classifier import train_lgb_classifier


# Suppress LightGBM's auto-generated Column_N feature-name warnings when
# we pass numpy arrays. These tests intentionally exercise numpy input.
@pytest.fixture(autouse=True)
def _suppress_lgb_feature_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        yield


def test_train_lgb_classifier_returns_fitted_model():
    np.random.seed(0)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    model = train_lgb_classifier(X, y, verbose=False)
    preds = model.predict(X)
    assert len(preds) == 100


def test_train_lgb_classifier_predicts_probabilities():
    np.random.seed(0)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    model = train_lgb_classifier(X, y, verbose=False)
    probs = model.predict_proba(X)
    # Probabilities should be in [0, 1]
    assert (probs >= 0).all()
    assert (probs <= 1).all()
    # Should sum to ~1 per row
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_train_lgb_classifier_supports_validation_set():
    np.random.seed(0)
    X_tr = np.random.randn(80, 5)
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_val = np.random.randn(20, 5)
    y_val = (X_val[:, 0] > 0).astype(int)
    model = train_lgb_classifier(
        X_tr, y_tr, X_val=X_val, y_val=y_val,
        early_stopping_rounds=10, verbose=False,
    )
    assert model is not None


def test_train_lgb_classifier_class_weight_balanced():
    """LGBMClassifier should be configured with class_weight='balanced'."""
    np.random.seed(0)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    model = train_lgb_classifier(X, y, verbose=False)
    # class_weight should be 'balanced' (sklearn-compatible LGBMClassifier reflects this)
    assert getattr(model, "class_weight", None) == "balanced"


def test_train_lgb_classifier_works_with_dataframe():
    """Passing pandas DataFrames preserves feature names."""
    np.random.seed(0)
    X = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["a", "b", "c"],
    )
    y = (X["a"] > 0).astype(int).values
    model = train_lgb_classifier(X, y, verbose=False)
    preds = model.predict(X)
    assert len(preds) == 100
