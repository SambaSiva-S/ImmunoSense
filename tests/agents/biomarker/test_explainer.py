"""Tests for Layer 2 Pillar C — XGBoost classifier + SHAP."""

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.layer2.explainer import (
    compute_shap_values,
    top_shap_drivers,
    train_xgb_classifier,
)


def test_train_xgb_classifier_returns_fitted_model():
    np.random.seed(0)
    X = np.random.randn(120, 6)
    y = np.random.randint(0, 3, 120)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    preds = model.predict(X)
    assert len(preds) == 120


def test_train_xgb_classifier_predicts_probabilities():
    np.random.seed(0)
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 3, 60)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    probs = model.predict_proba(X)
    assert probs.shape == (60, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_train_xgb_classifier_supports_validation_set():
    np.random.seed(0)
    X_tr = np.random.randn(80, 4)
    y_tr = np.random.randint(0, 3, 80)
    X_val = np.random.randn(20, 4)
    y_val = np.random.randint(0, 3, 20)
    model = train_xgb_classifier(
        X_tr, y_tr, n_classes=3, X_val=X_val, y_val=y_val,
        early_stopping_rounds=10, verbose=False,
    )
    assert model is not None


def test_compute_shap_values_runs_without_error():
    np.random.seed(0)
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 3, 40)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    shap_vals = compute_shap_values(model, X[:5])
    # shap may return either a (5, 5, 3) ndarray or a list of three (5, 5) arrays
    assert shap_vals is not None


def test_top_shap_drivers_returns_k_items():
    np.random.seed(0)
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 3, 40)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    shap_vals = compute_shap_values(model, X[:1])
    drivers = top_shap_drivers(
        shap_vals, sample_idx=0, pred_class=0,
        feature_names=["a", "b", "c", "d", "e"],
        sample_values=X[0], top_k=3,
    )
    assert len(drivers) == 3
    for d in drivers:
        assert "feature_name" in d
        assert "feature_value" in d
        assert "shap_value" in d
        assert "direction" in d
        assert d["direction"] in ("toward", "away_from")


def test_top_shap_drivers_sorted_by_absolute_value():
    """Top drivers must be sorted by |shap_value| descending."""
    np.random.seed(0)
    X = np.random.randn(40, 6)
    y = np.random.randint(0, 3, 40)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    shap_vals = compute_shap_values(model, X[:1])
    drivers = top_shap_drivers(
        shap_vals, sample_idx=0, pred_class=0,
        feature_names=[f"f{i}" for i in range(6)],
        sample_values=X[0], top_k=6,
    )
    abs_vals = [abs(d["shap_value"]) for d in drivers]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_top_shap_drivers_with_dataframe_input():
    """Pipeline should work end-to-end with DataFrame input too."""
    np.random.seed(0)
    X = pd.DataFrame(
        np.random.randn(40, 5),
        columns=["a", "b", "c", "d", "e"],
    )
    y = np.random.randint(0, 3, 40)
    model = train_xgb_classifier(X, y, n_classes=3, verbose=False)
    # SHAP on numpy
    shap_vals = compute_shap_values(model, X.values[:1])
    drivers = top_shap_drivers(
        shap_vals, sample_idx=0, pred_class=0,
        feature_names=["a", "b", "c", "d", "e"],
        sample_values=X.values[0], top_k=2,
    )
    assert len(drivers) == 2
