"""Pillar C: Explanatory — XGBoost classifier + SHAP TreeExplainer.

Trains an XGBoost multiclass classifier and computes SHAP TreeExplainer
values for per-prediction feature attribution.

This is the "why does the model think this diagnosis" pillar of Layer 2.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from immunosense.agents.biomarker.constants import LAYER2_HYPERPARAMS


def train_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
    early_stopping_rounds: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Any:
    """Train an XGBoost multiclass classifier.

    Args:
        X_train, y_train: training features and integer class labels
        n_classes: number of disease classes (passed as num_class to XGB)
        X_val, y_val: optional validation set for early stopping
        n_estimators, max_depth, learning_rate, early_stopping_rounds:
            hyperparams (defaults from LAYER2_HYPERPARAMS)
        random_state: seed for reproducibility
        verbose: print training progress

    Returns:
        Trained XGBClassifier.
    """
    import xgboost as xgb

    n_estimators = n_estimators or LAYER2_HYPERPARAMS["xgb_n_estimators"]
    max_depth = max_depth or LAYER2_HYPERPARAMS["xgb_max_depth"]
    learning_rate = learning_rate or LAYER2_HYPERPARAMS["xgb_learning_rate"]
    early_stopping_rounds = (
        early_stopping_rounds or LAYER2_HYPERPARAMS["xgb_early_stopping_rounds"]
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=(
            early_stopping_rounds if X_val is not None else None
        ),
        verbosity=0,
        random_state=random_state,
    )

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    if verbose and X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"  Pillar C (XGBoost) val accuracy: {acc * 100:.1f}%")

    return model


def compute_shap_values(
    xgb_model: Any,
    X: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Compute SHAP values for X using TreeExplainer.

    Args:
        xgb_model: trained XGBClassifier
        X: (n_samples, n_features) feature matrix
        verbose: print progress

    Returns:
        Either:
            shape (n_samples, n_features, n_classes) for multiclass models with
            modern shap versions, OR
            list of arrays (one per class), each shape (n_samples, n_features),
            for older shap versions.

        The caller (typically top_shap_drivers) handles both shapes.
    """
    import shap

    if verbose:
        print(f"  Computing SHAP values on {len(X)} samples...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    return shap_values


def top_shap_drivers(
    shap_values: Any,
    sample_idx: int,
    pred_class: int,
    feature_names: list,
    sample_values: np.ndarray,
    top_k: int = 3,
) -> list:
    """Extract top-k SHAP drivers for one sample's predicted class.

    Args:
        shap_values: output of compute_shap_values (np.ndarray 3D or list of 2D)
        sample_idx: index into the original sample matrix
        pred_class: integer class label predicted for this sample
        feature_names: ordered list of feature names matching X columns
        sample_values: (n_features,) feature values for this sample
        top_k: number of top drivers to return

    Returns:
        list of dicts with keys: feature_name, feature_value, shap_value, direction
        Sorted by |shap_value| descending.
    """
    # Handle both shap output shapes
    if isinstance(shap_values, list):
        # Older shap: list of (n_samples, n_features) arrays, one per class
        sv = shap_values[pred_class][sample_idx]
    else:
        # Newer shap: (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:
            sv = shap_values[sample_idx, :, pred_class]
        else:
            # Binary or single-class output
            sv = shap_values[sample_idx]

    top_idx = list(np.argsort(np.abs(sv))[-top_k:][::-1])
    drivers = []
    for j in top_idx:
        shap_val = float(sv[j])
        drivers.append({
            "feature_name": feature_names[j] if j < len(feature_names) else f"feature_{j}",
            "feature_value": float(sample_values[j]) if j < len(sample_values) else 0.0,
            "shap_value": shap_val,
            "direction": "toward" if shap_val > 0 else "away_from",
        })
    return drivers
