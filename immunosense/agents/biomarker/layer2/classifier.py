"""Pillar B: Probabilistic — LightGBM multiclass classifier.

Trains a LightGBM classifier on the Rheumatic biomarker dataset to predict
disease class probabilities. Uses class_weight='balanced' to handle the
~7-class imbalance.

This is the "what disease is most likely" pillar of Layer 2.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from immunosense.agents.biomarker.constants import LAYER2_HYPERPARAMS


def train_lgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: Optional[int] = None,
    num_leaves: Optional[int] = None,
    learning_rate: Optional[float] = None,
    early_stopping_rounds: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Any:
    """Train a multiclass LightGBM classifier with class_weight='balanced'.

    Args:
        X_train, y_train: training features and integer class labels
        X_val, y_val: optional validation set for early stopping
        n_estimators, num_leaves, learning_rate, early_stopping_rounds:
            hyperparams (defaults from LAYER2_HYPERPARAMS)
        random_state: seed for reproducibility
        verbose: print training progress

    Returns:
        Trained LGBMClassifier.
    """
    import lightgbm as lgb

    n_estimators = n_estimators or LAYER2_HYPERPARAMS["lgb_n_estimators"]
    num_leaves = num_leaves or LAYER2_HYPERPARAMS["lgb_num_leaves"]
    learning_rate = learning_rate or LAYER2_HYPERPARAMS["lgb_learning_rate"]
    early_stopping_rounds = (
        early_stopping_rounds or LAYER2_HYPERPARAMS["lgb_early_stopping_rounds"]
    )

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        class_weight="balanced",
        verbose=-1,
        random_state=random_state,
    )

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ]

    model.fit(X_train, y_train, **fit_kwargs)

    if verbose and X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"  Pillar B (LightGBM) val accuracy: {acc * 100:.1f}%")

    return model
