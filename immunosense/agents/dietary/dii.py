"""Dietary Inflammatory Index (Shivappa 2014) computation and percentile lookup.

DII scoring:
    For each nutrient component:
        z = (intake - global_mean) / global_sd
        centered_percentile = 2 * Phi(z) - 1     # ranges [-1, +1]
        contribution = centered_percentile * inflammatory_effect_score
    DII = sum(contributions over components with non-NaN intake)

Higher DII = more pro-inflammatory diet.

Percentile lookup:
    Layer 1 trains quantile regressors that predict DII at q=[0.10, 0.25, 0.50,
    0.75, 0.90, 0.95, 0.99] given (age, sex, bmi). Place an observed DII on
    that demographic-specific distribution by interpolating between quantiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from immunosense.agents.dietary.constants import DII_REF, LAYER1_FEATURE_COLS


def compute_dii_row(intake_row: pd.Series, ref: dict = DII_REF) -> float:
    """Compute DII for one participant given their nutrient intake row.

    Args:
        intake_row: pandas Series with one entry per component in `ref`.
                     NaN values are skipped.
        ref: DII reference table (component -> (mean, sd, effect))

    Returns:
        DII score (float). NaN if no components had data.
    """
    total = 0.0
    n_used = 0
    for component, (mu, sd, effect) in ref.items():
        v = intake_row.get(component, np.nan)
        if pd.isna(v):
            continue
        z = (v - mu) / sd
        centered = (norm.cdf(z) * 2.0) - 1.0
        total += centered * effect
        n_used += 1
    return total if n_used > 0 else np.nan


def compute_meal_dii(nutrients: dict, ref: dict = DII_REF) -> float:
    """Compute DII from a daily nutrient dict.

    NOTE: This should be applied to DAILY nutrient totals, NOT per-meal.
    The DII transform is non-linear (z-score + Phi + multiply), so
    sum(per-meal DII) != DII(sum(per-meal nutrients)).

    Args:
        nutrients: dict mapping DII component name -> total intake.
        ref: DII reference table.

    Returns:
        DII score (float). NaN if no components had data.
    """
    total = 0.0
    n_used = 0
    for component, (mu, sd, effect) in ref.items():
        v = nutrients.get(component)
        if v is None or pd.isna(v):
            continue
        z = (v - mu) / sd
        centered = (norm.cdf(z) * 2.0) - 1.0
        total += centered * effect
        n_used += 1
    return total if n_used > 0 else float("nan")


def get_population_dii(
    age: float,
    sex: int,
    bmi: float,
    dii_models: dict,
    feature_cols: list = LAYER1_FEATURE_COLS,
) -> dict:
    """Predict DII quantiles for a given demographic.

    Args:
        age: Age in years
        sex: 1 = male, 2 = female (NHANES convention)
        bmi: Body mass index
        dii_models: Dict of {quantile: trained model}
        feature_cols: Feature column ordering used at training time

    Returns:
        Dict {quantile: predicted_dii_value} for each trained quantile
    """
    X_one = pd.DataFrame([[age, sex, bmi]], columns=feature_cols)
    return {q: float(m.predict(X_one)[0]) for q, m in dii_models.items()}


def get_dii_percentile(
    age: float,
    sex: int,
    bmi: float,
    dii_value: float,
    dii_models: dict,
    feature_cols: list = LAYER1_FEATURE_COLS,
) -> float:
    """Place an observed DII on the demographic-specific population distribution.

    Args:
        age, sex, bmi: Demographics for percentile lookup
        dii_value: Observed DII to place
        dii_models: Trained quantile regressors {q: model}
        feature_cols: Feature column ordering used at training time

    Returns:
        Percentile in [0, 1]. Clipped to [min(q), max(q)] outside the
        trained quantile range. Defensively re-sorts predicted quantiles
        because quantile regression can produce non-monotonic predictions.
    """
    quantile_preds = get_population_dii(age, sex, bmi, dii_models, feature_cols)
    qs = np.array(list(quantile_preds.keys()))
    vs = np.array(list(quantile_preds.values()))

    # Defensive sort: quantile regression can produce non-monotonic predictions
    order = np.argsort(vs)
    qs, vs = qs[order], vs[order]

    if dii_value <= vs[0]:
        return float(qs[0])
    if dii_value >= vs[-1]:
        return float(qs[-1])
    return float(np.interp(dii_value, vs, qs))


def load_layer1_artifacts(artifact_dir: Path) -> dict:
    """Load Layer 1 trained quantile models + scoring reference.

    Args:
        artifact_dir: Path to directory containing dii_quantile_*.pkl files
                       and dii_scoring_reference.pkl

    Returns:
        Dict with keys:
            'dii_models':  dict of {quantile: trained_model}
            'dii_ref':     DII reference table used at training
            'feature_cols': demographic feature column ordering

    Raises:
        FileNotFoundError if expected artifacts are missing.
    """
    import joblib

    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Layer 1 artifacts directory not found: {artifact_dir}. "
            "Train them via `python -m immunosense.agents.dietary.layer1_train`."
        )

    ref_path = artifact_dir / "dii_scoring_reference.pkl"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"DII scoring reference not found at {ref_path}. "
            "Train Layer 1 first."
        )

    ref = joblib.load(ref_path)
    dii_models = {}
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        model_path = artifact_dir / f"dii_quantile_{int(q * 100):02d}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Quantile model missing: {model_path}. "
                "Re-train Layer 1 (some models are missing)."
            )
        dii_models[q] = joblib.load(model_path)

    return {
        "dii_models": dii_models,
        "dii_ref": ref.get("dii_ref", DII_REF),
        "feature_cols": ref.get("feature_cols", LAYER1_FEATURE_COLS),
    }
