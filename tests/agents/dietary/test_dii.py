"""Tests for dietary.dii: DII scoring + percentile lookup."""

import math

import numpy as np
import pandas as pd

from immunosense.agents.dietary.constants import DII_REF
from immunosense.agents.dietary.dii import (
    compute_dii_row,
    compute_meal_dii,
    get_dii_percentile,
    get_population_dii,
)


def test_compute_dii_row_neutral():
    """Intake exactly at global mean -> z=0 -> centered=0 -> DII=0."""
    # Build an intake row where every component equals its global mean
    intake_row = pd.Series({c: ref[0] for c, ref in DII_REF.items()})
    dii = compute_dii_row(intake_row)
    assert abs(dii) < 1e-6


def test_compute_dii_row_all_nan_returns_nan():
    """If every component is NaN, DII should be NaN."""
    intake_row = pd.Series({c: np.nan for c in DII_REF})
    dii = compute_dii_row(intake_row)
    assert pd.isna(dii)


def test_compute_dii_row_partial_nan_uses_others():
    """Components with NaN intake should be silently skipped."""
    intake_row = pd.Series({c: np.nan for c in DII_REF})
    # Give it one anti-inflammatory hit
    intake_row["fiber_g"] = 100.0   # way above mean -> anti-inflammatory
    dii = compute_dii_row(intake_row)
    assert dii < 0  # anti-inflammatory


def test_compute_dii_row_pro_inflammatory_high_satfat():
    """High saturated fat should yield positive DII contribution."""
    intake_row = pd.Series({c: ref[0] for c, ref in DII_REF.items()})  # neutral
    intake_row["saturated_fat_g"] = 100.0  # way above 28.6 mean
    dii = compute_dii_row(intake_row)
    assert dii > 0


def test_compute_dii_row_anti_inflammatory_high_omega3():
    """High omega-3 should yield negative DII contribution."""
    intake_row = pd.Series({c: ref[0] for c, ref in DII_REF.items()})  # neutral
    intake_row["omega3_g"] = 5.0  # way above 1.06 mean
    dii = compute_dii_row(intake_row)
    assert dii < 0


def test_compute_meal_dii_matches_row():
    """compute_meal_dii(dict) should match compute_dii_row(Series) on same data."""
    data = {c: ref[0] for c, ref in DII_REF.items()}
    data["saturated_fat_g"] = 50.0
    dii_dict = compute_meal_dii(data)
    dii_row = compute_dii_row(pd.Series(data))
    assert abs(dii_dict - dii_row) < 1e-9


def test_compute_meal_dii_empty():
    """Empty nutrients dict returns NaN."""
    dii = compute_meal_dii({})
    assert pd.isna(dii)


def test_compute_meal_dii_unrelated_keys_ignored():
    """Keys not in DII_REF should be ignored."""
    data = {"random_key": 99999.0, "fiber_g": 100.0}
    dii = compute_meal_dii(data)
    # Should still be negative (anti-inflammatory from fiber)
    assert dii < 0


# === Percentile lookup ===


class _MockQuantileModel:
    """A mock quantile model with a fixed prediction."""

    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, X) -> np.ndarray:
        return np.array([self.prediction])


def _build_mock_models(quantile_to_dii_pred: dict) -> dict:
    return {q: _MockQuantileModel(v) for q, v in quantile_to_dii_pred.items()}


def test_get_population_dii_returns_all_quantiles():
    """Should return one prediction per trained quantile."""
    mock_models = _build_mock_models({
        0.10: -1.0, 0.50: 0.5, 0.90: 2.0,
    })
    out = get_population_dii(30, 1, 22, mock_models)
    assert set(out.keys()) == {0.10, 0.50, 0.90}


def test_get_dii_percentile_below_min():
    """DII below the smallest trained quantile -> returns min quantile."""
    mock_models = _build_mock_models({0.10: -1.0, 0.50: 0.0, 0.90: 1.0})
    pct = get_dii_percentile(30, 1, 22, -5.0, mock_models)
    assert pct == 0.10


def test_get_dii_percentile_above_max():
    """DII above the largest trained quantile -> returns max quantile."""
    mock_models = _build_mock_models({0.10: -1.0, 0.50: 0.0, 0.90: 1.0})
    pct = get_dii_percentile(30, 1, 22, 99.0, mock_models)
    assert pct == 0.90


def test_get_dii_percentile_interpolation():
    """DII between two quantiles interpolates linearly."""
    mock_models = _build_mock_models({0.10: 0.0, 0.50: 1.0, 0.90: 2.0})
    pct = get_dii_percentile(30, 1, 22, 0.5, mock_models)
    # halfway between 0.10 and 0.50 -> ~0.30
    assert 0.25 <= pct <= 0.35


def test_get_dii_percentile_non_monotonic_models_handled():
    """If quantile predictions are non-monotonic, function should sort defensively."""
    # Note: q=0.50 predicts higher than q=0.90 (non-monotonic)
    mock_models = _build_mock_models({0.10: 0.0, 0.50: 2.0, 0.90: 1.0})
    pct = get_dii_percentile(30, 1, 22, 1.5, mock_models)
    # Should still return a valid percentile between 0.10 and 0.90
    assert 0.10 <= pct <= 0.90
