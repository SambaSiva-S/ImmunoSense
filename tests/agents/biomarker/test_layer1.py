"""Tests for Layer 1 (CRP baseline)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.biomarker.constants import LAYER1_FEATURE_COLS, QUANTILES
from immunosense.agents.biomarker.layer1.crp_baseline import (
    CRPBaseline,
    get_crp_percentile,
    load_layer1_artifacts,
)


@pytest.fixture
def fake_crp_models():
    """Build a CRPBaseline with mock models that return increasing thresholds."""
    class FakeModel:
        def __init__(self, threshold: float):
            self.threshold = threshold

        def predict(self, X):
            return np.full(len(X), self.threshold)

    # Trained quantile thresholds (increasing with q)
    return CRPBaseline(models={
        0.10: FakeModel(0.5),
        0.25: FakeModel(1.0),
        0.50: FakeModel(2.0),
        0.75: FakeModel(4.0),
        0.90: FakeModel(8.0),
        0.95: FakeModel(12.0),
        0.99: FakeModel(20.0),
    })


def test_crp_baseline_returns_correct_quantile_below_threshold(fake_crp_models):
    """CRP below 10th percentile threshold should return 0.10."""
    p = fake_crp_models.percentile(age=30, sex=2, bmi=22, crp_value=0.3)
    assert p == 0.10


def test_crp_baseline_returns_99th_when_above_all_thresholds(fake_crp_models):
    """CRP above 99th percentile threshold should return 0.99."""
    p = fake_crp_models.percentile(age=30, sex=2, bmi=22, crp_value=100.0)
    assert p == 0.99


def test_crp_baseline_returns_correct_middle_quantile(fake_crp_models):
    """CRP between 50th and 75th thresholds should return 0.75."""
    p = fake_crp_models.percentile(age=30, sex=2, bmi=22, crp_value=3.0)
    assert p == 0.75


def test_crp_baseline_predict_quantiles_returns_all(fake_crp_models):
    quantiles = fake_crp_models.predict_quantiles(age=30, sex=2, bmi=22)
    assert set(quantiles.keys()) == set(QUANTILES)


def test_crp_baseline_load_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        CRPBaseline.load(Path("/nonexistent/path"))


def test_get_crp_percentile_wrapper(fake_crp_models):
    """The standalone function wraps CRPBaseline.percentile()."""
    p1 = get_crp_percentile(30, 2, 22, 0.3, fake_crp_models)
    p2 = fake_crp_models.percentile(30, 2, 22, 0.3)
    assert p1 == p2


def test_load_layer1_artifacts_alias_calls_load():
    """load_layer1_artifacts is just a renamed CRPBaseline.load."""
    with pytest.raises(FileNotFoundError):
        load_layer1_artifacts(Path("/nonexistent/path"))


def test_crp_baseline_uses_provided_feature_cols():
    """If feature_cols differs from default, predict_quantiles still works."""
    class FakeModel:
        def __init__(self, t):
            self.t = t
        def predict(self, X):
            return np.full(len(X), self.t)

    baseline = CRPBaseline(
        models={0.5: FakeModel(2.0)},
        feature_cols=["age", "sex", "bmi"],
    )
    out = baseline.predict_quantiles(45, 1, 25)
    assert out == {0.5: 2.0}


def test_crp_baseline_quantiles_are_sorted(fake_crp_models):
    """Internal _sorted_quantiles is monotonic."""
    qs = fake_crp_models._sorted_quantiles
    assert qs == sorted(qs)
