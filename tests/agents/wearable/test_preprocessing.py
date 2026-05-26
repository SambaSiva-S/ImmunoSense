"""Tests for wearable.preprocessing - Hampel, Akima, ENMO."""

from __future__ import annotations

import numpy as np
import pytest

from immunosense.agents.wearable.preprocessing import (
    HampelFilter,
    akima_interpolate,
    enmo_sleep_wake,
)


class TestHampelFilter:
    def test_passes_normal_signal(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(900, 30, 100)  # normal RR intervals
        f = HampelFilter(window_size=7, k_threshold=3.0)
        filtered, mask = f.filter(signal)
        # Few or no outliers in clean Gaussian data
        assert mask.sum() < 10

    def test_replaces_outlier_with_median(self):
        rng = np.random.default_rng(0)
        # Surrounding values have some variation so MAD > 0
        signal = np.concatenate([
            rng.normal(950, 5, 10),
            np.array([9999.0]),
            rng.normal(950, 5, 10),
        ])
        f = HampelFilter(window_size=7, k_threshold=3.0)
        filtered, mask = f.filter(signal)
        # The outlier at index 10 should be flagged and replaced
        assert mask[10]
        # And replaced with something close to 950
        assert abs(filtered[10] - 950.0) < 20.0

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            HampelFilter(window_size=2)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k_threshold"):
            HampelFilter(k_threshold=0)

    def test_returns_correct_shape(self):
        signal = np.array([1, 2, 3, 4, 5], dtype=float)
        f = HampelFilter()
        filtered, mask = f.filter(signal)
        assert filtered.shape == signal.shape
        assert mask.shape == signal.shape
        assert mask.dtype == bool


class TestAkimaInterpolate:
    def test_fills_gap_without_overshoot(self):
        ts = np.arange(20).astype(float)
        vals = np.array([60.0, 61, 62, 63, np.nan, np.nan, np.nan,
                         80, 81, 82, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74])
        filled = akima_interpolate(ts, vals, ts)
        # No NaN remaining
        assert not np.isnan(filled).any()
        # Filled values should be in or near [60, 83] - no overshoot
        assert filled.min() >= 58
        assert filled.max() <= 85

    def test_too_few_valid_returns_nan(self):
        ts = np.arange(5).astype(float)
        vals = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        filled = akima_interpolate(ts, vals, ts)
        assert np.isnan(filled).all()


class TestENMOSleepWake:
    def test_low_motion_low_hr_is_sleep(self):
        n = 60
        enmo = np.full(n, 0.005)  # very low motion
        hr = np.full(n, 60.0)     # below threshold
        labels = enmo_sleep_wake(enmo, hr)
        assert (labels == "sleep").all()

    def test_high_motion_or_high_hr_is_wake(self):
        rng = np.random.default_rng(0)
        n = 60
        # Variable motion above threshold (constants would have variance 0)
        enmo = np.abs(rng.normal(0.2, 0.05, n))
        hr = np.full(n, 60.0)
        labels = enmo_sleep_wake(enmo, hr)
        # Variable high motion should mostly be classified wake
        assert (labels == "wake").sum() > n * 0.8

    def test_high_hr_is_wake_even_if_still(self):
        n = 60
        enmo = np.full(n, 0.001)  # very low motion
        hr = np.full(n, 100.0)     # but high HR
        labels = enmo_sleep_wake(enmo, hr)
        assert (labels == "wake").all()
