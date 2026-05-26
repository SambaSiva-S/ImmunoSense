"""Tests for wearable feature computation: HRV, sleep, TADI, stress."""

from __future__ import annotations

import numpy as np

from immunosense.agents.wearable.hrv import compute_hrv_features
from immunosense.agents.wearable.sleep import compute_sleep_features
from immunosense.agents.wearable.stress import compute_wearable_stress_score
from immunosense.agents.wearable.tadi import compute_tadi


class TestHRVFeatures:
    def test_returns_dict_with_expected_keys(self):
        rng = np.random.default_rng(0)
        rr = rng.normal(950, 30, 200)
        result = compute_hrv_features(rr)
        for key in ["rmssd", "sdnn", "pnn50", "hf_power", "lf_hf_ratio", "sd1", "sd2"]:
            assert key in result

    def test_low_data_returns_nans(self):
        rr = np.array([900.0, 910.0])
        result = compute_hrv_features(rr)
        assert np.isnan(result["rmssd"])

    def test_filters_unphysiological_values(self):
        """Values outside (300, 2000) should be filtered out."""
        rr_valid = np.full(100, 900.0)
        rr_with_bad = np.concatenate([rr_valid, np.array([100.0, 5000.0])])
        result = compute_hrv_features(rr_with_bad)
        # Should still compute features from valid 100 values
        assert not np.isnan(result["rmssd"])
        # The filtered bad values mean RMSSD should be near 0 (constant input)
        # Actually with constant rr=900 RMSSD = 0
        assert result["rmssd"] == 0.0

    def test_higher_variability_yields_higher_rmssd(self):
        rng = np.random.default_rng(0)
        rr_low = rng.normal(950, 5, 200)    # tight HRV
        rr_high = rng.normal(950, 50, 200)  # loose HRV
        low = compute_hrv_features(rr_low)
        high = compute_hrv_features(rr_high)
        assert high["rmssd"] > low["rmssd"]


class TestSleepFeatures:
    def test_typical_sleep_architecture(self):
        # 60% light, 20% deep, 15% rem, 5% wake (480 minutes)
        stages = np.array(
            ["light"] * 288 + ["deep"] * 96 + ["rem"] * 72 + ["wake"] * 24
        )
        result = compute_sleep_features(stages)
        assert result["duration_hrs"] == (288 + 96 + 72) / 60.0
        assert abs(result["efficiency"] - (288 + 96 + 72) / 480) < 1e-6
        assert abs(result["deep_pct"] - 96 / (288 + 96 + 72)) < 1e-6

    def test_empty_returns_nans(self):
        result = compute_sleep_features(np.array([]))
        assert np.isnan(result["duration_hrs"])

    def test_all_wake_zero_efficiency(self):
        stages = np.array(["wake"] * 100)
        result = compute_sleep_features(stages)
        assert result["efficiency"] == 0.0


class TestTADI:
    def test_perfectly_coupled_yields_low_tadi(self):
        # HRV and temp move together (same z-scored shape)
        t = np.linspace(0, 4 * np.pi, 50)
        hrv = np.sin(t)
        temp = np.sin(t)
        tadi = compute_tadi(hrv, temp)
        # Strong positive correlation -> tadi near 0
        assert tadi < 0.05

    def test_decoupled_yields_high_tadi(self):
        # Truly uncorrelated noise (anti-correlated sinusoids are still strongly
        # correlated after a lag shift; for full decoupling we need pure noise)
        rng = np.random.default_rng(0)
        hrv = rng.normal(0, 1, 50)
        temp = rng.normal(0, 1, 50)
        tadi = compute_tadi(hrv, temp)
        # Random noise -> low positive correlation -> high TADI
        # Allowing some slack because random correlation can leak through
        assert tadi > 0.5

    def test_zero_variance_returns_nan(self):
        hrv = np.full(50, 30.0)  # constant
        temp = np.linspace(33.0, 34.0, 50)
        assert np.isnan(compute_tadi(hrv, temp))

    def test_too_short_returns_nan(self):
        assert np.isnan(compute_tadi(np.array([1.0, 2.0]), np.array([3.0, 4.0])))


class TestStressScore:
    def test_no_components_returns_zero(self):
        assert compute_wearable_stress_score({}) == 0.0

    def test_low_hrv_increases_stress(self):
        reading = {"hrv_sleep_vs_baseline_ratio": 0.5}  # 50% baseline
        score = compute_wearable_stress_score(reading)
        # 1 - 0.5 = 0.5, weight 0.35, total = 0.5 (renormalized to 1.0)
        # Actually weights renormalize so 0.5 stays
        assert score == 0.5

    def test_clipped_to_unit_interval(self):
        # Extreme negative values clipped to 0
        reading = {
            "hrv_sleep_vs_baseline_ratio": 5.0,  # super high (low stress)
            "sleep_efficiency": 0.99,
            "skin_temp_deviation": -1.0,  # very low (clipped to 0)
        }
        score = compute_wearable_stress_score(reading)
        assert 0.0 <= score <= 1.0

    def test_high_score_with_all_bad(self):
        reading = {
            "hrv_sleep_vs_baseline_ratio": 0.0,
            "sleep_efficiency": 0.0,
            "skin_temp_deviation": 1.0,  # high
            "resting_hr_vs_baseline": 5.0,  # high
            "activity_vs_baseline": 0.0,
        }
        score = compute_wearable_stress_score(reading)
        assert score > 0.8
