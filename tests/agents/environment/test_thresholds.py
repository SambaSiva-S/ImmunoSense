"""Tests for environment.thresholds - EPA classifiers and AQI conversion."""

from __future__ import annotations

import pytest

from immunosense.agents.environment.thresholds import (
    aqi_to_ozone_ppb,
    aqi_to_pm25,
    classify_all_thresholds,
    classify_barometric_threshold,
    classify_ozone_threshold,
    classify_pm25_threshold,
    classify_pollen_threshold,
    classify_uv_threshold,
)
from immunosense.agents.environment.types import FetchedFeatures


class TestPM25Classification:
    @pytest.mark.parametrize("value,expected", [
        (8.0, "good"),
        (12.0, "good"),
        (12.1, "moderate"),
        (35.4, "moderate"),
        (35.5, "unhealthy_sensitive"),
        (55.4, "unhealthy_sensitive"),
        (55.5, "unhealthy"),
        (150.4, "unhealthy"),
        (200.0, "very_unhealthy"),
        (300.0, "hazardous"),
    ])
    def test_classification(self, value, expected):
        assert classify_pm25_threshold(value) == expected

    def test_none_returns_none(self):
        assert classify_pm25_threshold(None) is None


class TestOzoneClassification:
    @pytest.mark.parametrize("value,expected", [
        (30, "good"),
        (54, "good"),
        (55, "moderate"),
        (70, "moderate"),
        (71, "unhealthy_sensitive"),
        (85, "unhealthy_sensitive"),
        (86, "unhealthy"),
        (105, "unhealthy"),
        (120, "very_unhealthy"),
    ])
    def test_classification(self, value, expected):
        assert classify_ozone_threshold(value) == expected


class TestUVClassification:
    @pytest.mark.parametrize("value,expected", [
        (0, "low"),
        (2, "low"),
        (3, "moderate"),
        (5.9, "moderate"),
        (6, "high"),
        (7.9, "high"),
        (8, "very_high"),
        (10.9, "very_high"),
        (11, "extreme"),
    ])
    def test_classification(self, value, expected):
        assert classify_uv_threshold(value) == expected


class TestBarometricClassification:
    @pytest.mark.parametrize("value,expected", [
        (0.2, "stable"),
        (-0.3, "stable"),
        (0.6, "moderate_change"),
        (-0.9, "moderate_change"),
        (1.5, "large_change"),
        (-2.0, "large_change"),
    ])
    def test_classification(self, value, expected):
        assert classify_barometric_threshold(value) == expected


class TestPollenClassification:
    @pytest.mark.parametrize("value,expected", [
        (0.5, "low"),
        (1.4, "low"),
        (1.5, "moderate"),
        (2.9, "moderate"),
        (3.0, "high"),
        (4.4, "high"),
        (4.5, "very_high"),
        (5.0, "very_high"),
    ])
    def test_classification(self, value, expected):
        assert classify_pollen_threshold(value) == expected


class TestClassifyAllThresholds:
    def test_full_set(self):
        features = FetchedFeatures(
            pm25_ug_m3=15.0,
            ozone_ppb=60.0,
            uv_index=7.5,
            barometric_change_kpa=0.8,
            pollen_index=3.5,
        )
        result = classify_all_thresholds(features)
        assert result == {
            "pm25": "moderate",
            "ozone": "moderate",
            "uv": "high",
            "barometric": "moderate_change",
            "pollen": "high",
        }

    def test_missing_features_return_none(self):
        features = FetchedFeatures()  # all None
        result = classify_all_thresholds(features)
        for v in result.values():
            assert v is None


class TestAQIConversion:
    def test_aqi_to_pm25_breakpoints(self):
        """Verify EPA breakpoint inversion for PM2.5."""
        assert aqi_to_pm25(0) == 0.0
        assert aqi_to_pm25(50) == 12.0
        # Midpoint of AQI 0-50 should be midpoint of 0-12
        midpoint = aqi_to_pm25(25)
        assert 5 < midpoint < 7  # ~6

    def test_aqi_to_ozone_ppb_breakpoints(self):
        """Verify EPA breakpoint inversion for ozone."""
        assert aqi_to_ozone_ppb(0) == 0
        assert aqi_to_ozone_ppb(50) == 54
        # AQI 50-100 -> ozone 55-70
        midpoint = aqi_to_ozone_ppb(75)
        assert 60 < midpoint < 65  # ~62-63

    def test_none_aqi_returns_none(self):
        assert aqi_to_pm25(None) is None
        assert aqi_to_ozone_ppb(None) is None

    def test_aqi_above_500_returns_none(self):
        """AQI outside breakpoint range returns None."""
        assert aqi_to_pm25(600) is None
