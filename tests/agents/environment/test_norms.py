"""Tests for environment.norms - regional baselines and percentile lookups."""

from __future__ import annotations

import pytest

from immunosense.agents.environment.norms import (
    FEATURES,
    REGIONAL_NORMS,
    REGIONS,
    SEASONS,
    get_population_percentile,
    infer_region_from_zip,
    infer_season_from_date,
)


class TestRegionInference:
    @pytest.mark.parametrize("zip_code,expected", [
        ("28202", "SE"),
        ("85001", "SW"),
        ("02101", "NE"),
        ("60601", "MW"),
        ("90001", "W"),
        ("10001", "NE"),
        ("94102", "W"),
    ])
    def test_zip_to_region(self, zip_code, expected):
        assert infer_region_from_zip(zip_code) == expected

    def test_invalid_zip_defaults_to_se(self):
        assert infer_region_from_zip(None) == "SE"
        assert infer_region_from_zip("abcde") == "SE"
        # Note: empty string zfills to "00000" -> first digit '0' -> "NE" (by design)
        # Truly invalid (non-numeric) defaults to "SE"

    def test_int_zip_works(self):
        """Integer ZIP codes should work."""
        assert infer_region_from_zip(28202) == "SE"

    def test_short_zip_padded(self):
        """Short ZIP padded with leading zeros."""
        assert infer_region_from_zip("2101") == "NE"  # treated as 02101


class TestSeasonInference:
    @pytest.mark.parametrize("date,expected", [
        ("2026-01-15", "winter"),
        ("2026-02-28", "winter"),
        ("2026-03-15", "spring"),
        ("2026-04-15", "spring"),
        ("2026-05-15", "spring"),
        ("2026-06-15", "summer"),
        ("2026-07-15", "summer"),
        ("2026-08-15", "summer"),
        ("2026-09-15", "fall"),
        ("2026-10-15", "fall"),
        ("2026-11-15", "fall"),
        ("2026-12-15", "winter"),
    ])
    def test_date_to_season(self, date, expected):
        assert infer_season_from_date(date) == expected


class TestRegionalNorms:
    def test_all_cells_populated(self):
        """5 regions x 4 seasons x 5 features = 100 cells, all populated."""
        missing = [
            (r, s, f)
            for r in REGIONS for s in SEASONS for f in FEATURES
            if f not in REGIONAL_NORMS[r][s]
        ]
        assert not missing, f"Missing cells: {missing}"

    def test_all_norms_are_tuples_with_mean_and_std(self):
        for r in REGIONS:
            for s in SEASONS:
                for f in FEATURES:
                    val = REGIONAL_NORMS[r][s][f]
                    assert isinstance(val, tuple), f"{r}/{s}/{f}: not a tuple"
                    assert len(val) == 2, f"{r}/{s}/{f}: not length 2"
                    mean, std = val
                    assert std > 0, f"{r}/{s}/{f}: std must be positive"


class TestPopulationPercentile:
    def test_median_value_returns_50th_percentile(self):
        """Value at the regional mean should be ~50th percentile."""
        # SE/spring/PM2.5 mean is 9.0
        pct = get_population_percentile("SE", "spring", "pm25", 9.0)
        assert abs(pct - 0.5) < 0.01

    def test_high_value_returns_high_percentile(self):
        # SE/spring/PM2.5 mean=9.0, std=3.5. Value=20 is ~3 stdev high.
        pct = get_population_percentile("SE", "spring", "pm25", 20.0)
        assert pct > 0.99

    def test_low_value_returns_low_percentile(self):
        pct = get_population_percentile("SE", "spring", "pm25", 0.0)
        assert pct < 0.05

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            get_population_percentile("ZZ", "spring", "pm25", 10.0)

    def test_unknown_season_raises(self):
        with pytest.raises(ValueError, match="Unknown season"):
            get_population_percentile("SE", "foo", "pm25", 10.0)

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            get_population_percentile("SE", "spring", "co2", 10.0)
