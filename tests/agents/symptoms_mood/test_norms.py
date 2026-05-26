"""Tests for symptoms_mood.norms - disease-stratified population baselines."""

from __future__ import annotations

import pytest

from immunosense.agents.symptoms_mood.norms import (
    DISEASE_NORMS,
    get_population_percentile,
    normalize_disease,
)
from immunosense.agents.symptoms_mood.types import ALL_FEATURES, DISEASE_TYPES


class TestNormalizeDisease:
    @pytest.mark.parametrize("alias,expected", [
        ("Rheumatoid Arthritis", "RA"),
        ("rheumatoid arthritis", "RA"),
        ("rheumatoid", "RA"),
        ("RA", "RA"),
        ("ra", "RA"),
        ("Lupus", "SLE"),
        ("systemic lupus", "SLE"),
        ("SLE", "SLE"),
        ("Multiple Sclerosis", "MS"),
        ("MS", "MS"),
        ("Sjogren's", "Sjogrens"),
        ("sjogrens", "Sjogrens"),
        ("Psoriatic Arthritis", "PsA"),
        ("PsA", "PsA"),
        ("psa", "PsA"),
        ("Unknown", "Mixed"),
        ("Mixed", "Mixed"),
        ("", "Mixed"),
    ])
    def test_aliases(self, alias, expected):
        assert normalize_disease(alias) == expected

    def test_none_returns_mixed(self):
        assert normalize_disease(None) == "Mixed"

    def test_unknown_string_returns_mixed(self):
        assert normalize_disease("not_a_disease") == "Mixed"


class TestDiseaseNorms:
    def test_all_cells_populated(self):
        """6 diseases x 10 features = 60 cells, all populated."""
        missing = [
            (d, f) for d in DISEASE_TYPES for f in ALL_FEATURES
            if f not in DISEASE_NORMS[d]
        ]
        assert not missing, f"Missing cells: {missing}"

    def test_all_norms_have_positive_std(self):
        for d in DISEASE_TYPES:
            for f in ALL_FEATURES:
                _, std = DISEASE_NORMS[d][f]
                assert std > 0, f"{d}/{f}: std must be positive"


class TestPopulationPercentile:
    def test_median_value_returns_50th_percentile(self):
        # RA fatigue mean is 5.8
        pct = get_population_percentile("RA", "fatigue", 5.8)
        assert abs(pct - 0.5) < 0.01

    def test_high_value_returns_high_percentile(self):
        # RA fatigue mean=5.8, std=2.0. Value=12 is way high.
        pct = get_population_percentile("RA", "fatigue", 12.0)
        assert pct > 0.99

    def test_low_value_returns_low_percentile(self):
        pct = get_population_percentile("RA", "fatigue", 0.0)
        assert pct < 0.01

    def test_disease_alias_works(self):
        """Should normalize alias before lookup."""
        pct1 = get_population_percentile("RA", "fatigue", 5.8)
        pct2 = get_population_percentile("Rheumatoid Arthritis", "fatigue", 5.8)
        assert pct1 == pct2

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            get_population_percentile("RA", "co2_level", 10.0)
