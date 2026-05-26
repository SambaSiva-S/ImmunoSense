"""Tests for symptoms_mood.thresholds - clinical classifiers."""

from __future__ import annotations

import pytest

from immunosense.agents.symptoms_mood.thresholds import (
    classify_all_thresholds,
    classify_gad7,
    classify_phq8,
    classify_symptom_severity,
)
from immunosense.agents.symptoms_mood.types import FetchedSymptoms


class TestPHQ8Classification:
    @pytest.mark.parametrize("score,expected", [
        (0, "minimal"),
        (4, "minimal"),
        (5, "mild"),
        (9, "mild"),
        (10, "moderate"),
        (14, "moderate"),
        (15, "moderately_severe"),
        (19, "moderately_severe"),
        (20, "severe"),
        (24, "severe"),
    ])
    def test_classification(self, score, expected):
        assert classify_phq8(score) == expected

    def test_none_returns_none(self):
        assert classify_phq8(None) is None


class TestGAD7Classification:
    @pytest.mark.parametrize("score,expected", [
        (0, "minimal"),
        (4, "minimal"),
        (5, "mild"),
        (9, "mild"),
        (10, "moderate"),
        (14, "moderate"),
        (15, "severe"),
        (21, "severe"),
    ])
    def test_classification(self, score, expected):
        assert classify_gad7(score) == expected


class TestSymptomSeverity:
    @pytest.mark.parametrize("value,expected", [
        (0, "mild"),
        (3, "mild"),
        (4, "moderate"),
        (6, "moderate"),
        (7, "severe"),
        (10, "severe"),
    ])
    def test_classification(self, value, expected):
        assert classify_symptom_severity(value) == expected


class TestClassifyAllThresholds:
    def test_full_set(self):
        f = FetchedSymptoms(
            fatigue=7.0,
            joint_pain=5.0,
            brain_fog_severity=8.0,
            gi_distress=2.0,
            skin_severity=4.0,
            sleep_severity=6.0,
            energy_severity=5.0,
            wellness_severity=3.0,
            phq8_score=15,
            gad7_score=8,
        )
        r = classify_all_thresholds(f)
        assert r["fatigue"] == "severe"
        assert r["joint_pain"] == "moderate"
        assert r["brain_fog"] == "severe"
        assert r["gi_distress"] == "mild"
        assert r["phq8"] == "moderately_severe"
        assert r["gad7"] == "mild"

    def test_missing_returns_none(self):
        r = classify_all_thresholds(FetchedSymptoms())
        for v in r.values():
            assert v is None
