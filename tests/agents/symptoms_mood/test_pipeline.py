"""Tests for symptoms_mood.pipeline - flare score and daily summary."""

from __future__ import annotations

import pytest

from immunosense.agents.symptoms_mood import (
    CompositeSymptomSource,
    MockSymptomSource,
    StructuredFormSource,
    process_symptom_day,
)
from immunosense.agents.symptoms_mood.pipeline import (
    FLARE_WEIGHTS,
    compute_daily_flare_score,
)
from immunosense.agents.symptoms_mood.types import FetchedSymptoms


class TestFlareWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(FLARE_WEIGHTS.values()) - 1.0) < 1e-9

    def test_has_8_symptom_weights(self):
        assert len(FLARE_WEIGHTS) == 8


class TestComputeDailyFlareScore:
    def test_zero_inputs_yields_zero(self):
        f = FetchedSymptoms(
            fatigue=0, joint_pain=0, brain_fog_severity=0,
            gi_distress=0, skin_severity=0, sleep_severity=0,
            energy_severity=0, wellness_severity=0,
        )
        assert compute_daily_flare_score(f) == 0.0

    def test_max_inputs_yields_one(self):
        f = FetchedSymptoms(
            fatigue=10, joint_pain=10, brain_fog_severity=10,
            gi_distress=10, skin_severity=10, sleep_severity=10,
            energy_severity=10, wellness_severity=10,
        )
        assert compute_daily_flare_score(f) == 1.0

    def test_partial_inputs(self):
        """Score = sum(weight * value/10) for non-None features."""
        f = FetchedSymptoms(fatigue=5.0)  # only fatigue (weight 0.20)
        # 0.20 * (5.0 / 10) = 0.10
        assert abs(compute_daily_flare_score(f) - 0.10) < 1e-6

    def test_explicit_flare_floor(self):
        """Pressing flare button forces score >= 0.80."""
        f = FetchedSymptoms(explicit_flare=True)
        # No symptom data but flare button
        score = compute_daily_flare_score(f)
        assert score >= 0.80

    def test_explicit_flare_with_severity(self):
        """If severity provided, use max(weighted_score, floor, severity)."""
        f = FetchedSymptoms(explicit_flare=True, explicit_flare_severity=0.95)
        assert compute_daily_flare_score(f) >= 0.95

    def test_score_clipped_to_unit_interval(self):
        f = FetchedSymptoms(
            fatigue=15.0,  # invalid value > 10
            joint_pain=10.0,
        )
        # fatigue gets clipped to 10
        score = compute_daily_flare_score(f)
        assert 0.0 <= score <= 1.0


class TestProcessSymptomDay:
    def test_with_mock_source(self):
        mock = MockSymptomSource(disease="RA", seed_offset=1)
        composite = CompositeSymptomSource(mock_fallback=mock)
        summary = process_symptom_day(
            "patient_001", "2026-04-01", "RA",
            composite_source=composite,
        )
        assert summary.date == "2026-04-01"
        assert summary.patient_id == "patient_001"
        assert summary.disease == "RA"
        assert summary.fatigue is not None
        assert 0.0 <= summary.flare_score <= 1.0
        assert summary.overall_confidence == 0.0  # all synthetic

    def test_with_structured_form(self):
        composite = CompositeSymptomSource()
        form_data = {
            "fatigue": 7.0,
            "joint_pain": 5.0,
            "phq8_score": 12,
        }
        summary = process_symptom_day(
            "patient_001", "2026-04-01", "RA",
            form_data=form_data,
            composite_source=composite,
        )
        assert summary.fatigue == 7.0
        assert summary.joint_pain == 5.0
        # Confidence should reflect 3 structured features / 10 total
        assert summary.overall_confidence == 0.3

    def test_with_flare_event(self):
        composite = CompositeSymptomSource()
        summary = process_symptom_day(
            "p001", "2026-04-01", "Mixed",
            flare_event_severity=0.9,
            composite_source=composite,
        )
        assert summary.flare_button_pressed is True
        assert summary.flare_score >= 0.9

    def test_percentiles_populated(self):
        composite = CompositeSymptomSource(
            mock_fallback=MockSymptomSource(disease="RA"),
        )
        summary = process_symptom_day(
            "p001", "2026-04-01", "RA",
            composite_source=composite,
        )
        assert len(summary.percentiles) == 10  # all features
        for p in summary.percentiles.values():
            if p is not None:
                assert 0.0 <= p <= 1.0
