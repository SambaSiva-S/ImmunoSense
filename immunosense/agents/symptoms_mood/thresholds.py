"""Clinical threshold classifiers for symptom + mood features.

PHQ-8: standard depression severity categories (validated).
GAD-7: standard anxiety severity categories (validated).
Symptom severity: 0-10 scale binned into mild / moderate / severe.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.symptoms_mood.types import FetchedSymptoms


def classify_phq8(score: Optional[float]) -> Optional[str]:
    """Standard PHQ-8 severity categories (validated clinical scale).

    Categories:
        0-4:   minimal depression
        5-9:   mild
        10-14: moderate
        15-19: moderately_severe
        20+:   severe
    """
    if score is None:
        return None
    if score <= 4:
        return "minimal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    if score <= 19:
        return "moderately_severe"
    return "severe"


def classify_gad7(score: Optional[float]) -> Optional[str]:
    """Standard GAD-7 severity categories (validated clinical scale).

    Categories:
        0-4:   minimal anxiety
        5-9:   mild
        10-14: moderate
        15+:   severe
    """
    if score is None:
        return None
    if score <= 4:
        return "minimal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    return "severe"


def classify_symptom_severity(value: Optional[float]) -> Optional[str]:
    """Generic 0-10 symptom severity classifier.

    Categories:
        0-3:  mild
        4-6:  moderate
        7+:   severe
    """
    if value is None:
        return None
    if value <= 3:
        return "mild"
    if value <= 6:
        return "moderate"
    return "severe"


def classify_all_thresholds(fetched: FetchedSymptoms) -> dict:
    """Apply all threshold classifiers to a FetchedSymptoms struct."""
    return {
        "fatigue":      classify_symptom_severity(fetched.fatigue),
        "joint_pain":   classify_symptom_severity(fetched.joint_pain),
        "brain_fog":    classify_symptom_severity(fetched.brain_fog_severity),
        "gi_distress":  classify_symptom_severity(fetched.gi_distress),
        "skin":         classify_symptom_severity(fetched.skin_severity),
        "sleep":        classify_symptom_severity(fetched.sleep_severity),
        "energy":       classify_symptom_severity(fetched.energy_severity),
        "wellness":     classify_symptom_severity(fetched.wellness_severity),
        "phq8":         classify_phq8(fetched.phq8_score),
        "gad7":         classify_gad7(fetched.gad7_score),
    }
