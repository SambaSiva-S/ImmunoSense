"""Conductor-facing wellness signature.

Per-feature weighted 0-1 score representing overall patient wellness deviation
from personal baseline. Mirrors Agent 3's flare_signature pattern.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from immunosense.agents.symptoms_mood.types import DailySymptomMoodSummary


WELLNESS_EVIDENCE_WEIGHTS = {
    "fatigue":             0.20,
    "joint_pain":          0.20,
    "brain_fog_severity":  0.15,
    "wellness_severity":   0.10,
    "energy_severity":     0.10,
    "sleep_severity":      0.10,
    "gi_distress":         0.05,
    "skin_severity":       0.05,
    "phq8_score":          0.025,
    "gad7_score":          0.025,
}


def compute_wellness_signature(
    summary: DailySymptomMoodSummary,
    anomaly_scores: dict,
    detected_patterns: Optional[list] = None,
) -> dict:
    """0-1 wellness deviation score with contributing factors.

    Args:
        summary: Today's DailySymptomMoodSummary.
        anomaly_scores: Per-feature anomaly scores from agent's tracker.
        detected_patterns: Optional DetectedSymptomPattern list for trigger-boost.

    Returns:
        Dict with:
            score: 0-1 wellness deviation
            contributing_factors: list of per-feature contribution dicts
            clinical_alerts: features with moderate-or-severe threshold alerts
            data_quality_confidence: from summary.overall_confidence
            flare_score: from summary.flare_score
            flare_button_pressed: from summary.flare_button_pressed
    """
    detected_patterns = detected_patterns or []

    triggered_features = set()
    for p in detected_patterns:
        # Extract base feature name (e.g., 'fatigue (>p85)' -> 'fatigue')
        base = p.feature.split(" ")[0]
        if p.confidence in ("high", "medium"):
            triggered_features.add(base)

    contributions = []
    total_score = 0.0

    for feature, weight in WELLNESS_EVIDENCE_WEIGHTS.items():
        anomaly = anomaly_scores.get(feature, float("nan"))
        normalized = (
            0.0 if pd.isna(anomaly) else max(0.0, min(3.0, anomaly)) / 3.0
        )
        effective_weight = weight * (1.5 if feature in triggered_features else 1.0)
        total_score += normalized * effective_weight

        contributions.append({
            "feature": feature,
            "anomaly_score": float(anomaly) if not pd.isna(anomaly) else None,
            "normalized": normalized,
            "evidence_weight": weight,
            "effective_weight": effective_weight,
            "is_personal_trigger": feature in triggered_features,
        })

    total_score = min(1.0, total_score)
    contributions.sort(
        key=lambda c: c["normalized"] * c["effective_weight"], reverse=True
    )

    # Clinical alerts: moderate-or-worse threshold categories
    alert_categories = {"moderate", "severe", "moderately_severe"}
    clinical_alerts = [
        f for f, alert in summary.threshold_alerts.items()
        if alert in alert_categories
    ]

    return {
        "score": total_score,
        "contributing_factors": contributions,
        "clinical_alerts": clinical_alerts,
        "data_quality_confidence": summary.overall_confidence,
        "flare_score": summary.flare_score,
        "flare_button_pressed": summary.flare_button_pressed,
    }
