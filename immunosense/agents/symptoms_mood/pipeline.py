"""Layer 2 pipeline - process_symptom_day and compute_daily_flare_score.

process_symptom_day orchestrates:
    1. Source assembly (voice/form/button/mock)
    2. Disease-stratified percentile lookup
    3. Clinical threshold classification
    4. Daily flare score computation
    5. Overall confidence reporting
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.symptoms_mood.norms import (
    get_population_percentile,
    normalize_disease,
)
from immunosense.agents.symptoms_mood.sources.composite import CompositeSymptomSource
from immunosense.agents.symptoms_mood.sources.voice import VoiceTranscriptSource
from immunosense.agents.symptoms_mood.thresholds import classify_all_thresholds
from immunosense.agents.symptoms_mood.types import (
    ALL_FEATURES,
    DailySymptomMoodSummary,
    FetchedSymptoms,
)


# Weights for compute_daily_flare_score. Sum to 1.0.
FLARE_WEIGHTS = {
    "fatigue":             0.20,
    "joint_pain":          0.20,
    "brain_fog_severity":  0.15,
    "gi_distress":         0.10,
    "skin_severity":       0.10,
    "sleep_severity":      0.10,
    "energy_severity":     0.10,
    "wellness_severity":   0.05,
}
assert abs(sum(FLARE_WEIGHTS.values()) - 1.0) < 1e-9


# Confidence stamps that count as "real" (non-synthetic)
_NON_SYNTHETIC = {"structured", "voice_extracted", "flare_button"}


def compute_daily_flare_score(
    fetched: FetchedSymptoms,
    button_override_floor: float = 0.80,
) -> float:
    """Compute the canonical Agent 5 -> Conductor flare score.

    Args:
        fetched: Daily symptom data.
        button_override_floor: If patient pressed flare button, ensure score >=
            this value (regardless of computed weighted score).

    Returns:
        Flare score in [0.0, 1.0].
    """
    score = 0.0
    for feat, w in FLARE_WEIGHTS.items():
        v = getattr(fetched, feat, None)
        if v is not None:
            score += w * (max(0.0, min(10.0, v)) / 10.0)

    if fetched.explicit_flare:
        explicit_sev = (
            fetched.explicit_flare_severity
            if fetched.explicit_flare_severity is not None
            else 0.85
        )
        score = max(score, button_override_floor, explicit_sev)

    return min(1.0, max(0.0, score))


def process_symptom_day(
    patient_id: str,
    target_date: str,
    disease: str,
    transcript: Optional[str] = None,
    form_data: Optional[dict] = None,
    flare_event_severity: Optional[float] = None,
    composite_source: Optional[CompositeSymptomSource] = None,
) -> DailySymptomMoodSummary:
    """Layer 2 pipeline: assemble + classify + score one day.

    Args:
        patient_id: Patient identifier.
        target_date: ISO date string 'YYYY-MM-DD'.
        disease: Disease name (canonical or alias).
        transcript: Optional voice transcript text.
        form_data: Optional structured form data.
        flare_event_severity: Optional flare button severity (0-1).
        composite_source: Override source. If None, creates one with VoiceTranscriptSource.

    Returns:
        DailySymptomMoodSummary ready for agent.observe().
    """
    if composite_source is None:
        composite_source = CompositeSymptomSource(voice=VoiceTranscriptSource())

    fetched = composite_source.assemble(
        patient_id=patient_id,
        target_date=target_date,
        transcript=transcript,
        form_data=form_data,
        flare_event_severity=flare_event_severity,
    )

    norm_disease = normalize_disease(disease)

    # Layer 1 percentile lookups
    percentiles = {}
    for feat in ALL_FEATURES:
        v = getattr(fetched, feat)
        percentiles[feat] = (
            None if v is None
            else get_population_percentile(norm_disease, feat, v)
        )

    threshold_alerts = classify_all_thresholds(fetched)
    flare_score = compute_daily_flare_score(fetched)

    real_count = sum(
        1 for feat in ALL_FEATURES
        if fetched.confidence.get(feat) in _NON_SYNTHETIC
    )
    overall_confidence = real_count / 10.0

    return DailySymptomMoodSummary(
        date=target_date,
        patient_id=patient_id,
        disease=norm_disease,
        fatigue=fetched.fatigue,
        joint_pain=fetched.joint_pain,
        brain_fog_severity=fetched.brain_fog_severity,
        gi_distress=fetched.gi_distress,
        skin_severity=fetched.skin_severity,
        sleep_severity=fetched.sleep_severity,
        energy_severity=fetched.energy_severity,
        wellness_severity=fetched.wellness_severity,
        phq8_score=fetched.phq8_score,
        gad7_score=fetched.gad7_score,
        emotional_valence=fetched.emotional_valence,
        new_symptom_mentions=list(fetched.new_symptom_mentions),
        percentiles=percentiles,
        threshold_alerts=threshold_alerts,
        feature_confidence=dict(fetched.confidence),
        sources=dict(fetched.sources),
        errors=list(fetched.errors),
        overall_confidence=overall_confidence,
        flare_score=flare_score,
        flare_button_pressed=bool(fetched.explicit_flare),
    )
