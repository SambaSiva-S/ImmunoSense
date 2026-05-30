"""Symptom builder — raw UI input -> DailySymptomMoodSummary.

The simplest builder: the symptoms agent's domain object maps near-directly from
what the user logs (tap sliders, text, voice transcript). One or more
SymptomLog rows for a (user, date) are folded into a single
DailySymptomMoodSummary the agent consumes.

When multiple logs exist for the same day (e.g. a morning tap + an evening
voice note), we take the most severe non-null value per field — a flare day is
defined by its worst point, not its average. This is a deliberate, documented
aggregation choice (overridable later).
"""

from __future__ import annotations

from typing import Iterable, Optional

from immunosense.agents.symptoms_mood.types import DailySymptomMoodSummary


# DailySymptomMoodSummary symptom fields populated from SymptomLog rows.
_SYMPTOM_FIELDS = (
    "fatigue",
    "joint_pain",
    "brain_fog_severity",
    "gi_distress",
    "skin_severity",
    "sleep_severity",
    "energy_severity",
    "wellness_severity",
)
_MOOD_FIELDS = ("phq8_score", "gad7_score")


def build_symptom_summary(
    logs: Iterable,
    date: str,
    user_id: str,
    disease: str,
    flare_button_pressed: bool = False,
    explicit_flare_severity: Optional[float] = None,
) -> DailySymptomMoodSummary:
    """Fold one day's SymptomLog rows into a DailySymptomMoodSummary.

    Args:
        logs: iterable of SymptomLog ORM rows (or any objects exposing the
            symptom/mood fields as attributes) for this (user, date).
        date: ISO date string.
        user_id: the user this summary belongs to (kept as the agent's
            internal patient_id field — see note below).
        disease: the user's condition (e.g. "SLE").
        flare_button_pressed: whether the flare button was pressed this day.
        explicit_flare_severity: optional severity if the user marked a flare.

    Returns:
        DailySymptomMoodSummary ready for the symptoms agent.

    Note on identity: DailySymptomMoodSummary has a `patient_id` field (the
    agent's internal domain field, intentionally unchanged in the rename). We
    pass user_id into it because that is the value the caller has; the agent
    uses it only as its own memory key and never emits it.
    """
    logs = list(logs)

    # Aggregate: most-severe (max) non-null value per field across the day's logs.
    agg: dict = {}
    for field in _SYMPTOM_FIELDS + _MOOD_FIELDS:
        values = [
            getattr(log, field) for log in logs
            if getattr(log, field, None) is not None
        ]
        agg[field] = max(values) if values else None

    # Collect any free-text mentions for the voice-extracted-extras field.
    mentions = [
        getattr(log, "free_text") for log in logs
        if getattr(log, "free_text", None)
    ]

    # Track which sources contributed (provenance).
    sources = {}
    for log in logs:
        src = getattr(log, "source", "tap")
        sources[src] = sources.get(src, 0) + 1

    summary = DailySymptomMoodSummary(
        date=date,
        patient_id=user_id,  # agent-internal field; see docstring note
        disease=disease,
        fatigue=agg["fatigue"],
        joint_pain=agg["joint_pain"],
        brain_fog_severity=agg["brain_fog_severity"],
        gi_distress=agg["gi_distress"],
        skin_severity=agg["skin_severity"],
        sleep_severity=agg["sleep_severity"],
        energy_severity=agg["energy_severity"],
        wellness_severity=agg["wellness_severity"],
        phq8_score=agg["phq8_score"],
        gad7_score=agg["gad7_score"],
        new_symptom_mentions=mentions,
        sources=sources,
        flare_button_pressed=bool(flare_button_pressed),
    )
    if explicit_flare_severity is not None:
        summary.flare_score = float(explicit_flare_severity)
    return summary
