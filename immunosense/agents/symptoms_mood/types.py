"""Data structures and feature constants for Agent 5.

Centralized here to avoid circular imports across the module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


# ============================================================
# Feature naming convention
# ============================================================

SYMPTOM_FEATURES = [
    "fatigue", "joint_pain", "brain_fog_severity", "gi_distress",
    "skin_severity", "sleep_severity", "energy_severity", "wellness_severity",
]
MOOD_FEATURES = ["phq8_score", "gad7_score"]
ALL_FEATURES = SYMPTOM_FEATURES + MOOD_FEATURES

DISEASE_TYPES = ["RA", "SLE", "MS", "Sjogrens", "PsA", "Mixed"]


# ============================================================
# Source contract
# ============================================================

@dataclass
class FetchedSymptoms:
    """Raw output from one symptom data source (voice, form, button, mock)."""

    fatigue: Optional[float] = None
    joint_pain: Optional[float] = None
    brain_fog_severity: Optional[float] = None
    gi_distress: Optional[float] = None
    skin_severity: Optional[float] = None
    sleep_severity: Optional[float] = None
    energy_severity: Optional[float] = None
    wellness_severity: Optional[float] = None
    phq8_score: Optional[float] = None
    gad7_score: Optional[float] = None
    emotional_valence: Optional[float] = None
    new_symptom_mentions: list = field(default_factory=list)
    explicit_flare: Optional[bool] = None
    explicit_flare_severity: Optional[float] = None
    confidence: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class SymptomDataSource(Protocol):
    """Protocol every symptom source implements."""

    def fetch(self, patient_id: str, target_date: str) -> FetchedSymptoms:
        ...


# ============================================================
# Daily summary - the agent's primary input
# ============================================================

@dataclass
class DailySymptomMoodSummary:
    """Layer 2 output - one per (patient, date), consumed by Layer 3."""

    date: str
    patient_id: str
    disease: str

    # Symptom severity (0-10)
    fatigue: Optional[float] = None
    joint_pain: Optional[float] = None
    brain_fog_severity: Optional[float] = None
    gi_distress: Optional[float] = None
    skin_severity: Optional[float] = None
    sleep_severity: Optional[float] = None
    energy_severity: Optional[float] = None
    wellness_severity: Optional[float] = None

    # Mood (validated scales)
    phq8_score: Optional[float] = None  # 0-24
    gad7_score: Optional[float] = None  # 0-21

    # Voice-extracted extras
    emotional_valence: Optional[float] = None  # -1 to +1
    new_symptom_mentions: list = field(default_factory=list)

    # Layer 1 outputs
    percentiles: dict = field(default_factory=dict)
    threshold_alerts: dict = field(default_factory=dict)

    # Provenance
    feature_confidence: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    overall_confidence: float = 0.0

    # Canonical Agent 5 outputs
    flare_score: float = 0.0
    flare_button_pressed: bool = False


# ============================================================
# Detection outputs
# ============================================================

@dataclass
class DetectedSymptomPattern:
    """One BH-FDR validated symptom-flare pattern."""

    feature: str
    lag_days: int
    effect_size: float
    p_value: float
    q_value: float
    n_observations: int
    confidence: str  # 'low' | 'medium' | 'high'


@dataclass
class HypothesisEvidence:
    """Single-hypothesis evidence exposed to Conductor for cross-agent corroboration.

    Includes ALL hypotheses tested, regardless of BH survival. Conductor uses
    these sub-threshold signals (e.g., raw_p=0.018 effects suppressed by BH
    at q=0.10) to make cross-agent corroboration decisions.
    """

    feature: str
    lag_days: int
    effect_size: float
    raw_p_value: float
    q_value: float
    n_observations: int
    biology_category: str  # 'predictive' | 'concurrent' | 'reactive'
    survives_fdr: bool      # True if q < fdr_target


# ============================================================
# Agent report
# ============================================================

@dataclass
class SymptomsMoodAgentReport:
    """Output from agent.analyze() - what the Conductor consumes."""

    n_days_observed: int
    n_flare_events: int
    n_hypotheses_tested: int
    baselines: dict
    today_anomaly_scores: Optional[dict]
    detected_patterns: list  # list of DetectedSymptomPattern
    tracker_activation: dict
