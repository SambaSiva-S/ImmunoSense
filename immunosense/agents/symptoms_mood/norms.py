"""Layer 1 - Disease-stratified population baselines.

Disease-specific (mean, std) for each feature, derived from autoimmune cohort
literature (LUMINA, CORRONA, NARCOMS approximations).

The 'Mixed' PHQ-8 baseline is OVERRIDDEN with real NHANES values via
``nhanes.update_disease_norms_with_nhanes()`` when available. All other
norms remain literature approximations until real-cohort calibration (v2).
"""

from __future__ import annotations

from typing import Optional

from scipy.stats import norm

from immunosense.agents.symptoms_mood.types import ALL_FEATURES, DISEASE_TYPES


# Disease-stratified (mean, std) norms per feature.
# Values are literature approximations from autoimmune cohort studies.
# 'Mixed' PHQ-8 gets overridden with NHANES real data when available.
DISEASE_NORMS = {
    "RA": {
        "fatigue":             (5.8, 2.0),  "joint_pain":         (5.5, 2.3),
        "brain_fog_severity":  (3.2, 2.0),  "gi_distress":        (2.5, 1.8),
        "skin_severity":       (2.0, 1.5),  "sleep_severity":     (4.5, 2.2),
        "energy_severity":     (5.5, 2.1),  "wellness_severity":  (5.0, 2.0),
        "phq8_score":          (7.5, 4.5),  "gad7_score":         (6.0, 4.0),
    },
    "SLE": {
        "fatigue":             (6.4, 2.1),  "joint_pain":         (4.8, 2.5),
        "brain_fog_severity":  (5.1, 2.2),  "gi_distress":        (3.0, 2.0),
        "skin_severity":       (4.5, 2.5),  "sleep_severity":     (5.0, 2.3),
        "energy_severity":     (6.0, 2.2),  "wellness_severity":  (5.5, 2.1),
        "phq8_score":          (8.0, 4.8),  "gad7_score":         (6.5, 4.2),
    },
    "MS": {
        "fatigue":             (6.1, 2.2),  "joint_pain":         (3.5, 2.0),
        "brain_fog_severity":  (5.5, 2.0),  "gi_distress":        (3.5, 2.2),
        "skin_severity":       (2.0, 1.5),  "sleep_severity":     (5.2, 2.3),
        "energy_severity":     (6.2, 2.0),  "wellness_severity":  (5.3, 2.2),
        "phq8_score":          (7.8, 4.6),  "gad7_score":         (6.2, 4.1),
    },
    "Sjogrens": {
        "fatigue":             (6.8, 2.0),  "joint_pain":         (4.5, 2.4),
        "brain_fog_severity":  (4.8, 2.1),  "gi_distress":        (4.0, 2.3),
        "skin_severity":       (3.0, 2.0),  "sleep_severity":     (5.5, 2.2),
        "energy_severity":     (6.5, 2.0),  "wellness_severity":  (5.8, 2.0),
        "phq8_score":          (8.2, 4.7),  "gad7_score":         (6.8, 4.3),
    },
    "PsA": {
        "fatigue":             (5.5, 2.1),  "joint_pain":         (5.8, 2.4),
        "brain_fog_severity":  (3.0, 1.8),  "gi_distress":        (2.5, 1.8),
        "skin_severity":       (5.5, 2.5),  "sleep_severity":     (4.8, 2.2),
        "energy_severity":     (5.2, 2.1),  "wellness_severity":  (4.8, 2.0),
        "phq8_score":          (7.0, 4.3),  "gad7_score":         (5.8, 3.9),
    },
    "Mixed": {
        "fatigue":             (5.8, 2.3),  "joint_pain":         (4.5, 2.5),
        "brain_fog_severity":  (4.0, 2.3),  "gi_distress":        (3.2, 2.2),
        "skin_severity":       (3.5, 2.5),  "sleep_severity":     (5.0, 2.4),
        "energy_severity":     (5.8, 2.2),  "wellness_severity":  (5.2, 2.2),
        # PHQ-8 will be overridden with NHANES values via nhanes.update_disease_norms_with_nhanes
        "phq8_score":          (7.5, 4.7),  "gad7_score":         (6.2, 4.2),
    },
}


# Sanity check: every disease has every feature
_missing = [(d, f) for d in DISEASE_TYPES for f in ALL_FEATURES
            if f not in DISEASE_NORMS[d]]
assert not _missing, f"Missing norm cells: {_missing}"


# Disease name aliases for normalization
DISEASE_ALIASES = {
    "rheumatoid arthritis": "RA",     "rheumatoid": "RA",         "ra": "RA",
    "lupus": "SLE",                    "systemic lupus": "SLE",    "sle": "SLE",
    "multiple sclerosis": "MS",        "ms": "MS",
    "sjogren's": "Sjogrens",           "sjogrens": "Sjogrens",     "sjogren": "Sjogrens",
    "psoriatic arthritis": "PsA",      "psa": "PsA",
    "unknown": "Mixed",                "multiple": "Mixed",         "": "Mixed",
}


def normalize_disease(disease: Optional[object]) -> str:
    """Convert various disease aliases to canonical DISEASE_TYPES string."""
    if disease is None:
        return "Mixed"
    key = str(disease).strip().lower()
    if key in DISEASE_ALIASES:
        return DISEASE_ALIASES[key]
    for d in DISEASE_TYPES:
        if d.lower() == key:
            return d
    return "Mixed"


def get_population_percentile(disease: str, feature: str, value: float) -> float:
    """Place observed value on disease-stratified population CDF.

    Args:
        disease: Disease name (canonical or alias).
        feature: One of ALL_FEATURES.
        value: Observed value (typically 0-10 for severity, 0-24 for PHQ-8).

    Returns:
        Percentile in [0, 1] under assumed normal CDF.

    Raises:
        ValueError: If feature is unknown for the resolved disease.
    """
    norm_d = normalize_disease(disease)
    if feature not in DISEASE_NORMS[norm_d]:
        raise ValueError(f"Unknown feature '{feature}'")
    mean, std = DISEASE_NORMS[norm_d][feature]
    if std <= 0:
        return 0.5
    return float(norm.cdf((value - mean) / std))
