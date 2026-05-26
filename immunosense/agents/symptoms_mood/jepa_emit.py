"""JEPA embedding emission - 36-dim daily dense vector for the World Model.

Layout (indices):
    0-7:   8 symptom severities, normalized to [0, 1]
    8:     PHQ-8 score / 24
    9:     GAD-7 score / 21
    10-15: Disease one-hot (RA, SLE, MS, Sjogrens, PsA, Mixed)
    16-25: 10 features' percentiles (Layer 1 outputs)
    26:    Emotional valence shifted to [0, 1]
    27:    New symptom mentions count, capped at 5 then normalized
    28:    Flare score
    29:    Overall data confidence
    30:    Has structured source (binary)
    31:    Has voice source (binary)
    32:    Has flare button source (binary)
    33:    Has synthetic source (binary)
    34-35: Reserved (zero)
"""

from __future__ import annotations

import numpy as np

from immunosense.agents.symptoms_mood.types import (
    ALL_FEATURES,
    DISEASE_TYPES,
    SYMPTOM_FEATURES,
    DailySymptomMoodSummary,
)


JEPA_EMBEDDING_DIM = 36


def compute_jepa_embedding(summary: DailySymptomMoodSummary) -> np.ndarray:
    """Compute the 36-dim JEPA-compatible embedding for one daily summary.

    Args:
        summary: DailySymptomMoodSummary from process_symptom_day().

    Returns:
        numpy array of shape (36,) with float32 dtype.
    """
    vec = np.zeros(JEPA_EMBEDDING_DIM, dtype=np.float32)

    # Indices 0-7: Symptom severities normalized to [0,1]
    for i, feat in enumerate(SYMPTOM_FEATURES):
        v = getattr(summary, feat, None)
        vec[i] = (max(0.0, min(10.0, v)) / 10.0) if v is not None else 0.0

    # Index 8-9: PHQ-8 / 24, GAD-7 / 21
    vec[8] = (summary.phq8_score / 24.0) if summary.phq8_score is not None else 0.0
    vec[9] = (summary.gad7_score / 21.0) if summary.gad7_score is not None else 0.0

    # Indices 10-15: Disease one-hot
    if summary.disease in DISEASE_TYPES:
        vec[10 + DISEASE_TYPES.index(summary.disease)] = 1.0

    # Indices 16-25: 10 feature percentiles
    for i, feat in enumerate(ALL_FEATURES):
        p = summary.percentiles.get(feat)
        vec[16 + i] = float(p) if p is not None else 0.0

    # Index 26: Emotional valence shifted to [0,1]
    vec[26] = (
        (summary.emotional_valence + 1.0) / 2.0
        if summary.emotional_valence is not None
        else 0.5
    )

    # Index 27: New symptom mentions count (capped at 5, normalized)
    vec[27] = min(1.0, len(summary.new_symptom_mentions) / 5.0)

    # Index 28: Flare score (already 0-1)
    vec[28] = float(summary.flare_score)

    # Index 29: Overall confidence (already 0-1)
    vec[29] = float(summary.overall_confidence)

    # Indices 30-33: Source flags (binary)
    sources_seen = set(summary.feature_confidence.values())
    vec[30] = 1.0 if "structured" in sources_seen else 0.0
    vec[31] = 1.0 if "voice_extracted" in sources_seen else 0.0
    vec[32] = 1.0 if "flare_button" in sources_seen else 0.0
    vec[33] = 1.0 if "synthetic" in sources_seen else 0.0

    # Indices 34-35: Reserved (left as 0)

    return vec
