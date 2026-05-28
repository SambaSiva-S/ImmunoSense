"""Data structures for Agent 1 (Biomarker).

Centralized to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Input
# ============================================================

@dataclass
class BiomarkerReading:
    """One biomarker reading from a patient.

    Fields beyond the named ones are stored in `extra`. The agent expects
    keys matching ALL_VALUE_FEATURES (Age, ESR, CRP, RF, Anti-CCP, C3, C4,
    Gender_enc, HLA-B27_enc, ANA_enc, Anti-Ro_enc, Anti-La_enc,
    Anti-dsDNA_enc, Anti-Sm_enc) plus optional trigger booleans.
    """

    day: int = 0
    CRP: Optional[float] = None
    ESR: Optional[float] = None
    RF: Optional[float] = None
    Anti_CCP: Optional[float] = None  # Anti-CCP
    C3: Optional[float] = None
    C4: Optional[float] = None
    is_flare: bool = False
    extra: dict = field(default_factory=dict)


# ============================================================
# Layer 1: Population percentile
# ============================================================

@dataclass
class Layer1Output:
    """Layer 1: where does this CRP value fall in the demographic population?"""

    biomarker: str
    value: float
    population_percentile: float           # 0-1 (e.g., 0.95 = 95th percentile)
    interpretation: str                    # "NORMAL" | "ELEVATED" | "ALARMING"


# ============================================================
# Layer 2: Disease intelligence (fused 3-pillar prediction)
# ============================================================

@dataclass
class Layer2Output:
    """Layer 2: which disease, with what confidence, and why?"""

    prediction: str                        # Predicted disease class name
    confidence: float                      # Fused probability (0-1)
    probabilities: dict                    # {disease: prob} for all 7 classes
    pillar_a_similarities: dict            # {disease: cosine_sim} from contrastive
    pillar_b_probabilities: dict           # {disease: prob} from LightGBM
    pillar_c_probabilities: dict           # {disease: prob} from XGBoost
    pillars_agree: bool                    # All 3 pillars picked the same class?
    contrastive_embedding: Optional[list] = None  # 128-dim, if encoder available
    top_drivers: list = field(default_factory=list)  # SHAP feature drivers


# ============================================================
# Layer 3: Personal adaptation
# ============================================================

@dataclass
class DetectedTriggerPattern:
    """One detected trigger -> biomarker correlation at a specific lag."""

    trigger: str
    biomarker: str
    lag_readings: int
    correlation: float
    effect_size: float
    effect_pct: float
    n_exposed: int
    strength: str                          # "STRONG" | "MODERATE"


@dataclass
class Layer3Output:
    """Layer 3: how does this reading compare to the patient's own baseline?"""

    has_personal_data: bool
    readings_count: int
    personal_weight: float                 # 0.0 -> 0.8 cold-start ramp
    biomarkers: dict                       # Per-biomarker context dict
    patterns: list = field(default_factory=list)  # list[DetectedTriggerPattern]
    flare_rule: Optional[dict] = None      # Per-biomarker flare thresholds


# ============================================================
# Top-level agent report
# ============================================================

@dataclass
class BiomarkerAgentReport:
    """Composite result from BiomarkerAgent.process()."""

    timestamp: int                         # day of reading
    layer1: dict                           # {biomarker_name: Layer1Output dict}
    layer2: Optional[Layer2Output]
    layer3: Layer3Output
    alerts: list                           # list[str] human-readable alerts
