"""Data structures for Agent 2 (Dietary).

Centralized to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Layer 2: per-meal processing outputs
# ============================================================

@dataclass
class FoodMatch:
    """Result of fuzzy-matching an extracted food name to NHANES food code."""

    extracted_name: str
    nhanes_code: int
    nhanes_description: str
    match_score: float


@dataclass
class MealResult:
    """Output of processing one meal: nutrients + triggers + glycemic load.

    Note: DII is computed at the DAILY level (via rollup_day), not per-meal.
    Computing DII per meal then averaging is mathematically incorrect because
    DII uses non-linear z-score transforms on intake quantities.
    """

    input_text: str
    timestamp: Optional[str]
    extracted_foods: list  # list[ExtractedFood]
    matches: list           # list[Optional[FoodMatch]]
    nutrients: dict
    triggers: dict          # {dairy: bool, gluten: bool, nightshade: bool, upf: bool}
    glycemic_load: float
    n_foods_extracted: int
    n_foods_matched: int
    n_foods_unmatched: int
    warnings: list = field(default_factory=list)


# ============================================================
# Layer 2: daily rollup (the 12-feature vector for Layer 3)
# ============================================================

@dataclass
class DailyRollup:
    """One day's aggregated dietary features, ready for the Layer 3 tracker.

    The 12-feature Layer 3 vector:
        Continuous (5): dii_score, omega6_omega3_ratio, glycemic_load,
                        sodium_mg, alcohol_g
        Boolean (4):    gluten_present, dairy_present, nightshade_present, upf_present
        Time-based (3): first_meal_timestamp, last_meal_timestamp,
                        longest_intraday_gap_hours

    Note: overnight_fast_hours (the 6th continuous feature) is computed by
    OvernightFastTracker.compute() in Layer 3, NOT here, because it requires
    yesterday's last_meal_timestamp.
    """

    date: str
    meal_count: int
    daily_nutrients: dict

    # Layer-3-facing feature vector
    dii_score: float
    omega6_omega3_ratio: float
    glycemic_load: float
    sodium_mg: float
    alcohol_g: float

    # Timestamps for Layer 3 to compute overnight fasting
    first_meal_timestamp: Optional[str]
    last_meal_timestamp: Optional[str]
    longest_intraday_gap_hours: float

    gluten_present: bool
    dairy_present: bool
    nightshade_present: bool
    upf_present: bool

    feature_confidence: dict
    daily_dii_percentile: Optional[float]
    meal_results: list


# ============================================================
# Layer 3: detection outputs
# ============================================================

@dataclass
class DetectedPattern:
    """One detected feature -> flare relationship at a specific lag."""

    feature: str
    lag_days: int
    effect_size: float    # Pearson r OR binarized mean-diff OR boolean mean-diff
    p_value: float
    n_observations: int
    confidence: str       # 'high' if p<0.01, 'medium' if p<0.05, 'low' if p<0.10


# ============================================================
# Layer 3: agent report
# ============================================================

@dataclass
class DietaryAgentReport:
    """Output from DietaryAgent.analyze() - what the Conductor consumes."""

    n_days_observed: int
    n_flare_events: int
    baselines: dict
    today_anomaly_scores: Optional[dict]
    today_overnight_fast: Optional[float]
    detected_patterns: list  # list[DetectedPattern]
    tracker_activation: dict
