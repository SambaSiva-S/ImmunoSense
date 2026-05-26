"""Agent 2 — Dietary Agent.

Three-layer architecture:
    Layer 1: NHANES quantile regression models for DII percentiles
             (trained offline via layer1_train; runtime loads pickled artifacts)
    Layer 2: Per-meal pipeline (text -> nutrients + triggers + GL) and daily
             rollup into a 12-feature vector
    Layer 3: DietaryAgent — overnight fast tracker + per-feature baselines +
             trigger detection with permutation tests

The agent emits a 12-feature daily vector to the Conductor:
    - 6 continuous: dii_score, omega6_omega3_ratio, glycemic_load,
                    sodium_mg, alcohol_g, overnight_fast_hours
    - 4 boolean triggers: gluten_present, dairy_present, nightshade_present, upf_present
    - 2 metadata: dii_percentile, longest_intraday_gap_hours

Public API::

    >>> from immunosense.agents.dietary import (
    ...     DietaryAgent, process_meal, rollup_day, MockExtractor
    ... )
    >>> agent = DietaryAgent(patient_id='p001')
    >>> meal1 = process_meal('Two eggs and toast', extractor=MockExtractor(),
    ...                       timestamp='2026-04-01T08:00:00')
    >>> meal2 = process_meal('Chicken with rice', extractor=MockExtractor(),
    ...                       timestamp='2026-04-01T12:30:00')
    >>> rollup = rollup_day([meal1, meal2], date='2026-04-01',
    ...                      age=45, sex=2, bmi=28)
    >>> agent.observe(rollup)
    {'overnight_fast_hours': nan, 'anomaly_scores': {...}}
"""

from immunosense.agents.dietary.agent import DietaryAgent
from immunosense.agents.dietary.detector import DietaryTriggerDetector
from immunosense.agents.dietary.dii import (
    compute_dii_row,
    compute_meal_dii,
    get_dii_percentile,
    get_population_dii,
    load_layer1_artifacts,
)
from immunosense.agents.dietary.pipeline import (
    compute_intraday_gap_and_timestamps,
    process_meal,
    rollup_day,
)
from immunosense.agents.dietary.sources import (
    ClaudeHaikuExtractor,
    Extractor,
    ExtractedFood,
    ExtractedMeal,
    MockExtractor,
    make_default_extractor,
)
from immunosense.agents.dietary.trackers import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
    DietaryRobustTracker,
    OvernightFastTracker,
)
from immunosense.agents.dietary.triggers import classify_food_triggers
from immunosense.agents.dietary.types import (
    DailyRollup,
    DetectedPattern,
    DietaryAgentReport,
    FoodMatch,
    MealResult,
)

__all__ = [
    # Main agent
    "DietaryAgent",
    # Pipeline
    "process_meal",
    "rollup_day",
    "compute_intraday_gap_and_timestamps",
    # DII
    "compute_dii_row",
    "compute_meal_dii",
    "get_dii_percentile",
    "get_population_dii",
    "load_layer1_artifacts",
    # Extractors
    "Extractor",
    "ExtractedFood",
    "ExtractedMeal",
    "MockExtractor",
    "ClaudeHaikuExtractor",
    "make_default_extractor",
    # Trackers + detector
    "OvernightFastTracker",
    "DietaryRobustTracker",
    "DietaryTriggerDetector",
    "CONTINUOUS_FEATURES",
    "BOOLEAN_TRIGGERS",
    # Triggers
    "classify_food_triggers",
    # Types
    "DailyRollup",
    "MealResult",
    "FoodMatch",
    "DetectedPattern",
    "DietaryAgentReport",
]
