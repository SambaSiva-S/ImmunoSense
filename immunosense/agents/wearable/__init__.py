"""Agent 4 — Wearable Agent.

Captures HRV, sleep architecture, skin temperature, cardio, activity, and
respiratory data from wearables. Produces a 29-dim feature vector per night
plus single-metric and composite alerts.

Six-layer architecture::

    L1: Mock data generator (production: HealthKit/Health Connect/Fitbit/Oura)
    L2: Hampel artifact removal, Akima interpolation, ENMO sleep/wake
    L3: 29-feature engineering — HRV (10) / sleep (5) / temp (2) / cardio (3) /
        activity (2) / respiratory (2) / TADI (2) / composite (2) / quality (1)
    L4: Personal baselines via shared RobustBaselineTracker
    L5: Single-metric + named composite alerts (autoimmune_prodrome,
        acute_stress_response)
    L6: 29-dim output vector + wearable_stress_score + state labels
    L7: BaseAgent contract (process, get_output_vector, get_status)

Public API::

    >>> from immunosense.agents.wearable import WearableAgent, MockWearableGenerator
    >>> agent = WearableAgent()
    >>> gen = MockWearableGenerator()
    >>> night_df, rr = gen.generate_night(0, flare_state='normal')
    >>> result = agent.process({
    ...     'night_df': night_df,
    ...     'rr_intervals': rr,
    ...     'night_idx': 0,
    ... })
    >>> result.vector.shape
    (29,)
"""

from immunosense.agents.wearable.agent import WearableAgent
from immunosense.agents.wearable.alerts import (
    COMPOSITE_RULES,
    SINGLE_METRIC_RULES,
    evaluate_composite_alerts,
    evaluate_single_metric_alerts,
)
from immunosense.agents.wearable.hrv import compute_hrv_features
from immunosense.agents.wearable.pattern import PatternDetector
from immunosense.agents.wearable.pipeline import (
    apply_baseline_fillin,
    build_output_vector,
    engineer_features,
)
from immunosense.agents.wearable.preprocessing import (
    HampelFilter,
    akima_interpolate,
    enmo_sleep_wake,
)
from immunosense.agents.wearable.sleep import compute_sleep_features
from immunosense.agents.wearable.sources.mock import MockWearableGenerator
from immunosense.agents.wearable.states import derive_physiological_states
from immunosense.agents.wearable.stress import compute_wearable_stress_score
from immunosense.agents.wearable.tadi import compute_tadi
from immunosense.agents.wearable.types import FEATURE_NAMES, TRACKED_FEATURES

__all__ = [
    "WearableAgent",
    # Sources
    "MockWearableGenerator",
    # Pipeline
    "engineer_features",
    "apply_baseline_fillin",
    "build_output_vector",
    # Preprocessing
    "HampelFilter",
    "akima_interpolate",
    "enmo_sleep_wake",
    # Feature computation
    "compute_hrv_features",
    "compute_sleep_features",
    "compute_tadi",
    "compute_wearable_stress_score",
    "derive_physiological_states",
    # Alerts
    "SINGLE_METRIC_RULES",
    "COMPOSITE_RULES",
    "evaluate_single_metric_alerts",
    "evaluate_composite_alerts",
    # Patterns
    "PatternDetector",
    # Constants
    "FEATURE_NAMES",
    "TRACKED_FEATURES",
]
