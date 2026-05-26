"""Data structures for Agent 3 (Environment).

All dataclasses Agent 3 produces or consumes live here. Other modules import
from this file to avoid circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class FetchedFeatures:
    """Raw output from one environmental data source (one API call).

    All five features are Optional because data sources may return partial
    information (e.g., AirNow returns PM2.5 + ozone but not pollen).
    """

    pm25_ug_m3: Optional[float] = None
    ozone_ppb: Optional[float] = None
    uv_index: Optional[float] = None
    barometric_change_kpa: Optional[float] = None
    pollen_index: Optional[float] = None

    # Provenance fields
    confidence: dict = field(default_factory=dict)  # feature -> 'real' | 'synthetic'
    sources: dict = field(default_factory=dict)     # feature -> source name string
    errors: list = field(default_factory=list)      # list of error messages


class EnvironmentDataSource(Protocol):
    """Protocol every environmental data source implements."""

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Fetch features for given location and date. Returns FetchedFeatures."""
        ...


@dataclass
class DailyEnvironmentSummary:
    """Layer 2 output - one per (patient, date), consumed by Layer 3.

    The agent's `observe()` method takes this as input.
    """

    date: str
    location: dict  # {'zip_code': str, 'lat': float, 'lng': float, 'region': str, 'season': str}

    # The five canonical environmental features
    pm25_ug_m3: Optional[float] = None
    ozone_ppb: Optional[float] = None
    uv_index: Optional[float] = None
    barometric_change_kpa: Optional[float] = None
    pollen_index: Optional[float] = None

    # Layer 1 outputs
    percentiles: dict = field(default_factory=dict)        # feature -> percentile [0,1]
    threshold_alerts: dict = field(default_factory=dict)   # feature -> EPA category

    # Provenance
    feature_confidence: dict = field(default_factory=dict)  # feature -> 'real' | 'synthetic'
    sources: dict = field(default_factory=dict)             # feature -> source string
    errors: list = field(default_factory=list)
    overall_confidence: float = 0.0                          # fraction of features that are 'real'


@dataclass
class DetectedPattern:
    """One BH FDR-validated environmental trigger pattern detected for this patient."""

    feature: str          # 'pm25' | 'ozone' | 'uv' | 'barometric' | 'pollen'
    lag_days: int         # how many days before flare the exposure occurred
    effect_size: float    # observed mean difference (exposed - unexposed flare scores)
    p_value: float        # raw permutation p-value
    q_value: float        # BH-corrected q-value
    confidence: str       # 'low' | 'medium' | 'high' (based on n_exposed)
    n_exposed: int        # number of days the feature exceeded the trigger threshold
    n_total: int          # total days in observation window


@dataclass
class EnvironmentAgentReport:
    """Output from agent.analyze() - what the Conductor consumes."""

    n_days_observed: int
    n_flare_events: int
    today_percentiles: Optional[dict] = None
    today_threshold_alerts: Optional[dict] = None
    detected_patterns: list = field(default_factory=list)  # list of DetectedPattern
    tracker_activation: dict = field(default_factory=dict)
    overall_data_confidence: float = 0.0
