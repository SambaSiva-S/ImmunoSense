"""Agent 3 — Environment Agent.

Captures environmental triggers (PM2.5, ozone, UV, barometric pressure, pollen)
that affect autoimmune disease. Three-layer architecture:

    Layer 1: Geographic regional baseline (EPA/NOAA reference tables)
    Layer 2: API ingestion (AirNow + OpenWeather + Google Pollen) -> daily summary
    Layer 3: Per-patient adaptation (robust tracker + BH FDR trigger detector)

Public API:
    >>> from immunosense.agents.environment import EnvironmentAgent
    >>> agent = EnvironmentAgent(patient_id='patient_001', zip_code='28202')
    >>> # ... build daily_summary via process_environment_day ...
    >>> agent.observe(daily_summary)
    >>> agent.observe_flare('2026-03-15', severity=0.7)
    >>> report = agent.analyze()
"""

from immunosense.agents.environment.agent import EnvironmentAgent
from immunosense.agents.environment.pipeline import (
    compute_flare_signature,
    process_environment_day,
)
from immunosense.agents.environment.sources.composite import CompositeEnvironmentSource
from immunosense.agents.environment.sources.mock import MockEnvironmentSource
from immunosense.agents.environment.types import (
    DailyEnvironmentSummary,
    DetectedPattern,
    EnvironmentAgentReport,
    FetchedFeatures,
)

__all__ = [
    "EnvironmentAgent",
    "process_environment_day",
    "compute_flare_signature",
    "CompositeEnvironmentSource",
    "MockEnvironmentSource",
    "DailyEnvironmentSummary",
    "DetectedPattern",
    "EnvironmentAgentReport",
    "FetchedFeatures",
]
