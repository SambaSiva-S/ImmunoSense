"""Wearable data sources.

Currently only the deterministic Mock generator. Production integration with
HealthKit / Health Connect / Fitbit / Oura is deferred post-extraction.
"""

from immunosense.agents.wearable.sources.mock import MockWearableGenerator

__all__ = ["MockWearableGenerator"]
