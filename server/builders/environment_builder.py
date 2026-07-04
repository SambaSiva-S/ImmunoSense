"""Environment builder — user's stored coordinates -> DailyEnvironmentSummary.

The environment Layer 2 pipeline (process_environment_day) is keyed on ZIP, but
we store lat/lng on the profile (resolved from the user's city/zip at save time).
This builder mirrors that pipeline but takes coordinates directly, so the
Environment agent can be fed from any stored location.

Uses CompositeEnvironmentSource: real APIs (AirNow air quality, Google Pollen,
OpenWeather pressure/UV) when API keys are configured, deterministic mock data
otherwise. So it works today with mock data and upgrades to real data when keys
are added — no code change needed.
"""
from __future__ import annotations

from typing import Optional

from immunosense.agents.environment.norms import (
    get_population_percentile,
    infer_season_from_date,
)
from immunosense.agents.environment.pipeline import _FF_TO_L1
from immunosense.agents.environment.sources.composite import CompositeEnvironmentSource
from immunosense.agents.environment.sources.mock import MockEnvironmentSource
from immunosense.agents.environment.thresholds import classify_all_thresholds
from immunosense.agents.environment.types import (
    DailyEnvironmentSummary,
    EnvironmentDataSource,
)


def _region_from_latlon(lat: float, lon: float) -> str:
    """Coarse US region from coordinates (reuses the mock source's mapping)."""
    return MockEnvironmentSource._region_from_latlon(lat, lon)


def build_environment_summary(
    lat: float,
    lon: float,
    target_date: str,
    label: Optional[str] = None,
    source: Optional[EnvironmentDataSource] = None,
) -> DailyEnvironmentSummary:
    """Build a DailyEnvironmentSummary for given coordinates + date.

    Args:
        lat, lon: location coordinates (from the user's stored home location).
        target_date: 'YYYY-MM-DD'.
        label: human-readable place name (stored on the summary's location).
        source: override data source; defaults to CompositeEnvironmentSource
            (real APIs w/ mock fallback).
    """
    if source is None:
        source = CompositeEnvironmentSource()

    region = _region_from_latlon(lat, lon)
    season = infer_season_from_date(target_date)
    fetched = source.fetch(lat, lon, target_date)

    percentiles = {}
    for ff_field, l1_name in _FF_TO_L1.items():
        value = getattr(fetched, ff_field)
        percentiles[l1_name] = (
            None if value is None
            else get_population_percentile(region, season, l1_name, value)
        )

    threshold_alerts = classify_all_thresholds(fetched)
    n_real = sum(1 for v in fetched.confidence.values() if v == "real")
    overall_confidence = n_real / 5.0 if fetched.confidence else 0.0

    return DailyEnvironmentSummary(
        date=target_date,
        location={
            "lat": lat, "lon": lon, "label": label or "",
            "region": region, "season": season,
        },
        pm25_ug_m3=fetched.pm25_ug_m3,
        ozone_ppb=fetched.ozone_ppb,
        uv_index=fetched.uv_index,
        barometric_change_kpa=fetched.barometric_change_kpa,
        pollen_index=fetched.pollen_index,
        percentiles=percentiles,
        threshold_alerts=threshold_alerts,
        feature_confidence=dict(fetched.confidence),
        sources=dict(fetched.sources),
        errors=list(fetched.errors),
        overall_confidence=overall_confidence,
    )
