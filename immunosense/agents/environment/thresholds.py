"""EPA threshold classifiers and AQI ↔ concentration conversion.

Deterministic clinical thresholds applied per-feature.
AirNow returns AQI (0-500); we invert EPA's breakpoint table to ug/m^3 / ppb.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.environment.types import FetchedFeatures


# EPA AQI breakpoints: (aqi_lo, aqi_hi, conc_lo, conc_hi)
PM25_AQI_BREAKPOINTS = [
    (0, 50, 0.0, 12.0),
    (51, 100, 12.1, 35.4),
    (101, 150, 35.5, 55.4),
    (151, 200, 55.5, 150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
]

OZONE_AQI_BREAKPOINTS_8HR = [
    (0, 50, 0, 54),
    (51, 100, 55, 70),
    (101, 150, 71, 85),
    (151, 200, 86, 105),
    (201, 300, 106, 200),
]


# ============================================================
# Per-feature threshold classifiers
# ============================================================

def classify_pm25_threshold(pm25_ug_m3: Optional[float]) -> Optional[str]:
    """EPA NAAQS PM2.5 24-hour AQI breakpoints."""
    if pm25_ug_m3 is None:
        return None
    if pm25_ug_m3 <= 12.0:
        return "good"
    if pm25_ug_m3 <= 35.4:
        return "moderate"
    if pm25_ug_m3 <= 55.4:
        return "unhealthy_sensitive"  # autoimmune alert
    if pm25_ug_m3 <= 150.4:
        return "unhealthy"
    if pm25_ug_m3 <= 250.4:
        return "very_unhealthy"
    return "hazardous"


def classify_ozone_threshold(ozone_ppb: Optional[float]) -> Optional[str]:
    """EPA NAAQS 8-hour ozone AQI breakpoints."""
    if ozone_ppb is None:
        return None
    if ozone_ppb <= 54:
        return "good"
    if ozone_ppb <= 70:
        return "moderate"
    if ozone_ppb <= 85:
        return "unhealthy_sensitive"
    if ozone_ppb <= 105:
        return "unhealthy"
    return "very_unhealthy"


def classify_uv_threshold(uv_index: Optional[float]) -> Optional[str]:
    """WHO UV index classification."""
    if uv_index is None:
        return None
    if uv_index < 3:
        return "low"
    if uv_index < 6:
        return "moderate"
    if uv_index < 8:
        return "high"
    if uv_index < 11:
        return "very_high"
    return "extreme"


def classify_barometric_threshold(barometric_change_kpa: Optional[float]) -> Optional[str]:
    """Barometric pressure change magnitude (24-hour delta)."""
    if barometric_change_kpa is None:
        return None
    abs_change = abs(barometric_change_kpa)
    if abs_change < 0.5:
        return "stable"
    if abs_change < 1.0:
        return "moderate_change"
    return "large_change"


def classify_pollen_threshold(pollen_index: Optional[float]) -> Optional[str]:
    """Universal Pollen Index (UPI) classification, 0-5 scale."""
    if pollen_index is None:
        return None
    if pollen_index < 1.5:
        return "low"
    if pollen_index < 3.0:
        return "moderate"
    if pollen_index < 4.5:
        return "high"
    return "very_high"


def classify_all_thresholds(features: FetchedFeatures) -> dict:
    """Apply all threshold classifiers to a FetchedFeatures struct."""
    return {
        "pm25":       classify_pm25_threshold(features.pm25_ug_m3),
        "ozone":      classify_ozone_threshold(features.ozone_ppb),
        "uv":         classify_uv_threshold(features.uv_index),
        "barometric": classify_barometric_threshold(features.barometric_change_kpa),
        "pollen":     classify_pollen_threshold(features.pollen_index),
    }


# ============================================================
# AQI -> concentration conversion (linear interpolation within breakpoints)
# ============================================================

def aqi_to_concentration(
    aqi: Optional[float],
    breakpoints: list,
) -> Optional[float]:
    """Convert AirNow AQI to concentration via EPA breakpoint inversion."""
    if aqi is None:
        return None
    aqi = float(aqi)
    for (aqi_lo, aqi_hi, conc_lo, conc_hi) in breakpoints:
        if aqi_lo <= aqi <= aqi_hi:
            frac = (aqi - aqi_lo) / (aqi_hi - aqi_lo) if aqi_hi > aqi_lo else 0.0
            return conc_lo + frac * (conc_hi - conc_lo)
    return None


def aqi_to_pm25(aqi: Optional[float]) -> Optional[float]:
    """Convert PM2.5 AQI to ug/m^3."""
    return aqi_to_concentration(aqi, PM25_AQI_BREAKPOINTS)


def aqi_to_ozone_ppb(aqi: Optional[float]) -> Optional[float]:
    """Convert ozone AQI to ppb."""
    return aqi_to_concentration(aqi, OZONE_AQI_BREAKPOINTS_8HR)
