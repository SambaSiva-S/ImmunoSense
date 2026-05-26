"""AirNowSource - EPA AirNow API for PM2.5 and ozone.

The free AirNow API returns AQI values; we convert to ug/m^3 (PM2.5)
and ppb (ozone) using EPA breakpoint inversion.

Caching: 24-hour TTL per (lat, lon, date) tuple, stored as pickle files.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from immunosense.agents.environment.thresholds import aqi_to_ozone_ppb, aqi_to_pm25
from immunosense.agents.environment.types import FetchedFeatures


class AirNowSource:
    """EPA AirNow data source (PM2.5 + ozone).

    Args:
        api_key: Override API key. If None, reads AIRNOW_API_KEY env var.
        cache_dir: Override cache directory. Defaults to ./artifacts/agent3/api_cache/.
    """

    BASE_URL = "https://www.airnowapi.org/aq/observation/latLong/current/"
    DISTANCE_MILES = 25
    CACHE_TTL_HOURS = 24

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("AIRNOW_API_KEY")
        self.cache_dir = cache_dir or Path("./artifacts/agent3/api_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Fetch PM2.5 + ozone for given location. Falls back gracefully on errors."""
        cache_path = self._cache_path(latitude, longitude, target_date)
        if cache_path.exists() and self._cache_is_fresh(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if not self.api_key:
            return FetchedFeatures(errors=["AirNow: no AIRNOW_API_KEY"])

        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "format": "application/json",
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance": self.DISTANCE_MILES,
                    "API_KEY": self.api_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return FetchedFeatures(errors=[f"AirNow HTTP: {type(e).__name__}: {e}"])
        except ValueError as e:
            return FetchedFeatures(errors=[f"AirNow JSON: {e}"])

        result = self._parse_response(data)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            result.errors.append(f"AirNow cache write: {e}")
        return result

    def _parse_response(self, data: list) -> FetchedFeatures:
        if not isinstance(data, list) or not data:
            return FetchedFeatures(errors=["AirNow: empty or invalid response"])

        pm25_ug_m3 = None
        ozone_ppb = None
        for obs in data:
            param = obs.get("ParameterName", "").upper()
            aqi = obs.get("AQI")
            if param == "PM2.5" and aqi is not None:
                pm25_ug_m3 = aqi_to_pm25(aqi)
            elif param in ("O3", "OZONE") and aqi is not None:
                ozone_ppb = aqi_to_ozone_ppb(aqi)

        confidence, sources = {}, {}
        if pm25_ug_m3 is not None:
            confidence["pm25_ug_m3"] = "real"
            sources["pm25_ug_m3"] = "airnow"
        if ozone_ppb is not None:
            confidence["ozone_ppb"] = "real"
            sources["ozone_ppb"] = "airnow"

        return FetchedFeatures(
            pm25_ug_m3=pm25_ug_m3,
            ozone_ppb=ozone_ppb,
            confidence=confidence,
            sources=sources,
        )

    def _cache_path(self, lat: float, lon: float, target_date: str) -> Path:
        return self.cache_dir / f"airnow_{lat:.4f}_{lon:.4f}_{target_date}.pkl"

    def _cache_is_fresh(self, cache_path: Path) -> bool:
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        return age_hours < self.CACHE_TTL_HOURS
