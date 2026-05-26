"""OpenWeatherSource - barometric pressure (OpenWeather) + UV (Open-Meteo).

The free OpenWeather tier doesn't include UV index, so we use two upstream APIs:
    - OpenWeather /weather -> pressure (main.pressure, hPa)
    - Open-Meteo /forecast -> UV index (daily.uv_index_max, no API key)

Barometric semantics (24-hour delta):
    Day 1 for a location: pressure stored to disk, change=None
    Day 2+: change = today's pressure - yesterday's stored pressure
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from immunosense.agents.environment.types import FetchedFeatures


class OpenWeatherSource:
    """OpenWeather (pressure) + Open-Meteo (UV) data source.

    Args:
        api_key: Override OpenWeather API key. If None, reads OPENWEATHER_API_KEY.
        cache_dir: Override cache directory.
    """

    OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    CACHE_TTL_HOURS = 6

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        self.cache_dir = cache_dir or Path("./artifacts/agent3/api_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Fetch barometric pressure delta + UV index for given location."""
        cache_path = self._cache_path(latitude, longitude, target_date)
        if cache_path.exists() and self._cache_is_fresh(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        pressure_kpa, ow_errors = self._fetch_pressure(latitude, longitude)
        uv_index, om_errors = self._fetch_uv(latitude, longitude)

        # Store today's raw pressure for tomorrow's delta computation
        if pressure_kpa is not None:
            self._save_raw_pressure(latitude, longitude, target_date, pressure_kpa)

        barometric_change_kpa = self._compute_pressure_delta(
            latitude, longitude, target_date, pressure_kpa
        )

        confidence, sources = {}, {}
        if barometric_change_kpa is not None:
            confidence["barometric_change_kpa"] = "real"
            sources["barometric_change_kpa"] = "openweather"
        if uv_index is not None:
            confidence["uv_index"] = "real"
            sources["uv_index"] = "open-meteo"

        result = FetchedFeatures(
            barometric_change_kpa=(
                round(barometric_change_kpa, 2) if barometric_change_kpa is not None else None
            ),
            uv_index=round(uv_index, 1) if uv_index is not None else None,
            confidence=confidence,
            sources=sources,
            errors=ow_errors + om_errors,
        )

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            result.errors.append(f"OpenWeather cache write: {e}")
        return result

    def _fetch_pressure(self, latitude: float, longitude: float) -> tuple:
        if not self.api_key:
            return None, ["OpenWeather: no OPENWEATHER_API_KEY"]
        try:
            response = requests.get(
                self.OPENWEATHER_URL,
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            data = response.json()
        except requests.exceptions.RequestException as e:
            return None, [f"OpenWeather HTTP: {type(e).__name__}: {e}"]
        except ValueError as e:
            return None, [f"OpenWeather JSON: {e}"]

        if "cod" in data and str(data.get("cod")) != "200":
            return None, [f"OpenWeather API: {data.get('message', 'unknown')}"]

        pressure_hpa = data.get("main", {}).get("pressure")
        if pressure_hpa is None:
            return None, ["OpenWeather: main.pressure missing"]
        return pressure_hpa / 10.0, []  # hPa to kPa

    def _fetch_uv(self, latitude: float, longitude: float) -> tuple:
        try:
            response = requests.get(
                self.OPEN_METEO_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": "uv_index_max",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
                timeout=10,
            )
            data = response.json()
        except requests.exceptions.RequestException as e:
            return None, [f"Open-Meteo HTTP: {type(e).__name__}: {e}"]
        except ValueError as e:
            return None, [f"Open-Meteo JSON: {e}"]

        try:
            uv_max = data["daily"]["uv_index_max"][0]
            if uv_max is None:
                return None, ["Open-Meteo: uv_index_max null"]
            return float(uv_max), []
        except (KeyError, IndexError, TypeError) as e:
            return None, [f"Open-Meteo unexpected shape: {e}"]

    def _save_raw_pressure(
        self, lat: float, lon: float, target_date: str, pressure_kpa: float
    ) -> None:
        path = self._pressure_path(lat, lon, target_date)
        try:
            with open(path, "wb") as f:
                pickle.dump(pressure_kpa, f)
        except Exception:
            pass  # non-critical

    def _compute_pressure_delta(
        self,
        lat: float,
        lon: float,
        target_date: str,
        current_pressure_kpa: Optional[float],
    ) -> Optional[float]:
        if current_pressure_kpa is None:
            return None
        yesterday = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday_path = self._pressure_path(lat, lon, yesterday)
        if not yesterday_path.exists():
            return None  # first day for this location
        try:
            with open(yesterday_path, "rb") as f:
                prior = pickle.load(f)
            if not isinstance(prior, (int, float)):
                return None
            return current_pressure_kpa - prior
        except Exception:
            return None

    def _cache_path(self, lat: float, lon: float, target_date: str) -> Path:
        return self.cache_dir / f"openweather_{lat:.4f}_{lon:.4f}_{target_date}.pkl"

    def _pressure_path(self, lat: float, lon: float, target_date: str) -> Path:
        return self.cache_dir / f"pressure_{lat:.4f}_{lon:.4f}_{target_date}.pkl"

    def _cache_is_fresh(self, cache_path: Path) -> bool:
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        return age_hours < self.CACHE_TTL_HOURS
