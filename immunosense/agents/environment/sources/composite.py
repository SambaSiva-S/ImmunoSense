"""CompositeEnvironmentSource - production data source that combines all APIs.

Wraps AirNow + OpenWeather + Google Pollen + Mock fallback. Per-feature
graceful degradation: if a real source returns None for a feature, Mock
fills the gap and the confidence is stamped 'synthetic'.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.environment.sources.airnow import AirNowSource
from immunosense.agents.environment.sources.google_pollen import GooglePollenSource
from immunosense.agents.environment.sources.mock import MockEnvironmentSource
from immunosense.agents.environment.sources.openweather import OpenWeatherSource
from immunosense.agents.environment.types import FetchedFeatures


class CompositeEnvironmentSource:
    """Production composite data source with graceful real → mock fallback.

    Args:
        airnow: Override AirNow source. If None, uses default.
        openweather: Override OpenWeather source. If None, uses default.
        google_pollen: Override Google Pollen source. If None, uses default.
        mock: Override Mock source. If None, uses default.
    """

    AIRNOW_FEATURES = ["pm25_ug_m3", "ozone_ppb"]
    WEATHER_FEATURES = ["barometric_change_kpa", "uv_index"]
    POLLEN_FEATURES = ["pollen_index"]

    def __init__(
        self,
        airnow: Optional[AirNowSource] = None,
        openweather: Optional[OpenWeatherSource] = None,
        google_pollen: Optional[GooglePollenSource] = None,
        mock: Optional[MockEnvironmentSource] = None,
    ) -> None:
        self.airnow = airnow or AirNowSource()
        self.openweather = openweather or OpenWeatherSource()
        self.google_pollen = google_pollen or GooglePollenSource()
        self.mock = mock or MockEnvironmentSource()

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Fetch all features. Real values where available, synthetic where not."""
        airnow_result = self.airnow.fetch(latitude, longitude, target_date)
        weather_result = self.openweather.fetch(latitude, longitude, target_date)
        pollen_result = self.google_pollen.fetch(latitude, longitude, target_date)
        mock_result = self.mock.fetch(latitude, longitude, target_date)

        merged = FetchedFeatures()

        for feature in self.AIRNOW_FEATURES:
            self._merge_feature(merged, feature, airnow_result, mock_result, "airnow")

        for feature in self.WEATHER_FEATURES:
            real_source = weather_result.sources.get(feature, "openweather")
            self._merge_feature(merged, feature, weather_result, mock_result, real_source)

        for feature in self.POLLEN_FEATURES:
            self._merge_feature(merged, feature, pollen_result, mock_result, "google-pollen")

        merged.errors = (
            airnow_result.errors + weather_result.errors + pollen_result.errors
        )
        return merged

    @staticmethod
    def _merge_feature(
        merged: FetchedFeatures,
        feature: str,
        real_result: FetchedFeatures,
        mock_result: FetchedFeatures,
        real_source_name: str,
    ) -> None:
        """Merge one feature: prefer real value, fall back to mock."""
        real_val = getattr(real_result, feature)
        if real_val is not None:
            setattr(merged, feature, real_val)
            merged.confidence[feature] = "real"
            merged.sources[feature] = real_source_name
        else:
            setattr(merged, feature, getattr(mock_result, feature))
            merged.confidence[feature] = "synthetic"
            merged.sources[feature] = "mock"
