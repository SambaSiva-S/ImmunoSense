"""GooglePollenSource - Google Pollen API for tree/grass/weed indices.

Max-aggregates the three pollen types: patient feels the worst trigger,
not the average. Patients with specific allergen sensitivity (e.g., grass
only) would benefit from per-category tracking in v2.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from immunosense.agents.environment.types import FetchedFeatures


class GooglePollenSource:
    """Google Pollen API data source.

    Args:
        api_key: Override API key. If None, reads GOOGLE_POLLEN_API_KEY.
        cache_dir: Override cache directory.
    """

    BASE_URL = "https://pollen.googleapis.com/v1/forecast:lookup"
    CACHE_TTL_HOURS = 24

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_POLLEN_API_KEY")
        self.cache_dir = cache_dir or Path("./artifacts/agent3/api_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Fetch max pollen index (tree, grass, weed) for given location."""
        cache_path = self._cache_path(latitude, longitude, target_date)
        if cache_path.exists() and self._cache_is_fresh(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if not self.api_key:
            return FetchedFeatures(errors=["GooglePollen: no GOOGLE_POLLEN_API_KEY"])

        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "key": self.api_key,
                    "location.latitude": latitude,
                    "location.longitude": longitude,
                    "days": 1,
                },
                timeout=10,
            )
            data = response.json()
        except requests.exceptions.RequestException as e:
            return FetchedFeatures(errors=[f"GooglePollen HTTP: {type(e).__name__}: {e}"])
        except ValueError as e:
            return FetchedFeatures(errors=[f"GooglePollen JSON: {e}"])

        if "error" in data:
            err = data["error"]
            return FetchedFeatures(
                errors=[f"GooglePollen API {err.get('code')}: {err.get('message')}"]
            )

        result = self._parse_response(data)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            result.errors.append(f"GooglePollen cache write: {e}")
        return result

    def _parse_response(self, data: dict) -> FetchedFeatures:
        try:
            daily_info = data.get("dailyInfo", [])
            if not daily_info:
                return FetchedFeatures(errors=["GooglePollen: dailyInfo missing"])

            today = daily_info[0]
            indices = {}
            for entry in today.get("pollenTypeInfo", []):
                code = entry.get("code", "").upper()
                if code in ("TREE", "GRASS", "WEED"):
                    value = entry.get("indexInfo", {}).get("value")
                    if value is not None:
                        indices[code] = value

            if not indices:
                return FetchedFeatures(errors=["GooglePollen: no indices in response"])

            # Max aggregation: patient feels the worst trigger, not the average
            pollen_max = float(max(indices.values()))
            return FetchedFeatures(
                pollen_index=round(pollen_max, 1),
                confidence={"pollen_index": "real"},
                sources={"pollen_index": "google-pollen"},
            )
        except (KeyError, IndexError, TypeError) as e:
            return FetchedFeatures(
                errors=[f"GooglePollen unexpected shape: {type(e).__name__}: {e}"]
            )

    def _cache_path(self, lat: float, lon: float, target_date: str) -> Path:
        return self.cache_dir / f"pollen_{lat:.4f}_{lon:.4f}_{target_date}.pkl"

    def _cache_is_fresh(self, cache_path: Path) -> bool:
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        return age_hours < self.CACHE_TTL_HOURS
