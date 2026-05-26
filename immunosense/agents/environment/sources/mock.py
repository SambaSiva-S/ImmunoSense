"""MockEnvironmentSource - deterministic synthetic environmental data.

Used as fallback when real APIs are unavailable, and for testing.
Determinism: seed derived from (lat, lon, date, seed_offset) so reruns
produce identical results.
"""

from __future__ import annotations

import hashlib
import random as random_mod

from immunosense.agents.environment.norms import REGIONAL_NORMS, infer_season_from_date
from immunosense.agents.environment.types import FetchedFeatures


class MockEnvironmentSource:
    """Deterministic synthetic environmental data, sampled from REGIONAL_NORMS.

    Args:
        seed_offset: Add to the seed derivation to get different deterministic
            streams. Useful for generating multiple synthetic patient profiles
            at the same location.
    """

    def __init__(self, seed_offset: int = 0) -> None:
        self.seed_offset = seed_offset

    def fetch(self, latitude: float, longitude: float, target_date: str) -> FetchedFeatures:
        """Return deterministic synthetic features for the given location and date."""
        seed_key = f"{latitude:.4f},{longitude:.4f},{target_date},{self.seed_offset}"
        seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
        rng = random_mod.Random(seed)

        region = self._region_from_latlon(latitude, longitude)
        season = infer_season_from_date(target_date)
        norms = REGIONAL_NORMS[region][season]

        pm25 = max(0.0, rng.gauss(norms["pm25"][0], norms["pm25"][1]))
        ozone = max(0.0, rng.gauss(norms["ozone"][0], norms["ozone"][1]))
        uv = max(0.0, min(12.0, rng.gauss(norms["uv"][0], norms["uv"][1])))
        barometric = max(0.0, abs(rng.gauss(0.0, norms["barometric"][1])))
        pollen = max(0.0, min(10.0, rng.gauss(norms["pollen"][0], norms["pollen"][1])))

        # 5% chance of spike event (dust storm, pollution event)
        if rng.random() < 0.05:
            spike = rng.choice(["pm25", "ozone", "pollen"])
            if spike == "pm25":
                pm25 *= rng.uniform(2.5, 4.0)
            elif spike == "ozone":
                ozone *= rng.uniform(1.5, 2.0)
            elif spike == "pollen":
                pollen = min(10.0, pollen * rng.uniform(1.8, 2.5))

        feature_keys = [
            "pm25_ug_m3", "ozone_ppb", "uv_index",
            "barometric_change_kpa", "pollen_index",
        ]
        return FetchedFeatures(
            pm25_ug_m3=round(pm25, 1),
            ozone_ppb=round(ozone, 0),
            uv_index=round(uv, 1),
            barometric_change_kpa=round(barometric, 2),
            pollen_index=round(pollen, 1),
            confidence={f: "synthetic" for f in feature_keys},
            sources={f: "mock" for f in feature_keys},
        )

    @staticmethod
    def _region_from_latlon(lat: float, lon: float) -> str:
        """Approximate US region from (lat, lon)."""
        if lon > -82:
            return "SE" if lat < 40 else "NE"
        if lon > -95:
            return "SE" if lat < 36 else "MW"
        if lon > -105:
            return "SW" if lat < 40 else "MW"
        return "W"
