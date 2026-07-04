"""Geocoding — resolve a user-entered city name or US zip code to coordinates.

The environmental data sources (AirNow, Google Pollen, OpenWeather) all take
lat/lng, but users think in cities and zip codes. This bridges the two.

Strategy (all free, no API key):
  - looks like a US zip (5 digits)  -> Zippopotam.us
  - otherwise (city name)           -> Open-Meteo geocoding (global)

Design notes:
  - Network-resilient: any failure returns None rather than raising, so a
    geocoding outage never blocks profile-saving. Caller decides what to do.
  - Returns (lat, lng, normalized_label) so we can store a clean label too.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# (lat, lng, label) or None
GeocodeResult = Optional[Tuple[float, float, str]]

_US_ZIP = re.compile(r"^\s*(\d{5})(?:-\d{4})?\s*$")


def geocode(query: str, *, timeout: float = 6.0) -> GeocodeResult:
    """Resolve a city name or US zip to (lat, lng, label). None on failure."""
    if not query or not query.strip():
        return None
    q = query.strip()

    zip_match = _US_ZIP.match(q)
    if zip_match:
        return _geocode_us_zip(zip_match.group(1), timeout=timeout)
    return _geocode_city(q, timeout=timeout)


def _geocode_us_zip(zipcode: str, *, timeout: float) -> GeocodeResult:
    """US zip -> lat/lng via Zippopotam.us (free, no key)."""
    import httpx
    try:
        r = httpx.get(f"https://api.zippopotam.us/us/{zipcode}", timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        place = data["places"][0]
        lat = float(place["latitude"])
        lng = float(place["longitude"])
        label = f'{place["place name"]}, {place["state abbreviation"]} {zipcode}'
        return (lat, lng, label)
    except Exception:
        return None


def _geocode_city(name: str, *, timeout: float) -> GeocodeResult:
    """City name -> lat/lng via Open-Meteo geocoding (free, no key, global)."""
    import httpx
    try:
        r = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        lat = float(top["latitude"])
        lng = float(top["longitude"])
        # Build a clean label: "City, Admin1, CountryCode"
        parts = [top.get("name")]
        if top.get("admin1"):
            parts.append(top["admin1"])
        if top.get("country_code"):
            parts.append(top["country_code"])
        label = ", ".join(p for p in parts if p)
        return (lat, lng, label)
    except Exception:
        return None
