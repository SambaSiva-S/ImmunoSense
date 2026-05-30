"""Biomarker builder — raw UI input -> {demographics, reading}.

The biomarker agent's process() consumes:
    {'demographics': {'age', 'sex', 'bmi'}, 'reading': {biomarker values}}

The user self-enters lab values (CRP, ESR, CBC components) and their basic
demographics live in their profile. This builder assembles the two dicts the
agent expects. No external data needed at build time — the agent's NHANES
percentile baseline is its own trained Layer 1 (built separately from the
NHANES .XPT files), not something the builder supplies per call.
"""

from __future__ import annotations

from typing import Optional


def build_biomarker_input(
    reading_row,
    profile,
) -> dict:
    """Assemble the biomarker agent's {demographics, reading} input.

    Args:
        reading_row: a BiomarkerReading ORM row (or object) exposing crp, esr,
            and a `payload` dict of any additional biomarker fields.
        profile: the user's Profile (or object) exposing age/sex/bmi info.
            Sex is encoded as the NHANES convention (1 = male, 2 = female) to
            match the agent's percentile baseline.

    Returns:
        {'demographics': {...}, 'reading': {...}} ready for agent.process().
    """
    demographics = {
        "age": _safe_float(getattr(profile, "age", None), default=40.0),
        "sex": int(getattr(profile, "sex", 1) or 1),
        "bmi": _safe_float(getattr(profile, "bmi", None), default=25.0),
    }

    # Core self-entered labs.
    reading: dict = {}
    if getattr(reading_row, "crp", None) is not None:
        reading["CRP"] = float(reading_row.crp)
    if getattr(reading_row, "esr", None) is not None:
        reading["ESR"] = float(reading_row.esr)

    # Any additional biomarker fields the user entered live in payload.
    payload = getattr(reading_row, "payload", None) or {}
    for k, v in payload.items():
        if v is not None and k not in reading:
            reading[k] = v

    return {"demographics": demographics, "reading": reading}


def _safe_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
