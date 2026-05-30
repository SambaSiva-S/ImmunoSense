"""Pydantic request/response schemas + the server-side wellness framing.

The framing is centralized here (DECISION 3): the API decides whether to show a
probability number or wellness-phrased prose, so no client can accidentally
display a scary percentage under low confidence. This is a safety choice made
once, server-side.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class SymptomLogIn(BaseModel):
    source: str = "tap"  # tap | text | voice
    fatigue: Optional[float] = None
    joint_pain: Optional[float] = None
    brain_fog_severity: Optional[float] = None
    gi_distress: Optional[float] = None
    skin_severity: Optional[float] = None
    sleep_severity: Optional[float] = None
    energy_severity: Optional[float] = None
    wellness_severity: Optional[float] = None
    phq8_score: Optional[float] = None
    gad7_score: Optional[float] = None
    free_text: Optional[str] = None


class MealLogIn(BaseModel):
    source: str = "text"  # text | voice | photo | tap
    description: str
    photo_id: Optional[str] = None


class BiomarkerLogIn(BaseModel):
    crp: Optional[float] = None
    esr: Optional[float] = None
    extra: dict = Field(default_factory=dict)  # other biomarker fields


class FlareIn(BaseModel):
    severity: float = 0.8


class ConsentIn(BaseModel):
    consent_type: str
    granted: bool


class PhotoRequestIn(BaseModel):
    content_type: str = "image/jpeg"


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class LogAck(BaseModel):
    ok: bool = True
    log_id: Optional[str] = None
    bucket_id: str
    trace_id: str


class PatternOut(BaseModel):
    name: str
    label: str
    description: str


class DisplayBlock(BaseModel):
    """UI-facing pre-formatted block. The client renders this directly."""
    headline: str
    show_number: bool
    band_label: Optional[str] = None


class ReportOut(BaseModel):
    bucket_id: str
    evaluated_at: Optional[str] = None
    confidence_level: str
    status: str  # "ok" | "insufficient"
    flare_probability: Optional[float] = None
    severity_composite: Optional[float] = None
    severity_band: Optional[str] = None
    matched_patterns: list[PatternOut] = Field(default_factory=list)
    explanation: Optional[str] = None
    display: DisplayBlock
    trace_id: str


# --------------------------------------------------------------------------- #
# Confidence-aware framing (the safety-relevant translation)
# --------------------------------------------------------------------------- #
_BAND_LABELS = {
    "low": "steady",
    "moderate": "slightly elevated",
    "high": "notably elevated",
}


def build_display(confidence_level: str, flare_probability, severity_band) -> DisplayBlock:
    """Translate the raw inference into a wellness-framed display block.

    Rules:
      - INSUFFICIENT (probability gated to None): never show a number. Show a
        gentle "not enough data yet" message.
      - Otherwise: a wellness-phrased headline keyed off the severity band,
        and show_number=True only at moderate confidence or above.
    """
    if flare_probability is None or confidence_level == "insufficient":
        return DisplayBlock(
            headline="Not enough recent data to show a pattern yet — keep logging.",
            show_number=False,
            band_label=None,
        )

    band = (severity_band or "low").lower()
    descriptor = _BAND_LABELS.get(band, "steady")
    headline = f"Your indicators are {descriptor} compared with your baseline."

    # Only surface the numeric probability when confidence is moderate+.
    show_number = confidence_level in ("moderate", "high")
    return DisplayBlock(headline=headline, show_number=show_number, band_label=descriptor)


def report_to_out(report, trace_id: str) -> ReportOut:
    """Convert a ConductorReport into the API ReportOut, with framing applied."""
    conf = report.confidence_level.value
    status = "insufficient" if report.flare_probability is None else "ok"
    display = build_display(conf, report.flare_probability, report.severity_band)
    evaluated = getattr(report, "evaluated_at", None)
    evaluated_iso = evaluated.isoformat() if hasattr(evaluated, "isoformat") else evaluated
    return ReportOut(
        bucket_id=getattr(report, "bucket_id", ""),
        evaluated_at=evaluated_iso,
        confidence_level=conf,
        status=status,
        flare_probability=report.flare_probability,
        severity_composite=report.severity_composite,
        severity_band=report.severity_band,
        matched_patterns=[
            PatternOut(name=p.name, label=p.label, description=p.description)
            for p in report.matched_patterns
        ],
        explanation=report.explanation,
        display=display,
        trace_id=trace_id,
    )
