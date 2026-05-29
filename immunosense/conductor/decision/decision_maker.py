"""Decision policy — Challenge 3 Phase 3 + orchestration policy.

This is the POLICY layer. Fusion computes numbers, quality computes confidence;
the decision maker turns those into ACTIONS:

    - call_tfm:     should we spend a TFM call to generate a narrative
                    explanation this bucket? (TFM calls have cost, so we are
                    selective — Challenge 3 Phase 3 "selective TFM".)
    - raise_alert:  should we surface a clinical/patient alert?

Keeping policy OUT of the math keeps both auditable: you can change when alerts
fire without touching how probability is computed, and vice versa.

DECISION RULES (v1, deliberately conservative and transparent):

    Gating first — if confidence is INSUFFICIENT (probability is None), we do
    NOT raise an alert (we cannot stand behind a number) and we DO optionally
    call the TFM only to explain the "not enough data" state to the patient.

    Otherwise:
        raise_alert if ANY of:
            - severity_composite >= ALERT_SEVERITY
            - flare_probability >= ALERT_PROBABILITY
            - a flare button was pressed this bucket (handled upstream too)
        call_tfm if ANY of:
            - raise_alert is true (an alert always deserves an explanation)
            - severity_band == "moderate" or "high"
            - at least one corroboration pattern matched (worth narrating)
            - this is the patient's explicit request context (future hook)

All thresholds are module constants so they are easy to tune and, later,
calibrate via the Auto-Research loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from immunosense.events.types import ConfidenceLevel

# Tunable thresholds (candidates for Auto-Research calibration later).
ALERT_SEVERITY = 0.6
ALERT_PROBABILITY = 0.5


@dataclass
class Decision:
    """The orchestration decision for a bucket.

    Fields:
        call_tfm: whether to invoke the Thinking Machine for explanation.
        raise_alert: whether to surface a clinical/patient alert.
        audience: who the explanation/alert is for ("patient" | "clinician").
        reasons: human-readable reasons (audit trail).
    """

    call_tfm: bool = False
    raise_alert: bool = False
    audience: str = "patient"
    reasons: list = field(default_factory=list)


class DecisionMaker:
    """Turns a fused evaluation into TFM/alert actions."""

    def __init__(
        self,
        alert_severity: float = ALERT_SEVERITY,
        alert_probability: float = ALERT_PROBABILITY,
        explain_insufficient: bool = True,
    ):
        self.alert_severity = alert_severity
        self.alert_probability = alert_probability
        # Whether to call the TFM to explain a gated (insufficient) result.
        self.explain_insufficient = explain_insufficient

    def decide(
        self,
        flare_probability: Optional[float],
        severity_composite: Optional[float],
        matched_patterns: list,
        confidence_result,
        flare_button: Optional[float] = None,
        severity_band: Optional[str] = None,
    ) -> Decision:
        """Compute the decision for a bucket.

        Args:
            flare_probability: from fusion (None if gated).
            severity_composite: from risk engine (None if gated).
            matched_patterns: list of matched corroboration patterns.
            confidence_result: the ConfidenceResult (level).
            flare_button: severity if the patient pressed the button this bucket.
            severity_band: "low"/"moderate"/"high" or None.

        Returns:
            Decision.
        """
        reasons = []

        # --- Gated case: INSUFFICIENT confidence (no probability) ---
        if flare_probability is None:
            call_tfm = bool(self.explain_insufficient)
            if call_tfm:
                reasons.append(
                    "confidence insufficient — explain the 'not enough data' state"
                )
            # A flare button press is always honored even under low data.
            raise_alert = flare_button is not None and flare_button >= 0.5
            if raise_alert:
                reasons.append("flare button pressed (honored despite low data)")
            return Decision(
                call_tfm=call_tfm or raise_alert,
                raise_alert=raise_alert,
                audience="patient",
                reasons=reasons or ["insufficient confidence; no action"],
            )

        # --- Normal case ---
        raise_alert = False
        if severity_composite is not None and severity_composite >= self.alert_severity:
            raise_alert = True
            reasons.append(
                f"severity composite {severity_composite:.2f} >= {self.alert_severity}"
            )
        if flare_probability >= self.alert_probability:
            raise_alert = True
            reasons.append(
                f"flare probability {flare_probability:.2f} >= {self.alert_probability}"
            )
        if flare_button is not None and flare_button >= 0.5:
            raise_alert = True
            reasons.append("flare button pressed")

        call_tfm = False
        if raise_alert:
            call_tfm = True
            reasons.append("alert raised -> explain it")
        elif severity_band in ("moderate", "high"):
            call_tfm = True
            reasons.append(f"severity band '{severity_band}' warrants explanation")
        elif matched_patterns:
            call_tfm = True
            names = ", ".join(getattr(p, "name", str(p)) for p in matched_patterns)
            reasons.append(f"corroboration pattern(s) matched: {names}")

        if not reasons:
            reasons.append("no thresholds crossed; no action")

        return Decision(
            call_tfm=call_tfm,
            raise_alert=raise_alert,
            audience="patient",
            reasons=reasons,
        )
