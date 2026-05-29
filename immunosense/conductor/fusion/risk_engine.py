"""Severity composite — Challenge 3 Phase 4 (UI / decision-facing).

The severity composite is a single 0..1 number for display and decision-making
that blends THREE already-computed things:

    1. flare_probability  — from Phase 1 Bayesian fusion (the likelihood)
    2. acute_severity     — how severe the CURRENT signals are right now
                            (distinct from probability: a flare can be likely
                            but mild, or unlikely but the present symptoms are
                            already severe)
    3. confidence         — the Challenge 7 level, which DAMPENS the composite
                            when evidence is thin

CRITICAL: this module CONSUMES the fusion probability; it does NOT re-derive
risk from the agent signals independently. Re-deriving would double-count the
evidence already in the probability. acute_severity is a DIFFERENT quantity
(present intensity, not future likelihood), so combining them is legitimate.

GATING: if probability is None (INSUFFICIENT confidence gated Phase 1), the
composite is None too — we never present a risk number the evidence can't
support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from immunosense.conductor.fusion.statistical_fusion import extract_signal_strength
from immunosense.events.types import ConfidenceLevel

# How much each confidence level dampens the composite (multiplicative).
# INSUFFICIENT never reaches here (gated), but included for completeness.
_CONFIDENCE_DAMPING = {
    ConfidenceLevel.INSUFFICIENT: 0.0,
    ConfidenceLevel.LOW: 0.6,
    ConfidenceLevel.MODERATE: 0.85,
    ConfidenceLevel.HIGH: 1.0,
}

# Blend weights for probability vs acute severity (before damping).
_W_PROBABILITY = 0.6
_W_ACUTE = 0.4


@dataclass
class RiskResult:
    """The severity composite outcome.

    Fields:
        severity_composite: 0..1 blended score, or None if gated.
        flare_probability: echoed from fusion (the likelihood input).
        acute_severity: 0..1 present-intensity component.
        confidence_damping: the multiplier applied for the confidence level.
        band: coarse label for UI ("low" | "moderate" | "high"), or None.
    """

    severity_composite: Optional[float]
    flare_probability: Optional[float]
    acute_severity: float
    confidence_damping: float
    band: Optional[str]


def _acute_severity(agent_outputs: dict) -> float:
    """Present-intensity score: the max signal strength across agents.

    This captures 'how bad are things right now' independent of how likely a
    flare is. Max (not mean) so a single severe system isn't averaged away.
    """
    if not agent_outputs:
        return 0.0
    return max(extract_signal_strength(o) for o in agent_outputs.values())


def _band(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "moderate"
    return "low"


class RiskEngine:
    """Blends probability + acute severity + confidence into a composite."""

    def compute(
        self,
        flare_probability: Optional[float],
        confidence_result,
        agent_outputs: dict,
    ) -> RiskResult:
        """Compute the severity composite for a bucket.

        Args:
            flare_probability: From StatisticalFusion (None if gated).
            confidence_result: The ConfidenceResult (provides the level).
            agent_outputs: Reporting agents' outputs (for acute severity).

        Returns:
            RiskResult. severity_composite is None when probability is None.
        """
        acute = _acute_severity(agent_outputs)

        # Gate: no probability (insufficient confidence) -> no composite.
        if flare_probability is None:
            return RiskResult(
                severity_composite=None,
                flare_probability=None,
                acute_severity=round(acute, 4),
                confidence_damping=0.0,
                band=None,
            )

        damping = _CONFIDENCE_DAMPING.get(confidence_result.level, 1.0)
        blended = _W_PROBABILITY * flare_probability + _W_ACUTE * acute
        composite = blended * damping
        composite = float(min(1.0, max(0.0, composite)))

        return RiskResult(
            severity_composite=round(composite, 4),
            flare_probability=flare_probability,
            acute_severity=round(acute, 4),
            confidence_damping=damping,
            band=_band(composite),
        )
