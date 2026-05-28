"""Risk / severity composite — Phase 4 of Challenge 3.

STATUS: Sprint 6 (NOT YET IMPLEMENTED).
Sprint 5 ships this as a stub returning None.

Locked design (Challenge 3, Phase 4): the severity composite is a UI-FACING
score that combines the calibrated flare_probability (Phase 1), a severity
signal, and the confidence level (Challenge 7). It is computed FROM the fusion
output — it does NOT re-derive risk independently, and it does NOT feed back
into the probability. This is strictly a presentation/decision-support number,
kept separate from the Bayesian math to avoid double-counting.

Sprint 6 will implement:
    severity_composite = f(flare_probability, severity_signal, confidence_level)
with confidence gating (INSUFFICIENT suppresses the composite display).
"""

from __future__ import annotations

from typing import Optional


class RiskEngine:
    """Computes the UI-facing severity composite. Sprint 6 implementation pending."""

    def compute(
        self,
        flare_probability: Optional[float],
        confidence_result,
        agent_outputs: dict,
    ) -> Optional[float]:
        """Return the severity composite, or None until Sprint 6.

        Args:
            flare_probability: The calibrated probability from StatisticalFusion
                (None in Sprint 5 since fusion is stubbed).
            confidence_result: The ConfidenceResult (gates display).
            agent_outputs: Reporting agents' outputs (severity signals).

        Returns:
            None in Sprint 5 (stub). A float composite from Sprint 6 onward.
        """
        return None
