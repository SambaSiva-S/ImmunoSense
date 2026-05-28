"""Decision policy — when to call TFM, when to raise an alert.

STATUS: Sprint 6 (NOT YET IMPLEMENTED).
Sprint 5 ships this as a stub returning a no-op decision.

Locked design (Challenge 3, Phase 3 + orchestration policy): this is the
policy layer that decides, given a fused evaluation, whether to:
    - call the TFM (Thinking Machine) for a narrative explanation
      (selective — only when warranted, to control cost), and
    - raise a clinical/patient alert.

It is deliberately separated from the fusion math and the quality scoring:
fusion computes numbers, quality computes confidence, and the decision maker
turns those into ACTIONS. Keeping policy out of the math keeps both auditable.

Sprint 6 will implement triggers such as:
    - call TFM if severity_composite is high OR a corroboration pattern fired
      OR the result is a state transition
    - suppress everything if confidence is INSUFFICIENT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    """The orchestration decision for a bucket.

    Fields:
        call_tfm: Whether to invoke the Thinking Machine for explanation.
        raise_alert: Whether to surface a clinical/patient alert.
        reasons: Human-readable reasons for the decision (audit trail).
    """

    call_tfm: bool = False
    raise_alert: bool = False
    reasons: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class DecisionMaker:
    """Turns a fused evaluation into actions. Sprint 6 implementation pending."""

    def decide(
        self,
        flare_probability,
        severity_composite,
        matched_patterns: list,
        confidence_result,
    ) -> Decision:
        """Return the orchestration decision.

        Sprint 5 stub: always a no-op decision (no TFM call, no alert) with a
        reason noting that the decision layer is not yet active.
        """
        return Decision(
            call_tfm=False,
            raise_alert=False,
            reasons=["decision layer is a Sprint 6 stub (no-op)"],
        )
