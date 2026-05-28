"""Cross-agent corroboration patterns — Phase 2 of Challenge 3.

STATUS: Sprint 6 (NOT YET IMPLEMENTED).
Sprint 5 ships this as a stub returning an empty list.

Locked design (Challenge 3, Phase 2): SEMANTIC corroboration only. When
multiple agents independently point at the same story (e.g. wearable HRV crash
+ symptom fatigue spike + environmental pollen surge), that pattern is surfaced
as a named, human-readable corroboration. Crucially, this is semantic ONLY — it
does NOT modify the Bayesian probability from Phase 1. Phase 2 enriches the
explanation; it never re-weights the math. This separation is the fix for the
double-counting bug caught during the Challenge 3 design.

Sprint 6 will implement:
    - A library of corroboration patterns (co-firing signal templates)
    - Matching reporting agents' alerts/vectors against those templates
    - Returning matched pattern names for the report + TFM context
"""

from __future__ import annotations


class Corroboration:
    """Semantic cross-agent pattern matching. Sprint 6 implementation pending."""

    def match(self, agent_outputs: dict) -> list:
        """Return matched corroboration pattern names.

        Args:
            agent_outputs: Map agent_id -> AgentOutput for reporting agents.

        Returns:
            Empty list in Sprint 5 (stub). A list of pattern-name strings
            from Sprint 6 onward. Never affects the Bayesian probability.
        """
        return []
