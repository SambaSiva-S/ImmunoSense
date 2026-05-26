"""FlareButtonSource - explicit override signal.

The flare button is a high-priority override: when the patient presses it,
the system records an explicit flare event that bypasses the standard
flare_score computation and forces the score to at least 0.80.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.symptoms_mood.types import FetchedSymptoms


class FlareButtonSource:
    """Explicit flare event source. The patient pressed the 'I'm having a flare' button."""

    def from_event(self, severity: Optional[float] = None) -> FetchedSymptoms:
        """Create a FetchedSymptoms reflecting a flare button press.

        Args:
            severity: Optional 0-1 severity. If None, the daily_flare_score
                computation uses the button_override_floor (0.80) as the floor.
        """
        result = FetchedSymptoms(
            explicit_flare=True,
            explicit_flare_severity=severity,
        )
        result.confidence["explicit_flare"] = "flare_button"
        result.sources["explicit_flare"] = "patient_button"
        return result
