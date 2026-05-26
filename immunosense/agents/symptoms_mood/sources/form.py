"""StructuredFormSource - direct slider/checkbox input.

Higher confidence than voice extraction. Use when patient explicitly fills
a form rather than narrating symptoms.
"""

from __future__ import annotations

from immunosense.agents.symptoms_mood.types import ALL_FEATURES, FetchedSymptoms


class StructuredFormSource:
    """Direct structured input from patient-filled form (sliders, checkboxes)."""

    def from_dict(self, form_data: dict) -> FetchedSymptoms:
        """Convert a form dict to FetchedSymptoms with 'structured' confidence."""
        result = FetchedSymptoms(
            fatigue=form_data.get("fatigue"),
            joint_pain=form_data.get("joint_pain"),
            brain_fog_severity=form_data.get("brain_fog_severity"),
            gi_distress=form_data.get("gi_distress"),
            skin_severity=form_data.get("skin_severity"),
            sleep_severity=form_data.get("sleep_severity"),
            energy_severity=form_data.get("energy_severity"),
            wellness_severity=form_data.get("wellness_severity"),
            phq8_score=form_data.get("phq8_score"),
            gad7_score=form_data.get("gad7_score"),
            explicit_flare=form_data.get("explicit_flare", False),
            explicit_flare_severity=form_data.get("explicit_flare_severity"),
        )
        for feat in ALL_FEATURES:
            if getattr(result, feat) is not None:
                result.confidence[feat] = "structured"
                result.sources[feat] = "form"
        return result
