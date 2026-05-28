"""PersonalAdaptationEngine - orchestrates Layer 3 sub-components.

Composes BiomarkerBaselineTracker + PatternDetector and emits alerts based
on the combined personal context.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.biomarker.constants import (
    BIOMARKERS_FOR_TRACKING,
    BIOMARKER_TRIGGERS,
)
from immunosense.agents.biomarker.layer3.detector import PatternDetector
from immunosense.agents.biomarker.layer3.trackers import BiomarkerBaselineTracker


class PersonalAdaptationEngine:
    """Layer 3 orchestrator: tracker + detector + alert generation.

    Args:
        biomarkers: biomarkers to track (default BIOMARKERS_FOR_TRACKING)
        triggers:   triggers to analyze (default BIOMARKER_TRIGGERS)
        tracker_kwargs: passed to BiomarkerBaselineTracker
        detector_kwargs: passed to PatternDetector
    """

    def __init__(
        self,
        biomarkers: Optional[list] = None,
        triggers: Optional[list] = None,
        tracker_kwargs: Optional[dict] = None,
        detector_kwargs: Optional[dict] = None,
    ) -> None:
        self.biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)
        self.triggers = list(triggers or BIOMARKER_TRIGGERS)
        self.tracker = BiomarkerBaselineTracker(
            biomarkers=self.biomarkers, **(tracker_kwargs or {}),
        )
        self.detector = PatternDetector(
            biomarkers=self.biomarkers,
            triggers=self.triggers,
            **(detector_kwargs or {}),
        )
        self.trajectory_history: list = []

    def process_reading(self, reading: dict) -> dict:
        """Update state and return combined personal context.

        Args:
            reading: dict with biomarker + trigger keys.

        Returns:
            dict with keys 'has_personal_data', 'biomarkers', 'patterns', 'alerts'.
        """
        self.trajectory_history.append(reading)
        self.tracker.update(reading)

        context = self.tracker.get_personal_context(reading)

        # Pattern detection only kicks in at >=10 readings
        if len(self.trajectory_history) >= self.detector.min_readings:
            pattern_results = self.detector.analyze(self.trajectory_history)
            context["patterns"] = pattern_results
        else:
            context["patterns"] = {
                "has_patterns": False,
                "message": "Collecting data...",
                "patterns": [],
                "flare_rule": None,
                "n_readings_analyzed": len(self.trajectory_history),
            }

        context["alerts"] = self._generate_alerts(context, reading)
        return context

    def _generate_alerts(self, context: dict, reading: dict) -> list:
        """Generate human-readable alerts from the combined context."""
        alerts = []

        if not context.get("has_personal_data", False):
            return alerts

        # Biomarker-level alerts
        for bm, bm_ctx in context.get("biomarkers", {}).items():
            if bm_ctx["interpretation"] == "CRITICAL":
                alerts.append({
                    "level": "CRITICAL",
                    "message": (
                        f"{bm} is {bm_ctx['value']} - "
                        f"{bm_ctx['anomaly_score']:.1f}x IQR above YOUR baseline "
                        f"({bm_ctx['median_baseline']:.1f})"
                    ),
                    "biomarker": bm,
                })
            elif bm_ctx["interpretation"] == "ELEVATED" and bm_ctx["trend"] == "RISING":
                alerts.append({
                    "level": "WARNING",
                    "message": (
                        f"{bm} is elevated ({bm_ctx['value']}) and RISING - "
                        f"monitor closely"
                    ),
                    "biomarker": bm,
                })

        # Trigger-based alerts
        pattern_results = context.get("patterns", {})
        if pattern_results.get("has_patterns", False):
            for pattern in pattern_results.get("patterns", [])[:3]:
                trigger_name = pattern["trigger"].replace("_", " ")
                if reading.get(pattern["trigger"], False):
                    alerts.append({
                        "level": "INFO",
                        "message": (
                            f"{trigger_name.title()} detected - historically "
                            f"correlates with {pattern['biomarker']} change "
                            f"(lag: {pattern['lag_readings']} readings, "
                            f"r={pattern['correlation']:.2f})"
                        ),
                    })

        return alerts
