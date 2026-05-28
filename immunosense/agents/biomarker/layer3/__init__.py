"""Layer 3: Personal adaptation engine.

Three sub-components:
    BiomarkerBaselineTracker:  per-patient robust median + IQR baselines
                                (uses agents.common.trackers.RobustBaselineTracker)
    PatternDetector:           trigger -> biomarker correlation across time lags
    PersonalAdaptationEngine:  orchestrator + alert generation

Activates progressively as data accumulates:
    < 3 readings:   no personal data yet
    3+ readings:    baseline tracker active, personal_weight ramps 0 -> 0.8
    10+ readings:   pattern detector active
"""

from immunosense.agents.biomarker.layer3.detector import PatternDetector
from immunosense.agents.biomarker.layer3.engine import PersonalAdaptationEngine
from immunosense.agents.biomarker.layer3.trackers import BiomarkerBaselineTracker

__all__ = [
    "BiomarkerBaselineTracker",
    "PatternDetector",
    "PersonalAdaptationEngine",
]
