"""Cross-agent fusion (Challenge 3): Bayesian probability, corroboration, risk.

Phase 1 (statistical_fusion) is the math truth. Phase 2 (corroboration) is
semantic only and never affects the probability. Phase 4 (risk_engine) consumes
the probability for a UI composite and never re-derives it.
"""

from immunosense.conductor.fusion.corroboration import (
    Corroboration,
    CorroborationPattern,
    MatchedPattern,
)
from immunosense.conductor.fusion.risk_engine import RiskEngine, RiskResult
from immunosense.conductor.fusion.statistical_fusion import (
    AgentContribution,
    FusionResult,
    StatisticalFusion,
    extract_signal_strength,
)

__all__ = [
    "StatisticalFusion",
    "FusionResult",
    "AgentContribution",
    "extract_signal_strength",
    "Corroboration",
    "CorroborationPattern",
    "MatchedPattern",
    "RiskEngine",
    "RiskResult",
]
