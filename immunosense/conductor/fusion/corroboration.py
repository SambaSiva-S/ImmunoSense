"""Cross-agent corroboration patterns — Challenge 3 Phase 2 (semantic only).

When multiple agents INDEPENDENTLY point at the same physiological story, that
co-firing is more meaningful than any single signal. This module names those
stories as human-readable patterns for the report and as context for the TFM.

THE CRITICAL RULE (do not violate):
    Corroboration is SEMANTIC ONLY. It NEVER modifies the Bayesian flare
    probability from Phase 1. The probability already accounts for each agent's
    contribution; letting a "pattern" add more probability would double-count
    the very same signals. Phase 2 enriches the EXPLANATION, not the math.
    This separation is the structural fix for the double-counting bug found in
    the Challenge 3 design.

WHAT A PATTERN IS:
    A named template describing a set of agents that, when each shows an
    "elevated" signal in the same bucket, together suggest a recognizable
    mechanism (e.g. autonomic stress, inflammatory surge). Matching is based on
    each agent's signal strength crossing a threshold and/or carrying alerts.

PROVISIONAL STATUS:
    These 7 patterns are literature-informed where noted and reasoned defaults
    otherwise. They are CROSS-DISEASE for v1 (disease-specific variants are a
    later refinement) and are NOT clinically validated. They describe
    associations, never causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from immunosense.conductor.fusion.statistical_fusion import extract_signal_strength

# Signal-strength threshold for an agent to "participate" in a pattern.
_PARTICIPATION_THRESHOLD = 0.55


@dataclass(frozen=True)
class CorroborationPattern:
    """A named co-firing template.

    Fields:
        name: Stable identifier (snake_case).
        label: Human-readable name for display.
        required_agents: Agents that must ALL be elevated for a match.
        optional_agents: Agents that strengthen the match if also elevated
            but are not required.
        description: One-line plain-language meaning (for TFM/report).
        source: Provenance ("literature-informed:..." or "default:...").
    """

    name: str
    label: str
    required_agents: tuple
    optional_agents: tuple = ()
    description: str = ""
    source: str = "default"


# The v1 cross-disease pattern library (7 patterns).
_PATTERN_LIBRARY = [
    CorroborationPattern(
        name="autonomic_stress",
        label="Autonomic stress pattern",
        required_agents=("agent4_wearable", "agent5_symptoms_mood"),
        description=(
            "HRV suppression together with rising self-reported symptoms — "
            "an autonomic stress signature that often precedes flares."
        ),
        source="literature-informed: HRV + symptom co-movement near flare onset",
    ),
    CorroborationPattern(
        name="inflammatory_surge",
        label="Inflammatory surge pattern",
        required_agents=("agent1_biomarker", "agent5_symptoms_mood"),
        optional_agents=("agent4_wearable",),
        description=(
            "Elevated inflammatory biomarkers alongside worsening symptoms — "
            "convergent evidence of active inflammation."
        ),
        source="literature-informed: inflammatory markers track symptomatic activity",
    ),
    CorroborationPattern(
        name="environmental_trigger",
        label="Environmental trigger pattern",
        required_agents=("agent3_environment", "agent5_symptoms_mood"),
        description=(
            "An environmental exposure spike coinciding with symptom worsening — "
            "suggests an external trigger."
        ),
        source="default: environmental exposure as a proximate trigger",
    ),
    CorroborationPattern(
        name="dietary_inflammatory",
        label="Dietary inflammatory pattern",
        required_agents=("agent2_dietary", "agent5_symptoms_mood"),
        description=(
            "A pro-inflammatory dietary period overlapping symptom worsening."
        ),
        source="default: dietary inflammatory index association",
    ),
    CorroborationPattern(
        name="physiological_multisystem",
        label="Multi-system physiological pattern",
        required_agents=("agent1_biomarker", "agent4_wearable", "agent5_symptoms_mood"),
        description=(
            "Biomarkers, wearable physiology, and symptoms all elevated together "
            "— a broad multi-system signal warranting attention."
        ),
        source="default: multi-system convergence",
    ),
    CorroborationPattern(
        name="lifestyle_load",
        label="Lifestyle load pattern",
        required_agents=("agent2_dietary", "agent3_environment"),
        optional_agents=("agent4_wearable",),
        description=(
            "Dietary and environmental loads co-occurring — accumulating "
            "modifiable burden even before strong symptoms appear."
        ),
        source="default: modifiable-burden accumulation",
    ),
    CorroborationPattern(
        name="silent_physiological",
        label="Silent physiological pattern",
        required_agents=("agent1_biomarker", "agent4_wearable"),
        description=(
            "Biomarker and wearable physiology elevated WITHOUT strong symptoms "
            "— a potentially pre-symptomatic signal."
        ),
        source="default: pre-symptomatic physiological drift",
    ),
]


@dataclass
class MatchedPattern:
    """A pattern that fired for a bucket (audit + display)."""

    name: str
    label: str
    description: str
    participating_agents: list = field(default_factory=list)
    source: str = ""


class Corroboration:
    """Matches reporting agents' signals against the pattern library."""

    def __init__(self, patterns: Optional[list] = None,
                 participation_threshold: float = _PARTICIPATION_THRESHOLD):
        self.patterns = patterns if patterns is not None else list(_PATTERN_LIBRARY)
        self.participation_threshold = participation_threshold

    def _elevated_agents(self, agent_outputs: dict) -> set:
        """Set of agent_ids whose signal strength clears the threshold."""
        elevated = set()
        for agent_id, output in agent_outputs.items():
            if extract_signal_strength(output) >= self.participation_threshold:
                elevated.add(agent_id)
        return elevated

    def match(self, agent_outputs: dict) -> list:
        """Return the list of MatchedPattern objects that fired this bucket.

        A pattern matches when ALL its required agents are elevated. Optional
        agents that are also elevated are recorded as participating but are not
        required. This NEVER returns or affects a probability — semantic only.
        """
        elevated = self._elevated_agents(agent_outputs)
        matched = []
        for pat in self.patterns:
            if all(a in elevated for a in pat.required_agents):
                participating = list(pat.required_agents) + [
                    a for a in pat.optional_agents if a in elevated
                ]
                matched.append(
                    MatchedPattern(
                        name=pat.name,
                        label=pat.label,
                        description=pat.description,
                        participating_agents=sorted(participating),
                        source=pat.source,
                    )
                )
        return matched
