"""Versioned likelihood-ratio calibration table (Challenge 3 Phase 1, Q1).

The Bayesian fusion needs, for each agent, a likelihood ratio (LR): how much
more (or less) likely is an imminent flare GIVEN that agent's signal, compared
to baseline. This module holds those LRs in a VERSIONED table so that:

    1. The provenance of every number is explicit (default vs literature).
    2. The Auto-Research loop (Sprint 9) has a concrete object to recalibrate,
       writing new versions rather than mutating values in place.
    3. A patient's historical inferences can be re-interpreted against the LR
       version that was live when they were computed.

IMPORTANT HONESTY NOTE: these are PROVISIONAL starting values. A few are
informed by the direction and rough magnitude of published autoimmune /
rheumatology findings (e.g. elevated CRP/ESR associating with disease activity;
HRV suppression preceding flares); most are reasoned defaults. NONE should be
treated as clinically validated. They are deliberately conservative (LRs close
to 1.0) so the system does not make strong claims it cannot yet support. The
`source` field records the basis for each.

LR semantics:
    LR > 1  : signal raises flare probability
    LR = 1  : signal is uninformative
    LR < 1  : signal lowers flare probability (protective / reassuring)

An agent contributes its LR only to the extent of its quality (Challenge 7):
a low-quality signal is shrunk toward LR=1 (uninformative). See
StatisticalFusion for how quality tempers the LR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CALIBRATION_VERSION = "lr-v1"

# Baseline prior probability of an imminent flare in any given 6h bucket,
# absent any agent signal. Deliberately low (flares are not the common case).
# This is a reasoned default, NOT an epidemiological estimate.
DEFAULT_BASELINE_FLARE_PRIOR = 0.08


@dataclass(frozen=True)
class AgentLR:
    """Likelihood ratio specification for one agent.

    Fields:
        agent_id: Which agent.
        lr_positive: LR applied when the agent's signal is "elevated"
            (its scalar signal strength is high). >1 raises flare odds.
        lr_negative: LR applied when the agent's signal is clearly low /
            reassuring. <=1 lowers flare odds. Defaults to 1.0 (no effect).
        signal_threshold: The signal-strength value (0..1) above which
            lr_positive applies and below `low_threshold` lr_negative applies.
        low_threshold: Below this, the agent is considered reassuring.
        source: Provenance of these numbers ("default" or "literature:...").
    """

    agent_id: str
    lr_positive: float
    lr_negative: float = 1.0
    signal_threshold: float = 0.6
    low_threshold: float = 0.2
    source: str = "default"


# The v1 table. One entry per agent. Values are intentionally modest.
_LR_TABLE_V1 = {
    "agent1_biomarker": AgentLR(
        agent_id="agent1_biomarker",
        lr_positive=2.2,   # elevated inflammatory biomarkers (CRP/ESR) raise odds
        lr_negative=0.6,   # normal labs are mildly reassuring
        signal_threshold=0.6,
        low_threshold=0.2,
        source="literature-informed: inflammatory markers track disease activity",
    ),
    "agent2_dietary": AgentLR(
        agent_id="agent2_dietary",
        lr_positive=1.4,   # pro-inflammatory dietary pattern: weak signal
        lr_negative=0.9,
        signal_threshold=0.6,
        low_threshold=0.2,
        source="default: dietary inflammatory index is a weak short-horizon signal",
    ),
    "agent3_environment": AgentLR(
        agent_id="agent3_environment",
        lr_positive=1.5,   # environmental triggers (pollen, pollution, heat)
        lr_negative=0.95,
        signal_threshold=0.6,
        low_threshold=0.2,
        source="default: environmental exposure as a flare trigger",
    ),
    "agent4_wearable": AgentLR(
        agent_id="agent4_wearable",
        lr_positive=2.0,   # HRV suppression / autonomic stress precedes flares
        lr_negative=0.7,   # healthy HRV is reassuring
        signal_threshold=0.6,
        low_threshold=0.2,
        source="literature-informed: HRV suppression associates with flare onset",
    ),
    "agent5_symptoms_mood": AgentLR(
        agent_id="agent5_symptoms_mood",
        lr_positive=2.5,   # rising self-reported symptoms: strongest near-term signal
        lr_negative=0.6,
        signal_threshold=0.55,
        low_threshold=0.2,
        source="literature-informed: patient-reported symptoms are leading indicators",
    ),
}

# Registry of all calibration versions (Auto-Research appends new ones later).
_VERSIONS = {
    "lr-v1": _LR_TABLE_V1,
}


@dataclass
class CalibrationTable:
    """A loaded, versioned LR table the fusion engine consults."""

    version: str
    baseline_prior: float
    table: dict = field(default_factory=dict)

    def get(self, agent_id: str) -> Optional[AgentLR]:
        return self.table.get(agent_id)

    def has(self, agent_id: str) -> bool:
        return agent_id in self.table

    @property
    def agent_ids(self) -> list:
        return sorted(self.table.keys())


def load_calibration(version: str = CALIBRATION_VERSION) -> CalibrationTable:
    """Load a calibration table by version.

    Defaults to the current version. Raises if an unknown version is asked
    for (so a stale reference fails loudly rather than silently mis-scoring).
    """
    if version not in _VERSIONS:
        raise ValueError(
            f"Unknown calibration version {version!r}. "
            f"Known: {sorted(_VERSIONS)}"
        )
    return CalibrationTable(
        version=version,
        baseline_prior=DEFAULT_BASELINE_FLARE_PRIOR,
        table=dict(_VERSIONS[version]),
    )
