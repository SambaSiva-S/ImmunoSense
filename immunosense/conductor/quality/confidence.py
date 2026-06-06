"""Overall confidence aggregation — the 4-level rule from Challenge 7.

Takes the per-agent quality scores and collapses them into one of four
levels: INSUFFICIENT / LOW / MODERATE / HIGH. INSUFFICIENT is the safety
floor — when the system reaches it, the Conductor should suppress probability
display because there isn't enough trustworthy evidence to make a claim.

The aggregation rule (locked as the SIMPLEST defensible version, to be
refined later by the Auto-Research Loop once real data exists):

    Count agents whose freshness-adjusted quality lands in each band:
        high       : quality >= 0.75
        moderate   : 0.50 <= quality < 0.75
        low        : 0.25 <= quality < 0.50
        (below 0.25 contributes to neither — effectively "no signal")

    Then:
        HIGH         if >= 3 agents are high
        MODERATE     if >= 3 agents are at least moderate
        LOW          if >= 2 agents are at least low
        INSUFFICIENT otherwise

The "3 of 5" thresholds are a starting heuristic. All per-agent scores are
logged to Layer A so the rule can be recalibrated against outcomes without
re-running history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from immunosense.conductor.quality.scorer import AgentQuality
from immunosense.events.types import ConfidenceLevel

# Band thresholds on the freshness-adjusted quality score.
HIGH_BAND = 0.75
MODERATE_BAND = 0.50
LOW_BAND = 0.25

# Confidence requirements SCALE to the number of reporting agents, rather than
# using absolute counts. The system was designed for ~5 agents but Phase 1 ships
# with 3 active (biomarker, dietary, symptoms/mood); a hardcoded "needs 3" made
# HIGH/MODERATE unreachable in practice and capped diligent users at LOW. Scaling
# keeps the same intent — "most of the available agents are confident" — at any
# agent count, while preserving a safety floor: at least 2 agents must agree for
# any level above INSUFFICIENT (a single agent can never alone drive confidence).
import math

MIN_AGENTS_FOR_CONFIDENCE = 2  # absolute safety floor


def _scaled_requirements(n_reporting: int) -> tuple[int, int, int]:
    """Return (high_req, moderate_req, low_req) agent counts for n reporting
    agents. HIGH ~ ceil(0.8*n) (nearly all), MODERATE ~ ceil(0.6*n) (a majority),
    LOW ~ 2 (the floor). All clamped to >= the safety floor and <= n.

    Worked examples:
      n=2 -> high 2, moderate 2, low 2
      n=3 -> high 3, moderate 2, low 2
      n=5 -> high 4, moderate 3, low 2
    """
    high = max(MIN_AGENTS_FOR_CONFIDENCE, math.ceil(0.8 * n_reporting))
    moderate = max(MIN_AGENTS_FOR_CONFIDENCE, math.ceil(0.6 * n_reporting))
    low = MIN_AGENTS_FOR_CONFIDENCE
    return min(high, n_reporting), min(moderate, n_reporting), min(low, n_reporting)


@dataclass
class ConfidenceResult:
    """The aggregated confidence outcome for a bucket.

    Fields:
        level: The 4-level ConfidenceLevel.
        overall_quality: Mean quality across REPORTING agents (0 if none).
        n_high / n_moderate / n_low: Band counts (cumulative bands are
            derived in the rule, these are the exclusive-band counts).
        n_reporting: How many agents produced output this bucket.
        per_agent: The full list of AgentQuality records (for the audit log).
    """

    level: ConfidenceLevel
    overall_quality: float
    n_high: int
    n_moderate: int
    n_low: int
    n_reporting: int
    per_agent: list


class ConfidenceAggregator:
    """Collapses per-agent quality into the 4-level confidence."""

    def aggregate(self, qualities: Iterable[AgentQuality]) -> ConfidenceResult:
        qualities = list(qualities)
        reporting = [q for q in qualities if q.reported]

        # Band counts over REPORTING agents only, so they're consistent with the
        # reporting-count-scaled requirements below (a non-reporting agent must
        # not count toward a band).
        n_high = sum(1 for q in reporting if q.quality >= HIGH_BAND)
        n_moderate = sum(
            1 for q in reporting if MODERATE_BAND <= q.quality < HIGH_BAND
        )
        n_low = sum(1 for q in reporting if LOW_BAND <= q.quality < MODERATE_BAND)

        # Cumulative counts for the threshold rule.
        at_least_moderate = n_high + n_moderate
        at_least_low = n_high + n_moderate + n_low

        # Requirements scale to how many agents contributed MEANINGFUL signal
        # this bucket (quality at least the LOW band) — not raw reported count.
        # A near-zero-quality agent that technically reported shouldn't raise the
        # bar for genuinely-confident agents (otherwise two dead signals would
        # penalize three strong ones). Floor still applies to the reporting set.
        n_signal = n_high + n_moderate + n_low
        high_req, moderate_req, low_req = _scaled_requirements(n_signal)

        # Safety floor: never produce a positive confidence level from fewer than
        # MIN_AGENTS_FOR_CONFIDENCE agents with meaningful signal.
        if n_signal < MIN_AGENTS_FOR_CONFIDENCE:
            level = ConfidenceLevel.INSUFFICIENT
        elif n_high >= high_req:
            level = ConfidenceLevel.HIGH
        elif at_least_moderate >= moderate_req:
            level = ConfidenceLevel.MODERATE
        elif at_least_low >= low_req:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.INSUFFICIENT

        if reporting:
            overall = sum(q.quality for q in reporting) / len(reporting)
        else:
            overall = 0.0

        return ConfidenceResult(
            level=level,
            overall_quality=overall,
            n_high=n_high,
            n_moderate=n_moderate,
            n_low=n_low,
            n_reporting=len(reporting),
            per_agent=qualities,
        )
