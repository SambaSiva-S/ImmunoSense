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

# How many agents must clear a band for each overall level.
HIGH_REQUIRES = 3       # >=3 agents high
MODERATE_REQUIRES = 3   # >=3 agents at least moderate
LOW_REQUIRES = 2        # >=2 agents at least low


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

        n_high = sum(1 for q in qualities if q.quality >= HIGH_BAND)
        n_moderate = sum(
            1 for q in qualities if MODERATE_BAND <= q.quality < HIGH_BAND
        )
        n_low = sum(1 for q in qualities if LOW_BAND <= q.quality < MODERATE_BAND)

        # Cumulative counts for the threshold rule.
        at_least_moderate = n_high + n_moderate
        at_least_low = n_high + n_moderate + n_low

        if n_high >= HIGH_REQUIRES:
            level = ConfidenceLevel.HIGH
        elif at_least_moderate >= MODERATE_REQUIRES:
            level = ConfidenceLevel.MODERATE
        elif at_least_low >= LOW_REQUIRES:
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
