"""Per-agent quality scoring.

Each agent already self-reports a ``confidence`` (0..1) inside its AgentOutput.
The scorer's job (Challenge 7) is to take that self-reported confidence and
apply a FRESHNESS PENALTY relative to the agent's declared poll_frequency, so
a reading that is old *for that agent's cadence* counts for less.

    quality = confidence * freshness_weight(poll_frequency, produced_at, ref)

A 6-day-old biomarker reading (weekly cadence) keeps most of its weight; a
6-day-old wearable reading (hourly cadence) loses almost all of it. This is
the concrete realization of Challenge 7's freshness halflife idea, made
possible by the poll_frequency attribute the Sprint 5 audit surfaced.

The scorer never invents quality from scratch — it adjusts the agent's own
number. If an agent failed (degraded output, confidence 0.0), quality is 0.0
regardless of freshness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from immunosense.adapters.base import AdapterResult
from immunosense.events.bucket import AgentData, freshness_weight


@dataclass
class AgentQuality:
    """The quality breakdown for one agent in one bucket.

    Fields:
        agent_id: Which agent.
        raw_confidence: The agent's self-reported confidence (0..1).
        freshness: The freshness weight applied (0..1).
        quality: raw_confidence * freshness, the final per-agent score.
        reported: True if the agent produced output this bucket (even if
            low quality). False is reserved for agents absent from the bucket.
        ok: True if the agent's process() succeeded.
    """

    agent_id: str
    raw_confidence: float
    freshness: float
    quality: float
    reported: bool
    ok: bool


class QualityScorer:
    """Computes per-agent quality from adapter results + freshness."""

    def score(
        self,
        result: AdapterResult,
        agent_data: AgentData,
        poll_frequency: str,
        reference: datetime,
    ) -> AgentQuality:
        """Score one agent's contribution for a bucket.

        Args:
            result: The AdapterResult from running the agent.
            agent_data: The AgentData fed in (carries produced_at).
            poll_frequency: The agent's declared cadence (drives halflife).
            reference: The "now" for age computation (typically bucket end).

        Returns:
            AgentQuality with the freshness-adjusted score.
        """
        raw = float(result.output.confidence)
        # A failed agent contributes nothing, regardless of freshness.
        if not result.ok:
            return AgentQuality(
                agent_id=result.agent_id,
                raw_confidence=raw,
                freshness=0.0,
                quality=0.0,
                reported=True,
                ok=False,
            )

        fresh = freshness_weight(poll_frequency, agent_data.produced_at, reference)
        quality = max(0.0, min(1.0, raw * fresh))
        return AgentQuality(
            agent_id=result.agent_id,
            raw_confidence=raw,
            freshness=fresh,
            quality=quality,
            reported=True,
            ok=True,
        )

    @staticmethod
    def absent(agent_id: str) -> AgentQuality:
        """Quality record for an agent that had no data this bucket."""
        return AgentQuality(
            agent_id=agent_id,
            raw_confidence=0.0,
            freshness=0.0,
            quality=0.0,
            reported=False,
            ok=False,
        )
