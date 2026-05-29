"""Bayesian cross-agent fusion — Challenge 3 Phase 1 (the math truth).

This is the ONLY place flare probability is computed. Corroboration patterns
(Phase 2) are semantic and must never feed back here; the risk engine (Phase 4)
consumes this output but never re-derives it. Keeping the probability math
isolated to this module is the structural fix for the double-counting bug
identified during the Challenge 3 design.

THE MATH
--------
We combine evidence in LOG-ODDS space, which is the natural domain for Bayesian
updating with likelihood ratios:

    posterior_odds = prior_odds * product(LR_i)         (odds form of Bayes)
    log(posterior_odds) = log(prior_odds) + sum(log LR_i)

Steps:
    1. Start from the baseline prior probability of a flare in this bucket,
       converted to log-odds.
    2. For each REPORTING agent, derive a scalar "signal strength" in [0,1]
       from its AgentOutput, decide whether it is elevated / reassuring /
       neutral, and select the appropriate LR from the calibration table.
    3. TEMPER each LR by the agent's quality (Challenge 7): a low-quality
       signal is shrunk toward LR=1 (uninformative) in log space:
           effective_log_LR = quality * log(LR)
       So a quality-0 agent contributes nothing; a quality-1 agent contributes
       its full LR; partial quality contributes proportionally. This is how the
       confidence layer flows into the probability without a second mechanism.
    4. Sum the effective log-LRs onto the prior log-odds and convert back to a
       probability.

CONFIDENCE GATING
-----------------
If overall confidence is INSUFFICIENT (Challenge 7 safety floor), we return
None — the system must not display a probability it cannot support. This is
returned explicitly so the UI can show "not enough data" rather than a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from immunosense.conductor.calibration.likelihood_ratios import (
    CalibrationTable,
    load_calibration,
)
from immunosense.events.types import ConfidenceLevel


@dataclass
class AgentContribution:
    """Audit record of how one agent affected the fused probability."""

    agent_id: str
    signal_strength: float     # 0..1 scalar extracted from the output
    direction: str             # "elevated" | "reassuring" | "neutral"
    raw_lr: float              # LR selected from the table
    quality: float             # the agent's quality (tempering factor)
    effective_log_lr: float    # quality * log(raw_lr) — what actually applied


@dataclass
class FusionResult:
    """The output of Bayesian fusion for one bucket.

    Fields:
        flare_probability: Posterior probability in [0,1], or None if gated
            by INSUFFICIENT confidence.
        prior: The baseline prior used.
        posterior_log_odds: Final log-odds (for debugging/audit).
        contributions: Per-agent audit records.
        calibration_version: Which LR table version produced this.
        gated: True if returned None due to insufficient confidence.
    """

    flare_probability: Optional[float]
    prior: float
    posterior_log_odds: Optional[float]
    contributions: list = field(default_factory=list)
    calibration_version: str = ""
    gated: bool = False


# --------------------------------------------------------------------------- #
# Signal-strength extraction
# --------------------------------------------------------------------------- #
# Different agents emit different vectors, but two cross-cutting cues are
# available on every AgentOutput: (a) the alerts list, whose severities map to
# concern, and (b) the vector magnitude. We combine them into a single 0..1
# signal-strength scalar. This is intentionally simple and documented as
# refinable; per-agent bespoke extractors can replace it later without changing
# the fusion math.

_SEVERITY_WEIGHT = {
    "critical": 1.0,
    "severe": 1.0,
    "warning": 0.5,
    "moderate": 0.5,
}


def _alert_signal(alerts: list) -> Optional[float]:
    """Derive 0..1 concern from an agent's alerts, or None if no alerts."""
    if not alerts:
        return None
    weights = []
    for a in alerts:
        # Agents use heterogeneous keys: normalize 'severity' or 'level'.
        sev = str(a.get("severity") or a.get("level") or "").lower()
        weights.append(_SEVERITY_WEIGHT.get(sev, 0.3))
    # Strongest alert dominates, but multiple alerts nudge it up slightly.
    top = max(weights)
    bonus = min(0.15, 0.05 * (len(weights) - 1))
    return min(1.0, top + bonus)


def _vector_signal(vector) -> Optional[float]:
    """Derive 0..1 signal from vector magnitude, or None if unavailable.

    Uses a squashed mean-absolute-value so the scale is bounded and robust to
    dimension. This is a generic fallback; it is deliberately weak.
    """
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    mav = float(np.nanmean(np.abs(arr)))
    # Squash with a soft saturating curve; mav~0 -> 0, large -> ->1.
    return float(1.0 - math.exp(-mav))


def extract_signal_strength(output) -> float:
    """Combine alert and vector cues into one 0..1 signal strength.

    Alerts are the stronger cue when present (they are explicit clinical
    threshold crossings); vector magnitude is a weaker fallback.
    """
    a = _alert_signal(output.alerts)
    v = _vector_signal(output.vector)
    if a is not None and v is not None:
        return float(min(1.0, 0.7 * a + 0.3 * v))
    if a is not None:
        return a
    if v is not None:
        return v
    return 0.0


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _prob_to_log_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _log_odds_to_prob(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


class StatisticalFusion:
    """Quality-tempered Bayesian fusion of agent signals into flare probability."""

    def __init__(self, calibration: Optional[CalibrationTable] = None):
        self.calibration = calibration or load_calibration()

    def fuse(self, agent_outputs: dict, confidence_result) -> FusionResult:
        """Compute the posterior flare probability for a bucket.

        Args:
            agent_outputs: Map agent_id -> AgentOutput (reporting agents only).
            confidence_result: The ConfidenceResult from the quality layer.
                Provides the overall level (for gating) and per-agent quality.

        Returns:
            FusionResult. flare_probability is None when gated by INSUFFICIENT.
        """
        prior = self.calibration.baseline_prior

        # Confidence gating (Challenge 7 safety floor).
        if confidence_result.level == ConfidenceLevel.INSUFFICIENT:
            return FusionResult(
                flare_probability=None,
                prior=prior,
                posterior_log_odds=None,
                contributions=[],
                calibration_version=self.calibration.version,
                gated=True,
            )

        # Per-agent quality lookup from the confidence result.
        quality_by_agent = {
            q.agent_id: q.quality for q in confidence_result.per_agent
        }

        log_odds = _prob_to_log_odds(prior)
        contributions = []

        for agent_id, output in agent_outputs.items():
            lr_spec = self.calibration.get(agent_id)
            if lr_spec is None:
                continue  # no calibration for this agent -> contributes nothing

            signal = extract_signal_strength(output)
            quality = float(quality_by_agent.get(agent_id, 0.0))

            if signal >= lr_spec.signal_threshold:
                raw_lr = lr_spec.lr_positive
                direction = "elevated"
            elif signal <= lr_spec.low_threshold:
                raw_lr = lr_spec.lr_negative
                direction = "reassuring"
            else:
                raw_lr = 1.0
                direction = "neutral"

            # Temper by quality in log space: quality 0 -> no effect.
            effective_log_lr = quality * math.log(raw_lr) if raw_lr > 0 else 0.0
            log_odds += effective_log_lr

            contributions.append(
                AgentContribution(
                    agent_id=agent_id,
                    signal_strength=round(signal, 4),
                    direction=direction,
                    raw_lr=raw_lr,
                    quality=round(quality, 4),
                    effective_log_lr=round(effective_log_lr, 4),
                )
            )

        probability = _log_odds_to_prob(log_odds)
        return FusionResult(
            flare_probability=float(round(probability, 4)),
            prior=prior,
            posterior_log_odds=float(round(log_odds, 4)),
            contributions=contributions,
            calibration_version=self.calibration.version,
            gated=False,
        )
