"""DietaryAgent — Layer 3 orchestrator for Agent 2.

Composes OvernightFastTracker, DietaryRobustTracker, and DietaryTriggerDetector
into a per-patient streaming agent. Inherits BaseAgent for Conductor compat.

Output dimension: 10 (6 continuous + 4 boolean features, in the canonical
ordering of CONTINUOUS_FEATURES + BOOLEAN_TRIGGERS).

The agent expects DailyRollup objects (from Layer 2's rollup_day) and emits:
    - daily_flare_score equivalent → anomaly_scores dict via observe()
    - flare events via observe_flare(date, severity)
    - DietaryAgentReport via analyze() for the Conductor
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.agents.dietary.constants import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
)
from immunosense.agents.dietary.detector import DietaryTriggerDetector
from immunosense.agents.dietary.trackers import (
    DietaryRobustTracker,
    OvernightFastTracker,
)
from immunosense.agents.dietary.types import DietaryAgentReport


class DietaryAgent(BaseAgent):
    """Per-patient orchestrator for Agent 2 Layer 3.

    Lifecycle:
        1. ``observe(rollup)`` is called once per day with a DailyRollup
        2. Optionally ``observe_flare(date, severity)`` records observed flares
        3. ``analyze()`` runs trigger detection + returns DietaryAgentReport
        4. ``process(input_data)`` provides a BaseAgent-compatible adapter

    Output vector ordering (10-dim):
        Indices 0-5: dii_score, omega6_omega3_ratio, glycemic_load,
                     sodium_mg, alcohol_g, overnight_fast_hours (anomaly scores)
        Indices 6-9: gluten_present, dairy_present, nightshade_present,
                     upf_present (booleans cast to 0.0/1.0)
    """

    agent_id = "agent2_dietary"
    agent_version = "1.0.0"
    output_dim = 10  # 6 continuous + 4 boolean
    poll_frequency = "daily"

    def __init__(
        self,
        patient_id: Optional[str] = None,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
        trigger_min_days: int = 14,
        n_permutations: int = 500,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.patient_id = patient_id

        self.fast_tracker = OvernightFastTracker()
        self.baseline_tracker = DietaryRobustTracker(
            window=window,
            anomaly_threshold=anomaly_threshold,
            personalization_days=personalization_days,
        )
        self.trigger_detector = DietaryTriggerDetector(
            min_days=trigger_min_days,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )

        self.daily_records: list = []
        self.flare_events: dict = {}
        self._latest_anomalies: Optional[dict] = None
        self._latest_overnight_fast: Optional[float] = None
        self._latest_features: Optional[dict] = None
        self._latest_rollup: Any = None

    # =====================================================================
    # Layer 3 streaming API
    # =====================================================================

    def observe(self, rollup) -> dict:
        """Record one day's rollup; update trackers; return anomalies.

        Args:
            rollup: A DailyRollup (or duck-typed equivalent with required fields).

        Returns:
            Dict with keys 'overnight_fast_hours' and 'anomaly_scores'.
        """
        overnight_fast = self.fast_tracker.compute(rollup)

        daily_features = {
            "date": rollup.date,
            "dii_score": rollup.dii_score,
            "omega6_omega3_ratio": rollup.omega6_omega3_ratio,
            "glycemic_load": rollup.glycemic_load,
            "sodium_mg": rollup.sodium_mg,
            "alcohol_g": rollup.alcohol_g,
            "overnight_fast_hours": overnight_fast,
            "gluten_present": rollup.gluten_present,
            "dairy_present": rollup.dairy_present,
            "nightshade_present": rollup.nightshade_present,
            "upf_present": rollup.upf_present,
        }

        self.baseline_tracker.update(daily_features)
        self.daily_records.append(daily_features)

        self._latest_anomalies = self.baseline_tracker.anomaly_scores(daily_features)
        self._latest_overnight_fast = overnight_fast
        self._latest_features = daily_features
        self._latest_rollup = rollup

        return {
            "overnight_fast_hours": overnight_fast,
            "anomaly_scores": self._latest_anomalies,
        }

    def observe_flare(self, date: str, severity: float) -> None:
        """Record a flare event for `date` (distributed by Conductor)."""
        self.flare_events[date] = float(severity)

    def analyze(self) -> DietaryAgentReport:
        """Run trigger detection and return a structured report."""
        flare_dict = dict(self.flare_events)
        # Days without a recorded flare are assumed flare-free (severity 0)
        for rec in self.daily_records:
            flare_dict.setdefault(rec["date"], 0.0)

        patterns = self.trigger_detector.detect(self.daily_records, flare_dict)
        baselines = self.baseline_tracker.report()

        activation = {
            "baseline_continuous": {
                f: baselines["continuous"][f]["n_clean"] >= 3
                for f in CONTINUOUS_FEATURES
            },
            "trigger_detector": (
                len(self.daily_records) >= self.trigger_detector.min_days
            ),
            "overnight_fast": (
                self._latest_overnight_fast is not None
                and not pd.isna(self._latest_overnight_fast)
            ),
        }

        return DietaryAgentReport(
            n_days_observed=len(self.daily_records),
            n_flare_events=sum(1 for s in self.flare_events.values() if s > 0),
            baselines=baselines,
            today_anomaly_scores=self._latest_anomalies,
            today_overnight_fast=self._latest_overnight_fast,
            detected_patterns=patterns,
            tracker_activation=activation,
        )

    # =====================================================================
    # BaseAgent interface
    # =====================================================================

    def process(self, input_data: dict) -> AgentOutput:
        """BaseAgent-compatible adapter.

        Args:
            input_data: dict with required key 'rollup' (a DailyRollup).
                         Optional key 'flares' = list of (date, severity) tuples.

        Returns:
            AgentOutput with the 10-dim vector and any detected patterns
            as alerts.
        """
        start_time = datetime.now(timezone.utc)
        trace_id = self._new_trace_id()

        try:
            rollup = input_data["rollup"]
        except KeyError as e:
            self._error_count += 1
            raise ValueError(
                "DietaryAgent.process requires input_data['rollup']"
            ) from e

        # Optional: record any flares passed in
        for date, severity in input_data.get("flares", []):
            self.observe_flare(date, severity)

        # Run streaming update
        update = self.observe(rollup)

        # Optionally run detection if we have enough data
        patterns: list = []
        if len(self.daily_records) >= self.trigger_detector.min_days:
            report = self.analyze()
            patterns = report.detected_patterns

        # Build the 10-dim vector
        vector = self.get_output_vector()

        # Convert detected patterns to alert dicts
        alerts = [
            {
                "name": p.feature,
                "severity": p.confidence,
                "lag_days": p.lag_days,
                "effect_size": p.effect_size,
                "p_value": p.p_value,
            }
            for p in patterns
        ]

        latency_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000.0
        self._record_latency(latency_ms)
        self._last_success = datetime.now(timezone.utc)

        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={
                "patient_id": self.patient_id,
                "date": rollup.date,
                "rollup_summary": {
                    "meal_count": rollup.meal_count,
                    "dii_score": rollup.dii_score,
                    "daily_dii_percentile": rollup.daily_dii_percentile,
                    "glycemic_load": rollup.glycemic_load,
                    "sodium_mg": rollup.sodium_mg,
                    "alcohol_g": rollup.alcohol_g,
                    "gluten_present": rollup.gluten_present,
                    "dairy_present": rollup.dairy_present,
                    "nightshade_present": rollup.nightshade_present,
                    "upf_present": rollup.upf_present,
                },
                "overnight_fast_hours": update["overnight_fast_hours"],
                "anomaly_scores": update["anomaly_scores"],
                "n_patterns_detected": len(patterns),
            },
            vector=vector,
            vector_dim=self.output_dim,
            alerts=alerts,
            confidence=self._estimate_confidence(rollup),
            trace_id=trace_id,
        )

    def get_output_vector(self) -> np.ndarray:
        """Return the 10-dim vector: 6 anomaly scores + 4 trigger booleans.

        NaN anomaly scores (insufficient baseline) become 0.0. Booleans
        become 0.0 / 1.0.

        Returns:
            np.ndarray of shape (10,) with float64 entries.
        """
        if self._latest_features is None:
            return np.zeros(self.output_dim, dtype=np.float64)

        # Continuous: use anomaly scores (NaN -> 0.0)
        vec = np.zeros(self.output_dim, dtype=np.float64)
        if self._latest_anomalies is not None:
            for i, f in enumerate(CONTINUOUS_FEATURES):
                v = self._latest_anomalies.get(f, float("nan"))
                vec[i] = 0.0 if pd.isna(v) else float(v)

        # Boolean: cast to 0.0/1.0
        for i, f in enumerate(BOOLEAN_TRIGGERS):
            v = self._latest_features.get(f, False)
            vec[len(CONTINUOUS_FEATURES) + i] = 1.0 if bool(v) else 0.0

        return vec

    def _estimate_confidence(self, rollup) -> float:
        """Aggregate per-feature confidence into a single 0-1 score.

        Maps: high=1.0, medium=0.6, low=0.3, unavailable=0.0, then averages.
        """
        if not hasattr(rollup, "feature_confidence") or not rollup.feature_confidence:
            return 1.0  # No metadata to score from

        mapping = {"high": 1.0, "medium": 0.6, "low": 0.3, "unavailable": 0.0}
        scores = [mapping.get(c, 0.5) for c in rollup.feature_confidence.values()]
        return float(np.mean(scores)) if scores else 1.0
