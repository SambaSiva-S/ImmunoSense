"""EnvironmentAgent - Layer 3 orchestrator, Conductor-facing API.

Interface contract (same shape as DietaryAgent and SymptomsMoodAgent for
uniform Conductor consumption):
    observe(daily_summary)         - update Layer 3 trackers, record observation
    observe_flare(date, severity)  - record flare event for trigger detection
    analyze()                      - return EnvironmentAgentReport
    flare_signature(summary=None)  - Conductor-facing 0-1 score for one day

Inherits from BaseAgent (agents.base) for JEPA compatibility and
universal health monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.agents.environment.detector import EnvironmentTriggerDetector
from immunosense.agents.environment.pipeline import compute_flare_signature
from immunosense.agents.environment.trackers import (
    ENV_FEATURES,
    EnvironmentRobustTracker,
)
from immunosense.agents.environment.types import (
    DailyEnvironmentSummary,
    EnvironmentAgentReport,
)


class EnvironmentAgent(BaseAgent):
    """Agent 3 - Environment Agent.

    Tracks environmental triggers (PM2.5, ozone, UV, barometric pressure, pollen)
    and detects per-patient trigger patterns via BH FDR-corrected permutation tests.

    Args:
        window: Rolling window size for baseline trackers (default 14).
        anomaly_threshold: Outlier threshold for baseline trackers (default 2.0 IQRs).
        personalization_days: Days until full personal weight (default 25).
        trigger_min_days: Minimum observation days before detection runs (default 14).
        n_permutations: Permutations per hypothesis test (default 500).
        fdr_target: Target FDR level for BH correction (default 0.10).

    Example::

        >>> from immunosense.agents.environment import (
        ...     EnvironmentAgent, process_environment_day, MockEnvironmentSource
        ... )
        >>> agent = EnvironmentAgent()
        >>> source = MockEnvironmentSource()
        >>> for day in range(30):
        ...     summary = process_environment_day('28202', f'2026-04-{day+1:02d}', source=source)
        ...     agent.observe(summary)
        >>> agent.observe_flare('2026-04-15', severity=0.8)
        >>> report = agent.analyze()
        >>> sig = agent.flare_signature()
    """

    agent_id = "agent3_environment"
    agent_version = "1.0.0"
    output_dim = 5  # one per environmental feature
    poll_frequency = "6hr"

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
        trigger_min_days: int = 14,
        n_permutations: int = 500,
        fdr_target: float = 0.10,
    ) -> None:
        super().__init__()
        self.baseline_tracker = EnvironmentRobustTracker(
            window=window,
            anomaly_threshold=anomaly_threshold,
            personalization_days=personalization_days,
        )
        self.trigger_detector = EnvironmentTriggerDetector(
            min_days=trigger_min_days,
            n_permutations=n_permutations,
            fdr_target=fdr_target,
        )
        self.daily_records: list = []
        self.flare_events: dict = {}
        self._latest_anomalies: Optional[dict] = None
        self._latest_summary: Optional[DailyEnvironmentSummary] = None

    # ============================================================
    # Conductor-facing interface (observe/analyze pattern)
    # ============================================================

    def observe(self, daily_summary: DailyEnvironmentSummary) -> dict:
        """Record one day's environmental summary, update trackers, return anomaly scores."""
        self.baseline_tracker.update(daily_summary)
        self.daily_records.append({
            "date": daily_summary.date,
            **{f: getattr(daily_summary, f, None) for f in ENV_FEATURES},
        })
        self._latest_anomalies = self.baseline_tracker.anomaly_scores(daily_summary)
        self._latest_summary = daily_summary
        return {"anomaly_scores": self._latest_anomalies}

    def observe_flare(self, date: str, severity: float) -> None:
        """Record a flare event for downstream trigger detection."""
        self.flare_events[date] = float(severity)

    def analyze(self) -> EnvironmentAgentReport:
        """Run BH FDR trigger detection across observed days. Returns full report."""
        flare_dict = dict(self.flare_events)
        for rec in self.daily_records:
            flare_dict.setdefault(rec["date"], 0.0)

        patterns = self.trigger_detector.detect(self.daily_records, flare_dict)
        baselines = self.baseline_tracker.report()
        activation = {f: baselines[f]["n_clean"] >= 3 for f in ENV_FEATURES}
        activation["trigger_detector"] = (
            len(self.daily_records) >= self.trigger_detector.min_days
        )

        overall_conf = (
            float(np.mean([
                self._latest_summary.overall_confidence
                for _ in [None]  # use most recent
            ])) if self._latest_summary else 0.0
        )

        return EnvironmentAgentReport(
            n_days_observed=len(self.daily_records),
            n_flare_events=sum(1 for s in self.flare_events.values() if s > 0),
            today_percentiles=(
                self._latest_summary.percentiles if self._latest_summary else None
            ),
            today_threshold_alerts=(
                self._latest_summary.threshold_alerts if self._latest_summary else None
            ),
            detected_patterns=patterns,
            tracker_activation=activation,
            overall_data_confidence=overall_conf,
        )

    def flare_signature(
        self, daily_summary: Optional[DailyEnvironmentSummary] = None
    ) -> dict:
        """Conductor-facing 0-1 environmental flare risk score."""
        summary = daily_summary or self._latest_summary
        if summary is None:
            return {
                "score": 0.0,
                "contributing_factors": [],
                "threshold_breaches": [],
                "data_quality_confidence": 0.0,
            }
        anomalies = self.baseline_tracker.anomaly_scores(summary)
        report = self.analyze()
        return compute_flare_signature(summary, anomalies, report.detected_patterns)

    # ============================================================
    # BaseAgent interface (process pattern)
    # ============================================================

    def process(self, input_data: dict) -> AgentOutput:
        """BaseAgent.process() adapter.

        Bridges the BaseAgent's process() interface to the observe/analyze pattern.

        Args:
            input_data: Dict with key 'daily_summary' (DailyEnvironmentSummary) and
                optionally 'flare_event': {'date': str, 'severity': float}.

        Returns:
            AgentOutput with 5-feature anomaly vector and threshold alerts.
        """
        t_start = datetime.now(timezone.utc)
        trace_id = self._new_trace_id()

        daily_summary = input_data.get("daily_summary")
        if daily_summary is None:
            raise ValueError("input_data must contain 'daily_summary'")

        # Optional flare event
        flare_event = input_data.get("flare_event")
        if flare_event is not None:
            self.observe_flare(flare_event["date"], flare_event["severity"])

        # Observe the day
        obs_result = self.observe(daily_summary)
        anomaly_scores = obs_result["anomaly_scores"]

        # Build the 5-dim output vector
        vector = self.get_output_vector()

        # Latency tracking
        elapsed_ms = (datetime.now(timezone.utc) - t_start).total_seconds() * 1000.0
        self._record_latency(elapsed_ms)
        self._last_success = datetime.now(timezone.utc)

        # Alerts: convert threshold_alerts to alert dicts
        alerts = []
        for feature, alert_level in daily_summary.threshold_alerts.items():
            if alert_level in {
                "unhealthy_sensitive", "unhealthy", "very_unhealthy", "hazardous",
                "high", "very_high", "extreme", "large_change",
            }:
                alerts.append({
                    "name": f"{feature}_{alert_level}",
                    "severity": "warning" if alert_level in {
                        "unhealthy_sensitive", "moderate_change", "high"
                    } else "critical",
                    "alert_type": "population_threshold",
                    "feature": feature,
                    "value": getattr(daily_summary, f"{feature}_ug_m3", None)
                              or getattr(daily_summary, f"{feature}_ppb", None)
                              or getattr(daily_summary, f"{feature}_index", None),
                })

        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={
                "daily_summary": daily_summary,
                "anomaly_scores": anomaly_scores,
                "feature_names": ENV_FEATURES,
                "agent_version": self.agent_version,
            },
            vector=vector,
            vector_dim=self.output_dim,
            alerts=alerts,
            confidence=daily_summary.overall_confidence,
            trace_id=trace_id,
        )

    def get_output_vector(self) -> np.ndarray:
        """Return 5-dim anomaly score vector. NaN for features without baseline."""
        if self._latest_anomalies is None:
            return np.full(self.output_dim, np.nan, dtype=np.float64)

        arr = np.full(self.output_dim, np.nan, dtype=np.float64)
        for i, feature in enumerate(ENV_FEATURES):
            v = self._latest_anomalies.get(feature, float("nan"))
            if v is not None and not np.isnan(v):
                arr[i] = float(v)
        return arr
