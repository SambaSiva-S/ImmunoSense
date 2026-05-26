"""SymptomsMoodAgent - Conductor-facing orchestrator.

Special role: Agent 5 is the canonical source of flare events. Other agents
receive flare_score from here via the Conductor's distribution.

Inherits BaseAgent for JEPA compatibility and health monitoring.

Interface contract::

    observe(daily_summary)               # update Layer 3 trackers
    observe_flare(date, severity)        # rarely called - Agent 5 produces flares
    analyze() -> SymptomsMoodAgentReport
    flare_signature(summary) -> dict     # 0-1 wellness deviation
    daily_flare_score(summary) -> float  # canonical flare label for Conductor
    jepa_embedding(summary) -> np.array  # 36-dim dense vector
    raw_hypothesis_evidence() -> list    # ALL 10 hypothesis results for Conductor
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.agents.symptoms_mood.detector import SymptomsMoodTriggerDetector
from immunosense.agents.symptoms_mood.jepa_emit import (
    JEPA_EMBEDDING_DIM,
    compute_jepa_embedding,
)
from immunosense.agents.symptoms_mood.memory import MemoryStore, StubMemoryStore
from immunosense.agents.symptoms_mood.trackers import SymptomsMoodRobustTracker
from immunosense.agents.symptoms_mood.types import (
    ALL_FEATURES,
    DailySymptomMoodSummary,
    SymptomsMoodAgentReport,
)
from immunosense.agents.symptoms_mood.wellness import compute_wellness_signature


class SymptomsMoodAgent(BaseAgent):
    """Agent 5 - Symptoms & Mood Agent.

    Args:
        window: Rolling window size for baseline trackers (default 14).
        anomaly_threshold: Outlier threshold in IQR units (default 2.0).
        personalization_days: Days until full personal weight (default 25).
        trigger_min_days: Minimum observation days before detection runs.
        n_permutations: Permutations per hypothesis test (default 500).
        fdr_target: BH FDR target (default 0.10).
        memory_store: Memory store for cross-session continuity (default StubMemoryStore).
        patient_id: Patient identifier (default 'patient_001').
    """

    agent_id = "agent5_symptoms_mood"
    agent_version = "1.0.0"
    output_dim = JEPA_EMBEDDING_DIM  # 36
    poll_frequency = "daily"

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
        trigger_min_days: int = 14,
        n_permutations: int = 500,
        fdr_target: float = 0.10,
        memory_store: Optional[MemoryStore] = None,
        patient_id: str = "patient_001",
    ) -> None:
        super().__init__()
        self.baseline_tracker = SymptomsMoodRobustTracker(
            window=window,
            anomaly_threshold=anomaly_threshold,
            personalization_days=personalization_days,
        )
        self.trigger_detector = SymptomsMoodTriggerDetector(
            min_days=trigger_min_days,
            n_permutations=n_permutations,
            fdr_target=fdr_target,
        )
        self.memory: MemoryStore = memory_store or StubMemoryStore()
        self.patient_id = patient_id
        self.daily_records: list = []
        self.flare_events: dict = {}
        self._latest_anomalies: Optional[dict] = None
        self._latest_summary: Optional[DailySymptomMoodSummary] = None

    # ============================================================
    # Conductor-facing interface (observe/analyze pattern)
    # ============================================================

    def observe(self, daily_summary: DailySymptomMoodSummary) -> dict:
        """Record one day's symptom summary. Returns anomaly scores + flare_score."""
        self.baseline_tracker.update(daily_summary)
        self.daily_records.append({
            "date": daily_summary.date,
            **{f: getattr(daily_summary, f, None) for f in ALL_FEATURES},
        })

        # Agent 5 produces flares directly from daily_summary.flare_score
        self.flare_events[daily_summary.date] = daily_summary.flare_score
        self.memory.add_observation(self.patient_id, daily_summary)

        self._latest_anomalies = self.baseline_tracker.anomaly_scores(daily_summary)
        self._latest_summary = daily_summary
        return {
            "anomaly_scores": self._latest_anomalies,
            "flare_score": daily_summary.flare_score,
        }

    def observe_flare(self, date: str, severity: float) -> None:
        """Override the flare event for a date. Rarely needed - Agent 5 produces flares."""
        self.flare_events[date] = float(severity)

    def analyze(self) -> SymptomsMoodAgentReport:
        """Run BH FDR detection across observed days. Returns full report."""
        flare_dict = dict(self.flare_events)
        for rec in self.daily_records:
            flare_dict.setdefault(rec["date"], 0.0)

        patterns = self.trigger_detector.detect(self.daily_records, flare_dict)
        baselines = self.baseline_tracker.report()
        activation = {f: baselines[f]["n_clean"] >= 3 for f in ALL_FEATURES}
        activation["trigger_detector"] = (
            len(self.daily_records) >= self.trigger_detector.min_days
        )

        return SymptomsMoodAgentReport(
            n_days_observed=len(self.daily_records),
            n_flare_events=sum(1 for s in self.flare_events.values() if s >= 0.5),
            n_hypotheses_tested=self.trigger_detector.last_n_hypotheses,
            baselines=baselines,
            today_anomaly_scores=self._latest_anomalies,
            detected_patterns=patterns,
            tracker_activation=activation,
        )

    def flare_signature(
        self, daily_summary: Optional[DailySymptomMoodSummary] = None
    ) -> dict:
        """Conductor-facing 0-1 wellness deviation score."""
        summary = daily_summary or self._latest_summary
        if summary is None:
            return {
                "score": 0.0,
                "contributing_factors": [],
                "clinical_alerts": [],
                "data_quality_confidence": 0.0,
                "flare_score": 0.0,
                "flare_button_pressed": False,
            }
        anomalies = self.baseline_tracker.anomaly_scores(summary)
        report = self.analyze()
        return compute_wellness_signature(summary, anomalies, report.detected_patterns)

    # ============================================================
    # Agent 5's special role: canonical flare source + JEPA modality
    # ============================================================

    def daily_flare_score(
        self, daily_summary: Optional[DailySymptomMoodSummary] = None
    ) -> float:
        """Return the canonical 0-1 flare score that the Conductor distributes."""
        summary = daily_summary or self._latest_summary
        return summary.flare_score if summary else 0.0

    def jepa_embedding(
        self, daily_summary: Optional[DailySymptomMoodSummary] = None
    ) -> np.ndarray:
        """Return the 36-dim JEPA-compatible dense vector for one day."""
        summary = daily_summary or self._latest_summary
        if summary is None:
            return np.zeros(JEPA_EMBEDDING_DIM, dtype=np.float32)
        return compute_jepa_embedding(summary)

    def raw_hypothesis_evidence(self) -> list:
        """Return ALL hypothesis-level evidence from most recent analyze().

        Includes hypotheses that did NOT survive BH FDR — for Conductor's
        cross-agent corroboration. A signal at raw_p=0.018 doesn't survive
        Agent 5's own BH at q=0.10, but the Conductor may surface it if
        another agent (e.g., wearable) also reports anomalous activity on
        the same days.

        Returns empty list if analyze() has not been called yet.
        """
        if not hasattr(self.trigger_detector, "last_evidence"):
            return []
        return list(self.trigger_detector.last_evidence)

    # ============================================================
    # BaseAgent interface
    # ============================================================

    def process(self, input_data: dict) -> AgentOutput:
        """BaseAgent.process() adapter.

        Args:
            input_data: Dict with key 'daily_summary' (DailySymptomMoodSummary).

        Returns:
            AgentOutput with 36-dim JEPA vector and clinical alerts.
        """
        t_start = datetime.now(timezone.utc)
        trace_id = self._new_trace_id()

        daily_summary = input_data.get("daily_summary")
        if daily_summary is None:
            raise ValueError("input_data must contain 'daily_summary'")

        # Observe the day
        obs_result = self.observe(daily_summary)

        # Build the 36-dim JEPA vector
        vector = compute_jepa_embedding(daily_summary)

        # Latency tracking
        elapsed_ms = (datetime.now(timezone.utc) - t_start).total_seconds() * 1000.0
        self._record_latency(elapsed_ms)
        self._last_success = datetime.now(timezone.utc)

        # Alerts: convert threshold_alerts to alert dicts
        alerts = []
        alert_severity_map = {
            "moderate": "warning",
            "severe": "critical",
            "moderately_severe": "critical",
        }
        for feature, alert_level in daily_summary.threshold_alerts.items():
            if alert_level in alert_severity_map:
                alerts.append({
                    "name": f"{feature}_{alert_level}",
                    "severity": alert_severity_map[alert_level],
                    "alert_type": "clinical_threshold",
                    "feature": feature,
                    "category": alert_level,
                })

        if daily_summary.flare_button_pressed:
            alerts.append({
                "name": "flare_button_pressed",
                "severity": "critical",
                "alert_type": "explicit_flare",
                "feature": "explicit_flare",
            })

        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={
                "daily_summary": daily_summary,
                "anomaly_scores": obs_result["anomaly_scores"],
                "flare_score": daily_summary.flare_score,
                "feature_names": ALL_FEATURES,
                "agent_version": self.agent_version,
            },
            vector=vector,
            vector_dim=self.output_dim,
            alerts=alerts,
            confidence=daily_summary.overall_confidence,
            trace_id=trace_id,
        )

    def get_output_vector(self) -> np.ndarray:
        """Return the most recent 36-dim JEPA embedding."""
        return self.jepa_embedding()

    def emit_embedding(
        self, daily_summary: Optional[DailySymptomMoodSummary] = None
    ) -> np.ndarray:
        """JEPACompatible interface: emit the daily JEPA embedding."""
        return self.jepa_embedding(daily_summary)
