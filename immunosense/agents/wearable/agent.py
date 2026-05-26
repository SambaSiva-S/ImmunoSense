"""WearableAgent - the main orchestrator for Agent 4.

Inherits BaseAgent (immunosense.agents.base) for JEPA compatibility and
universal health monitoring.

Interface contract::

    process(input_data) -> AgentOutput   # full nightly inference pipeline
    get_output_vector() -> np.ndarray    # 29-dim vector
    get_status() -> AgentHealth          # from BaseAgent

Input shape for process()::

    {
        'night_df': pd.DataFrame,    # minute-level: timestamp, hr, skin_temp,
                                     # enmo, spo2, sleep_stage
        'rr_intervals': list[float], # RR intervals (ms) for the night
        'night_idx': int,            # night number
        'is_flare': bool (optional), # for synthetic-data testing
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.agents.common.trackers import RobustBaselineTracker
from immunosense.agents.wearable.alerts import (
    evaluate_composite_alerts,
    evaluate_single_metric_alerts,
)
from immunosense.agents.wearable.pattern import PatternDetector
from immunosense.agents.wearable.pipeline import (
    apply_baseline_fillin,
    build_output_vector,
    engineer_features,
)
from immunosense.agents.wearable.states import derive_physiological_states
from immunosense.agents.wearable.stress import compute_wearable_stress_score
from immunosense.agents.wearable.types import FEATURE_NAMES, TRACKED_FEATURES


class WearableAgent(BaseAgent):
    """Agent 4 - Wearable signal processor.

    Takes one night of raw wearable data per call, produces:
        - 29-dim AgentOutput.vector for JEPA
        - alerts (single-metric + named composite)
        - physiological states (in AgentOutput.data)
        - data_quality_overall and trace_id

    Args:
        rules_version: Version stamp for emitted alerts.
        baseline_window: Rolling window for the RobustBaselineTracker.
        baseline_outlier_threshold: Outlier threshold for the tracker (IQR units).
    """

    agent_id = "agent4_wearable"
    agent_version = "1.0.0"
    output_dim = 29
    poll_frequency = "1hr"  # nightly in notebook; hourly in production

    def __init__(
        self,
        rules_version: str = "2026.05.24-001",
        baseline_window: int = 10,
        baseline_outlier_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.tracker = RobustBaselineTracker(
            features=TRACKED_FEATURES,
            window=baseline_window,
            outlier_threshold=baseline_outlier_threshold,
        )
        self.pattern_detector = PatternDetector(
            features=[
                "hrv_rmssd_sleep", "sleep_efficiency", "skin_temp_deviation",
                "resting_hr", "thermo_autonomic_decoupling_index",
            ],
            triggers=[
                "poor_sleep_last_night", "late_caffeine", "high_stress_yesterday",
            ],
            lag_range=(1, 3),
        )
        self.trajectory: list = []
        self.rules_version = rules_version
        self._last_vector: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(42)

    def process(self, input_data: dict) -> AgentOutput:
        """Process one wearable input package.

        Args:
            input_data: Dict with keys:
                night_df: pd.DataFrame (minute-level)
                rr_intervals: list of RR intervals (ms)
                night_idx: int (optional - defaults to len(trajectory))
                is_flare: bool (optional - for synthetic data labeling)

        Returns:
            AgentOutput with 29-dim vector + alerts + physiological states.
        """
        t_start = datetime.now(timezone.utc)
        trace_id = self._new_trace_id()

        try:
            night_df = input_data["night_df"]
            rr_intervals = input_data["rr_intervals"]
            night_idx = input_data.get("night_idx", len(self.trajectory))

            # L2 handled internally by L3 (HRV's own preprocessing); explicit
            # Hampel/Akima available via the preprocessing module for callers
            # that need raw signal cleanup.

            # L3 — engineer 29-feature reading
            reading = engineer_features(
                night_df, rr_intervals, night_idx, rng=self._rng,
            )
            reading["is_flare"] = input_data.get("is_flare", False)

            # L4 — fill in personal-normalized fields (against PRE-update baseline)
            reading = apply_baseline_fillin(reading, self.tracker)
            self.tracker.update(reading)
            self.trajectory.append(reading)

            # L5 — alerts
            single_alerts = evaluate_single_metric_alerts(
                reading, self.tracker, self.rules_version,
            )
            composite_alerts = evaluate_composite_alerts(
                reading, self.tracker, self.trajectory[-3:], self.rules_version,
            )
            all_alerts = single_alerts + composite_alerts

            # L6 — composite stress + vector
            reading["wearable_stress_score"] = compute_wearable_stress_score(reading)
            reading["composite_alert_count"] = float(len(composite_alerts))
            vector = build_output_vector(reading)
            self._last_vector = vector

            # Semantic state labels
            states = derive_physiological_states(reading, all_alerts)

            # Confidence = data quality
            confidence = reading.get("data_quality_overall", 0.0) or 0.0

            # Latency tracking
            elapsed_ms = (
                datetime.now(timezone.utc) - t_start
            ).total_seconds() * 1000.0
            self._record_latency(elapsed_ms)
            self._last_success = datetime.now(timezone.utc)

            return AgentOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "reading": reading,
                    "physiological_states": states,
                    "feature_names": FEATURE_NAMES,
                    "rules_version": self.rules_version,
                    "agent_version": self.agent_version,
                },
                vector=vector,
                vector_dim=self.output_dim,
                alerts=all_alerts,
                confidence=float(confidence),
                trace_id=trace_id,
            )

        except Exception:
            self._error_count += 1
            raise

    def get_output_vector(self) -> np.ndarray:
        """Return the most recent 29-dim feature vector."""
        if self._last_vector is None:
            return np.full(self.output_dim, np.nan, dtype=np.float64)
        return self._last_vector
