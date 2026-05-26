"""Tests for EnvironmentAgent - the orchestrator class.

Includes end-to-end synthetic-patient test that validates BH FDR detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.base import BaseAgent
from immunosense.agents.environment import (
    EnvironmentAgent,
    MockEnvironmentSource,
    process_environment_day,
)


class TestEnvironmentAgentBasics:
    def test_inherits_baseagent(self):
        agent = EnvironmentAgent()
        assert isinstance(agent, BaseAgent)

    def test_agent_identity(self):
        agent = EnvironmentAgent()
        assert agent.agent_id == "agent3_environment"
        assert agent.output_dim == 5
        assert agent.embedding_version == "agent3_environment_v1.0.0"

    def test_cold_start_no_baseline(self):
        agent = EnvironmentAgent()
        report = agent.analyze()
        assert report.n_days_observed == 0
        assert report.n_flare_events == 0
        assert report.detected_patterns == []

    def test_get_output_vector_cold_start_is_nan(self):
        agent = EnvironmentAgent()
        v = agent.get_output_vector()
        assert v.shape == (5,)
        assert np.isnan(v).all()

    def test_initialize_stores_dependencies(self):
        agent = EnvironmentAgent()
        agent.initialize(config={"x": 1})
        assert agent.config == {"x": 1}


class TestEnvironmentAgentObserve:
    def _make_agent_with_days(self, n_days=14, source=None):
        agent = EnvironmentAgent()
        source = source or MockEnvironmentSource(seed_offset=42)
        start = pd.Timestamp("2026-03-01")
        for day in range(n_days):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_environment_day("28202", date, source=source)
            agent.observe(summary)
        return agent

    def test_observe_records_anomaly_scores(self):
        agent = self._make_agent_with_days(n_days=10)
        # After 10 days, anomaly scores should be available
        assert agent._latest_anomalies is not None
        assert "pm25_ug_m3" in agent._latest_anomalies

    def test_observe_updates_daily_records(self):
        agent = self._make_agent_with_days(n_days=14)
        assert len(agent.daily_records) == 14
        for rec in agent.daily_records:
            assert "date" in rec
            assert "pm25_ug_m3" in rec

    def test_baseline_tracker_active_after_3_days(self):
        agent = self._make_agent_with_days(n_days=5)
        report = agent.analyze()
        # PM2.5 tracker should be active
        assert report.tracker_activation["pm25_ug_m3"] is True

    def test_trigger_detector_inactive_before_min_days(self):
        agent = self._make_agent_with_days(n_days=10)
        report = agent.analyze()
        # 10 days < 14 day min
        assert report.tracker_activation["trigger_detector"] is False

    def test_trigger_detector_active_at_min_days(self):
        agent = self._make_agent_with_days(n_days=14)
        report = agent.analyze()
        assert report.tracker_activation["trigger_detector"] is True


class TestEnvironmentAgentFlareDetection:
    """End-to-end test: inject synthetic flares correlated with PM2.5 spikes."""

    def test_detects_planted_pm25_trigger(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        start = pd.Timestamp("2026-03-01")

        # 60-day observation
        for day in range(60):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_environment_day("28202", date, source=source)
            agent.observe(summary)

        # Plant flares correlated with high PM2.5 (synthetic test of detection)
        for rec in agent.daily_records:
            pm = rec.get("pm25_ug_m3")
            if pm and pm > 12:  # high PM
                agent.observe_flare(rec["date"], 0.7)

        report = agent.analyze()

        # Should detect at least one PM2.5-related pattern
        pm25_patterns = [
            p for p in report.detected_patterns
            if "pm25_ug_m3" in p.feature
        ]
        assert len(pm25_patterns) > 0, "Failed to detect planted PM2.5 trigger"

        # The detected pattern should have positive effect (high PM -> more flares)
        for p in pm25_patterns:
            assert p.effect_size > 0
            # Should be high or medium confidence given the clean synthetic signal
            assert p.confidence in ("high", "medium")

    def test_no_false_positive_with_random_flares(self):
        """Random flares uncorrelated with environment should rarely yield patterns."""
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=100)
        start = pd.Timestamp("2026-03-01")
        rng = np.random.RandomState(0)

        for day in range(60):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_environment_day("28202", date, source=source)
            agent.observe(summary)
            # Random flare with 20% probability, severity uniform
            if rng.rand() < 0.20:
                agent.observe_flare(date, float(rng.rand()))

        report = agent.analyze()

        # At FDR=0.10, expected false positives <= 10% of detected.
        # With ~20 hypotheses and no real signal, often 0 detections.
        # We accept up to 2 spurious detections.
        assert len(report.detected_patterns) <= 2, (
            f"Too many spurious detections: {len(report.detected_patterns)}"
        )


class TestEnvironmentAgentFlareSignature:
    def test_flare_signature_returns_score_in_range(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        # Need at least one observation
        summary = process_environment_day("28202", "2026-04-01", source=source)
        agent.observe(summary)
        sig = agent.flare_signature(summary)
        assert 0.0 <= sig["score"] <= 1.0

    def test_flare_signature_cold_start_returns_zero(self):
        agent = EnvironmentAgent()
        sig = agent.flare_signature()
        assert sig["score"] == 0.0
        assert sig["contributing_factors"] == []

    def test_flare_signature_has_contributors(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        for day in range(20):
            date = (pd.Timestamp("2026-03-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_environment_day("28202", date, source=source)
            agent.observe(summary)
        sig = agent.flare_signature()
        assert len(sig["contributing_factors"]) == 5  # one per feature


class TestEnvironmentAgentProcess:
    """Test the BaseAgent.process() adapter."""

    def test_process_returns_valid_agent_output(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        summary = process_environment_day("28202", "2026-04-01", source=source)

        result = agent.process({"daily_summary": summary})

        assert result.agent_id == "agent3_environment"
        assert result.vector.shape == (5,)
        assert result.vector_dim == 5
        assert result.trace_id.startswith("agent3_environment-")

    def test_process_records_flare_event(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        summary = process_environment_day("28202", "2026-04-01", source=source)

        agent.process({
            "daily_summary": summary,
            "flare_event": {"date": "2026-04-01", "severity": 0.8},
        })

        assert agent.flare_events["2026-04-01"] == 0.8

    def test_process_without_daily_summary_raises(self):
        agent = EnvironmentAgent()
        with pytest.raises(ValueError, match="daily_summary"):
            agent.process({})

    def test_process_tracks_latency(self):
        agent = EnvironmentAgent()
        source = MockEnvironmentSource(seed_offset=42)
        for day in range(3):
            date = (pd.Timestamp("2026-04-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_environment_day("28202", date, source=source)
            agent.process({"daily_summary": summary})

        status = agent.get_status()
        assert status.avg_latency_ms > 0
        assert status.status == "healthy"
