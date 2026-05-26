"""Tests for SymptomsMoodAgent - the orchestrator with planted-pattern validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.base import BaseAgent
from immunosense.agents.symptoms_mood import (
    CompositeSymptomSource,
    MockSymptomSource,
    StubMemoryStore,
    SymptomsMoodAgent,
    process_symptom_day,
)
from immunosense.agents.symptoms_mood.pipeline import compute_daily_flare_score
from immunosense.agents.symptoms_mood.types import (
    DailySymptomMoodSummary,
    FetchedSymptoms,
)


class TestSymptomsMoodAgentBasics:
    def test_inherits_baseagent(self):
        agent = SymptomsMoodAgent()
        assert isinstance(agent, BaseAgent)

    def test_agent_identity(self):
        agent = SymptomsMoodAgent()
        assert agent.agent_id == "agent5_symptoms_mood"
        assert agent.output_dim == 36
        assert agent.embedding_version == "agent5_symptoms_mood_v1.0.0"

    def test_default_memory_is_stub(self):
        agent = SymptomsMoodAgent()
        assert isinstance(agent.memory, StubMemoryStore)

    def test_custom_memory_store(self):
        store = StubMemoryStore()
        agent = SymptomsMoodAgent(memory_store=store)
        assert agent.memory is store

    def test_patient_id_stored(self):
        agent = SymptomsMoodAgent(patient_id="alice")
        assert agent.patient_id == "alice"

    def test_cold_start_returns_zeros(self):
        agent = SymptomsMoodAgent()
        assert agent.daily_flare_score() == 0.0
        emb = agent.jepa_embedding()
        assert emb.shape == (36,)
        assert (emb == 0).all()
        assert agent.raw_hypothesis_evidence() == []


class TestObserveAnalyze:
    def _build_agent_with_days(self, n_days=14, disease="RA"):
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource(disease=disease, seed_offset=42)
        composite = CompositeSymptomSource(mock_fallback=mock)
        start = pd.Timestamp("2026-03-01")
        for day in range(n_days):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_symptom_day(
                "p001", date, disease,
                composite_source=composite,
            )
            agent.observe(summary)
        return agent

    def test_observe_records_anomaly_scores(self):
        agent = self._build_agent_with_days(n_days=10)
        assert agent._latest_anomalies is not None
        assert "fatigue" in agent._latest_anomalies

    def test_observe_records_flare_score(self):
        agent = self._build_agent_with_days(n_days=14)
        assert len(agent.flare_events) == 14
        for date, score in agent.flare_events.items():
            assert 0.0 <= score <= 1.0

    def test_observe_writes_to_memory(self):
        agent = self._build_agent_with_days(n_days=5)
        # StubMemoryStore.n_observations returns count
        assert agent.memory.n_observations(agent.patient_id) == 5

    def test_baseline_tracker_active_after_min_days(self):
        agent = self._build_agent_with_days(n_days=5)
        report = agent.analyze()
        assert report.tracker_activation["fatigue"] is True

    def test_trigger_detector_inactive_before_min_days(self):
        agent = self._build_agent_with_days(n_days=10)
        report = agent.analyze()
        assert report.tracker_activation["trigger_detector"] is False

    def test_trigger_detector_active_at_min_days(self):
        agent = self._build_agent_with_days(n_days=14)
        report = agent.analyze()
        assert report.tracker_activation["trigger_detector"] is True


class TestPlantedPatternDetection:
    """End-to-end: inject sleep_severity → flare pattern at lag=+2."""

    def test_detects_planted_sleep_flare_pattern(self):
        agent = SymptomsMoodAgent(patient_id="test_patient")
        rng = np.random.RandomState(99)
        start = pd.Timestamp("2026-03-01")

        for day in range(60):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            bad_sleep_today = rng.rand() < 0.30
            bad_sleep_2_days_ago = (
                day >= 2 and agent.daily_records[day - 2].get("sleep_severity", 4) > 7
            )

            # Sleep severity bimodal: ~8 if bad sleep, ~4 otherwise
            # Symptoms boost by +3 after bad sleep (2-day lag)
            fetched = FetchedSymptoms(
                sleep_severity=(
                    float(rng.normal(8.0, 0.5)) if bad_sleep_today
                    else float(rng.normal(4.0, 1.5))
                ),
                fatigue=float(rng.normal(5.0, 1.5)) + (3.0 if bad_sleep_2_days_ago else 0.0),
                joint_pain=float(rng.normal(5.0, 1.5)) + (3.0 if bad_sleep_2_days_ago else 0.0),
                wellness_severity=float(rng.normal(5.0, 1.5)) + (3.0 if bad_sleep_2_days_ago else 0.0),
                energy_severity=float(rng.normal(5.0, 1.5)),
                brain_fog_severity=float(rng.normal(3.5, 1.5)),
                gi_distress=float(rng.normal(2.5, 1.0)),
                skin_severity=float(rng.normal(2.0, 1.0)),
                phq8_score=float(rng.normal(7.0, 2.5)),
                gad7_score=float(rng.normal(6.0, 2.5)),
                explicit_flare=False,
            )
            flare_score = compute_daily_flare_score(fetched)
            summary = DailySymptomMoodSummary(
                date=date,
                patient_id="test_patient",
                disease="RA",
                fatigue=fetched.fatigue,
                joint_pain=fetched.joint_pain,
                brain_fog_severity=fetched.brain_fog_severity,
                gi_distress=fetched.gi_distress,
                skin_severity=fetched.skin_severity,
                sleep_severity=fetched.sleep_severity,
                energy_severity=fetched.energy_severity,
                wellness_severity=fetched.wellness_severity,
                phq8_score=fetched.phq8_score,
                gad7_score=fetched.gad7_score,
                flare_score=flare_score,
            )
            agent.observe(summary)

        report = agent.analyze()

        sleep_patterns = [
            p for p in report.detected_patterns
            if "sleep_severity" in p.feature
        ]
        assert len(sleep_patterns) > 0, "Failed to detect planted sleep->flare pattern"

        for p in sleep_patterns:
            assert p.lag_days == 2  # pre-registered lag
            assert p.effect_size > 0
            assert p.confidence in ("high", "medium")


class TestRawHypothesisEvidence:
    def _build_60_day_agent(self):
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource(disease="Mixed", seed_offset=7)
        composite = CompositeSymptomSource(mock_fallback=mock)
        start = pd.Timestamp("2026-03-01")
        for day in range(60):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            summary = process_symptom_day(
                "p001", date, "Mixed",
                composite_source=composite,
            )
            agent.observe(summary)
        agent.analyze()  # populates raw evidence
        return agent

    def test_returns_all_10_hypotheses(self):
        agent = self._build_60_day_agent()
        evidence = agent.raw_hypothesis_evidence()
        assert len(evidence) == 10  # one per pre-registered (feature, lag)

    def test_evidence_has_biology_category(self):
        agent = self._build_60_day_agent()
        evidence = agent.raw_hypothesis_evidence()
        categories = {ev.biology_category for ev in evidence}
        assert categories == {"predictive", "concurrent", "reactive"}

    def test_survives_fdr_flag(self):
        agent = self._build_60_day_agent()
        evidence = agent.raw_hypothesis_evidence()
        # Each entry should have survives_fdr boolean
        for ev in evidence:
            assert isinstance(ev.survives_fdr, bool)


class TestJEPAEmbedding:
    def test_shape_and_dtype(self):
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource()
        composite = CompositeSymptomSource(mock_fallback=mock)
        summary = process_symptom_day("p001", "2026-04-01", "RA",
                                       composite_source=composite)
        agent.observe(summary)
        emb = agent.jepa_embedding()
        assert emb.shape == (36,)
        assert emb.dtype == np.float32

    def test_disease_onehot(self):
        """RA should set index 10 (RA position in DISEASE_TYPES)."""
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource(disease="RA")
        composite = CompositeSymptomSource(mock_fallback=mock)
        summary = process_symptom_day("p001", "2026-04-01", "RA",
                                       composite_source=composite)
        agent.observe(summary)
        emb = agent.jepa_embedding()
        # DISEASE_TYPES = ['RA', 'SLE', 'MS', 'Sjogrens', 'PsA', 'Mixed']
        # RA index 0, so emb[10 + 0] = 1.0
        assert emb[10] == 1.0
        for i in range(11, 16):
            assert emb[i] == 0.0


class TestProcessAdapter:
    def test_process_returns_valid_output(self):
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource()
        composite = CompositeSymptomSource(mock_fallback=mock)
        summary = process_symptom_day("p001", "2026-04-01", "Mixed",
                                       composite_source=composite)

        result = agent.process({"daily_summary": summary})

        assert result.agent_id == "agent5_symptoms_mood"
        assert result.vector.shape == (36,)
        assert result.vector_dim == 36
        assert result.trace_id.startswith("agent5_symptoms_mood-")

    def test_process_without_summary_raises(self):
        agent = SymptomsMoodAgent()
        with pytest.raises(ValueError, match="daily_summary"):
            agent.process({})


class TestWellnessSignature:
    def test_cold_start_returns_zero(self):
        agent = SymptomsMoodAgent()
        sig = agent.flare_signature()
        assert sig["score"] == 0.0
        assert sig["contributing_factors"] == []
        assert sig["clinical_alerts"] == []

    def test_has_canonical_outputs(self):
        agent = SymptomsMoodAgent()
        mock = MockSymptomSource()
        composite = CompositeSymptomSource(mock_fallback=mock)
        summary = process_symptom_day("p001", "2026-04-01", "Mixed",
                                       composite_source=composite)
        agent.observe(summary)
        sig = agent.flare_signature()
        # Must include flare_score (Agent 5's special output)
        assert "flare_score" in sig
        assert "flare_button_pressed" in sig
        assert 0.0 <= sig["score"] <= 1.0
