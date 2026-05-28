"""Tests for the quality scorer and 4-level confidence aggregation."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from immunosense.adapters.base import AdapterResult
from immunosense.agents.base import AgentOutput
from immunosense.conductor.quality.confidence import (
    ConfidenceAggregator,
    HIGH_BAND,
    MODERATE_BAND,
)
from immunosense.conductor.quality.scorer import AgentQuality, QualityScorer
from immunosense.events.bucket import AgentData
from immunosense.events.types import ConfidenceLevel


def _result(agent_id, confidence, ok=True, dim=7):
    out = AgentOutput(
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        data={},
        vector=np.zeros(dim),
        vector_dim=dim,
        confidence=confidence,
    )
    return AdapterResult(agent_id=agent_id, output=out, ok=ok,
                         error=None if ok else "err")


class TestQualityScorer:
    def setup_method(self):
        self.scorer = QualityScorer()
        self.ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)

    def test_fresh_high_confidence(self):
        r = _result("agent5_symptoms_mood", 0.9)
        data = AgentData(agent_id="agent5_symptoms_mood", domain_object="s",
                         produced_at=self.ref)
        q = self.scorer.score(r, data, "daily", self.ref)
        assert q.ok
        assert q.raw_confidence == 0.9
        assert q.freshness == pytest.approx(1.0)
        assert q.quality == pytest.approx(0.9)

    def test_failed_agent_zero_quality(self):
        r = _result("agent1_biomarker", 0.0, ok=False)
        data = AgentData(agent_id="agent1_biomarker", domain_object="x",
                         produced_at=self.ref)
        q = self.scorer.score(r, data, "weekly", self.ref)
        assert q.ok is False
        assert q.quality == 0.0

    def test_stale_hourly_reading_penalized(self):
        r = _result("agent4_wearable", 0.9, dim=29)
        old = self.ref - timedelta(days=2)
        data = AgentData(agent_id="agent4_wearable", domain_object="x",
                         produced_at=old)
        q = self.scorer.score(r, data, "1hr", self.ref)
        # 2-day-old hourly reading: heavily penalized.
        assert q.quality < 0.1

    def test_stale_weekly_reading_survives(self):
        r = _result("agent1_biomarker", 0.9)
        old = self.ref - timedelta(days=2)
        data = AgentData(agent_id="agent1_biomarker", domain_object="x",
                         produced_at=old)
        q = self.scorer.score(r, data, "weekly", self.ref)
        # 2-day-old weekly reading: barely penalized.
        assert q.quality > 0.7

    def test_absent_agent(self):
        q = QualityScorer.absent("agent3_environment")
        assert q.reported is False
        assert q.quality == 0.0


class TestConfidenceAggregator:
    def setup_method(self):
        self.agg = ConfidenceAggregator()

    def _q(self, quality, reported=True, ok=True):
        return AgentQuality(
            agent_id="a", raw_confidence=quality, freshness=1.0,
            quality=quality, reported=reported, ok=ok,
        )

    def test_high_requires_three(self):
        qs = [self._q(0.8), self._q(0.8), self._q(0.8), self._q(0.1), self._q(0.0)]
        result = self.agg.aggregate(qs)
        assert result.level == ConfidenceLevel.HIGH

    def test_moderate(self):
        # 3 at-least-moderate (one high, two moderate), not 3 high.
        qs = [self._q(0.8), self._q(0.6), self._q(0.6), self._q(0.1), self._q(0.0)]
        result = self.agg.aggregate(qs)
        assert result.level == ConfidenceLevel.MODERATE

    def test_low(self):
        # 2 at-least-low, fewer than 3 moderate.
        qs = [self._q(0.3), self._q(0.3), self._q(0.0), self._q(0.0), self._q(0.0)]
        result = self.agg.aggregate(qs)
        assert result.level == ConfidenceLevel.LOW

    def test_insufficient(self):
        qs = [self._q(0.1), self._q(0.0), self._q(0.0), self._q(0.0), self._q(0.0)]
        result = self.agg.aggregate(qs)
        assert result.level == ConfidenceLevel.INSUFFICIENT

    def test_all_absent_insufficient(self):
        qs = [self._q(0.0, reported=False) for _ in range(5)]
        result = self.agg.aggregate(qs)
        assert result.level == ConfidenceLevel.INSUFFICIENT
        assert result.overall_quality == 0.0
        assert result.n_reporting == 0

    def test_overall_quality_mean_of_reporting(self):
        qs = [self._q(0.8), self._q(0.6),
              self._q(0.0, reported=False)]
        result = self.agg.aggregate(qs)
        # mean of reporting (0.8, 0.6) = 0.7
        assert result.overall_quality == pytest.approx(0.7)
        assert result.n_reporting == 2

    def test_band_counts(self):
        qs = [self._q(0.9), self._q(0.6), self._q(0.3)]
        result = self.agg.aggregate(qs)
        assert result.n_high == 1
        assert result.n_moderate == 1
        assert result.n_low == 1
