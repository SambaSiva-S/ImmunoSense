"""Tests for Benjamini-Hochberg FDR utility."""

from __future__ import annotations

import numpy as np
import pytest

from immunosense.agents.common.fdr import bh_correction, permutation_test_diff


# ============================================================
# bh_correction tests
# ============================================================

class TestBHCorrection:
    def test_empty_input(self):
        q, rej = bh_correction([], alpha=0.10)
        assert len(q) == 0
        assert len(rej) == 0

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            bh_correction([0.5], alpha=0.0)
        with pytest.raises(ValueError):
            bh_correction([0.5], alpha=1.0)

    def test_single_hypothesis_significant(self):
        """One p-value, well below alpha → rejected."""
        q, rej = bh_correction([0.001], alpha=0.10)
        assert q[0] == pytest.approx(0.001)
        assert rej[0]

    def test_single_hypothesis_not_significant(self):
        q, rej = bh_correction([0.5], alpha=0.10)
        assert not rej[0]

    def test_q_values_monotonic(self):
        """Sorted q-values must be monotonically non-decreasing."""
        p_values = [0.001, 0.01, 0.04, 0.08, 0.2, 0.5]
        q, _ = bh_correction(p_values, alpha=0.10)
        # Since p-values are already sorted, q-values should be too
        for i in range(1, len(q)):
            assert q[i] >= q[i - 1] - 1e-10  # tolerate floating point

    def test_q_values_bounded_below_1(self):
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        q, _ = bh_correction(p_values, alpha=0.10)
        assert all(qi <= 1.0 for qi in q)

    def test_q_geq_p(self):
        """q-value should always be >= corresponding p-value (BH adjusts UP)."""
        p_values = [0.01, 0.02, 0.05, 0.10, 0.20]
        q, _ = bh_correction(p_values, alpha=0.10)
        for p, qi in zip(p_values, q):
            assert qi >= p - 1e-10  # tolerate floating point

    def test_smallest_p_smallest_q(self):
        """Smallest p-value gets smallest q-value (preserves ordering)."""
        p_values = [0.5, 0.001, 0.1, 0.01, 0.2]
        q, _ = bh_correction(p_values, alpha=0.10)
        # Index 1 has smallest p (0.001), should also have smallest q
        smallest_p_idx = int(np.argmin(p_values))
        smallest_q_idx = int(np.argmin(q))
        assert smallest_p_idx == smallest_q_idx

    def test_classic_bh_example(self):
        """Standard BH example: 6 p-values, FDR=0.05."""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060]
        q, rej = bh_correction(p_values, alpha=0.05)
        # At FDR=0.05 with n=6, we expect at least the smallest to be rejected
        assert rej[0]  # 0.001 should definitely be rejected
        # And 0.060 should not
        assert not rej[5]

    def test_handles_nan_p_values(self):
        """NaN p-values should propagate as NaN q-values, not be rejected."""
        p_values = [0.01, float("nan"), 0.5, float("nan")]
        q, rej = bh_correction(p_values, alpha=0.10)
        assert np.isnan(q[1])
        assert np.isnan(q[3])
        assert not rej[1]
        assert not rej[3]
        # Non-NaN ones get proper q-values
        assert not np.isnan(q[0])
        assert not np.isnan(q[2])

    def test_all_nan_input(self):
        q, rej = bh_correction([float("nan"), float("nan")], alpha=0.10)
        assert all(np.isnan(qi) for qi in q)
        assert not any(rej)


# ============================================================
# permutation_test_diff tests
# ============================================================

class TestPermutationTest:
    def test_clear_difference_detected(self):
        """Strong group difference → low p-value."""
        rng = np.random.default_rng(seed=42)
        # Group 1 clearly higher than group 0
        values = np.concatenate([
            rng.normal(0, 1, size=30),    # group 0
            rng.normal(3, 1, size=30),    # group 1, +3 stdev higher
        ])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        diff, p = permutation_test_diff(values, labels, n_permutations=500, rng=rng)
        assert diff > 2.0  # positive difference
        assert p < 0.05    # statistically significant

    def test_no_difference(self):
        """No real difference → p around 0.5."""
        rng = np.random.default_rng(seed=42)
        values = rng.normal(0, 1, size=60)
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        diff, p = permutation_test_diff(values, labels, n_permutations=500, rng=rng)
        # diff should be small
        assert abs(diff) < 0.5
        # p-value should be somewhere in middle (one-tailed)
        assert 0.1 < p < 0.9

    def test_one_tailed_negative_diff(self):
        """If group 1 is LOWER than group 0, p should be high."""
        rng = np.random.default_rng(seed=42)
        values = np.concatenate([
            rng.normal(3, 1, size=30),    # group 0, higher
            rng.normal(0, 1, size=30),    # group 1, lower
        ])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        diff, p = permutation_test_diff(values, labels, n_permutations=500, rng=rng)
        assert diff < 0  # negative difference
        assert p > 0.9   # one-tailed: very few permutations exceed observed

    def test_handles_small_groups(self):
        """Groups with < 2 members should return NaN p."""
        values = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 1])
        diff, p = permutation_test_diff(values, labels, n_permutations=100)
        assert np.isnan(p)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            permutation_test_diff(np.array([1, 2, 3]), np.array([0, 1]), n_permutations=10)

    def test_reproducible_with_rng(self):
        """Same RNG seed should produce same p-value."""
        values = np.random.default_rng(seed=1).normal(0, 1, size=60)
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        diff1, p1 = permutation_test_diff(
            values, labels, n_permutations=100,
            rng=np.random.default_rng(seed=99),
        )
        diff2, p2 = permutation_test_diff(
            values, labels, n_permutations=100,
            rng=np.random.default_rng(seed=99),
        )
        assert diff1 == diff2
        assert p1 == p2
