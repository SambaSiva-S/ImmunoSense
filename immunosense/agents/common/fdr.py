"""Benjamini-Hochberg False Discovery Rate utility.

Used by Agents 2 (Dietary), 3 (Environment), and 5 (Symptoms & Mood) for
statistically-rigorous trigger pattern detection across multiple hypotheses.

The BH procedure controls the expected proportion of false discoveries
(false positives) among all rejected hypotheses, which is more powerful
than Bonferroni correction when many hypotheses are tested.

References:
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing. JRSS-B.
"""

from __future__ import annotations

import numpy as np


def bh_correction(p_values: list[float], alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of raw p-values (one per hypothesis).
        alpha: Desired FDR level (default 0.10 = 10% FDR control).

    Returns:
        Tuple of:
            q_values: Array of BH-adjusted q-values, same order as input.
            rejected: Boolean array, True where hypothesis is rejected at alpha.

    Example:
        >>> p_values = [0.01, 0.02, 0.05, 0.10, 0.50]
        >>> q_values, rejected = bh_correction(p_values, alpha=0.10)
        >>> rejected[0]  # smallest p-value
        True

    Notes:
        - The BH procedure is more powerful than Bonferroni when many tests
          are run and at least some true effects exist.
        - q-values are non-monotonic with raw p-values in general; the
          procedure ensures monotonicity by taking running minima from
          the top.
        - Hypotheses with NaN p-values are propagated as NaN q-values
          (not rejected).
    """
    if not p_values:
        return np.array([]), np.array([], dtype=bool)

    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p = np.asarray(p_values, dtype=float)
    n = len(p)

    # Track NaN positions (skip them in correction)
    nan_mask = np.isnan(p)
    valid_mask = ~nan_mask
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

    # Sort valid p-values, remember original positions
    valid_p = p[valid_mask]
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # BH-adjusted p-values: p_(k) * n_valid / k
    # Apply running min from the right to enforce monotonicity
    bh_factors = np.arange(1, n_valid + 1, dtype=float)
    raw_q = sorted_p * n_valid / bh_factors
    sorted_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    sorted_q = np.minimum(sorted_q, 1.0)  # cap at 1.0

    # Unsort back to original positions of valid values
    valid_q = np.empty(n_valid)
    valid_q[sorted_idx] = sorted_q

    # Insert q-values into full-length array (NaN positions preserve NaN)
    q_values = np.full(n, np.nan)
    q_values[valid_mask] = valid_q

    # Rejection: q < alpha (NaN never rejected)
    rejected = np.zeros(n, dtype=bool)
    rejected[valid_mask] = valid_q < alpha

    return q_values, rejected


def permutation_test_diff(
    values: np.ndarray,
    group_labels: np.ndarray,
    n_permutations: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """One-tailed permutation test for difference in means between two groups.

    Args:
        values: Array of observed values.
        group_labels: Binary array (0/1) labeling each value's group.
        n_permutations: Number of permutations to estimate null distribution.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple of:
            observed_diff: mean(group=1) - mean(group=0)
            p_value: fraction of permuted diffs >= observed_diff

    Notes:
        - Permutes group labels while keeping values fixed.
        - One-tailed: tests if group=1 has HIGHER mean than group=0.
        - For two-tailed, double the p-value or use abs(diff).
        - Returns NaN p-value if either group has < 2 members.
    """
    if rng is None:
        rng = np.random.default_rng()

    values = np.asarray(values, dtype=float)
    group_labels = np.asarray(group_labels, dtype=int)

    if len(values) != len(group_labels):
        raise ValueError("values and group_labels must have same length")

    mask_one = group_labels == 1
    mask_zero = group_labels == 0
    n_one = int(mask_one.sum())
    n_zero = int(mask_zero.sum())

    if n_one < 2 or n_zero < 2:
        return 0.0, float("nan")

    observed_diff = float(values[mask_one].mean() - values[mask_zero].mean())

    # Permutation distribution
    n = len(values)
    n_extreme = 0
    for _ in range(n_permutations):
        permuted_labels = rng.permutation(group_labels)
        perm_one = permuted_labels == 1
        perm_zero = permuted_labels == 0
        perm_diff = values[perm_one].mean() - values[perm_zero].mean()
        if perm_diff >= observed_diff:
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_permutations + 1)  # add 1 to avoid p=0
    return observed_diff, p_value
