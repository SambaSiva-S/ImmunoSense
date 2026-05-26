"""NHANES PHQ-8 integration for real population baselines.

Downloads NHANES P_DPQ.XPT (Public Use, 2017-March 2020 pre-pandemic, adults 18+),
computes PHQ-8 distribution from real US population data, and OVERRIDES the
literature-derived 'Mixed' PHQ-8 baseline in DISEASE_NORMS.

CRITICAL: filters on items 1-8 ONLY (drops item 9 suicide screen). NHANES
skip-logic means many respondents have missing values on item 9 that aren't
real refusals - filtering on all 9 items would eliminate most valid responses.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from immunosense.agents.symptoms_mood.norms import DISEASE_NORMS


NHANES_DPQ_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DPQ.XPT"

# Default to project-local data/ directory
def _default_data_dir() -> Path:
    if Path("./data").exists():
        return Path("./data/nhanes")
    return Path("../data/nhanes")


def download_nhanes_dpq(
    local_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download NHANES depression screening file, or return cached copy.

    Args:
        local_path: Override local cache path. Defaults to ./data/nhanes/P_DPQ.XPT.
        force: If True, re-download even if cache exists.

    Returns:
        Path to the cached NHANES file.

    Raises:
        RuntimeError: If download fails and no cache exists.
    """
    if local_path is None:
        local_path = _default_data_dir() / "P_DPQ.XPT"

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not force:
        return local_path

    try:
        urllib.request.urlretrieve(NHANES_DPQ_URL, local_path)
        return local_path
    except Exception as e:
        if local_path.exists():
            return local_path  # use existing cache despite download failure
        raise RuntimeError(f"NHANES download failed and no cache: {e}") from e


def compute_phq8_norms_from_nhanes(local_path: Path) -> dict:
    """Parse NHANES DPQ file, compute PHQ-8 distribution statistics.

    Filters on items 1-8 ONLY (DPQ010 through DPQ080), deliberately dropping
    item 9 (suicide screen). NHANES skip-logic causes many respondents to
    have missing values on item 9 that aren't real refusals - including
    that filter would eliminate most valid PHQ-8 responses.

    Args:
        local_path: Path to local NHANES P_DPQ.XPT file.

    Returns:
        Dict with mean, std, median, p25, p75, n, severity_pct, source.

    Raises:
        ValueError: If file is missing expected columns or has no valid rows.
    """
    df = pd.read_sas(local_path, format="xport")
    phq_items_all = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    phq8_items = phq_items_all[:8]  # items 1-8 only

    missing = [c for c in phq8_items if c not in df.columns]
    if missing:
        raise ValueError(f"NHANES DPQ missing columns: {missing}")

    # Filter to valid responses (0-3) on items 1-8 only
    valid = df.copy()
    for c in phq8_items:
        valid = valid[valid[c].isin([0, 1, 2, 3])]
    if len(valid) == 0:
        raise ValueError("No valid PHQ-8 responses found")

    valid = valid.copy()
    valid["phq8_total"] = valid[phq8_items].sum(axis=1)

    def _cat(s: float) -> str:
        if s <= 4:
            return "minimal"
        if s <= 9:
            return "mild"
        if s <= 14:
            return "moderate"
        if s <= 19:
            return "mod_severe"
        return "severe"

    return {
        "mean": float(valid["phq8_total"].mean()),
        "std": float(valid["phq8_total"].std()),
        "median": float(valid["phq8_total"].median()),
        "p25": float(valid["phq8_total"].quantile(0.25)),
        "p75": float(valid["phq8_total"].quantile(0.75)),
        "n": len(valid),
        "severity_pct": (
            valid["phq8_total"].apply(_cat).value_counts(normalize=True).to_dict()
        ),
        "source": "NHANES 2017-March 2020 pre-pandemic, adults 18+",
    }


def update_disease_norms_with_nhanes(
    local_path: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Download NHANES, compute PHQ-8 norms, update DISEASE_NORMS['Mixed'].

    Args:
        local_path: Override local cache path.
        verbose: If True, print stats and update info.

    Returns:
        The computed stats dict, or {'error': ...} if loading failed.
    """
    try:
        local = download_nhanes_dpq(local_path)
        stats = compute_phq8_norms_from_nhanes(local)
    except Exception as e:
        if verbose:
            print(f"WARNING: Could not load NHANES PHQ-8 data: {e}")
            print("Falling back to literature-approximated value for Mixed/PHQ-8.")
        return {"error": str(e)}

    if verbose:
        print(f"NHANES PHQ-8 statistics (N={stats['n']:,}):")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Std:    {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.1f}")

    old_value = DISEASE_NORMS["Mixed"]["phq8_score"]
    new_value = (round(stats["mean"], 2), round(stats["std"], 2))
    DISEASE_NORMS["Mixed"]["phq8_score"] = new_value

    if verbose:
        print(f'\nDISEASE_NORMS["Mixed"]["phq8_score"]:')
        print(f"  Before (literature):  {old_value}")
        print(f"  After  (NHANES real): {new_value}")

    return stats
