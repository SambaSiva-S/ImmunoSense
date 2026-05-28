"""Synthetic patient trajectory generator.

Generates realistic longitudinal biomarker trajectories with planted flare
patterns (2-3 distinct flares per trajectory, each lasting 3 readings).

Used by:
    - notebook validation (verify Layer 3 detects planted patterns)
    - test_agent.py (end-to-end agent validation)

NOT used at production inference time. This is purely for testing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.constants import BIOMARKERS_FOR_TRACKING


class ImprovedPatientGenerator:
    """Generate realistic patient trajectories with proper flare patterns.

    Builds per-disease biomarker statistics from a reference dataframe, then
    generates trajectories with:
        - Personal baseline at 30-60% of disease mean (patient in remission)
        - 2-3 distinct flare windows (3 readings each)
        - Flare-relevant biomarker spikes (CRP/ESR multiply, C3/C4 drop for lupus)
        - Triggers correlate with upcoming flares (70% trigger probability
          in the 2 readings before a flare)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        biomarkers: Optional[list] = None,
        disease_col: str = "Disease",
    ) -> None:
        self.biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)
        self.disease_col = disease_col

        # Learn per-disease mean and std for each biomarker
        self.disease_stats = {}
        for disease in df[disease_col].dropna().unique():
            mask = df[disease_col] == disease
            self.disease_stats[disease] = {
                bm: {
                    "mean": float(df.loc[mask, bm].dropna().mean()),
                    "std": float(df.loc[mask, bm].dropna().std()),
                }
                for bm in self.biomarkers
                if bm in df.columns
            }

    def generate_trajectory(
        self,
        disease: str,
        n_readings: int = 30,
        months: int = 12,
        random_seed: Optional[int] = None,
    ) -> tuple:
        """Generate a trajectory with 2-3 distinct flares.

        Args:
            disease: disease class name (must be in self.disease_stats)
            n_readings: total number of readings to generate
            months: trajectory span in months (sets days between readings)
            random_seed: for reproducibility

        Returns:
            Tuple (trajectory, personal_baseline) where:
                trajectory: list of dicts with biomarker + trigger keys
                personal_baseline: dict {biomarker: baseline_value}
        """
        if disease not in self.disease_stats:
            raise ValueError(
                f"Unknown disease '{disease}'. Known: {list(self.disease_stats.keys())}"
            )

        rng = np.random.RandomState(random_seed)
        stats = self.disease_stats[disease]

        # Personal baseline = 30-60% of disease mean (patient in remission)
        # Floor at 0.5 to prevent negative or near-zero baselines that would
        # cause downstream flare-spike math to collapse to the floor.
        personal_baseline = {}
        for bm in self.biomarkers:
            if bm not in stats:
                continue
            remission_factor = rng.uniform(0.3, 0.6)
            personal_baseline[bm] = max(0.5, stats[bm]["mean"] * remission_factor)

        # Plan 2-3 flares at specific time points (each lasts 3 readings)
        n_flares = rng.randint(2, 4)
        # Safe range: leave room for 3-reading flare windows
        if n_readings >= 10:
            flare_starts = sorted(rng.choice(
                range(5, n_readings - 4), size=n_flares, replace=False,
            ))
        else:
            flare_starts = []
        flare_duration = 3
        flare_readings = set()
        for start in flare_starts:
            for j in range(flare_duration):
                if start + j < n_readings:
                    flare_readings.add(start + j)

        trajectory = []
        days_between = (months * 30) / max(n_readings, 1)

        for i in range(n_readings):
            reading = {
                "day": int(i * days_between),
                "is_flare": i in flare_readings,
            }

            for bm in self.biomarkers:
                if bm not in stats:
                    continue
                base = personal_baseline[bm]
                noise = rng.normal(0, stats[bm]["std"] * 0.08)

                if i in flare_readings:
                    # Find position within flare: 0=rising, 1=peak, 2=falling
                    pos = 0
                    for start in flare_starts:
                        if start <= i < start + flare_duration:
                            pos = i - start
                            break

                    if bm in ["CRP", "ESR"]:
                        # Inflammatory markers spike during flare
                        multipliers = [2.0, 3.5, 2.0]
                        value = base * multipliers[min(pos, 2)] + noise
                    elif bm in ["C3", "C4"] and disease == "Systemic Lupus Erythematosus":
                        # Complement drops during lupus flare
                        drop_factors = [0.7, 0.5, 0.7]
                        value = base * drop_factors[min(pos, 2)] + noise
                    else:
                        value = base * 1.3 + noise
                else:
                    value = base + noise

                reading[bm] = round(max(0.1, value), 2)

            # Triggers correlate with upcoming flares
            if i in flare_readings:
                reading["gluten_exposure"] = False
                reading["poor_sleep"] = False
                reading["high_stress"] = False
                reading["high_aqi"] = False
            else:
                upcoming_flare = any(
                    (i + offset) in flare_readings for offset in [1, 2]
                )
                if upcoming_flare:
                    reading["gluten_exposure"] = bool(rng.rand() < 0.7)
                    reading["poor_sleep"] = bool(rng.rand() < 0.6)
                    reading["high_stress"] = bool(rng.rand() < 0.5)
                    reading["high_aqi"] = bool(rng.rand() < 0.4)
                else:
                    reading["gluten_exposure"] = bool(rng.rand() < 0.15)
                    reading["poor_sleep"] = bool(rng.rand() < 0.15)
                    reading["high_stress"] = bool(rng.rand() < 0.1)
                    reading["high_aqi"] = bool(rng.rand() < 0.1)

            trajectory.append(reading)

        return trajectory, personal_baseline


def generate_synthetic_trajectory(
    disease: str,
    biomarkers: Optional[list] = None,
    n_readings: int = 30,
    months: int = 12,
    random_seed: int = 42,
) -> tuple:
    """Convenience: generate a synthetic trajectory without a reference dataframe.

    Uses hardcoded reasonable distributions per disease (based on the
    Rheumatic dataset stats). For more realistic generation, build an
    ImprovedPatientGenerator from a real dataframe.

    Args:
        disease: one of the DISEASE_CLASSES
        biomarkers: biomarkers to include (default BIOMARKERS_FOR_TRACKING)
        n_readings, months, random_seed: as in ImprovedPatientGenerator

    Returns:
        Tuple (trajectory, personal_baseline).
    """
    biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)

    # Hardcoded per-disease stats based on observed Rheumatic dataset means
    PRESET_STATS = {
        "Rheumatoid Arthritis": {
            "CRP": {"mean": 25.0, "std": 18.0},
            "ESR": {"mean": 45.0, "std": 22.0},
            "RF": {"mean": 90.0, "std": 70.0},
            "Anti-CCP": {"mean": 90.0, "std": 65.0},
            "C3": {"mean": 110.0, "std": 25.0},
            "C4": {"mean": 25.0, "std": 8.0},
        },
        "Systemic Lupus Erythematosus": {
            "CRP": {"mean": 18.0, "std": 14.0},
            "ESR": {"mean": 50.0, "std": 25.0},
            "RF": {"mean": 35.0, "std": 30.0},
            "Anti-CCP": {"mean": 25.0, "std": 25.0},
            "C3": {"mean": 75.0, "std": 25.0},
            "C4": {"mean": 15.0, "std": 8.0},
        },
        "Normal": {
            "CRP": {"mean": 2.0, "std": 1.5},
            "ESR": {"mean": 12.0, "std": 6.0},
            "RF": {"mean": 10.0, "std": 5.0},
            "Anti-CCP": {"mean": 3.0, "std": 2.0},
            "C3": {"mean": 120.0, "std": 20.0},
            "C4": {"mean": 28.0, "std": 6.0},
        },
    }

    if disease not in PRESET_STATS:
        # Fall back to Normal distribution for any disease we don't have presets for
        stats = PRESET_STATS["Normal"]
    else:
        stats = PRESET_STATS[disease]

    # Build a fake DataFrame with the stats, then use ImprovedPatientGenerator
    # for consistency with the testing path. We use a deterministic but varied
    # seed per disease (NOT recreated inside the row loop, which would make all
    # 50 rows for that disease identical).
    fake_rows = []
    for d_idx, (d, d_stats) in enumerate(PRESET_STATS.items()):
        # One RandomState per disease, varied so std() > 0 across rows
        d_rng = np.random.RandomState(random_seed + d_idx * 100)
        for _ in range(50):
            row = {"Disease": d}
            for bm, bm_stats in d_stats.items():
                row[bm] = max(0.1, float(d_rng.normal(bm_stats["mean"], bm_stats["std"])))
            fake_rows.append(row)
    fake_df = pd.DataFrame(fake_rows)

    gen = ImprovedPatientGenerator(fake_df, biomarkers=biomarkers)
    return gen.generate_trajectory(
        disease=disease, n_readings=n_readings, months=months,
        random_seed=random_seed,
    )
