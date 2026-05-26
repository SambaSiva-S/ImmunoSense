"""MockWearableGenerator - realistic synthetic wearable data.

Produces minute-level HR, RR intervals, skin temperature, accelerometer (ENMO),
SpO2, and sleep stages for a single night (~8 hours / 480 minutes).

WESAD/MESA-derived population baselines with embedded flare states:
    normal: baseline biology, no inflammation
    prodrome: 30% HRV drop, mild temp elevation, decoupled HRV-temp
    acute_stress: 35% HRV drop, much higher HR, poor sleep
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


class MockWearableGenerator:
    """Produces realistic minute-level wearable data with embedded flares."""

    # WESAD/MESA-derived population baselines
    BASELINE = {
        "resting_hr": 62.0,        # bpm
        "rr_interval": 950.0,       # ms (RR = 60000 / HR roughly)
        "hrv_rmssd": 45.0,          # ms during sleep
        "skin_temp": 33.5,          # degrees C overnight
        "spo2": 97.5,               # %
        "sleep_efficiency": 0.87,   # fraction
    }

    def __init__(self, patient_id: str = "p001", seed: int = 42) -> None:
        self.patient_id = patient_id
        self.rng = np.random.default_rng(seed)
        # Personal offsets — fixed per patient
        self.personal = {
            "resting_hr": self.BASELINE["resting_hr"] + self.rng.normal(0, 4),
            "hrv_rmssd_base": self.BASELINE["hrv_rmssd"] + self.rng.normal(0, 5),
            "skin_temp_base": self.BASELINE["skin_temp"] + self.rng.normal(0, 0.1),
        }

    def _nightly_jitter(self) -> dict:
        """Re-jitter the personal baseline each night.

        Real human biology has natural inter-night variability — same person,
        same sensor, naturally varies by a few ms / bpm / 0.1 C from one night
        to the next. Without this, baseline IQR collapses to near-zero and
        anomaly scores explode on flare nights.
        """
        return {
            "hrv": float(self.rng.normal(0, 4.0)),       # ~4 ms inter-night variance
            "hr": float(self.rng.normal(0, 1.5)),         # ~1.5 bpm inter-night variance
            "skin_temp": float(self.rng.normal(0, 0.15)), # ~0.15 C inter-night variance
        }

    def generate_night(
        self,
        night_idx: int,
        flare_state: str = "normal",
        start_time: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, list[float]]:
        """Generate one night of minute-level data (8 hours = 480 minutes).

        Args:
            night_idx: Night number (used to derive default start_time).
            flare_state: One of 'normal' | 'prodrome' | 'acute_stress'.
            start_time: Optional override for night start (defaults to Apr 1, 2026 + night_idx).

        Returns:
            Tuple of (DataFrame indexed by timestamp, list of RR intervals).
        """
        if start_time is None:
            start_time = (
                datetime(2026, 4, 1, 23, 0, 0, tzinfo=timezone.utc)
                + timedelta(days=night_idx)
            )
        n_minutes = 480  # 8 hours

        # Per-night biological jitter
        jitter = self._nightly_jitter()

        # Flare modifiers
        if flare_state == "prodrome":
            hrv_multiplier = 0.70           # 30% HRV drop
            sleep_quality_mult = 0.85       # poorer sleep
            temp_offset = 0.4               # 0.4 C elevation
            tadi_decouple = True            # break HRV-temp coupling
            hr_offset = 4
        elif flare_state == "acute_stress":
            hrv_multiplier = 0.65
            sleep_quality_mult = 0.75
            temp_offset = 0.1
            tadi_decouple = False
            hr_offset = 8
        else:
            hrv_multiplier = 1.0
            sleep_quality_mult = 1.0
            temp_offset = 0.0
            tadi_decouple = False
            hr_offset = 0

        # Sleep stages
        stages = self._simulate_sleep_stages(n_minutes, sleep_quality_mult)

        # Heart rate (minute-level)
        hr = np.full(n_minutes, self.personal["resting_hr"] + hr_offset + jitter["hr"])
        hr += self.rng.normal(0, 1.5, n_minutes)
        for i, st in enumerate(stages):
            if st == "deep":
                hr[i] -= 6
            elif st == "rem":
                hr[i] += 4
            elif st == "wake":
                hr[i] += 12
        hr = np.maximum(hr, 35.0)

        # RR intervals derived from HR
        rr_intervals = []
        hrv_per_minute = (
            (self.personal["hrv_rmssd_base"] + jitter["hrv"]) * hrv_multiplier
        )
        hrv_per_minute = max(hrv_per_minute, 5.0)  # floor — HRV can't go to zero
        for i in range(n_minutes):
            n_beats = max(int(hr[i]), 30)
            mean_rr = 60000.0 / hr[i]
            rr = self.rng.normal(mean_rr, hrv_per_minute, n_beats)
            rr = np.maximum(rr, 300)  # physiological floor
            rr_intervals.extend(rr.tolist())

        # Skin temperature
        time_hours = np.arange(n_minutes) / 60.0
        base_temp = self.personal["skin_temp_base"] + temp_offset + jitter["skin_temp"]
        if tadi_decouple:
            # Decoupled: temp stays elevated, doesn't follow normal cycle
            skin_temp = base_temp + self.rng.normal(0, 0.15, n_minutes)
        else:
            # Normal cycle: gentle rise from sleep onset, peak ~mid-sleep
            skin_temp_pattern = 0.5 * np.sin(np.pi * time_hours / 8.0)
            skin_temp = (
                base_temp + skin_temp_pattern + self.rng.normal(0, 0.1, n_minutes)
            )

        # Accelerometer (ENMO)
        enmo = np.full(n_minutes, 0.01)
        for i, st in enumerate(stages):
            if st == "wake":
                enmo[i] = self.rng.uniform(0.05, 0.4)
            elif st == "rem":
                enmo[i] = self.rng.uniform(0.005, 0.03)
            else:
                enmo[i] = self.rng.uniform(0.001, 0.015)

        # SpO2
        spo2 = (
            np.full(n_minutes, self.BASELINE["spo2"])
            + self.rng.normal(0, 0.4, n_minutes)
        )
        # Occasional dips during REM
        for i, st in enumerate(stages):
            if st == "rem" and self.rng.random() < 0.02:
                spo2[i] -= self.rng.uniform(2, 5)
        spo2 = np.clip(spo2, 80, 100)

        timestamps = [start_time + timedelta(minutes=i) for i in range(n_minutes)]
        df = pd.DataFrame({
            "timestamp": timestamps,
            "hr": hr,
            "skin_temp": skin_temp,
            "enmo": enmo,
            "spo2": spo2,
            "sleep_stage": stages,
        })

        return df, rr_intervals

    def _simulate_sleep_stages(self, n_minutes: int, quality_mult: float) -> list:
        """Produce a realistic sleep-stage sequence.

        Rough cycle: light(20) → deep(15) → light(20) → rem(15), repeat ~5 cycles.
        Then inject wake periods proportionally to (1 - quality_mult).
        """
        stages = []
        cycle_template = ["light"] * 20 + ["deep"] * 15 + ["light"] * 20 + ["rem"] * 15
        while len(stages) < n_minutes:
            stages.extend(cycle_template)
        stages = stages[:n_minutes]

        # Inject wake periods (poor sleep quality = more wake)
        wake_rate = (1.0 - quality_mult) * 0.15 + 0.02
        for i in range(n_minutes):
            if self.rng.random() < wake_rate:
                stages[i] = "wake"

        return stages
