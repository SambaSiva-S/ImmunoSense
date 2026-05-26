"""MockSymptomSource - deterministic synthetic symptom data.

Samples from disease-stratified DISEASE_NORMS. Used for testing and as
fallback when no real patient input is available.
"""

from __future__ import annotations

import hashlib
import random as random_mod

from immunosense.agents.symptoms_mood.norms import DISEASE_NORMS, normalize_disease
from immunosense.agents.symptoms_mood.types import ALL_FEATURES, FetchedSymptoms


class MockSymptomSource:
    """Deterministic synthetic symptom data, sampled from disease norms.

    Args:
        disease: Disease type (canonical or alias). Defaults to 'Mixed'.
        seed_offset: Adjust to get different deterministic streams.
    """

    def __init__(self, disease: str = "Mixed", seed_offset: int = 0) -> None:
        self.disease = normalize_disease(disease)
        self.seed_offset = seed_offset

    def fetch(self, patient_id: str, target_date: str) -> FetchedSymptoms:
        """Return deterministic synthetic features for the given patient and date."""
        seed_key = f"{patient_id},{target_date},{self.seed_offset}"
        seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
        rng = random_mod.Random(seed)

        norms = DISEASE_NORMS[self.disease]

        def _sample(feat: str, lo: float = 0.0, hi: float = 10.0) -> float:
            mean, std = norms[feat]
            return max(lo, min(hi, rng.gauss(mean, std)))

        result = FetchedSymptoms(
            fatigue=round(_sample("fatigue"), 1),
            joint_pain=round(_sample("joint_pain"), 1),
            brain_fog_severity=round(_sample("brain_fog_severity"), 1),
            gi_distress=round(_sample("gi_distress"), 1),
            skin_severity=round(_sample("skin_severity"), 1),
            sleep_severity=round(_sample("sleep_severity"), 1),
            energy_severity=round(_sample("energy_severity"), 1),
            wellness_severity=round(_sample("wellness_severity"), 1),
            phq8_score=round(max(0, min(24, rng.gauss(*norms["phq8_score"]))), 0),
            gad7_score=round(max(0, min(21, rng.gauss(*norms["gad7_score"]))), 0),
            explicit_flare=False,
        )
        for f in ALL_FEATURES:
            result.confidence[f] = "synthetic"
            result.sources[f] = "mock"
        return result
