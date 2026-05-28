"""BiomarkerAgent - orchestrator for Agent 1 Layer 1 + 2 + 3.

Inherits BaseAgent for Conductor integration. Output dimension is 7
(disease class probabilities). The 128-dim contrastive embedding is
exposed separately via emit_embedding().

Usage::

    agent = BiomarkerAgent(patient_id='p001')
    agent.load_models(
        layer1_dir=Path('./artifacts/agent1_layer1'),
        layer2_dir=Path('./artifacts/agent1_layer2'),
    )

    result = agent.process({
        'demographics': {'age': 45, 'sex': 2, 'bmi': 28},
        'reading': {
            'day': 0,
            'Age': 45, 'CRP': 12.0, 'ESR': 45, 'RF': 80,
            'Anti-CCP': 65, 'C3': 95, 'C4': 22,
            'Gender_enc': 1, 'HLA-B27_enc': 0, 'ANA_enc': 0,
            'Anti-Ro_enc': 0, 'Anti-La_enc': 0,
            'Anti-dsDNA_enc': 0, 'Anti-Sm_enc': 0,
            'gluten_exposure': False, 'poor_sleep': True,
            'high_stress': False, 'high_aqi': False,
        },
    })
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.agents.biomarker.constants import (
    ALL_INPUT_FEATURES,
    ALL_VALUE_FEATURES,
    BIOMARKERS_FOR_TRACKING,
    BIOMARKER_TRIGGERS,
    DISEASE_CLASSES,
)
from immunosense.agents.biomarker.layer1.crp_baseline import CRPBaseline
from immunosense.agents.biomarker.layer2.encoder import get_similarities
from immunosense.agents.biomarker.layer2.explainer import (
    compute_shap_values,
    top_shap_drivers,
)
from immunosense.agents.biomarker.layer2.fusion import fuse_predictions
from immunosense.agents.biomarker.layer2.pipeline import Layer2Bundle
from immunosense.agents.biomarker.layer3.engine import PersonalAdaptationEngine
from immunosense.agents.biomarker.types import (
    Layer1Output,
    Layer2Output,
    Layer3Output,
)


class BiomarkerAgent(BaseAgent):
    """Per-patient orchestrator for Agent 1 (Biomarker).

    Output vector ordering (7-dim):
        Indices 0-6: disease probabilities in DISEASE_CLASSES order
                     (alphabetical, matching sklearn LabelEncoder default)

    The full 128-dim contrastive embedding is accessible via emit_embedding()
    and Layer2Output.contrastive_embedding.
    """

    agent_id = "agent1_biomarker"
    agent_version = "1.0.0"
    output_dim = 7  # disease class probabilities
    poll_frequency = "weekly"  # biomarker labs are typically not daily

    def __init__(
        self,
        patient_id: Optional[str] = None,
        biomarkers: Optional[list] = None,
        triggers: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.patient_id = patient_id
        self.biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)
        self.triggers = list(triggers or BIOMARKER_TRIGGERS)

        # Models — set via load_models()
        self.crp_baseline: Optional[CRPBaseline] = None
        self.layer2: Optional[Layer2Bundle] = None

        # Layer 3 personal adaptation
        self.layer3 = PersonalAdaptationEngine(
            biomarkers=self.biomarkers, triggers=self.triggers,
        )

        # Cache for last output (used by get_output_vector + emit_embedding)
        self._latest_layer2: Optional[Layer2Output] = None
        self._latest_embedding: Optional[np.ndarray] = None

    # =====================================================================
    # Model loading
    # =====================================================================

    def load_models(
        self,
        layer1_dir: Optional[Path] = None,
        layer2_dir: Optional[Path] = None,
        require_layer1: bool = True,
        require_layer2: bool = True,
    ) -> None:
        """Load trained Layer 1 and Layer 2 artifacts from disk.

        Args:
            layer1_dir: directory with crp_quantile_*.pkl files
            layer2_dir: directory with the 6 Layer 2 artifact files
            require_layer1: if True, raise if Layer 1 artifacts are missing.
                            Set False if you intend to run with Layer 2 + 3 only.
            require_layer2: similar for Layer 2.
        """
        if layer1_dir is not None:
            try:
                self.crp_baseline = CRPBaseline.load(layer1_dir)
            except FileNotFoundError:
                if require_layer1:
                    raise
                self.crp_baseline = None

        if layer2_dir is not None:
            try:
                self.layer2 = Layer2Bundle.load(layer2_dir)
            except FileNotFoundError:
                if require_layer2:
                    raise
                self.layer2 = None

    # =====================================================================
    # Layer 1 inference
    # =====================================================================

    def _run_layer1(self, demographics: dict, reading: dict) -> dict:
        """Compute population CRP percentile if Layer 1 and CRP are available."""
        if self.crp_baseline is None or "CRP" not in reading or reading["CRP"] is None:
            return {}

        crp_value = float(reading["CRP"])
        age = demographics.get("age", 40)
        sex = demographics.get("sex", 1)
        bmi = demographics.get("bmi", 25)

        percentile = self.crp_baseline.percentile(age, sex, bmi, crp_value)
        if percentile >= 0.95:
            interpretation = "ALARMING"
        elif percentile >= 0.75:
            interpretation = "ELEVATED"
        else:
            interpretation = "NORMAL"

        return {
            "CRP": {
                "value": crp_value,
                "population_percentile": float(percentile),
                "interpretation": interpretation,
            }
        }

    # =====================================================================
    # Layer 2 inference (3-pillar fusion)
    # =====================================================================

    def _reading_to_feature_vector(self, reading: dict) -> np.ndarray:
        """Convert a reading dict into the 28-feature vector Layer 2 expects."""
        values = []
        for feat in ALL_VALUE_FEATURES:
            v = reading.get(feat)
            if v is None:
                values.append(0.0)
            else:
                values.append(float(v))
        # Append missing flags
        for feat in ALL_VALUE_FEATURES:
            values.append(0.0 if reading.get(feat) is not None else 1.0)
        return np.array(values, dtype=float).reshape(1, -1)

    def _reading_to_feature_frame(self, reading: dict):
        """Convert a reading dict into a pandas DataFrame with named columns.

        Used to suppress sklearn 'X does not have valid feature names' warning
        when calling LGBMClassifier.predict_proba on a numpy array.
        """
        import pandas as pd
        x_arr = self._reading_to_feature_vector(reading)
        return pd.DataFrame(x_arr, columns=ALL_INPUT_FEATURES)

    def _run_layer2(self, reading: dict) -> Optional[Layer2Output]:
        """Run all 3 pillars + fusion."""
        if self.layer2 is None:
            return None

        # Build both: numpy for encoder/SHAP (needs ndarray), DataFrame for tree models
        x_arr = self._reading_to_feature_vector(reading)
        x_df = self._reading_to_feature_frame(reading)
        class_names = self.layer2.class_names

        # Pillar B: LightGBM probabilities (use DataFrame to keep feature names)
        lgb_probs = self.layer2.lgb_model.predict_proba(x_df)[0]

        # Pillar C: XGBoost probabilities + SHAP
        xgb_probs = self.layer2.xgb_model.predict_proba(x_df)[0]
        xgb_pred = int(np.argmax(xgb_probs))
        shap_values = compute_shap_values(self.layer2.xgb_model, x_arr, verbose=False)
        drivers = top_shap_drivers(
            shap_values, sample_idx=0, pred_class=xgb_pred,
            feature_names=ALL_INPUT_FEATURES,
            sample_values=x_arr[0],
            top_k=3,
        )

        # Pillar A: scale + encode + similarities to centroids
        x_scaled = self.layer2.scaler.transform(x_df)
        embedding = self.layer2.encoder.encode_single(x_scaled[0])
        sims = get_similarities(embedding, self.layer2.centroids)

        # Fusion
        pred_idx, confidence, agree, fused_probs, sim_probs = fuse_predictions(
            sims, lgb_probs, xgb_probs, class_names,
        )

        return Layer2Output(
            prediction=class_names[pred_idx],
            confidence=confidence,
            probabilities={
                class_names[i]: float(fused_probs[i]) for i in range(len(class_names))
            },
            pillar_a_similarities={k: float(v) for k, v in sims.items()},
            pillar_b_probabilities={
                class_names[i]: float(lgb_probs[i]) for i in range(len(class_names))
            },
            pillar_c_probabilities={
                class_names[i]: float(xgb_probs[i]) for i in range(len(class_names))
            },
            pillars_agree=agree,
            contrastive_embedding=embedding.tolist(),
            top_drivers=drivers,
        )

    # =====================================================================
    # Layer 3 inference (personal adaptation)
    # =====================================================================

    def _run_layer3(self, reading: dict) -> tuple:
        """Update tracker and return (Layer3Output, alerts) tuple."""
        ctx = self.layer3.process_reading(reading)
        layer3_out = Layer3Output(
            has_personal_data=ctx.get("has_personal_data", False),
            readings_count=ctx.get("readings_count", 0),
            personal_weight=ctx.get("personal_weight", 0.0),
            biomarkers=ctx.get("biomarkers", {}),
            patterns=ctx.get("patterns", {}).get("patterns", []),
            flare_rule=ctx.get("patterns", {}).get("flare_rule"),
        )
        return layer3_out, list(ctx.get("alerts", []))

    # =====================================================================
    # BaseAgent interface
    # =====================================================================

    def process(self, input_data: dict) -> AgentOutput:
        """Run all 3 layers and return an AgentOutput.

        Args:
            input_data: dict with required keys 'demographics' and 'reading'.
                demographics: {'age': float, 'sex': int (1 or 2), 'bmi': float}
                reading: dict of biomarker values + trigger booleans

        Returns:
            AgentOutput with:
                vector: 7-dim disease probability vector (in DISEASE_CLASSES order)
                data:   dict containing layer1, layer2, layer3 outputs + alerts
                alerts: list of alert dicts from Layer 3
                confidence: fused Layer 2 confidence (or 0.0 if Layer 2 missing)
        """
        start_time = datetime.now(timezone.utc)
        trace_id = self._new_trace_id()

        try:
            demographics = input_data["demographics"]
            reading = input_data["reading"]
        except KeyError as e:
            self._error_count += 1
            raise ValueError(
                "BiomarkerAgent.process requires "
                "input_data['demographics'] and input_data['reading']"
            ) from e

        # Run all 3 layers
        layer1_out = self._run_layer1(demographics, reading)
        layer2_out = self._run_layer2(reading)
        layer3_out, layer3_alerts = self._run_layer3(reading)

        # Compose alerts: Layer 3 personal alerts + Layer 1 population alert
        alerts = list(layer3_alerts)

        # Layer 1 alert if CRP is in alarming territory
        if "CRP" in layer1_out and layer1_out["CRP"]["interpretation"] == "ALARMING":
            alerts.append({
                "level": "WARNING",
                "message": (
                    f"CRP={layer1_out['CRP']['value']:.1f} is at the "
                    f"{layer1_out['CRP']['population_percentile'] * 100:.0f}th "
                    f"percentile for this demographic"
                ),
                "biomarker": "CRP",
            })

        # Build the 7-dim disease probability vector
        vector = self._build_output_vector(layer2_out)

        # Cache for emit_embedding / get_output_vector
        self._latest_layer2 = layer2_out
        if layer2_out is not None and layer2_out.contrastive_embedding is not None:
            self._latest_embedding = np.array(layer2_out.contrastive_embedding)

        # Compute confidence
        confidence = layer2_out.confidence if layer2_out is not None else 0.0

        # Record latency
        latency_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000.0
        self._record_latency(latency_ms)
        self._last_success = datetime.now(timezone.utc)

        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={
                "patient_id": self.patient_id,
                "day": reading.get("day", 0),
                "layer1": layer1_out,
                "layer2": (
                    {
                        "prediction": layer2_out.prediction,
                        "confidence": layer2_out.confidence,
                        "probabilities": layer2_out.probabilities,
                        "pillars_agree": layer2_out.pillars_agree,
                        "top_drivers": layer2_out.top_drivers,
                    } if layer2_out is not None else None
                ),
                "layer3": {
                    "has_personal_data": layer3_out.has_personal_data,
                    "readings_count": layer3_out.readings_count,
                    "personal_weight": layer3_out.personal_weight,
                    "biomarkers": layer3_out.biomarkers,
                    "n_patterns": len(layer3_out.patterns),
                },
            },
            vector=vector,
            vector_dim=self.output_dim,
            alerts=alerts,
            confidence=confidence,
            trace_id=trace_id,
        )

    def _build_output_vector(self, layer2_out: Optional[Layer2Output]) -> np.ndarray:
        """Build the 7-dim disease probability vector for the Conductor."""
        if layer2_out is None:
            return np.zeros(self.output_dim, dtype=np.float64)

        class_names = (
            self.layer2.class_names if self.layer2 is not None else DISEASE_CLASSES
        )
        vec = np.zeros(self.output_dim, dtype=np.float64)
        for i, c in enumerate(class_names[: self.output_dim]):
            vec[i] = float(layer2_out.probabilities.get(c, 0.0))
        return vec

    def get_output_vector(self) -> np.ndarray:
        """Return the most recent 7-dim disease probability vector."""
        return self._build_output_vector(self._latest_layer2)

    def emit_embedding(self, daily_summary: Any = None) -> np.ndarray:
        """Return the 128-dim contrastive embedding from the last process() call.

        If no inference has happened yet OR the encoder was unavailable,
        returns a zero vector of length LAYER2_EMBEDDING_DIM.

        The `daily_summary` argument is part of the BaseAgent interface but
        unused here — biomarker readings are sparse (weekly at most) so the
        embedding is sourced from the most recent inference, not summarized.
        """
        from immunosense.agents.biomarker.constants import LAYER2_EMBEDDING_DIM
        if self._latest_embedding is None:
            return np.zeros(LAYER2_EMBEDDING_DIM, dtype=np.float64)
        return self._latest_embedding.astype(np.float64)
