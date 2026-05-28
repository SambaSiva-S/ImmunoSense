"""Agent 1 — Biomarker Agent.

Three-layer ML architecture:
    Layer 1: Population-adjusted CRP baseline (LightGBM quantile regression
             on NHANES). Predicts demographic-specific CRP percentiles.
    Layer 2: Disease intelligence via three cognitive pillars on the
             Rheumatic dataset:
               Pillar A (Spatial):     Contrastive encoder + nearest centroid
               Pillar B (Probabilistic): LightGBM multiclass classifier
               Pillar C (Explanatory):   XGBoost classifier + SHAP
             Fusion: 0.3 * Pillar A + 0.35 * Pillar B + 0.35 * Pillar C
    Layer 3: Personal adaptation via RobustBaselineTracker (median + IQR,
             outlier-resistant) + PatternDetector (trigger -> biomarker
             correlation across time lags).

The agent emits a 7-dim disease-probability vector for the Conductor and a
128-dim contrastive embedding via emit_embedding().

Public API::

    >>> from immunosense.agents.biomarker import BiomarkerAgent
    >>> agent = BiomarkerAgent(patient_id='p001')
    >>> agent.load_models(layer1_dir='./artifacts/agent1_layer1',
    ...                    layer2_dir='./artifacts/agent1_layer2')
    >>> result = agent.process({
    ...     'demographics': {'age': 45, 'sex': 2, 'bmi': 28},
    ...     'reading': {'day': 0, 'CRP': 12.0, 'ESR': 45, ...},
    ... })

Training the models is done offline via standalone scripts:
    python -m immunosense.agents.biomarker.layer1.train
    python -m immunosense.agents.biomarker.layer2.train
    python -m immunosense.agents.biomarker.train          # both
"""

from immunosense.agents.biomarker.agent import BiomarkerAgent
from immunosense.agents.biomarker.constants import (
    ALL_INPUT_FEATURES,
    BIOMARKERS_FOR_TRACKING,
    BIOMARKER_TRIGGERS,
    DISEASE_CLASSES,
    LAYER1_FEATURE_COLS,
    QUANTILES,
)
from immunosense.agents.biomarker.layer1.crp_baseline import (
    CRPBaseline,
    get_crp_percentile,
    load_layer1_artifacts,
)
from immunosense.agents.biomarker.layer2.fusion import fuse_predictions
from immunosense.agents.biomarker.layer2.pipeline import Layer2Bundle
from immunosense.agents.biomarker.layer3.detector import PatternDetector
from immunosense.agents.biomarker.layer3.engine import PersonalAdaptationEngine
from immunosense.agents.biomarker.layer3.trackers import BiomarkerBaselineTracker
from immunosense.agents.biomarker.synthetic import (
    ImprovedPatientGenerator,
    generate_synthetic_trajectory,
)
from immunosense.agents.biomarker.types import (
    BiomarkerAgentReport,
    BiomarkerReading,
    DetectedTriggerPattern,
    Layer1Output,
    Layer2Output,
    Layer3Output,
)

__all__ = [
    # Main agent
    "BiomarkerAgent",
    # Layer 1
    "CRPBaseline",
    "get_crp_percentile",
    "load_layer1_artifacts",
    # Layer 2
    "Layer2Bundle",
    "fuse_predictions",
    # Layer 3
    "BiomarkerBaselineTracker",
    "PatternDetector",
    "PersonalAdaptationEngine",
    # Synthetic data
    "ImprovedPatientGenerator",
    "generate_synthetic_trajectory",
    # Types
    "BiomarkerReading",
    "Layer1Output",
    "Layer2Output",
    "Layer3Output",
    "DetectedTriggerPattern",
    "BiomarkerAgentReport",
    # Constants
    "ALL_INPUT_FEATURES",
    "BIOMARKERS_FOR_TRACKING",
    "BIOMARKER_TRIGGERS",
    "DISEASE_CLASSES",
    "LAYER1_FEATURE_COLS",
    "QUANTILES",
]
