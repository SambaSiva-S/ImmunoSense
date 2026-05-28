"""Static constants for Agent 1 (Biomarker).

Feature column ordering, disease class names, model hyperparameters.
"""

from __future__ import annotations


# ============================================================
# Layer 1: NHANES population baseline
# ============================================================

# NHANES demographic features for CRP quantile regression
LAYER1_FEATURE_COLS = ["age", "sex", "bmi"]

# Quantiles trained for percentile lookup
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


# ============================================================
# Layer 2: Rheumatic disease classification
# ============================================================

# Numeric biomarker features
NUMERIC_BIOMARKER_FEATURES = [
    "Age",
    "ESR",
    "CRP",
    "RF",
    "Anti-CCP",
    "C3",
    "C4",
]

# Categorical features (binary positive/negative tests)
CATEGORICAL_BIOMARKER_FEATURES = [
    "Gender_enc",
    "HLA-B27_enc",
    "ANA_enc",
    "Anti-Ro_enc",
    "Anti-La_enc",
    "Anti-dsDNA_enc",
    "Anti-Sm_enc",
]

# All value features (numeric + categorical)
ALL_VALUE_FEATURES = NUMERIC_BIOMARKER_FEATURES + CATEGORICAL_BIOMARKER_FEATURES

# Per-feature missingness indicators (1 if missing in source, 0 otherwise)
MISSING_FEATURES = [f"{f}_missing" for f in ALL_VALUE_FEATURES]

# Full input feature ordering for Layer 2 models (14 values + 14 missing flags = 28)
ALL_INPUT_FEATURES = ALL_VALUE_FEATURES + MISSING_FEATURES

# Disease classes (alphabetical, matching sklearn LabelEncoder default)
DISEASE_CLASSES = [
    "Ankylosing Spondylitis",
    "Normal",
    "Psoriatic Arthritis",
    "Rheumatoid Arthritis",
    "Sjogren's Syndrome",
    "Systemic Lupus Erythematosus",
    "Systemic Sclerosis",
]

# Embedding dimensionality for Pillar A contrastive encoder
LAYER2_EMBEDDING_DIM = 128

# Layer 2 default hyperparameters
LAYER2_HYPERPARAMS = {
    "lgb_n_estimators": 500,
    "lgb_num_leaves": 31,
    "lgb_learning_rate": 0.05,
    "lgb_early_stopping_rounds": 50,
    "xgb_n_estimators": 500,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.05,
    "xgb_early_stopping_rounds": 50,
    "encoder_lr": 0.001,
    "encoder_weight_decay": 0.01,
    "encoder_epochs": 100,
    "encoder_batch_size": 64,
    "ntxent_temperature": 0.1,
    "fusion_weight_a": 0.30,
    "fusion_weight_b": 0.35,
    "fusion_weight_c": 0.35,
    "fusion_similarity_temperature": 5.0,  # multiplier on cosine sims before softmax
}


# ============================================================
# Layer 3: Personal adaptation
# ============================================================

# Biomarkers tracked per-patient over time
BIOMARKERS_FOR_TRACKING = ["CRP", "ESR", "RF", "Anti-CCP", "C3", "C4"]

# Triggers correlated against biomarker changes
BIOMARKER_TRIGGERS = ["gluten_exposure", "poor_sleep", "high_stress", "high_aqi"]

# Layer 3 hyperparameters
LAYER3_HYPERPARAMS = {
    "tracker_window": 10,
    "tracker_outlier_threshold": 2.0,
    "tracker_personalization_days": 25,
    "detector_lag_range": (1, 3),
    "detector_min_readings": 10,
    "detector_correlation_threshold": 0.25,
    "detector_strong_correlation": 0.5,
}


# ============================================================
# NHANES data sources for Layer 1 training
# ============================================================

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

NHANES_FILES = {
    "P_HSCRP.XPT": "High-sensitivity CRP",
    "P_DEMO.XPT": "Demographics",
    "P_BMX.XPT": "Body measurements",
    "P_CBC.XPT": "Complete blood count",
}
