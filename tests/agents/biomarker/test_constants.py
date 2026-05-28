"""Tests for biomarker constants — feature ordering, hyperparameters, class names."""

from immunosense.agents.biomarker.constants import (
    ALL_INPUT_FEATURES,
    ALL_VALUE_FEATURES,
    BIOMARKERS_FOR_TRACKING,
    BIOMARKER_TRIGGERS,
    CATEGORICAL_BIOMARKER_FEATURES,
    DISEASE_CLASSES,
    LAYER1_FEATURE_COLS,
    LAYER2_EMBEDDING_DIM,
    LAYER2_HYPERPARAMS,
    LAYER3_HYPERPARAMS,
    MISSING_FEATURES,
    NUMERIC_BIOMARKER_FEATURES,
    QUANTILES,
)


def test_layer1_feature_cols_are_demographics():
    assert LAYER1_FEATURE_COLS == ["age", "sex", "bmi"]


def test_quantiles_are_seven():
    assert QUANTILES == [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def test_seven_disease_classes():
    assert len(DISEASE_CLASSES) == 7


def test_disease_classes_alphabetical():
    """LabelEncoder defaults to alphabetical ordering; class names must match."""
    assert DISEASE_CLASSES == sorted(DISEASE_CLASSES)


def test_disease_classes_include_normal():
    assert "Normal" in DISEASE_CLASSES


def test_numeric_features_seven():
    """7 numeric features: Age, ESR, CRP, RF, Anti-CCP, C3, C4."""
    assert len(NUMERIC_BIOMARKER_FEATURES) == 7
    for feat in ["Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"]:
        assert feat in NUMERIC_BIOMARKER_FEATURES


def test_categorical_features_seven():
    """7 categorical features (Gender + 6 antibody tests)."""
    assert len(CATEGORICAL_BIOMARKER_FEATURES) == 7


def test_all_value_features_combine():
    """Total value features = numeric + categorical = 14."""
    assert len(ALL_VALUE_FEATURES) == 14
    assert ALL_VALUE_FEATURES == NUMERIC_BIOMARKER_FEATURES + CATEGORICAL_BIOMARKER_FEATURES


def test_missing_features_match_value_features():
    """One missing flag per value feature."""
    assert len(MISSING_FEATURES) == len(ALL_VALUE_FEATURES)
    for feat in ALL_VALUE_FEATURES:
        assert f"{feat}_missing" in MISSING_FEATURES


def test_all_input_features_is_28():
    """14 value features + 14 missing flags = 28 total input features."""
    assert len(ALL_INPUT_FEATURES) == 28


def test_embedding_dim_is_128():
    assert LAYER2_EMBEDDING_DIM == 128


def test_layer2_fusion_weights_sum_to_one():
    """0.30 + 0.35 + 0.35 = 1.0."""
    weights = (
        LAYER2_HYPERPARAMS["fusion_weight_a"]
        + LAYER2_HYPERPARAMS["fusion_weight_b"]
        + LAYER2_HYPERPARAMS["fusion_weight_c"]
    )
    assert abs(weights - 1.0) < 1e-9


def test_layer2_pillars_b_and_c_have_equal_weight():
    """Tree models get equal weight (0.35 each)."""
    assert (
        LAYER2_HYPERPARAMS["fusion_weight_b"]
        == LAYER2_HYPERPARAMS["fusion_weight_c"]
    )


def test_layer3_biomarkers_for_tracking():
    """6 longitudinal biomarkers."""
    assert len(BIOMARKERS_FOR_TRACKING) == 6
    for bm in ["CRP", "ESR", "RF", "Anti-CCP", "C3", "C4"]:
        assert bm in BIOMARKERS_FOR_TRACKING


def test_layer3_triggers_include_main_four():
    for trigger in ["gluten_exposure", "poor_sleep", "high_stress", "high_aqi"]:
        assert trigger in BIOMARKER_TRIGGERS


def test_layer3_tracker_window_at_least_3():
    """Robust tracker needs minimum window of 3."""
    assert LAYER3_HYPERPARAMS["tracker_window"] >= 3


def test_layer3_detector_lag_range_is_tuple():
    lag_range = LAYER3_HYPERPARAMS["detector_lag_range"]
    assert isinstance(lag_range, tuple)
    assert lag_range[0] <= lag_range[1]
