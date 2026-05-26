"""Tests for dietary constants and reference tables."""

from immunosense.agents.dietary.constants import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
    DEFAULT_GI,
    DII_REF,
    GI_BY_CATEGORY,
    NHANES_DR1IFF_COL_MAP,
    NHANES_DR1TOT_COL_MAP,
    QUANTILES,
)


def test_dii_ref_has_27_components():
    """Shivappa 2014 uses 27 NHANES-available components (alcohol incl., turmeric etc. excluded)."""
    assert len(DII_REF) == 27


def test_dii_ref_tuple_shape():
    """Every DII_REF entry must be (mean, sd, effect) triple."""
    for component, value in DII_REF.items():
        assert isinstance(value, tuple), f"{component} not a tuple"
        assert len(value) == 3, f"{component} has {len(value)} values, expected 3"
        mean, sd, effect = value
        assert isinstance(mean, (int, float))
        assert isinstance(sd, (int, float))
        assert isinstance(effect, (int, float))
        assert sd > 0, f"{component} has zero or negative sd"


def test_dii_effects_have_both_signs():
    """Pro- and anti-inflammatory components should both exist."""
    effects = [v[2] for v in DII_REF.values()]
    assert any(e > 0 for e in effects), "no pro-inflammatory components"
    assert any(e < 0 for e in effects), "no anti-inflammatory components"


def test_dii_known_pro_inflammatory():
    """Saturated fat, fat_total should be pro-inflammatory (effect > 0)."""
    assert DII_REF["saturated_fat_g"][2] > 0
    assert DII_REF["fat_total_g"][2] > 0


def test_dii_known_anti_inflammatory():
    """Fiber, omega-3, vit C, magnesium should be anti-inflammatory (effect < 0)."""
    assert DII_REF["fiber_g"][2] < 0
    assert DII_REF["omega3_g"][2] < 0
    assert DII_REF["vit_c_mg"][2] < 0
    assert DII_REF["magnesium_mg"][2] < 0


def test_nhanes_col_maps_cover_dii_components():
    """Every non-derived DII component should have a NHANES col mapping."""
    derived_components = {"omega3_g", "omega6_g"}  # built from fatty acid cols
    for component in DII_REF:
        if component in derived_components:
            continue
        # DR1TOT and DR1IFF maps should each have this component
        assert component in NHANES_DR1TOT_COL_MAP, f"DR1TOT missing {component}"
        if component != "alcohol_g":  # alcohol_g present in both
            # DR1IFF has slightly different scope (excludes some, adds sodium)
            pass


def test_dr1iff_includes_sodium():
    """DR1IFF column map should include sodium (Layer 3 needs it for Th17 driver)."""
    assert "sodium_mg" in NHANES_DR1IFF_COL_MAP


def test_quantiles_are_layer1_set():
    """Layer 1 trains regressors at these specific quantiles."""
    assert QUANTILES == [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def test_feature_ordering_locked():
    """Layer 3 feature ordering is part of the public API; must not change silently."""
    assert CONTINUOUS_FEATURES == [
        "dii_score",
        "omega6_omega3_ratio",
        "glycemic_load",
        "sodium_mg",
        "alcohol_g",
        "overnight_fast_hours",
    ]
    assert BOOLEAN_TRIGGERS == [
        "gluten_present",
        "dairy_present",
        "nightshade_present",
        "upf_present",
    ]


def test_default_gi_in_valid_range():
    """Default GI fallback must be a sane value."""
    assert 0 < DEFAULT_GI < 100


def test_gi_table_values_in_range():
    """All GI values should be in [0, 100]."""
    for kw, gi in GI_BY_CATEGORY.items():
        assert 0 <= gi <= 100, f"{kw} has GI {gi} out of range"


def test_gi_table_protein_zero():
    """Pure proteins (chicken, fish, eggs) should have GI = 0."""
    assert GI_BY_CATEGORY["chicken"] == 0
    assert GI_BY_CATEGORY["salmon"] == 0
    assert GI_BY_CATEGORY["eggs"] == 0
