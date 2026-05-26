"""Static reference data for Agent 2 (Dietary).

Constants here are pure data with no runtime dependencies. Keep them
isolated so other modules can import them without circular imports.

References:
    DII_REF: Shivappa N, et al. (2014) Public Health Nutr 17(8):1689-96.
             "Designing and developing a literature-derived, population-based
              dietary inflammatory index."
    NHANES col map: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DR1TOT.htm
"""

from __future__ import annotations


# ============================================================
# DII Reference (Shivappa 2014)
# Format: component -> (global_mean, global_sd, inflammatory_effect_score)
# Negative effect = anti-inflammatory. Positive = pro-inflammatory.
# ============================================================
DII_REF = {
    "energy_kcal":          (2056.0,    338.0,   0.180),
    "carbohydrate_g":       ( 272.2,     40.0,   0.097),
    "protein_g":            (  79.4,     13.9,   0.021),
    "fat_total_g":          (  71.4,     19.4,   0.298),
    "saturated_fat_g":      (  28.6,      8.0,   0.373),
    "mufa_g":               (  27.0,      6.1,  -0.009),
    "pufa_g":               (  13.9,      3.9,  -0.337),
    "omega3_g":             (   1.06,     1.06, -0.436),
    "omega6_g":             (  10.8,      7.5,  -0.159),
    "cholesterol_mg":       ( 279.4,     51.2,   0.110),
    "fiber_g":              (  18.8,      4.9,  -0.663),
    "vit_a_mcg_rae":        ( 983.9,    518.6, -0.401),
    "vit_b1_thiamin_mg":    (   1.70,     0.66, -0.098),
    "vit_b2_riboflavin_mg": (   1.70,     0.79, -0.068),
    "vit_b3_niacin_mg":     (  25.90,    11.77, -0.246),
    "vit_b6_mg":            (   1.47,     0.74, -0.365),
    "vit_b12_mcg":          (   5.15,     2.70,  0.106),
    "folate_mcg":           ( 273.0,     70.7, -0.190),
    "vit_c_mg":             ( 118.2,     43.46, -0.424),
    "vit_d_mcg":            (   6.26,     2.21, -0.446),
    "vit_e_mg":             (   8.73,     1.49, -0.419),
    "iron_mg":              (  13.35,     3.71,  0.032),
    "magnesium_mg":         ( 310.1,    139.4, -0.484),
    "selenium_mcg":         (  67.0,     25.1, -0.191),
    "zinc_mg":              (   9.84,     2.19, -0.313),
    "caffeine_g":           (   8.05,     6.67, -0.110),
    "alcohol_g":            (  13.98,     3.72, -0.278),
}


# ============================================================
# NHANES DR1TOT column mapping (per-participant 24hr recall totals)
# Used by Layer 1 training pipeline (NHANES -> DII percentile model).
# ============================================================
NHANES_DR1TOT_COL_MAP = {
    "energy_kcal":          "DR1TKCAL",
    "carbohydrate_g":       "DR1TCARB",
    "protein_g":            "DR1TPROT",
    "fat_total_g":          "DR1TTFAT",
    "saturated_fat_g":      "DR1TSFAT",
    "mufa_g":               "DR1TMFAT",
    "pufa_g":               "DR1TPFAT",
    "omega3_g":             None,   # derived from individual fatty acids
    "omega6_g":             None,   # derived from individual fatty acids
    "cholesterol_mg":       "DR1TCHOL",
    "fiber_g":              "DR1TFIBE",
    "vit_a_mcg_rae":        "DR1TVARA",
    "vit_b1_thiamin_mg":    "DR1TVB1",
    "vit_b2_riboflavin_mg": "DR1TVB2",
    "vit_b3_niacin_mg":     "DR1TNIAC",
    "vit_b6_mg":            "DR1TVB6",
    "vit_b12_mcg":          "DR1TVB12",
    "folate_mcg":           "DR1TFOLA",
    "vit_c_mg":             "DR1TVC",
    "vit_d_mcg":            "DR1TVD",
    "vit_e_mg":             "DR1TATOC",
    "iron_mg":              "DR1TIRON",
    "magnesium_mg":         "DR1TMAGN",
    "selenium_mcg":         "DR1TSELE",
    "zinc_mg":              "DR1TZINC",
    "caffeine_g":           "DR1TCAFF",   # NHANES reports mg -> caller converts to g
    "alcohol_g":            "DR1TALCO",
}


# ============================================================
# NHANES DR1IFF column mapping (per-food individual records)
# Used by Layer 2 nutrient density cache builder.
# ============================================================
NHANES_DR1IFF_COL_MAP = {
    "energy_kcal":          "DR1IKCAL",
    "carbohydrate_g":       "DR1ICARB",
    "protein_g":            "DR1IPROT",
    "fat_total_g":          "DR1ITFAT",
    "saturated_fat_g":      "DR1ISFAT",
    "mufa_g":               "DR1IMFAT",
    "pufa_g":               "DR1IPFAT",
    "cholesterol_mg":       "DR1ICHOL",
    "fiber_g":              "DR1IFIBE",
    "vit_a_mcg_rae":        "DR1IVARA",
    "vit_b1_thiamin_mg":    "DR1IVB1",
    "vit_b2_riboflavin_mg": "DR1IVB2",
    "vit_b3_niacin_mg":     "DR1INIAC",
    "vit_b6_mg":            "DR1IVB6",
    "vit_b12_mcg":          "DR1IVB12",
    "folate_mcg":           "DR1IFOLA",
    "vit_c_mg":             "DR1IVC",
    "vit_d_mcg":            "DR1IVD",
    "vit_e_mg":             "DR1IATOC",
    "iron_mg":              "DR1IIRON",
    "magnesium_mg":         "DR1IMAGN",
    "selenium_mcg":         "DR1ISELE",
    "zinc_mg":              "DR1IZINC",
    "caffeine_g":           "DR1ICAFF",
    "alcohol_g":            "DR1IALCO",
    "sodium_mg":            "DR1ISODI",     # Th17 driver, used by Layer 3
}


# Quantiles for Layer 1 DII percentile regression
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


# Demographics features for Layer 1 quantile regression
LAYER1_FEATURE_COLS = ["age", "sex", "bmi"]


# Minimum DII components a participant must contribute for inclusion in Layer 1 training
MIN_DII_COMPONENTS = 20


# ============================================================
# Trigger classification keywords
# ============================================================

# First digit of NHANES food code: 1 = dairy
DAIRY_FIRST_DIGITS = {1}
DAIRY_KEYWORDS = {
    "milk", "cheese", "yogurt", "butter", "cream", "ice cream",
    "cottage", "ricotta", "feta", "mozzarella", "cheddar",
    "whey", "casein", "kefir", "ghee",
}

# First digit of NHANES food code: 5 = grain products
GLUTEN_FIRST_DIGITS = {5}
GLUTEN_KEYWORDS = {
    "wheat", "flour", "bread", "pasta", "noodle", "cracker",
    "cereal", "oat", "barley", "rye", "bulgur", "couscous",
    "tortilla", "pizza", "cake", "cookie", "muffin", "pancake",
    "waffle", "pretzel", "crouton", "biscuit", "pastry",
    "spaghetti", "macaroni", "ravioli",
}
GLUTEN_FREE_KEYWORDS = {
    "rice", "corn", "quinoa", "millet", "buckwheat",
    "sorghum", "amaranth", "gluten free", "gluten-free",
}

NIGHTSHADE_KEYWORDS = {
    "tomato", "potato", "eggplant", "pepper", "bell pepper",
    "paprika", "cayenne", "chili", "chile", "goji",
    "tomatillo", "pimento", "pimiento",
}
NIGHTSHADE_EXCLUSIONS = {"sweet potato", "yam"}

UPF_KEYWORDS = {
    "cookie", "cake", "muffin", "pastry", "donut", "doughnut", "brownie",
    "twinkie", "pop tart", "pop-tart", "snack cake",
    "hot dog", "sausage", "bologna", "salami", "pepperoni", "bacon",
    "spam", "luncheon meat", "chicken nugget", "fish stick", "corn dog",
    "soda", "cola", "soft drink", "energy drink", "sports drink",
    "sweetened beverage", "fruit drink",
    "frozen meal", "frozen pizza", "frozen dinner", "instant noodle",
    "macaroni and cheese", "instant soup",
    "chip", "crisp", "cheez", "goldfish",
    "candy", "chocolate bar", "gummy", "marshmallow",
}


# ============================================================
# Glycemic index lookup table by food category keyword
# ============================================================
GI_BY_CATEGORY = {
    "white rice": 73, "rice": 73,
    "white bread": 75, "bread": 70, "toast": 70,
    "baked potato": 85, "mashed potato": 85, "potato": 78, "french fries": 75,
    "corn flakes": 81, "instant oat": 79,
    "soda": 65, "cola": 65,
    "cake": 70, "cookie": 65, "donut": 75,
    "brown rice": 68, "pasta": 50, "oat": 55, "oatmeal": 55,
    "banana": 51, "mango": 51, "pineapple": 59,
    "apple": 36, "pear": 38, "orange": 43, "grape": 53,
    "milk": 31, "yogurt": 35,
    "lentil": 32, "lentils": 32, "dal": 32, "bean": 30, "beans": 30,
    "chickpea": 28,
    "broccoli": 15, "spinach": 15, "lettuce": 15, "salad": 15,
    "avocado": 15, "tomato": 15, "cucumber": 15,
    "egg": 0, "eggs": 0,
    "chicken": 0, "beef": 0, "pork": 0, "fish": 0, "salmon": 0, "tuna": 0,
    "nut": 15, "almond": 15, "walnut": 15,
    "coffee": 0, "tea": 0, "water": 0,
    "beer": 25, "wine": 0,
}
DEFAULT_GI = 50


# ============================================================
# Feature ordering for Layer 3 daily vector
# ============================================================
CONTINUOUS_FEATURES = [
    "dii_score",
    "omega6_omega3_ratio",
    "glycemic_load",
    "sodium_mg",
    "alcohol_g",
    "overnight_fast_hours",
]
BOOLEAN_TRIGGERS = [
    "gluten_present",
    "dairy_present",
    "nightshade_present",
    "upf_present",
]


# ============================================================
# NHANES download config (for Layer 1 training)
# ============================================================
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

NHANES_FILES = {
    "P_DR1TOT.XPT": "Day-1 total nutrient intakes",
    "P_DR1IFF.XPT": "Day-1 individual foods",
    "P_DRXFCD.XPT": "Food code dictionary",
    "P_HSCRP.XPT":  "High-sensitivity CRP",
    "P_DEMO.XPT":   "Demographics",
    "P_BMX.XPT":    "Body measurements",
}
