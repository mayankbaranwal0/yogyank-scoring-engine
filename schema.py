"""
Single source of truth for the dataset's schema and the as-of-date contract.

Both the trainer (`fixed_yogyank_training.py`) and the data explorer
(`build_yogyank_data_explorer.py`) import from here, so the feature lists and the
leakage contract cannot drift apart. The contract is also exported into
`artifacts/model_metadata.json` at training time.
"""

# Column roles
TARGET = "target_entitlement_score"
TIME_COL = "application_year"
LEAK_COL = "defaulted_in_next_12_months"
ID_COL = "farmer_id"
POLICY_COL = "pm_kisan_status"

# Feature groups (the only columns the model is allowed to see)
NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
]
CATEGORICAL_FEATURES = [
    "district",
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Per-column ROLE and an availability ASSUMPTION (the as-of-date contract).
# availability: "yes"   -> known at/before scoring
#               "no"    -> future outcome (leak)
#               "assume"-> plausibly fine but MUST be verified as as-of-scoring-date
#               "na"    -> not a feature
COLUMN_META = {
    "farmer_id":                   ("identifier", "na",     "Row key. Drop from features."),
    "application_year":            ("temporal",   "yes",    "Application timestamp; defines the as-of date."),
    "district":                    ("feature",    "yes",    "Location is known at application."),
    "land_area_acres":             ("feature",    "yes",    "Declared at application."),
    "crop_type":                   ("feature",    "assume", "Assume declared/planted at application."),
    "pm_kisan_status":             ("policy",     "yes",    "Enrolment status known at application."),
    "historical_repayment_score":  ("feature",    "assume", "Only valid if built from history BEFORE the scoring date."),
    "irrigation_type":             ("feature",    "yes",    "Static attribute of the holding."),
    "land_ownership":              ("feature",    "yes",    "Known at application."),
    "soil_type":                   ("feature",    "yes",    "Static attribute of the holding."),
    "sales_channel":               ("feature",    "assume", "May be measured over the season -> could bleed past scoring date."),
    "annual_income_inr":           ("feature",    "assume", "Self-reported at application = OK; reconciled later = leak."),
    "liability_ratio_pct":         ("feature",    "assume", "Must be an as-of-scoring-date snapshot."),
    "rainfall_deviation_pct":      ("feature",    "assume", "Season aggregate may extend past the scoring date."),
    "ndvi_score":                  ("feature",    "assume", "Satellite reading: as-of date OK; season composite = leak."),
    "defaulted_in_next_12_months": ("leak",       "no",     "Outcome over [T, T+12mo] -> UNKNOWABLE at scoring time."),
    "target_entitlement_score":    ("target",     "na",     "What we estimate (use UNMODIFIED; apply policy afterwards)."),
}

# Human-readable labels for the availability codes (used in metadata + dashboard).
AVAILABILITY_LABEL = {
    "yes":    "available",
    "no":     "leak_drop",
    "assume": "verify_as_of",
    "na":     "not_a_feature",
}
