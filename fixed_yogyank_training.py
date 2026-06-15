"""
Yogyank Entitlement Score - Training Pipeline (clean rewrite)

Predicts `target_entitlement_score` (how much support a farmer is entitled to).

What this fixes vs. the original draft:
  - Label is NOT mutated. The previous `-= 150` rule poisoned the target and
    made the model memorize a business rule instead of learning. PM-Kisan is now
    a plain input feature; the model learns its effect from clean data.
  - `defaulted_in_next_12_months` is DROPPED. It is a future outcome, unknown at
    scoring time -> target leakage.
  - All preprocessing lives inside a single sklearn Pipeline (impute + encode),
    fit on TRAIN ONLY, and is persisted with the model so it can be reloaded.
  - Time-based split (train <= 2023, test = 2024) to simulate deployment.
  - Honest evaluation: a mean baseline plus MAE and R2 on the held-out year.
  - Feature importances exported for explainability.
"""

import joblib
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_PATH = "farmer_scoring_sample_yogyank_round1.csv"
MODEL_PATH = "entitlement_model.pkl"
TARGET = "target_entitlement_score"
TIME_COL = "application_year"
TRAIN_MAX_YEAR = 2023  # train on <= this year, test on the year(s) after

# Columns known at scoring time. Deliberately excluded:
#   farmer_id                     -> identifier, not predictive
#   defaulted_in_next_12_months   -> FUTURE outcome (leakage)
#   application_year              -> time index, used only for the split
#   target_entitlement_score      -> the label
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


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def time_based_split(df):
    """Train on the past, test on the most recent year - mirrors deployment."""
    train = df[df[TIME_COL] <= TRAIN_MAX_YEAR]
    test = df[df[TIME_COL] > TRAIN_MAX_YEAR]
    return (
        train[FEATURES],
        test[FEATURES],
        train[TARGET],
        test[TARGET],
    )


def build_pipeline():
    """Preprocessing + model in one object so encoding travels with the model."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def feature_importance_frame(pipeline):
    """Map XGBoost importances back to human-readable feature names."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train():
    df = load_data()
    X_train, X_test, y_train, y_test = time_based_split(df)
    print(f"Train rows: {len(X_train)} (<= {TRAIN_MAX_YEAR})  |  Test rows: {len(X_test)} (> {TRAIN_MAX_YEAR})")

    # Baseline: always predict the training mean. Any real model must beat this.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_preds = baseline.predict(X_test)
    print("\n--- Baseline (predict mean) ---")
    print(f"MAE: {mean_absolute_error(y_test, base_preds):.2f}   R2: {r2_score(y_test, base_preds):.4f}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("\n--- XGBoost pipeline ---")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}   R2: {r2_score(y_test, preds):.4f}")

    importances = feature_importance_frame(pipeline)
    print("\n--- Top feature importances ---")
    print(importances.head(10).to_string(index=False))

    fig = px.bar(
        importances.head(15).iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Entitlement model - feature importances",
    )
    fig.write_html("feature_importances.html")
    print("\nSaved importance chart to feature_importances.html")

    # Persist the WHOLE pipeline + metadata so it can be reloaded and rebuilt.
    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "train_max_year": TRAIN_MAX_YEAR,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    train()
