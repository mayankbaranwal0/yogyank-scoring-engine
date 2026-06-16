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

import datetime as dt
import hashlib
import json
import os
import platform

import joblib
import pandas as pd
import plotly.express as px
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Schema and the as-of-date contract live in one place, shared with the explorer.
from schema import (
    AVAILABILITY_LABEL,
    CATEGORICAL_FEATURES,
    COLUMN_META,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TIME_COL,
)

DATA_PATH = "farmer_scoring_sample_yogyank.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "entitlement_model.pkl")
IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, "feature_importances.html")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
TRAIN_MAX_YEAR = 2023  # train on <= this year, test on the year(s) after


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


def _sha256(path):
    """Hash the data file so the metadata records exactly what trained the model."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_contract():
    """The as-of-date contract as plain dicts, ready for JSON."""
    return {
        col: {"role": role, "availability": AVAILABILITY_LABEL[avail], "note": note}
        for col, (role, avail, note) in COLUMN_META.items()
    }


def write_metadata(df, X_train, X_test, metrics):
    """Self-describing sidecar: provenance, schema/contract, versions, validation."""
    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": "XGBoost entitlement-score regressor",
        "data": {
            "file": DATA_PATH,
            "n_rows": int(len(df)),
            "sha256": _sha256(DATA_PATH),
        },
        "target": TARGET,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all": FEATURES,
        },
        "split": {
            "strategy": "out-of-time",
            "time_column": TIME_COL,
            "train_max_year": TRAIN_MAX_YEAR,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "schema_contract": schema_contract(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "validation": metrics,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


def train():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_data()
    X_train, X_test, y_train, y_test = time_based_split(df)
    print(f"Train rows: {len(X_train)} (<= {TRAIN_MAX_YEAR})  |  Test rows: {len(X_test)} (> {TRAIN_MAX_YEAR})")

    # Baseline: always predict the training mean. Any real model must beat this.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_preds = baseline.predict(X_test)
    base_mae = mean_absolute_error(y_test, base_preds)
    base_r2 = r2_score(y_test, base_preds)
    print("\n--- Baseline (predict mean) ---")
    print(f"MAE: {base_mae:.2f}   R2: {base_r2:.4f}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    model_r2 = r2_score(y_test, preds)
    print("\n--- XGBoost pipeline ---")
    print(f"MAE: {model_mae:.2f}   R2: {model_r2:.4f}")

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
    fig.write_html(IMPORTANCE_PATH)
    print(f"\nSaved importance chart to {IMPORTANCE_PATH}")

    # Persist the WHOLE pipeline + metadata so it can be reloaded and rebuilt.
    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "train_max_year": TRAIN_MAX_YEAR,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")

    metrics = {
        "metric_note": "out-of-time holdout (test year > train_max_year)",
        "baseline_mean": {"mae": round(float(base_mae), 4), "r2": round(float(base_r2), 4)},
        "model": {"mae": round(float(model_mae), 4), "r2": round(float(model_r2), 4)},
    }
    write_metadata(df, X_train, X_test, metrics)


if __name__ == "__main__":
    train()
