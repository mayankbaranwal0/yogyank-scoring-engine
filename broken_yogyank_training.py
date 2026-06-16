"""
Yogyank Entitlement Score - Baseline Training Script (Draft v1)
Author: Junior Data Scientist
Notes: Model is performing well. Validation score looks good. Ready for production.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib

# This (flawed) baseline model is written here. See AUDIT_MEMO.md for why it
# is not trustworthy. Output goes to artifacts/ alongside the other generated files.
MODEL_PATH = os.path.join("artifacts", "xgboost_baseline.pkl")


def load_and_prep_data(path="farmer_scoring_sample_yogyank.csv"):
    return pd.read_csv(path)


def train_model():
    df = load_and_prep_data()

    print("Applying PM Kisan business policy...")
    df.loc[df["pm_kisan_status"] == "No", "target_entitlement_score"] -= 150

    features = [
        "land_area_acres",
        "crop_type",
        "pm_kisan_status",
        "historical_repayment_score",
        "defaulted_in_next_12_months",
    ]

    X = df[features].copy()
    y = df["target_entitlement_score"]

    print("Encoding categorical variables...")
    encoder = LabelEncoder()
    X["crop_type"] = encoder.fit_transform(X["crop_type"])
    X["pm_kisan_status"] = encoder.fit_transform(X["pm_kisan_status"])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"Validation R2 Score: {score:.4f} (Wow!)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
