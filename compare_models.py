"""
compare_models.py
=================
Side-by-side comparison of the BROKEN baseline vs. the FIXED pipeline, plus a
step-by-step decomposition that shows *which* fix recovered how much of the
inflated score.

Why this script exists
----------------------
The broken script reports a high R2. Taken alone, that looks better than the
fixed model's honest R2 ~ 0.74 - which is exactly the trap. This script makes
the comparison fair by:

  1. Reproducing the broken model exactly as written (poisoned label, future
     leak, LabelEncoder, random split) and reading its self-reported metrics.
  2. Reporting the fixed pipeline's honest, out-of-time metrics.
  3. Peeling the broken model's crutches ONE AT A TIME - fix the split, drop the
     leak, clean the label - so you can see the reported number collapse toward
     the honest one. That decomposition is the real lesson.

Outputs a console table and a self-contained `compare_models.html`.

Run:
    python compare_models.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Reuse the real fixed pipeline + constants so the comparison uses the actual
# production code, not a re-implementation of it.
from fixed_yogyank_training import (
    ARTIFACTS_DIR,
    DATA_PATH,
    FEATURES,
    TARGET,
    TIME_COL,
    TRAIN_MAX_YEAR,
    build_pipeline,
    load_data,
    time_based_split,
)

OUTPUT_HTML = os.path.join(ARTIFACTS_DIR, "compare_models.html")

# Broken-lineage config (verbatim from broken_yogyank_training.py)
POLICY_COL = "pm_kisan_status"
LEAK_COL = "defaulted_in_next_12_months"
BROKEN_FEATURES = [
    "land_area_acres",
    "crop_type",
    "pm_kisan_status",
    "historical_repayment_score",
    "defaulted_in_next_12_months",
]
BROKEN_PARAMS = dict(
    n_estimators=60, max_depth=4, learning_rate=0.1,
    random_state=42, n_jobs=1, tree_method="hist",
)

# Palette
TEAL = "#2F6F69"   # honest / good
LEAK = "#C0392B"   # inflated / danger
CLAY = "#B5733A"
INK = "#1B2420"
BORDER = "#D8DCD6"


def _label_encode(frame):
    """Broken-style encoding: a fresh LabelEncoder per non-numeric column."""
    out = frame.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = LabelEncoder().fit_transform(out[c].astype(str))
    return out


def evaluate_lineage(df, features, *, poison_label, split):
    """
    Train a broken-lineage XGBoost (LabelEncoder + raw XGB) under a chosen
    data-hygiene configuration and return (r2, mae) on the held-out set.

    Holding the model family constant and varying only `features` / `poison_label`
    / `split` is what makes the decomposition an apples-to-apples peel.
    """
    y = df[TARGET].copy()
    if poison_label:
        y.loc[df[POLICY_COL] == "No"] -= 150
    X = _label_encode(df[features])

    if split == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
    else:  # out-of-time
        mask = df[TIME_COL] <= TRAIN_MAX_YEAR
        X_tr, X_te, y_tr, y_te = X[mask], X[~mask], y[mask], y[~mask]

    model = XGBRegressor(**BROKEN_PARAMS)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return r2_score(y_te, preds), mean_absolute_error(y_te, preds)


def evaluate_fixed(df):
    """The real fixed pipeline: OHE + impute + XGB, out-of-time split, clean label."""
    X_tr, X_te, y_tr, y_te = time_based_split(df)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return r2_score(y_te, preds), mean_absolute_error(y_te, preds)


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    broken4 = [f for f in BROKEN_FEATURES if f != LEAK_COL]

    # Headline: each script as actually written.
    broken_r2, broken_mae = evaluate_lineage(
        df, BROKEN_FEATURES, poison_label=True, split="random"
    )
    fixed_r2, fixed_mae = evaluate_fixed(df)

    # Decomposition: peel the broken model's crutches one at a time.
    stages = [
        ("Reported\n(random split, leak in, poisoned label)",
         *evaluate_lineage(df, BROKEN_FEATURES, poison_label=True, split="random")),
        ("Fix the split\n(out-of-time)",
         *evaluate_lineage(df, BROKEN_FEATURES, poison_label=True, split="oot")),
        ("Drop the future leak\n(remove defaulted_in_next_12_months)",
         *evaluate_lineage(df, broken4, poison_label=True, split="oot")),
        ("Clean the label\n(no -150 injection)",
         *evaluate_lineage(df, broken4, poison_label=False, split="oot")),
        ("Add the missing features\n(13 features, still LabelEncoder)",
         *evaluate_lineage(df, FEATURES, poison_label=False, split="oot")),
        ("Fix encoding & model\n(OneHotEncoder + impute + tuned XGB)",
         fixed_r2, fixed_mae),
    ]

    # ---- Console report ----
    print("\n" + "=" * 64)
    print("HEADLINE  -  each script as written")
    print("=" * 64)
    print(f"{'model':<34}{'R2':>10}{'MAE':>12}")
    print("-" * 64)
    print(f"{'Broken (self-reported)':<34}{broken_r2:>10.4f}{broken_mae:>12.2f}")
    print(f"{'Fixed (honest, out-of-time)':<34}{fixed_r2:>10.4f}{fixed_mae:>12.2f}")

    print("\n" + "=" * 64)
    print("DECOMPOSITION  -  peeling the crutches one at a time")
    print("=" * 64)
    print(f"{'stage':<52}{'R2':>10}")
    print("-" * 64)
    for label, r2, _mae in stages:
        print(f"{label.replace(chr(10), ' '):<52}{r2:>10.4f}")
    print("=" * 64)
    print(f"Inflation explained by the random split alone: "
          f"{stages[0][1] - stages[1][1]:+.3f} R2")
    print(f"Total drop from reported to honest:            "
          f"{stages[0][1] - stages[-1][1]:+.3f} R2\n")

    # ---- Visuals ----
    labels = [s[0] for s in stages]
    r2s = [s[1] for s in stages]

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.30, 0.30, 0.40],
        subplot_titles=(
            "Headline R&#178;",
            "Headline MAE (points)",
            "R&#178; decomposition: reported &#8594; honest",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "waterfall"}]],
    )

    # Headline R2
    fig.add_trace(go.Bar(
        x=["Broken<br>(reported)", "Fixed<br>(honest)"],
        y=[broken_r2, fixed_r2], marker_color=[LEAK, TEAL],
        text=[f"{broken_r2:.3f}", f"{fixed_r2:.3f}"], textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    # Headline MAE (lower is better)
    fig.add_trace(go.Bar(
        x=["Broken<br>(reported)", "Fixed<br>(honest)"],
        y=[broken_mae, fixed_mae], marker_color=[LEAK, TEAL],
        text=[f"{broken_mae:.1f}", f"{fixed_mae:.1f}"], textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    # Waterfall: contribution of each fix to R2
    deltas = [r2s[0]] + [r2s[i] - r2s[i - 1] for i in range(1, len(r2s))]
    fig.add_trace(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(r2s) - 1),
        x=[l.replace("\n", "<br>") for l in labels],
        y=deltas,
        text=[f"{r2s[0]:.3f}"] + [f"{d:+.3f}" for d in deltas[1:]],
        textposition="outside",
        connector=dict(line=dict(color=BORDER)),
        decreasing=dict(marker=dict(color=LEAK)),
        increasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=INK)),
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        title="Broken vs. Fixed - the reported score was mostly leakage & a bad split",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK),
        height=520, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=90, b=120),
    )
    fig.update_yaxes(title_text="R&#178;", row=1, col=1)
    fig.update_yaxes(title_text="MAE (points)", row=1, col=2)
    fig.update_yaxes(title_text="R&#178; on held-out farmers", row=1, col=3)

    fig.write_html(OUTPUT_HTML)
    print(f"Saved visual comparison to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
