# Yogyank Entitlement Score: Audit Memo

**Scope:** Review and remediation of `broken_yogyank_training.py`, the baseline training
script for the farmer entitlement score.
**Dataset:** `farmer_scoring_sample_yogyank_round1.csv` (5,000 rows, 17 columns,
application years 2022–2024, target `target_entitlement_score` ∈ [421, 980]).
**Outcome:** original script replaced by `fixed_yogyank_training.py` (original kept
untouched for reference).

---

## 1. What was dangerous in the original script

The original reported a high validation R² and was annotated *"Model is performing well…
Ready for production."* That number is **not trustworthy**, for several compounding
reasons. They are listed upstream-first, because the worst problems sit before training
ever starts.

### 1.1 The label was poisoned with a business rule (target leakage)
```python
df.loc[df["pm_kisan_status"] == "No", "target_entitlement_score"] -= 150
```
A deterministic policy (`pm_kisan_status == "No" → −150`) was **baked into the label**, and
`pm_kisan_status` was then **also used as a feature**. The model's cheapest path to a low
error is to memorize that rule rather than learn anything. This is circular: the model is
graded on its ability to reproduce an answer it was handed. It inflates R² on its own.

The data confirms the rule was *invented*, not *recovered*: the real mean gap between
PM-Kisan Yes vs. No in the raw data is far smaller than the 150-point rule the script
subtracts.

### 1.2 A feature from the future was used (forward leakage)
`defaulted_in_next_12_months` is, by its own name, an outcome measured **after** the
scoring date. It cannot exist when a farmer is actually scored, yet it is the single strongest
correlate of the target (**|corr| ≈ 0.48**). Training on it lets the model "see the future,"
which inflates validation accuracy and **guarantees the model cannot be reproduced at
inference time** (the column simply isn't available).

### 1.3 The validation split did not simulate deployment
`train_test_split(..., shuffle=True)` randomly mixed application years 2022–2024. In
production the model scores **future** applicants, so the correct test is out-of-time:
train on the past, score the most recent year. A random split lets the model train and test
on the same time period, which can leak temporal structure. This is a genuine
**methodological** error and must be fixed regardless. That said, when we measured it
directly on this dataset (see the decomposition in §2), the split contributed almost none of the inflation
(≈ 0.01 R²). The label poisoning and missing legitimate features were the real story. The
lesson: a wrong split is still wrong even when it happens not to bite on a given sample.

### 1.4 Preprocessing was incorrect and unreproducible
```python
encoder = LabelEncoder()
X["crop_type"]      = encoder.fit_transform(X["crop_type"])
X["pm_kisan_status"] = encoder.fit_transform(X["pm_kisan_status"])
```
- A **single** `LabelEncoder` was re-fit on the second column, discarding the first
  mapping, and then **thrown away** and never saved. The persisted model is therefore
  **unusable**: nothing can turn a raw farmer record into the integers the model expects.
- `LabelEncoder` is meant for **targets**, not features. On nominal categories it invents a
  fake ordinal ranking (e.g. Cotton=1 < Rice=4) and **crashes on any unseen category** at
  inference time.
- It was fit on the **full dataset** (fit-on-train-only discipline violated).

### 1.5 No baseline, single metric
R² was reported with no reference point. Without a baseline (e.g. "always predict the
mean") a number in isolation says nothing about real skill, and a single metric hides error
magnitude in the units that matter (points).

### 1.6 Persistence and explainability gaps
Only the model object was saved, with no encoder, no feature list, and no metadata. And there was
no feature-importance or reason-code output, so there is **no honest way to tell a farmer
why they received a given score**. Even if importances were printed, the top drivers
would have been the leaked default flag and the injected policy rule.

**Net effect:** the headline R² rested mostly on a poisoned label and a future-leak feature,
propped up further by a weak, under-specified feature set. Strip the crutches and the
broken approach's *own* honest skill is low; §2 quantifies exactly where the inflation came
from.

---

## 2. What was changed

All fixes live in `fixed_yogyank_training.py`. Honest, out-of-time results:

| Model | MAE | R² |
|---|---|---|
| Baseline (predict mean) | 87.35 | ~0.00 |
| **Fixed XGBoost pipeline** | **43.59** | **0.7413** |

The model roughly **halves the baseline error** on a held-out future year, a genuine,
defensible result rather than an inflated one.

**Where the inflation actually came from.** `compare_models.py` peels the broken model's
crutches one at a time, all on the *same* out-of-time 2024 holdout, holding the model family
constant until the final two rows:

| Stage | R² | Δ vs. previous |
|---|---|---|
| Reported (random split, leak in, poisoned label) | 0.689 | baseline |
| Fix the split (out-of-time) | 0.680 | −0.009 |
| Drop the future leak (`defaulted_in_next_12_months`) | 0.616 | −0.064 |
| Clean the label (no −150 injection) | 0.376 | **−0.241** |
| Add the missing features (13 features, still LabelEncoder) | 0.713 | **+0.337** |
| Fix encoding & model (OneHotEncoder + impute + tuned XGB) | **0.741** | +0.029 |

Reading this honestly:
- **The poisoned label was the single biggest inflator** (−0.241 R²): it let the model
  reproduce the −150 rule from `pm_kisan_status` instead of learning.
- **The future leak was the second** (−0.064 R²).
- **The bad split barely mattered here** (−0.009 R²): wrong in principle, but not the
  source of the inflated number on this data.
- **The broken model's own clean features are genuinely weak** (R² 0.376). The recovery to
  0.741 comes almost entirely from **adding legitimate, scoring-time features** (+0.337),
  with proper encoding and light tuning adding a final +0.029. In other words, the fix is
  not "same model, leak removed"; it is "honest feature engineering on a clean label."

### 2.1 Leakage control
- **Dropped `defaulted_in_next_12_months`** entirely, so the forward-looking outcome is gone.
- **Stopped poisoning the label** by removing the `-= 150` mutation; the target is now the
  raw entitlement score.
- **Curated the feature set to scoring-time-available fields only**: 13 features
  (6 numeric, 7 categorical), explicitly excluding the identifier, the leak column, the
  time index, and the label.

### 2.2 Validation
- **Time-based (out-of-time) split**: train on `application_year ≤ 2023` (3,606 rows),
  test on 2024 (1,394 rows). This mirrors how the model is actually used (score the next
  cohort) instead of peeking at the same period it trains on.

### 2.3 Preprocessing
- All preprocessing moved **inside a single `sklearn` `Pipeline`** (`ColumnTransformer` →
  `XGBRegressor`), so encoding is fit **on train only** and travels with the model.
- Replaced `LabelEncoder` with **`OneHotEncoder(handle_unknown="ignore")`**, which avoids
  inventing fake order on nominal categories and **does not crash on unseen categories** at
  inference.
- Added **median imputation** for the numeric columns with missing values
  (`rainfall_deviation_pct` and `ndvi_score`, ~15% missing each).

### 2.4 Reproducibility
- **The whole pipeline is persisted**, plus a metadata sidecar (feature list, target name,
  `train_max_year`) in a single `joblib` bundle, so anyone can reload it and score a raw
  record end-to-end.
- Deterministic configuration retained (`random_state=42`, `n_jobs=1`).
- `.pkl` artifacts are **gitignored** (reproducible from the script; binary, security-
  sensitive to unpickle); the recipe is versioned, not the cake.

### 2.5 Model / policy separation
- `pm_kisan_status` is now a **plain input feature** the model learns from clean data, not
  a hardcoded label edit. The PM-Kisan business adjustment, if still required, belongs in a
  **transparent, auditable post-prediction layer**, kept separate from the learned model so
  the two can be reasoned about and changed independently.

### 2.6 Explainability
- Added an **honest baseline (`DummyRegressor`)** and report both **MAE and R²**, so the
  number is anchored and expressed in points.
- **Feature importances** are printed and exported to `feature_importances.html`. The top
  drivers are now legitimate, scoring-time signals (`liability_ratio_pct`,
  `historical_repayment_score`, `land_area_acres`, `pm_kisan_status`, `irrigation_type`,
  and `annual_income_inr`), none of which leaks. There is finally a defensible answer to
  "why did this farmer get this score?"

---

## 3. Limitations that remain

The fixed pipeline is sound, but it is a **baseline**, not a production-validated system.

### 3.1 Things I would not trust yet
- **The "as-of-scoring-date" status of several features is unverified.** Columns like
  `historical_repayment_score`, `liability_ratio_pct`, `annual_income_inr`,
  `ndvi_score`, and `rainfall_deviation_pct` are *plausibly* available at scoring time, but
  some could be season aggregates or later reconciliations that secretly bleed past the
  scoring date, i.e. **subtler leakage than the obvious default flag**. Each needs its
  definition confirmed against the real data dictionary before I'd trust the score.
- **The data is synthetic.** R² 0.74 is honest *for this dataset*, but says nothing about
  real-world performance, label quality, or whether `target_entitlement_score` itself was
  generated by a known formula (which would make the whole exercise circular).
- **A single out-of-time fold (2024 only).** One held-out year can be lucky or unlucky; it
  is not yet evidence of stable performance across cohorts.
- **No fairness / bias review.** For a farmer-facing entitlement decision, performance
  across `district`, `land_ownership`, crop type, and income bands must be checked before
  trusting the score operationally.

### 3.2 Things I would improve with time
- **Rolling / expanding-window time-series cross-validation** instead of one fold, to get a
  stable performance range and confidence intervals.
- **A formal feature contract**, codifying each feature's source and as-of-date guarantee,
  enforced in code so leakage cannot creep back in.
- **SHAP-based per-applicant reason codes**, not just global importances, so each farmer
  gets a specific, individualized explanation.
- **The explicit post-prediction policy layer** (PM-Kisan and any other rules) with its own
  tests, plus light hyperparameter tuning, missing-data and fairness monitoring, and a thin
  inference script that loads the bundle and scores raw records.

---

*Original script (`broken_yogyank_training.py`) retained unchanged for comparison.
Fixed pipeline: `fixed_yogyank_training.py`. Data audit visual: `build_dashboard.py` →
`yogyank_data_explorer.html`.*
