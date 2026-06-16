# LLM Notes: Prompts & Replies

This file documents how the project was built and reasoned about as a sequence
of **prompts** and the **replies / outputs** that moved the work forward.

**Tool used:** [Claude Code](https://claude.com/claude-code), Anthropic's
command-line coding agent, running the Claude Opus model. All scripts, the audit
memo, and this file were produced through an interactive session with it, with
the decisions below driven by me.

---

## Part 1: Building the project

### 1. Audit the pipeline upstream-first

> **Prompt:** Walk the pipeline in the order data flows and answer one question
> at each stage: problem framing, target construction, feature selection,
> preprocessing/encoding, split/validation, training, evaluation, persistence,
> explainability, and process/meta. Then let's start fixing the product.

**Reply:** Traced the data from CSV to model. Found the fatal errors live
**upstream** of training: a label poisoned by a `-= 150` policy rule, a
future-outcome feature (`defaulted_in_next_12_months`) used as input, a random
split instead of out-of-time, a thrown-away `LabelEncoder`, no baseline, and no
saved preprocessing or explainability. The model's hyperparameters were fine.
Decisions taken: keep `target_entitlement_score` as the label (de-poisoned),
treat `pm_kisan_status` as a feature only, and do a full rewrite.

### 2. Write the corrected training pipeline

> **Prompt:** Rewrite the trainer into a clean, leakage-free, reproducible
> pipeline based on those decisions.

**Reply:** Produced `fixed_yogyank_training.py`: drops the leak, de-poisons the
label, encodes inside a single `sklearn` `Pipeline` (`OneHotEncoder` + median
impute) fit on train only, uses a time-based split (train <= 2023, test 2024),
adds a `DummyRegressor` baseline and reports both MAE and R2, persists the whole
pipeline plus metadata, and exports feature importances. Honest result: R2 0.741
/ MAE 43.6, roughly half the baseline error.

### 3. Build the offline HTML data explorer

> **Prompt:** Build a single-file Python script that generates a self-contained,
> offline-viewable HTML "data explorer" for a CSV dataset, designed to audit it
> for ML data-quality traps before any model is trusted.
>
> Data: `farmer_scoring_sample_yogyank.csv`, farmer entitlement-scoring data.
> Target is `target_entitlement_score`. `defaulted_in_next_12_months` is a
> forward-looking outcome (unknowable at scoring time = leakage).
> `application_year` (2022-2024) is the time index. `pm_kisan_status` is a policy
> field.
>
> Output: one `.html` file with the full Plotly library embedded (no server, no
> internet, no Python needed to view it). Accept optional CLI args for input CSV
> and output path; default the output into an `artifacts/` folder and create it
> if missing.
>
> Build these panels, each answering a real question (not decoration):
> 1. Schema / as-of-date contract: a table tagging every column with role
>    (feature/target/leak/policy/identifier/temporal) and availability (available
>    / LEAK: drop / verify as-of / n/a), plus dtype, uniqueness, and % missing.
> 2. Target distribution: histogram with the mean marked.
> 3. Signal vs. leakage: bar chart of each feature's absolute correlation with
>    the target, with the leak column highlighted to show the strongest signal is
>    the one you can't use.
> 4. Leakage spotlight: target distribution split by the leak flag, showing how
>    much it "knows" the outcome.
> 5. R2 decomposition: a directional waterfall reproducing how the reported score
>    collapses as you fix the split, drop the leak, and clean the label (degrade
>    gracefully to a text note if xgboost/scikit-learn are missing).
> 6. Missingness: % missing per column, plus a co-missingness check (do two gappy
>    columns vanish together?).
> 7. Temporal: rows per year and target distribution per year, highlighting the
>    held-out test year.
> 8. Policy injection: compare the real data gap for the policy field against the
>    magnitude the draft script injects into the label.
> 9. Interactive explorers: dropdown-driven per-column histograms (numeric) and
>    count bars (categorical).
>
> Design: a calm editorial theme; reserve red exclusively to mean "leakage /
> disqualified" as a visual through-line wherever the leak column appears. Use
> numeric HTML entities (not named ones) for any special glyphs so Plotly
> renders them. Make output deterministic and provenance-stamped (source filename
> + timestamp in the footer).

**Reply:** Produced `build_yogyank_data_explorer.py`, which generates
`artifacts/yogyank_data_explorer.html`.

### 4. Compare broken vs. fixed and decompose the inflation

> **Prompt:** Write a small `compare_models.py` that puts the broken and fixed
> metrics side by side, for thorough comparison and visual analysis.

**Reply:** Produced `compare_models.py`: it reproduces the broken model exactly,
then peels its crutches one at a time on the same 2024 holdout (fix split, drop
leak, clean label, add real features, fix encoding), printing a table and a
waterfall to `artifacts/compare_models.html`. The measured decomposition
surprised us: the poisoned label was the biggest inflator (-0.241 R2), the leak
second (-0.064), and the bad split barely mattered (-0.009). This corrected an
earlier assumption that the split was the main culprit.

### 5. Write the audit memo

> **Prompt:** Create `AUDIT_MEMO.md` covering what was dangerous in the original
> script (and why the reported result is untrustworthy), what changed (leakage
> control, validation, preprocessing, reproducibility, model/policy separation,
> explainability), and what limitations remain.

**Reply:** Produced `AUDIT_MEMO.md` with the measured decomposition table, and
corrected its own split-inflation claim once the numbers from `compare_models.py`
contradicted it.

### 6. Organize generated outputs

> **Prompt:** Put all generated artifacts in an `artifacts/` folder; track them
> rather than ignoring them.

**Reply:** Routed every script's output into `artifacts/` (each creates the
folder if missing), removed `*.pkl` from `.gitignore`, and documented each file's
source in the README. The broken baseline writes
`artifacts/xgboost_baseline.pkl`; the fixed pipeline writes
`artifacts/entitlement_model.pkl`.

---

## Part 1b: Suggestions accepted and rejected

The assistant proposed many options; keeping control meant accepting some and
overriding others. Concrete examples:

**Accepted:**
- **Keep the entitlement label, PM-Kisan as a feature.** Accepted the framing of
  these as explicit decisions rather than letting the assistant guess.
- **Out-of-time validation over a random split.** Accepted training on 2022-2023
  and testing on 2024 to mimic scoring a future cohort, instead of the random
  shuffle that lets the model peek at its own test period.
- **An honest baseline plus MAE, not R2 alone.** Accepted adding a
  `DummyRegressor` reference and reporting error in points, so the headline number
  is anchored and meaningful rather than impressive in isolation.
- **Persist the whole pipeline, not just the model.** Accepted saving the encoder,
  imputer, model, and feature metadata as one bundle, so a raw record can be
  scored end-to-end (the original threw the encoder away).
- **Splitting the comparison waterfall** into two isolated steps ("add features"
  vs. "fix encoding") so each effect is measured separately.

**Rejected or overridden:**
- **Rejected keeping the strongest feature.** `defaulted_in_next_12_months` had
  the highest correlation with the target (-0.48), and a naive "keep your best
  predictors" instinct would retain it. Rejected it purely on timing: a value
  that cannot exist at scoring time is worthless no matter how predictive it
  looks in training.
- **Rejected letting the model carry business policy.** Encoding the PM-Kisan
  rule into the training label (and feeding the field as a feature) was rejected.
  A governance lever belongs in a transparent post-model layer with a number the
  product owner sets, never memorized by the model from a poisoned target.
- **Rejected ordinal label-encoding for nominal categories.** `LabelEncoder`
  invents a fake order (Cotton < Rice) and crashes on unseen categories at
  inference. Rejected it in favour of `OneHotEncoder(handle_unknown="ignore")`
  inside the pipeline.
- **Rejected tuning the model to rescue the score.** The instinct to reach for
  XGBoost hyperparameters was declined; the original settings were fine and the
  real damage was upstream (label, features, split). Effort went to data hygiene,
  not model knobs.

**A correction worth noting:** the assistant's first draft of the audit memo
claimed the bad split caused much of the score inflation. Forcing it to *measure*
this (via `compare_models.py`) proved the opposite (the split mattered least, the
poisoned label most), and the memo was corrected to match the evidence.

---

## Part 2: Key lessons (the understanding prompts)

### Q1. Why drop `defaulted_in_next_12_months` when it was the strongest signal (correlation -0.48)?

**Reply:** A feature is disqualified on **timing, not effect size**. It is an
outcome measured *after* the scoring date, so it cannot exist when you actually
score a farmer. This is the **as-of-date contract**: for every feature, ask "is
this knowable at or before the moment I score?" The question is never whether a
feature correlates, it is whether it exists yet. This is the single most
transferable habit, because leakage is the most common reason a model looks great
in validation and fails in production.

### Q2. Why must the validation split simulate deployment, and how much of the inflated score did the split actually cause?

**Reply:** The model scores future applicants, so validation must mimic that:
train on the past (2022-2023), score the latest year (2024). The original random
shuffle let the model peek at the same period it was tested on, which is a genuine
methodological error. But measured on this data, the split contributed almost
nothing (-0.009 R2); the **poisoned label** was the dominant inflator (-0.241).
Lesson: fix the split because it is wrong in principle, but *measure* each fix's
effect rather than assuming. (A confounded waterfall that cleans the label inside
the "fix split" step will wrongly blame the split. Isolate one variable at a
time.)

### Q3. Why can a factor never be both a model feature and a policy rule?

**Reply:** Mixing them caused two failures in the original `target -= 150` rule.
First, a deterministic rule was injected into a learned target while
`pm_kisan_status` was also a feature, so the model just memorized the answer key.
Second, the magnitude was invented: the real Yes/No gap in the data is about 34
points, not 150. The principle: a factor is **either a feature or a policy lever,
never both**. Statistical effects belong in the model; governance choices belong
in a transparent post-model layer with a number the product owner supplies.

### Q4. How should you treat a validation result that looks great?

**Reply:** With suspicion. The original's "(Wow!)" and "ready for production"
were the tells. Default stance: "this number is probably wrong until I understand
why it is right." Two habits follow: **quantify the damage** with a decomposition
rather than asserting it, and **look upstream first**, because the rot is almost
always in the label, features, or split, not in the model hyperparameters where
the eye instinctively goes.

### Q5. What makes a model an actual deliverable?

**Reply:** A trained model is not a deliverable; a **rebuildable artifact** is.
The original saved only the `.pkl` while throwing away the encoder, making it
unusable at inference. Reproducibility means persisting the **whole pipeline**
(impute + encode + model) plus the feature list, target, and split metadata, so a
raw record can be scored end-to-end. The same discipline extends to repo hygiene:
keep generated outputs in a regenerable `artifacts/` folder so git tracks the
recipe, not the cake.

---

## Part 3: Personally erified by me

Each point below was checked against `fixed_yogyank_training.py`.

- **Leakage reasoning.** `FEATURES`, the only columns the model
  receives (via `train[FEATURES]` / `test[FEATURES]`), is 13 columns and excludes
  `defaulted_in_next_12_months`, `farmer_id`, `application_year`, and
  `target_entitlement_score`. A direct check confirmed none of those four reach
  the model.

- **Validation split.** `time_based_split` sends
  `application_year <= 2023` to train and `> 2023` to test. The two sets share no
  year (train = {2022, 2023}, test = {2024}, empty overlap) and split 3,606 vs
  1,394 rows, with no shuffle to mix periods.

- **Preprocessing boundary.** `SimpleImputer` and `OneHotEncoder` live
  inside the `ColumnTransformer` / `Pipeline`, and `pipeline.fit` is called only
  on `X_train`. The test set is never seen during fitting, so no test statistics
  enter imputation or encoding.

- **Saved artifacts.** The bundle holds exactly
  `{pipeline, features, target, train_max_year}`; reloading
  `artifacts/entitlement_model.pkl` and calling `pipeline.predict(record[features])`
  on a raw row returns an in-range score with no manual preprocessing.

- **Reason-code logic.** `feature_importance_frame` maps importances
  back to readable names, and the top global drivers are legitimate scoring-time
  signals (liability ratio, repayment history, land area, PM-Kisan), with no
  future-leak or injected-policy artifact among them. (Explanations are global;
  per-applicant SHAP is not part of this build.)

- **Run output.** All four scripts run to completion; the fixed pipeline
  reproduces R2 0.7413 / MAE 43.59 against a mean baseline of MAE 87.35, and
  repeated runs gave identical numbers (fixed `random_state=42`, `n_jobs=1`).

---

## The one to carry beyond this project

The **as-of-date contract** (Q1). Interrogating every feature's timing prevents
the most common and most expensive failure mode: a model that validates
beautifully and then collapses in production.
