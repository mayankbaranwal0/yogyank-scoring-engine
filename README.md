# Yogyank Scoring Engine

A **bank-agnostic farmer entitlement-scoring model:** a
continuous, explainable score that estimates a farmer's entitlement from
application-time information, with policy applied transparently *afterwards*
rather than baked into the model.

This repository is, at heart, a **case study in doing that scoring correctly**.
It pairs a deliberately flawed baseline trainer with a data-audit tool that
exposes the traps (data leakage, a misleading validation split, and policy
injected into the label) so they can be understood and fixed before any model
is trusted.

---

## What's in here

| File | Role |
|------|------|
| `farmer_scoring_sample_yogyank_round1.csv` | Synthetic farmer dataset (~features + target + a forward-looking outcome). |
| `broken_yogyank_training.py` | A **deliberately flawed** baseline XGBoost trainer (the "looks great, ship it" draft). Demonstrates the mistakes. |
| `build_dashboard.py` | Builds the **HTML Data Explorer**, a self-contained, offline data audit. |
| `artifacts/` | Generated outputs (model `.pkl` and HTML reports). Created automatically when you run the scripts. |
| `artifacts/yogyank_data_explorer.html` | The report **you generate** by running `build_dashboard.py` (the default output location). This is the preferred way to view it. |
| `yogyank_data_explorer_example.html` | A prebuilt example report, as a fallback for when you can't generate your own. |
| `requirements.txt` | Python dependencies. |

The dataset's target is `target_entitlement_score`. The column
`defaulted_in_next_12_months` is a **forward-looking outcome** that cannot be
known at scoring time. Using it as a feature is leakage, and the explorer marks
it in red everywhere it appears.

---

## Project setup

You need Python 3. From the project folder, install the dependencies:

```powershell
pip install -r requirements.txt
```

> Using a virtual environment is recommended so the pinned versions don't clash
> with other projects:
>
> ```powershell
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> ```

---

## The baseline trainer (`broken_yogyank_training.py`)

This is the *starting point*, not the goal. Running it trains an XGBoost
regressor and reports a suspiciously high R²:

```powershell
python broken_yogyank_training.py
```

It writes a model to `xgboost_baseline.pkl` and prints a glowing validation
score, which is exactly the problem. The script:

- uses `defaulted_in_next_12_months` (a future outcome) as a feature → **leakage**,
- splits the data **randomly** instead of out-of-time, and
- subtracts 150 from the label for non-PM-Kisan farmers → **policy baked into the target**.

The Data Explorer below quantifies how much of that "great" score is illusion.

---

## The HTML Data Explorer (`build_dashboard.py`)

A single, self-contained, **offline-viewable** HTML report that audits the data
before you trust it: schema/leakage contract, target shape, signal-vs-leakage,
missingness, temporal drift, and the policy-injection illustration. The full
Plotly library is embedded in the file, so no server or internet is needed to view it.

**The preferred way is to generate it yourself.** Run `build_dashboard.py` to
build the report from the CSV, then open the result. A prebuilt
`yogyank_data_explorer_example.html` is included only as a **fallback** for when
you can't generate your own.

### Generate it (preferred)

```powershell
python build_dashboard.py
```

This reads the default CSV and writes `artifacts/yogyank_data_explorer.html`
(the `artifacts/` folder is created automatically). You can also pass a custom
input CSV and/or output filename:

```powershell
python build_dashboard.py path/to/data.csv out.html
```

> The "R² collapse" waterfall panel needs `xgboost` + `scikit-learn` (already in
> `requirements.txt`). If they were missing, that one panel degrades to a short
> note instead of crashing.

### Open it

The report is a single `.html` file you can open in any browser. Open the report
you just generated:

```powershell
# Your own generated report, from the project folder
start artifacts\yogyank_data_explorer.html

# Or with a full path
start D:\trynew\yogyank-scoring-engine\artifacts\yogyank_data_explorer.html
```

Fallback: if you couldn't generate your own, open the prebuilt example instead:

```powershell
start yogyank_data_explorer_example.html
```

Open in a specific browser instead of the default:

```powershell
Start-Process chrome  .\artifacts\yogyank_data_explorer.html   # Chrome
Start-Process msedge  .\artifacts\yogyank_data_explorer.html   # Edge
Start-Process firefox .\artifacts\yogyank_data_explorer.html   # Firefox
```

You can also just **double-click** the file in File Explorer.

### Generate and open in one line

```powershell
python build_dashboard.py && start artifacts\yogyank_data_explorer.html
```

### What the report covers

| # | Section | Question it answers |
|---|---------|---------------------|
| 01 | Schema & as-of-date contract | What's in the table, and which fields are we allowed to use? |
| 02 | Target | What are we predicting? |
| 03 | Signal vs. leakage | Is there real signal, and where is the trap? |
| 04 | Gaps | Where is the data thin or absent? |
| 05 | Temporal | Can we validate the future (out-of-time split)? |
| 06 | Policy | Model vs. injected policy rule |
| 07 | Explore | Interactive per-column histograms / counts |

---

## Notes

The data is **synthetic** and the R² figures in the report are a *directional*
reproduction of the draft trainer, not exact. Nothing here is a production
validation result; the point is the methodology, not the numbers.
