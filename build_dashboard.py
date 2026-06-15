"""
build_dashboard.py
==================
Builds a single, self-contained, OFFLINE-viewable HTML explorer for the
Yogyank farmer entitlement-scoring dataset.

What it does
------------
Reads the synthetic farmer CSV and emits one .html file that you can open by
double-clicking — no server, no internet, no Python needed to *view* it. The
full Plotly library is embedded once inside the file.

The views are chosen to answer questions this project actually needs, not to
decorate: schema/contract, target shape, signal-vs-leakage, gaps (missingness),
temporal drift (does the validation split make sense?), and the policy-injection
illustration. The leakage column is marked in red everywhere it appears, as a
deliberate visual through-line meaning "disqualified — cannot be used at scoring
time."

Run
---
    pip install pandas plotly            # (xgboost + scikit-learn optional, see below)
    python build_dashboard.py            # uses defaults below
    python build_dashboard.py path/to/data.csv out.html

The R^2 "collapse" waterfall needs xgboost + scikit-learn (you already have
them). If they are missing, the script still builds — that one panel degrades
to a short note instead of crashing.

Author: (your name) — adapted as a learning project.
"""

from __future__ import annotations

import sys
import html
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CSV_PATH = "farmer_scoring_sample_yogyank_round1.csv"
OUTPUT_HTML = "yogyank_data_explorer.html"

TARGET = "target_entitlement_score"
LEAK_COL = "defaulted_in_next_12_months"          # forward-looking outcome
TEMPORAL = "application_year"
ID_COL = "farmer_id"
POLICY_COL = "pm_kisan_status"

# Explicit column groups (avoids dtype-sniffing surprises across pandas versions)
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

# Per-column ROLE and an availability ASSUMPTION (the seed of your schema/contract).
# availability: "yes" (known at/before scoring), "no" (future -> leak),
#               "assume" (plausibly fine but MUST be verified as as-of-scoring-date),
#               "na"  (not a feature)
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

# --------------------------------------------------------------------------- #
# Palette / theme  (one bold, meaningful choice: red == leak, everywhere)
# --------------------------------------------------------------------------- #
INK     = "#1B2420"
MUTED   = "#6B7670"
PAPER   = "#EDEFEC"
SURFACE = "#FBFCFA"
BORDER  = "#D8DCD6"
TEAL    = "#2F6F69"   # signal / trustworthy
TEAL_2  = "#5B958F"
CLAY    = "#B5733A"   # temporal / earth
GOLD    = "#C8A24B"   # policy
LEAK    = "#C0392B"   # danger / leakage — reserved
LEAK_BG = "#F6E3E0"

ROLE_COLORS = {
    "identifier": MUTED,
    "temporal":   CLAY,
    "feature":    TEAL,
    "policy":     GOLD,
    "leak":       LEAK,
    "target":     INK,
}
AVAIL_LABEL = {
    "yes":    ("available",      TEAL),
    "no":     ("LEAK — drop",    LEAK),
    "assume": ("verify as-of",   CLAY),
    "na":     ("—",              MUTED),
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace"

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the shared house style to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=60, r=24, t=48, b=48),
        colorway=[TEAL, CLAY, TEAL_2, GOLD, MUTED],
        title=dict(font=dict(size=15, color=INK)),
        hoverlabel=dict(font=dict(family=MONO, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def _div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=div_id, config=PLOT_CONFIG)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def schema_table_html(df: pd.DataFrame) -> str:
    rows = []
    for col in df.columns:
        role, avail, note = COLUMN_META.get(col, ("feature", "assume", ""))
        role_c = ROLE_COLORS[role]
        avail_text, avail_c = AVAIL_LABEL[avail]
        n_missing = df[col].isna().mean() * 100
        n_unique = df[col].nunique(dropna=True)
        dtype = str(df[col].dtype)
        leak_row = ' class="leak-row"' if role == "leak" else ""
        rows.append(f"""
          <tr{leak_row}>
            <td class="mono">{html.escape(col)}</td>
            <td><span class="chip" style="background:{role_c}1A;color:{role_c};border-color:{role_c}55">{role}</span></td>
            <td><span class="chip" style="background:{avail_c}1A;color:{avail_c};border-color:{avail_c}55">{avail_text}</span></td>
            <td class="mono num">{dtype}</td>
            <td class="mono num">{n_unique:,}</td>
            <td class="mono num">{n_missing:.0f}%</td>
            <td class="note">{html.escape(note)}</td>
          </tr>""")
    return f"""
    <table class="schema">
      <thead><tr>
        <th>column</th><th>role</th><th>available before scoring?</th>
        <th>dtype</th><th>unique</th><th>missing</th><th>assumption / note</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def fig_target(df: pd.DataFrame) -> go.Figure:
    s = df[TARGET]
    fig = go.Figure(go.Histogram(x=s, nbinsx=45, marker_color=TEAL,
                                 marker_line_color=SURFACE, marker_line_width=0.5,
                                 hovertemplate="score %{x}<br>%{y} farmers<extra></extra>"))
    fig.add_vline(x=s.mean(), line_dash="dash", line_color=INK,
                  annotation_text=f"mean {s.mean():.0f}", annotation_position="top")
    fig.update_layout(title="Target — entitlement score (what we predict)",
                      xaxis_title="target_entitlement_score", yaxis_title="farmers")
    return _style(fig, 360)


def fig_signal(df: pd.DataFrame) -> go.Figure:
    cols = NUMERIC_FEATURES + [LEAK_COL]
    corr = df[cols].corrwith(df[TARGET]).abs().sort_values()
    colors = [LEAK if c == LEAK_COL else TEAL for c in corr.index]
    labels = [f"{c}  ⛔" if c == LEAK_COL else c for c in corr.index]
    fig = go.Figure(go.Bar(
        x=corr.values, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>|corr| %{x:.3f}<extra></extra>"))
    fig.update_layout(
        title="Signal vs. leakage — |correlation| with the target",
        xaxis_title="absolute correlation with target", yaxis_title="")
    fig.add_annotation(x=corr[LEAK_COL], y=f"{LEAK_COL}  ⛔",
                       text="strongest signal is the one you can't have",
                       showarrow=True, arrowhead=2, ax=60, ay=0,
                       font=dict(color=LEAK, family=MONO, size=11),
                       arrowcolor=LEAK)
    return _style(fig, 380)


def fig_leakage_spotlight(df: pd.DataFrame) -> go.Figure:
    g0 = df.loc[df[LEAK_COL] == 0, TARGET]
    g1 = df.loc[df[LEAK_COL] == 1, TARGET]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=g0, nbinsx=40, name=f"no default (mean {g0.mean():.0f})",
                               marker_color=TEAL, opacity=0.75))
    fig.add_trace(go.Histogram(x=g1, nbinsx=40, name=f"defaulted (mean {g1.mean():.0f})",
                               marker_color=LEAK, opacity=0.75))
    fig.update_layout(
        title=f"Why it's leakage — the score already 'knows' the future ({g0.mean()-g1.mean():.0f}-pt gap)",
        barmode="overlay", xaxis_title="target_entitlement_score", yaxis_title="farmers")
    return _style(fig, 360)


def fig_r2_waterfall(df: pd.DataFrame) -> tuple[go.Figure | None, str]:
    """Quantify how much of the reported R^2 is illusion. Needs xgboost+sklearn."""
    try:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
    except Exception:
        return None, ("R² collapse panel skipped: install xgboost + scikit-learn "
                      "to reproduce it (pip install xgboost scikit-learn).")

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = LabelEncoder().fit_transform(out[c].astype(str))
        return out

    f_leak = ["land_area_acres", "crop_type", "pm_kisan_status",
              "historical_repayment_score", LEAK_COL]
    f_clean = ["land_area_acres", "crop_type", "pm_kisan_status",
               "historical_repayment_score"]

    def r2(frame_cols, y, mode):
        X = encode(df[frame_cols])
        if mode == "random":
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                  random_state=42, shuffle=True)
        else:  # out-of-time
            m = df[TEMPORAL] <= 2023
            Xtr, Xte, ytr, yte = X[m], X[~m], y[m], y[~m]
        mdl = xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.1,
                               random_state=42, n_jobs=1, tree_method="hist")
        mdl.fit(Xtr, ytr)
        return r2_score(yte, mdl.predict(Xte))

    y_policy = df[TARGET].copy()
    y_policy.loc[df[POLICY_COL] == "No"] -= 150          # junior bakes policy into label
    y_clean = df[TARGET].copy()

    r_reported = r2(f_leak, y_policy, "random")
    r_split = r2(f_leak, y_clean, "oot")
    r_honest = r2(f_clean, y_clean, "oot")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["reported<br>(random split,<br>leak in)",
           "fix the<br>split",
           "drop the<br>leak",
           "honest<br>baseline"],
        y=[r_reported, r_split - r_reported, r_honest - r_split, r_honest],
        text=[f"{r_reported:.2f}", f"{r_split-r_reported:+.2f}",
              f"{r_honest-r_split:+.2f}", f"{r_honest:.2f}"],
        textposition="outside",
        connector=dict(line=dict(color=BORDER)),
        decreasing=dict(marker=dict(color=LEAK)),
        increasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=INK)),
    ))
    fig.update_layout(
        title="What the reported R² is actually worth (directional repro)",
        yaxis_title="R² on held-out farmers", showlegend=False)
    note = (f"Reported ≈ {r_reported:.2f}. Fixing only the split → {r_split:.2f}; "
            f"also removing the leak → {r_honest:.2f}. "
            "Nearly half the inflation came from the split alone.")
    return _style(fig, 380), note


def fig_missingness(df: pd.DataFrame) -> tuple[go.Figure, str]:
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h",
                           marker_color=CLAY,
                           hovertemplate="%{y}<br>%{x:.1f}% missing<extra></extra>"))
    fig.update_layout(title="Gaps — where values are absent",
                      xaxis_title="% missing", yaxis_title="")
    # Co-missingness: do the two gappy columns vanish together?
    a, b = "ndvi_score", "rainfall_deviation_pct"
    note = ""
    if a in df and b in df:
        pa, pb = df[a].isna().mean(), df[b].isna().mean()
        both = (df[a].isna() & df[b].isna()).mean()
        expected = pa * pb
        rel = both / expected if expected else float("nan")
        note = (f"{a} and {b} are each ~{pa*100:.0f}% missing. They go missing together "
                f"{both*100:.1f}% of the time vs. {expected*100:.1f}% if independent "
                f"(×{rel:.1f}) — {'patterned, treat missingness as signal' if rel>1.3 else 'roughly independent'}.")
    return _style(fig, 320), note


def fig_temporal(df: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    years = sorted(df[TEMPORAL].dropna().unique())
    counts = df[TEMPORAL].value_counts().sort_index()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                        subplot_titles=("rows per year", "target distribution per year"))
    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values,
                         marker_color=CLAY, showlegend=False,
                         hovertemplate="%{x}: %{y} rows<extra></extra>"), row=1, col=1)
    for i, y in enumerate(years):
        col = TEAL if y <= 2023 else LEAK  # held-out year stands out
        fig.add_trace(go.Box(y=df.loc[df[TEMPORAL] == y, TARGET], name=str(y),
                             marker_color=col, boxmean=True, showlegend=False),
                      row=1, col=2)
    fig.update_layout(title="Temporal — can we validate the future? (2024 held out)")
    return _style(fig, 360)


def fig_policy(df: pd.DataFrame) -> go.Figure:
    means = df.groupby(POLICY_COL)[TARGET].mean()
    actual_gap = means.get("Yes", np.nan) - means.get("No", np.nan)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["actual gap in data<br>(Yes − No)", "injected by script<br>(−150 on the label)"],
                         y=[actual_gap, 150],
                         marker_color=[TEAL, LEAK],
                         text=[f"{actual_gap:.0f} pts", "150 pts"],
                         textposition="outside",
                         hovertemplate="%{x}: %{y:.0f}<extra></extra>"))
    fig.update_layout(
        title="Policy injection — the script invents a rule, it doesn't recover one",
        yaxis_title="points", showlegend=False)
    return _style(fig, 320)


def fig_numeric_explorer(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(NUMERIC_FEATURES):
        fig.add_trace(go.Histogram(x=df[c], nbinsx=40, marker_color=TEAL,
                                   name=c, visible=(i == 0),
                                   hovertemplate=f"{c} %{{x}}<br>%{{y}}<extra></extra>"))
    buttons = []
    for i, c in enumerate(NUMERIC_FEATURES):
        vis = [j == i for j in range(len(NUMERIC_FEATURES))]
        buttons.append(dict(label=c, method="update",
                            args=[{"visible": vis},
                                  {"title": f"Explore — {c}", "xaxis": {"title": c}}]))
    fig.update_layout(
        title=f"Explore — {NUMERIC_FEATURES[0]}",
        xaxis_title=NUMERIC_FEATURES[0], yaxis_title="farmers", showlegend=False,
        updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.18,
                          yanchor="top", bgcolor=SURFACE, bordercolor=BORDER)])
    return _style(fig, 380)


def fig_categorical_explorer(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(CATEGORICAL_FEATURES):
        vc = df[c].value_counts().sort_values()
        fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation="h",
                             marker_color=TEAL_2, name=c, visible=(i == 0),
                             hovertemplate="%{y}: %{x}<extra></extra>"))
    buttons = []
    for i, c in enumerate(CATEGORICAL_FEATURES):
        vis = [j == i for j in range(len(CATEGORICAL_FEATURES))]
        buttons.append(dict(label=c, method="update",
                            args=[{"visible": vis}, {"title": f"Explore — {c}"}]))
    fig.update_layout(
        title=f"Explore — {CATEGORICAL_FEATURES[0]}", showlegend=False,
        xaxis_title="count",
        updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.18,
                          yanchor="top", bgcolor=SURFACE, bordercolor=BORDER)])
    return _style(fig, 400)


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
def stat_cards(df: pd.DataFrame) -> str:
    cards = [
        ("rows", f"{len(df):,}"),
        ("columns", f"{df.shape[1]}"),
        ("feature candidates", f"{len(NUMERIC_FEATURES)+len(CATEGORICAL_FEATURES)}"),
        ("leak columns", "1"),
        ("target range", f"{df[TARGET].min():.0f}–{df[TARGET].max():.0f}"),
        ("years", f"{df[TEMPORAL].min()}–{df[TEMPORAL].max()}"),
    ]
    return "".join(
        f'<div class="card"><div class="k mono">{html.escape(v)}</div>'
        f'<div class="l">{html.escape(k)}</div></div>' for k, v in cards)


def section(num: str, eyebrow: str, title: str, body: str, lead: str = "") -> str:
    lead_html = f'<p class="lead">{lead}</p>' if lead else ""
    return f"""
    <section class="sec">
      <div class="sec-head">
        <span class="eyebrow mono">{num} · {eyebrow}</span>
        <h2>{title}</h2>
        {lead_html}
      </div>
      {body}
    </section>"""


def build_html(df: pd.DataFrame) -> str:
    plotlyjs = get_plotlyjs()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    target = _div(fig_target(df), "target")
    signal = _div(fig_signal(df), "signal")
    leak_spot = _div(fig_leakage_spotlight(df), "leakspot")
    wf_fig, wf_note = fig_r2_waterfall(df)
    waterfall = (_div(wf_fig, "waterfall") if wf_fig
                 else f'<p class="note-box">{html.escape(wf_note)}</p>')
    miss_fig, miss_note = fig_missingness(df)
    missing = _div(miss_fig, "missing")
    temporal = _div(fig_temporal(df), "temporal")
    policy = _div(fig_policy(df), "policy")
    num_exp = _div(fig_numeric_explorer(df), "numexp")
    cat_exp = _div(fig_categorical_explorer(df), "catexp")

    css = f"""
    :root{{--ink:{INK};--muted:{MUTED};--paper:{PAPER};--surface:{SURFACE};
           --border:{BORDER};--teal:{TEAL};--clay:{CLAY};--leak:{LEAK};--leakbg:{LEAK_BG}}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--paper);color:var(--ink);
          font-family:{FONT};line-height:1.5;-webkit-font-smoothing:antialiased}}
    .wrap{{max-width:1080px;margin:0 auto;padding:0 24px 80px}}
    header.top{{padding:56px 0 28px;border-bottom:1px solid var(--border)}}
    .kicker{{font-family:{MONO};font-size:12px;letter-spacing:.14em;
             text-transform:uppercase;color:var(--teal)}}
    h1{{font-size:38px;line-height:1.08;margin:10px 0 8px;letter-spacing:-.02em}}
    header.top p{{color:var(--muted);max-width:60ch;margin:0}}
    .cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:28px 0 8px}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
           padding:14px 14px 12px}}
    .card .k{{font-size:20px;font-weight:600;letter-spacing:-.01em}}
    .card .l{{font-size:11px;color:var(--muted);text-transform:uppercase;
              letter-spacing:.08em;margin-top:3px}}
    .sec{{padding:44px 0;border-bottom:1px solid var(--border)}}
    .sec-head{{margin-bottom:18px}}
    .eyebrow{{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--clay)}}
    h2{{font-size:23px;margin:6px 0 4px;letter-spacing:-.01em}}
    .lead{{color:var(--muted);max-width:74ch;margin:6px 0 0}}
    .panel{{background:var(--surface);border:1px solid var(--border);
            border-radius:12px;padding:8px 8px 4px;margin-top:14px}}
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
    .note{{color:var(--muted);font-size:12.5px}}
    .note-box{{background:var(--surface);border:1px solid var(--border);
               border-left:3px solid var(--clay);border-radius:8px;
               padding:12px 14px;color:var(--muted);font-size:13px;margin-top:14px}}
    .note-box.leak{{border-left-color:var(--leak);background:var(--leakbg)}}
    table.schema{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;
                  background:var(--surface);border:1px solid var(--border);border-radius:12px;
                  overflow:hidden}}
    .schema th{{text-align:left;font-family:{MONO};font-size:11px;text-transform:uppercase;
                letter-spacing:.06em;color:var(--muted);padding:11px 12px;
                border-bottom:1px solid var(--border);background:#F4F6F3}}
    .schema td{{padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:top}}
    .schema tr:last-child td{{border-bottom:none}}
    .schema tr.leak-row{{background:var(--leakbg)}}
    .mono{{font-family:{MONO}}}
    td.num{{text-align:right;color:var(--muted)}}
    td.note{{color:var(--muted);max-width:34ch}}
    .chip{{display:inline-block;font-family:{MONO};font-size:11px;padding:2px 8px;
           border-radius:999px;border:1px solid;white-space:nowrap}}
    footer{{padding:32px 0 0;color:var(--muted);font-size:12.5px}}
    footer code{{font-family:{MONO};background:var(--surface);border:1px solid var(--border);
                 padding:1px 6px;border-radius:5px}}
    @media (max-width:880px){{.cards{{grid-template-columns:repeat(2,1fr)}}
      .grid2{{grid-template-columns:1fr}} h1{{font-size:30px}}}}
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yogyank — data explorer</title>
<style>{css}</style>
<script>{plotlyjs}</script>
</head><body>
<div class="wrap">

  <header class="top">
    <div class="kicker">Yogyank · entitlement scoring · data audit</div>
    <h1>Reading the data before we trust it</h1>
    <p>A synthetic farmer dataset for a bank-agnostic entitlement score. Every panel
       answers a question the project needs. <strong style="color:{LEAK}">Red marks the
       leak</strong> — a field that cannot be known at scoring time — wherever it appears.</p>
    <div class="cards">{stat_cards(df)}</div>
  </header>

  {section("01", "orientation", "Schema &amp; the as-of-date contract",
           f'<div class="panel" style="padding:0;background:transparent;border:none">{schema_table_html(df)}</div>'
           '<p class="note" style="margin-top:10px">This table is the first draft of your schema/contract deliverable. '
           '"Available before scoring?" is the only question that matters for leakage — assumptions in clay must be verified against real feature definitions.</p>',
           "What is in the table, and which fields are we even allowed to use.")}

  {section("02", "target", "What we are predicting",
           f'<div class="panel">{target}</div>',
           "The score is continuous and bounded — so report error in points (MAE), not just R².")}

  {section("03", "signal vs. leakage", "Is there real signal — and where is the trap?",
           f'<div class="grid2"><div class="panel">{signal}</div><div class="panel">{leak_spot}</div></div>'
           f'<div class="panel">{waterfall}</div>'
           f'<p class="note-box leak">{html.escape(wf_note)}</p>',
           "There is genuine signal (repayment history, land, liability). But the tallest bar is a field from the future.")}

  {section("04", "gaps", "Where the data is thin or absent",
           f'<div class="panel">{missing}</div>'
           + (f'<p class="note-box">{html.escape(miss_note)}</p>' if miss_note else ""),
           "Missingness that follows a pattern is itself information — and a potential leak if the pattern depends on the outcome.")}

  {section("05", "temporal", "Can we validate the future?",
           f'<div class="panel">{temporal}</div>',
           "Three application years let us train on 2022–2023 and score 2024 — an out-of-time split that mimics deployment. Watch whether 2024 drifts.")}

  {section("06", "policy", "Model vs. policy",
           f'<div class="panel">{policy}</div>',
           "The draft script subtracts 150 from the label for non-PM-Kisan farmers. The real gap in the data is far smaller — so the rule is injected, not learned. Policy belongs in a transparent post-processing layer.")}

  {section("07", "explore", "Poke at the columns",
           f'<div class="grid2"><div class="panel">{num_exp}</div><div class="panel">{cat_exp}</div></div>',
           "Use the dropdowns to switch fields. Rare categories give unstable encodings and shaky reason codes — worth noting now.")}

  <footer>
    <p>Generated {stamp} from <code>{html.escape(Path(CSV_PATH).name)}</code> by <code>build_dashboard.py</code>.
       Self-contained &amp; offline: the full Plotly library is embedded in this file.</p>
    <p>Synthetic data; the R² figures are a directional reproduction of the draft script, not exact.
       Nothing here is a production validation result.</p>
  </footer>

</div></body></html>"""


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_HTML
    df = load_data(csv_path)
    html_str = build_html(df)
    Path(out_path).write_text(html_str, encoding="utf-8")
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB, self-contained, open in any browser)")


if __name__ == "__main__":
    main()
