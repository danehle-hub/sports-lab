# streamlit_app.py — FULL REPLACEMENT (toggles + sliders only)

import sys
import json
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

TOGGLES_CSV = RUNS / "controls_toggles.csv"
THRESH_CSV  = RUNS / "controls_thresholds.csv"
WEIGHTS_CSV = RUNS / "controls_weights.csv"

PRED_FILE        = RUNS / "upcoming_predictions_plus.csv"
BT_INPUT_CSV     = RUNS / "backtest_input.csv"
BT_RESULTS_CSV   = RUNS / "backtest_results.csv"
BT_SUMMARY_JSON  = RUNS / "backtest_summary.json"

# ---------------- Control schema ----------------
TOGGLE_DEFS = [
    ("enable_unit_floors", "Enforce unit floors by tier", True),
    ("use_ev",             "Use EV for tiering",          True),
    ("use_points",         "Use value_pts for tiering",   True),
]

THRESH_DEFS = [
    ("kelly_fraction",       "Kelly fraction",                    0.25,  (0.00, 1.00)),
    ("max_units",            "Max units per play",                3.00,  (0.00, 10.00)),
    ("default_spread_price", "Default spread price (American)",  -110.0, (-500.0, 500.0)),
    ("strong_ev_threshold",  "STRONG EV threshold",               0.030, (0.000, 0.200)),
    ("strong_pts_threshold", "STRONG points threshold",           3.0,   (0.0, 10.0)),
    ("lean_ev_threshold",    "LEAN EV threshold",                 0.0125,(0.000, 0.200)),
    ("lean_pts_threshold",   "LEAN points threshold",             1.5,   (0.0, 10.0)),
    ("ats_sigma",            "ATS sigma (spread SD, pts)",       13.5,   (8.0, 20.0)),
]

WEIGHT_DEFS = [
    ("meta_blend",   "Shrink factor (0..1): higher = more aggressive", 0.60, (0.00, 1.00)),
    ("w_trench",     "Weight: Trench",                                 0.40, (0.00, 2.00)),
    ("w_qb",         "Weight: QB",                                     0.30, (0.00, 2.00)),
    ("w_rest",       "Weight: Rest/Schedule",                          0.15, (0.00, 2.00)),
    ("w_travel",     "Weight: Travel/Body clock",                      0.05, (0.00, 2.00)),
    ("w_hfa",        "Weight: Home-field",                             0.05, (0.00, 2.00)),
    ("w_inj_cluster","Weight: Injury clusters",                        0.05, (0.00, 2.00)),
]

AGG_PRESETS = {
    "Conservative": {
        "weights":   {"meta_blend": 0.50},
        "thresholds": {
            "strong_pts_threshold": 3.5, "strong_ev_threshold": 0.035,
            "lean_pts_threshold":   2.0, "lean_ev_threshold":   0.015,
            "kelly_fraction":       0.20,
        },
    },
    "Balanced": {
        "weights":   {"meta_blend": 0.60},
        "thresholds": {
            "strong_pts_threshold": 3.0, "strong_ev_threshold": 0.030,
            "lean_pts_threshold":   1.5, "lean_ev_threshold":   0.0125,
            "kelly_fraction":       0.25,
        },
    },
    "Aggressive": {
        "weights":   {"meta_blend": 0.70},
        "thresholds": {
            "strong_pts_threshold": 2.5, "strong_ev_threshold": 0.025,
            "lean_pts_threshold":   1.0, "lean_ev_threshold":   0.010,
            "kelly_fraction":       0.30,
        },
    },
}

# ---------------- State helpers ----------------
def _init_state():
    if "toggles" not in st.session_state:
        st.session_state.toggles = {k: d for (k, _, d) in TOGGLE_DEFS}
    if "thresholds" not in st.session_state:
        st.session_state.thresholds = {k: d for (k, _, d, _) in THRESH_DEFS}
    if "weights" not in st.session_state:
        st.session_state.weights = {k: d for (k, _, d, _) in WEIGHT_DEFS}

def _apply_preset(preset_name: str):
    pre = AGG_PRESETS.get(preset_name, {})
    for k, v in pre.get("thresholds", {}).items():
        if k in st.session_state.thresholds:
            st.session_state.thresholds[k] = float(v)
    for k, v in pre.get("weights", {}).items():
        if k in st.session_state.weights:
            st.session_state.weights[k] = float(v)

def _save_controls_to_csv():
    t_rows = [{"key": k, "label": next(lbl for (kk, lbl, _) in TOGGLE_DEFS if kk==k), "value": v, "help": ""} for k, v in st.session_state.toggles.items()]
    th_rows= [{"key": k, "label": next(lbl for (kk, lbl, _, _) in THRESH_DEFS if kk==k), "value": st.session_state.thresholds[k],
               "min": rng[0], "max": rng[1], "step": "", "help": ""} for (k, _, __, rng) in THRESH_DEFS]
    w_rows = [{"key": k, "label": next(lbl for (kk, lbl, _, _) in WEIGHT_DEFS if kk==k), "value": st.session_state.weights[k],
               "min": rng[0], "max": rng[1], "step": "", "help": ""} for (k, _, __, rng) in WEIGHT_DEFS]
    pd.DataFrame(t_rows).to_csv(TOGGLES_CSV, index=False)
    pd.DataFrame(th_rows).to_csv(THRESH_CSV,  index=False)
    pd.DataFrame(w_rows).to_csv(WEIGHTS_CSV, index=False)

# ---------------- UI renderers ----------------
def render_toggle_group():
    st.subheader("Toggles")
    cols = st.columns(3)
    for i, (key, label, default) in enumerate(TOGGLE_DEFS):
        with cols[i % 3]:
            st.session_state.toggles[key] = st.checkbox(label, value=st.session_state.toggles[key])

def render_threshold_sliders():
    st.subheader("Thresholds & Sizing")
    for (key, label, default, (lo, hi)) in THRESH_DEFS:
        st.session_state.thresholds[key] = st.slider(
            label, min_value=float(lo), max_value=float(hi),
            value=float(st.session_state.thresholds[key])
        )

def render_weight_sliders():
    st.subheader("Weights")
    cols = st.columns(2)
    for i, (key, label, default, (lo, hi)) in enumerate(WEIGHT_DEFS):
        with cols[i % 2]:
            st.session_state.weights[key] = st.slider(
                label, min_value=float(lo), max_value=float(hi),
                value=float(st.session_state.weights[key])
            )

# ---------------- App ----------------
st.set_page_config(page_title="NFL Model Dashboard", layout="wide")
st.title("🏈 NFL Prediction Model Dashboard")

import datetime as _dt
st.markdown(f"**UI mode:** SLIDERS ONLY · build { _dt.datetime.now():%Y-%m-%d %H:%M:%S }")

_init_state()

tab_picks, tab_backtest = st.tabs(["📈 Picks", "📊 Backtest"])

# ==== PICKS TAB ====
with tab_picks:
    st.sidebar.header("Run Controls (Picks)")
    preset_choice = st.sidebar.selectbox(
        "Model aggressiveness (preset)",
        ["Conservative", "Balanced", "Aggressive"], index=1,
        help="Preset updates sliders instantly; you can still tweak after."
    )
    if st.sidebar.button(f"Apply '{preset_choice}' Preset"):
        _apply_preset(preset_choice)
        st.success(f"Applied {preset_choice} preset.")

    fetch_markets = st.sidebar.toggle("Fetch fresh markets", value=True)
    run_picks     = st.sidebar.button("🔁 Save Controls & Recompute (Picks)")

    render_toggle_group()
    render_threshold_sliders()
    render_weight_sliders()

    if run_picks:
        _save_controls_to_csv()
        st.info("Running model…")
        cmd = [sys.executable, "-m", "models.predict_upcoming_plus"]
        if fetch_markets:
            cmd.append("--fetch")
        try:
            completed = subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
            st.toast("Picks complete.", icon="✅")
            with st.expander("Run log"):
                st.code(completed.stdout + "\n" + completed.stderr)
        except subprocess.CalledProcessError as e:
            st.error("Model run failed.")
            st.exception(e)

    st.subheader("Spread Model Picks")
    if not PRED_FILE.exists():
        st.warning("No predictions yet. Click **Save Controls & Recompute (Picks)**.")
    else:
        preds = pd.read_csv(PRED_FILE)
        fmt = {
            "spread_home": "{:.1f}",
            "model_home_line": "{:.2f}",
            "home_cover_prob": "{:.3f}",
            "market_line_pts": "{:.1f}",
            "model_line_pts": "{:.2f}",
            "value_pts": "{:.2f}",
            "ats_ev": "{:.2f}",
            "units": "{:.2f}",
        }
        st.dataframe(preds.style.format(fmt), use_container_width=True, hide_index=True)

# ==== BACKTEST TAB ====
with tab_backtest:
    st.sidebar.header("Run Controls (Backtest)")
    season_input = st.sidebar.text_input("Seasons (e.g., 2018-2024 or 2022,2023,2024)", value="2022-2024")
    line_source  = st.sidebar.selectbox("Line source", ["closing", "open", "best_of"], index=0)
    only_tiered  = st.sidebar.checkbox("Only count LEAN/STRONG bets", value=True)
    fetch_needed = st.sidebar.toggle("Fetch new metrics (if wired)", value=False)
    run_bt       = st.sidebar.button("🔁 Save Controls & Run Backtest")

    st.subheader("Backtest Input")
    st.markdown(
        "Upload a historical CSV with columns: "
        "`season,week,commence_time,team_home,team_away,home_score,away_score,spread_home` "
        "and optional: prices `market_home_spread_price, market_away_spread_price`, "
        "alternate lines `spread_home_open, spread_home_closing, spread_home_best_of`."
    )
    file = st.file_uploader("Upload backtest_input.csv", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        df_up.to_csv(BT_INPUT_CSV, index=False)
        st.success(f"Uploaded → {BT_INPUT_CSV.name} ({len(df_up)} rows).")

    if BT_INPUT_CSV.exists():
        st.dataframe(pd.read_csv(BT_INPUT_CSV).head(12), use_container_width=True)
    else:
        st.info("No backtest input found. Upload above.")

    if run_bt:
        _save_controls_to_csv()  # ensures backtest uses current sliders/toggles/weights
        st.info("Running backtest…")
        cmd = [
            sys.executable, "-m", "models.backtest",
            "--seasons", season_input,
            "--line_source", line_source,
        ]
        if only_tiered:
            cmd.append("--only_tiered")
        if fetch_needed:
            cmd.append("--fetch")
        try:
            completed = subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
            st.toast("Backtest complete.", icon="✅")
            with st.expander("Backtest log"):
                st.code(completed.stdout + "\n" + completed.stderr)
        except subprocess.CalledProcessError as e:
            st.error("Backtest run failed.")
            st.exception(e)

    st.subheader("Backtest Results")
    if BT_SUMMARY_JSON.exists():
        try:
            summary = json.loads(BT_SUMMARY_JSON.read_text(encoding="utf-8"))
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Bets",        f"{summary.get('bets', 0):,}")
            c2.metric("Win %",       f"{100*summary.get('win_rate', 0):.1f}%")
            c3.metric("ROI",         f"{100*summary.get('roi', 0):.2f}%")
            c4.metric("Avg CLV (pt)",f"{summary.get('avg_clv_pts', 0):.2f}")
            c5.metric("Max Drawdown",f"{100*summary.get('max_drawdown', 0):.1f}%")
        except Exception:
            st.warning("Summary present but could not be parsed.")

    if BT_RESULTS_CSV.exists():
        bt = pd.read_csv(BT_RESULTS_CSV)
        fmt = {
            "spread_home": "{:.1f}",
            "model_home_line": "{:.2f}",
            "home_cover_prob": "{:.3f}",
            "market_line_pts": "{:.1f}",
            "model_line_pts": "{:.2f}",
            "value_pts": "{:.2f}",
            "roi": "{:.3f}",
            "units": "{:.2f}",
            "clv_pts": "{:.2f}",
        }
        st.dataframe(bt.head(500).style.format(fmt), use_container_width=True, hide_index=True)
        st.download_button(
            "Download full backtest results CSV",
            bt.to_csv(index=False).encode("utf-8"),
            file_name="backtest_results.csv"
        )
    else:
        st.info("Run a backtest to see results here.")
