# streamlit_app.py — FULL REPLACEMENT (v2025-11-10e pass-through)
# - Sliders saved to runs/controls_*.csv
# - Run button calls python -m models.predict_upcoming_plus --fetch
# - Pass-through if CSV is already the final 14-column export
# - Otherwise, transform raw columns
# - Strict column_config (no 6-dec drift)

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ---------------- Paths ----------------
APP_ROOT = Path(r"C:\Users\DanEhle\Documents\sports-lab")
MODELS_DIR = APP_ROOT / "models"
RUNS_DIR = APP_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = RUNS_DIR / "upcoming_predictions_plus.csv"
TOGGLES_CSV = RUNS_DIR / "controls_toggles.csv"
THRESH_CSV  = RUNS_DIR / "controls_thresholds.csv"
WEIGHTS_CSV = RUNS_DIR / "controls_weights.csv"

st.set_page_config(page_title="Sports Quant: NFL Model", layout="wide")

# ---------------- Utils ----------------
REQUIRED_EXPORT_COLS = [
    "Home Team","Away Team","Kickoff","Home Spread","Model Home Line","Home Cover %",
    "Model Line (pts)","Edge (pts)","ATS EV (%)","Units","Pick Tier","Pick Team","Pick Price","Market Home ML",
]

def round_to_half(x):
    if pd.isna(x):
        return np.nan
    try:
        return np.round(float(x) * 2) / 2.0
    except Exception:
        return np.nan

def fmt_percentage_1dp(x):
    if pd.isna(x):
        return np.nan
    try:
        val = float(x)
        return val * 100.0 if val <= 1.0 else val
    except Exception:
        return np.nan

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_home_spread(row):
    sh = row.get("spread_home", np.nan)
    if pd.notna(sh):
        try:
            return float(sh)
        except Exception:
            pass
    ml = row.get("market_line", None)
    if isinstance(ml, str) and len(ml.strip()) > 0:
        last = ml.strip().split()[-1]
        try:
            return float(last)
        except Exception:
            return np.nan
    return np.nan

def parse_kickoff(df: pd.DataFrame):
    candidates = ["kickoff", "kickoff_dt", "kickoff_time", "game_time", "start_time", "commence_time"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def series_or_default(df: pd.DataFrame, col: str, default_value):
    if col in df.columns:
        s = df[col]
        if not isinstance(s, pd.Series):
            return pd.Series([s] * len(df), index=df.index)
        return s
    return pd.Series([default_value] * len(df), index=df.index)

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def build_display_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    # --- PASS-THROUGH: already in final 14-column format ---
    if set(REQUIRED_EXPORT_COLS).issubset(set(raw.columns)):
        out = raw[REQUIRED_EXPORT_COLS].copy()
        # enforce numeric types and round (belt + suspenders)
        for c in ["Home Spread","Model Home Line","Model Line (pts)"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").apply(round_to_half)
        for c in ["Edge (pts)","ATS EV (%)","Units","Pick Price","Market Home ML","Home Cover %"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # --- FALLBACK: transform from wide/raw model output ---
    df = raw.copy()
    df["Home Team"] = series_or_default(df, "team_home", "")
    df["Away Team"] = series_or_default(df, "team_away", "")

    kickoff_col = parse_kickoff(df)
    if kickoff_col is None:
        df["Kickoff"] = ""
    else:
        k = pd.to_datetime(df[kickoff_col], errors="coerce", utc=False)
        df["Kickoff"] = k.dt.strftime("%b-%d-%Y, %H:%M").fillna("")

    if "spread_home" not in df.columns:
        df["spread_home"] = np.nan
    df["Home Spread"] = df.apply(parse_home_spread, axis=1)
    df["Home Spread"] = df["Home Spread"].apply(round_to_half)

    if "model_home_line" in df.columns:
        df["Model Home Line"] = coerce_numeric(df["model_home_line"]).apply(round_to_half)
    else:
        if "model_spread" in df.columns:
            df["Model Home Line"] = (-coerce_numeric(df["model_spread"])).apply(round_to_half)
        else:
            df["Model Home Line"] = np.nan

    if "model_line" in df.columns:
        tmp_ml = coerce_numeric(df["model_line"]).apply(round_to_half)
        df["Model Line (pts)"] = tmp_ml.where(~tmp_ml.isna(), df["Model Home Line"])
    else:
        df["Model Line (pts)"] = df["Model Home Line"]

    if "edge_vs_market" in df.columns:
        df["Edge (pts)"] = coerce_numeric(df["edge_vs_market"]).round(1)
    else:
        df["Edge (pts)"] = (
            coerce_numeric(df["Model Home Line"]) - coerce_numeric(df["Home Spread"])
        ).round(1)

    home_cover_col = None
    for c in ["home_cover_prob", "home_cover_prob_raw", "home_ats_prob", "home_spread_prob"]:
        if c in df.columns:
            home_cover_col = c
            break
    df["Home Cover %"] = coerce_numeric(df[home_cover_col]).apply(fmt_percentage_1dp) if home_cover_col else np.nan

    ats_ev_col = None
    for c in df.columns:
        if "ats" in c.lower() and "ev" in c.lower():
            ats_ev_col = c
            break
    df["ATS EV (%)"] = coerce_numeric(df[ats_ev_col]).apply(fmt_percentage_1dp).round(1) if ats_ev_col else np.nan

    df["Units"] = coerce_numeric(df.get("units", np.nan)).round(1)

    df["Pick Tier"] = series_or_default(df, "pick_tier", "")
    if "pick" in df.columns:
        df["Pick Team"] = series_or_default(df, "pick", "")
    else:
        df["Pick Team"] = series_or_default(df, "pick_team", "")

    if "pick_price" in df.columns:
        df["Pick Price"] = coerce_numeric(df["pick_price"]).round(0)
    else:
        mhml = coerce_numeric(series_or_default(df, "market_home_ml", np.nan))
        maml = coerce_numeric(series_or_default(df, "market_away_ml", np.nan))
        pick = series_or_default(df, "Pick Team", "")
        home = series_or_default(df, "Home Team", "")
        away = series_or_default(df, "Away Team", "")
        vals = []
        for p, h, a, hml, aml in zip(pick, home, away, mhml, maml):
            if isinstance(p, str) and isinstance(h, str) and isinstance(a, str):
                if p == h: vals.append(hml)
                elif p == a: vals.append(aml)
                else: vals.append(np.nan)
            else:
                vals.append(np.nan)
        df["Pick Price"] = pd.to_numeric(pd.Series(vals, index=df.index), errors="coerce").round(0)

    df["Market Home ML"] = coerce_numeric(df.get("market_home_ml", np.nan)).round(0)

    out = df[REQUIRED_EXPORT_COLS].copy()
    for c in ["Home Spread","Model Home Line","Model Line (pts)"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").apply(round_to_half)
    for c in ["Edge (pts)","ATS EV (%)","Units","Pick Price","Market Home ML","Home Cover %"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def run_model():
    completed = subprocess.run(
        [sys.executable, "-m", "models.predict_upcoming_plus", "--fetch"],
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
    )
    return completed

# ---------------- Controls: load/save ----------------
def _csv_map(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    d = {}
    for _, r in df.iterrows():
        d[str(r.get("key",""))] = r.get("value", "")
    return d

def _save_map(path: Path, d: dict):
    if not d:
        pd.DataFrame(columns=["key","value"]).to_csv(path, index=False)
        return
    pd.DataFrame([{"key":k, "value":v} for k, v in d.items()]).to_csv(path, index=False)

def sidebar_sliders():
    with st.sidebar:
        st.subheader("Adjustments")

        weights_map = _csv_map(WEIGHTS_CSV)
        thresh_map  = _csv_map(THRESH_CSV)
        toggles_map = _csv_map(TOGGLES_CSV)

        with st.form("controls_form", clear_on_submit=False):
            st.markdown("**Model Weights**")
            w_trench = st.slider("Trench weight", 0.0, 1.5, float(weights_map.get("w_trench", 0.40)), 0.01)
            w_qb     = st.slider("QB weight",     0.0, 1.5, float(weights_map.get("w_qb",     0.30)), 0.01)
            w_rest   = st.slider("Rest weight",   0.0, 1.5, float(weights_map.get("w_rest",   0.15)), 0.01)
            w_travel = st.slider("Travel weight", 0.0, 1.5, float(weights_map.get("w_travel", 0.05)), 0.01)
            w_hfa    = st.slider("HFA weight",    0.0, 1.5, float(weights_map.get("w_hfa",    0.05)), 0.01)
            w_inj    = st.slider("Injury cluster weight", 0.0, 1.5, float(weights_map.get("w_inj_cluster", 0.05)), 0.01)
            meta_bl  = st.slider("Meta blend",    0.0, 1.0, float(weights_map.get("meta_blend", 0.60)), 0.01)

            st.markdown("**Position Multipliers**")
            pos_qb   = st.slider("QB multiplier",  0.0, 2.0, float(weights_map.get("pos_qb",  1.0)), 0.05)
            pos_rb   = st.slider("RB multiplier",  0.0, 2.0, float(weights_map.get("pos_rb",  0.5)), 0.05)
            pos_wr   = st.slider("WR multiplier",  0.0, 2.0, float(weights_map.get("pos_wr",  0.6)), 0.05)
            pos_te   = st.slider("TE multiplier",  0.0, 2.0, float(weights_map.get("pos_te",  0.3)), 0.05)
            pos_cb   = st.slider("CB multiplier",  0.0, 2.0, float(weights_map.get("pos_cb",  0.4)), 0.05)
            pos_edge = st.slider("EDGE multiplier",0.0, 2.0, float(weights_map.get("pos_edge",0.5)), 0.05)
            pos_ol   = st.slider("OL multiplier",  0.0, 2.0, float(weights_map.get("pos_ol",  0.4)), 0.05)

            st.markdown("**Bet Sizing / Thresholds**")
            kelly    = st.slider("Kelly fraction", 0.0, 1.0, float(thresh_map.get("kelly_fraction", 0.25)), 0.01)
            maxu     = st.slider("Max units",      0.0, 10.0, float(thresh_map.get("max_units", 3.0)), 0.1)
            dflt_px  = st.number_input("Default spread price (American)", value=float(thresh_map.get("default_spread_price", -110.0)), step=1.0, format="%.0f")

            st.markdown("**EV / Edge Tiers**")
            strong_ev = st.slider("Strong EV threshold (%)", 0.0, 20.0, float(thresh_map.get("strong_ev_threshold", 3.0)*100.0), 0.1)
            strong_pts= st.slider("Strong edge threshold (pts)", 0.0, 10.0, float(thresh_map.get("strong_pts_threshold", 3.0)), 0.1)
            lean_ev   = st.slider("Lean EV threshold (%)",   0.0, 20.0, float(thresh_map.get("lean_ev_threshold",   1.5)*100.0), 0.1)
            lean_pts  = st.slider("Lean edge threshold (pts)",0.0, 10.0, float(thresh_map.get("lean_pts_threshold",  1.5)), 0.1)

            submitted = st.form_submit_button("Save Controls", use_container_width=True)
            if submitted:
                weights_out = {
                    "w_trench": w_trench, "w_qb": w_qb, "w_rest": w_rest, "w_travel": w_travel,
                    "w_hfa": w_hfa, "w_inj_cluster": w_inj, "meta_blend": meta_bl,
                    "pos_qb": pos_qb, "pos_rb": pos_rb, "pos_wr": pos_wr, "pos_te": pos_te,
                    "pos_cb": pos_cb, "pos_edge": pos_edge, "pos_ol": pos_ol
                }
                pd.DataFrame([{"key":k, "value":v} for k, v in weights_out.items()]).to_csv(WEIGHTS_CSV, index=False)

                thresh_out = {
                    "kelly_fraction": kelly,
                    "max_units": maxu,
                    "default_spread_price": dflt_px,
                    "strong_ev_threshold": strong_ev/100.0,
                    "strong_pts_threshold": strong_pts,
                    "lean_ev_threshold":   lean_ev/100.0,
                    "lean_pts_threshold":  lean_pts,
                }
                pd.DataFrame([{"key":k, "value":v} for k, v in thresh_out.items()]).to_csv(THRESH_CSV, index=False)

                if not TOGGLES_CSV.exists():
                    pd.DataFrame(columns=["key","value"]).to_csv(TOGGLES_CSV, index=False)

                st.success("Controls saved")
                st.cache_data.clear()

        st.divider()
        if st.button("Run / Refresh Predictions", use_container_width=True):
            with st.spinner("Running model..."):
                res = run_model()
            if res.returncode != 0:
                st.error("Model run failed. See details below.")
                with st.expander("Show stderr"):
                    st.code(res.stderr)
                with st.expander("Show stdout"):
                    st.code(res.stdout)
            else:
                st.cache_data.clear()
                st.success("Model completed. Reloading predictions...")
                st.rerun()

        if st.button("Reload Table Only", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ---------------- UI ----------------
st.title("Streamlit NFL Model — fresh session (kept context)")
sidebar_sliders()

# Load CSV and build display frame
raw_df = load_csv(PRED_CSV)
if raw_df.empty:
    st.info("No predictions found yet. Use the sidebar to **Save Controls** (if needed) and click **Run / Refresh Predictions**.")
    st.stop()

display_df = build_display_df(raw_df)

# Strict column_config
col_cfg = {
    "Home Team": st.column_config.TextColumn("Home Team", width="medium"),
    "Away Team": st.column_config.TextColumn("Away Team", width="medium"),
    "Kickoff":   st.column_config.TextColumn("Kickoff", width="medium"),
    "Home Spread":     st.column_config.NumberColumn("Home Spread", format="%.1f", width="small"),
    "Model Home Line": st.column_config.NumberColumn("Model Home Line", format="%.1f", width="small"),
    "Model Line (pts)":st.column_config.NumberColumn("Model Line (pts)", format="%.1f", width="small"),
    "Home Cover %": st.column_config.NumberColumn("Home Cover %", format="%.1f%%", width="small"),
    "ATS EV (%)":   st.column_config.NumberColumn("ATS EV (%)",   format="%.1f%%", width="small"),
    "Edge (pts)":     st.column_config.NumberColumn("Edge (pts)", format="%.1f", width="small"),
    "Units":          st.column_config.NumberColumn("Units",      format="%.1f", width="small"),
    "Pick Price":     st.column_config.NumberColumn("Pick Price", format="%.0f", width="small"),
    "Market Home ML": st.column_config.NumberColumn("Market Home ML", format="%.0f", width="small"),
    "Pick Tier": st.column_config.TextColumn("Pick Tier", width="small"),
    "Pick Team": st.column_config.TextColumn("Pick Team", width="medium"),
}

# Optional debug panel
with st.expander("Debug: data pipeline (temporary)"):
    st.write({"PRED_CSV": str(PRED_CSV)})
    try:
        import time
        exists = PRED_CSV.exists()
        size = PRED_CSV.stat().st_size if exists else 0
        mtime = time.ctime(PRED_CSV.stat().st_mtime) if exists else "n/a"
        st.write({"exists": exists, "size_bytes": size, "last_modified": mtime})
    except Exception as e:
        st.write({"stat_error": str(e)})
    st.write({"raw_df_shape": raw_df.shape})
    st.write({"raw_df_columns": list(raw_df.columns)})
    st.dataframe(raw_df.head(25), use_container_width=True, hide_index=True)

# Final table
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config=col_cfg,
)
