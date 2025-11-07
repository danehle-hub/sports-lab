# models/backtest.py — NEW FILE
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# Controls (same files Streamlit edits)
TOGGLES_CSV = RUNS / "controls_toggles.csv"
THRESH_CSV  = RUNS / "controls_thresholds.csv"
WEIGHTS_CSV = RUNS / "controls_weights.csv"

# Backtest IO
BT_INPUT_CSV    = RUNS / "backtest_input.csv"
BT_RESULTS_CSV  = RUNS / "backtest_results.csv"
BT_SUMMARY_JSON = RUNS / "backtest_summary.json"

# Import your existing model utilities
from models.features import build_component_lines
from models.model import (
    apply_dynamic_component_weights,
    finalize_lines_and_probs,
    add_tiers_and_units,
)

def _read_controls():
    def _safe_read(path, defaults=None):
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return defaults
    toggles_df    = _safe_read(TOGGLES_CSV, pd.DataFrame([("enable_unit_floors","",True,"")], columns=["key","label","value","help"]))
    thresholds_df = _safe_read(THRESH_CSV,  pd.DataFrame([("kelly_fraction","",0.25,0,1,0.01,"")], columns=["key","label","value","min","max","step","help"]))
    weights_df    = _safe_read(WEIGHTS_CSV, pd.DataFrame([("meta_blend","",0.60,0,1,0.01,"")], columns=["key","label","value","min","max","step","help"]))
    toggles    = {r.key: (bool(r.value) if str(r.value).lower() in ("true","false") else bool(r.value)) for _, r in toggles_df.iterrows()}
    thresholds = {r.key: float(r.value) for _, r in thresholds_df.iterrows()}
    weights    = {r.key: float(r.value) for _, r in weights_df.iterrows()}
    return toggles, thresholds, weights

def _parse_seasons(s: str):
    s = (s or "").replace(" ", "")
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return list(range(min(a,b), max(a,b)+1))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out

def build_market_line(df: pd.DataFrame, line_source: str) -> pd.Series:
    """
    Select which spread column to use as 'market spread' for backtest.
    Falls back to 'spread_home' if specific source not present.
    """
    line_source = (line_source or "closing").lower()
    cand_cols = {
        "closing": ["spread_home_closing", "spread_home"],
        "open":    ["spread_home_open",    "spread_home"],
        "best_of": ["spread_home_best_of", "spread_home"],
    }.get(line_source, ["spread_home"])
    for c in cand_cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.to_numeric(df.get("spread_home", np.nan), errors="coerce")

def compute_outcomes(bt: pd.DataFrame) -> pd.DataFrame:
    """
    Determine ATS result (1 win / 0.5 push / 0 loss) of the pick selected by tiers.
    We assume the pick side is encoded in 'value_side' ('HOME'/'AWAY') and we lay/take the market_line_pts.
    """
    # Final margin vs spread from the HOME perspective
    # Home covers if (home_score - away_score + market_line_pts) > 0; ==0 push; <0 loss
    margin = (pd.to_numeric(bt["home_score"], errors="coerce")
              - pd.to_numeric(bt["away_score"], errors="coerce")
              + pd.to_numeric(bt["market_line_pts"], errors="coerce"))
    # Make boolean for the direction the bet takes:
    bet_home = (bt["value_side"] == "HOME")
    # If bet is HOME, we need home to cover (>0). If bet is AWAY, away cover means home fails to cover (<0).
    cover = np.where(bet_home, margin > 0, margin < 0).astype(float)
    push  = (margin == 0).astype(float)

    # Win=1, Push=0.5, Loss=0
    result = cover * 1.0 + push * 0.5
    bt["result"] = result

    # Simple CLV proxy: model vs market points on bet side (positive means beat the close)
    # If betting HOME, model_line_pts should be <= market_line_pts for value (more negative number indicates fav stronger).
    # For simplicity we define clv_pts = |market| - |model| on the selected side’s sign convention.
    mlp = pd.to_numeric(bt["model_line_pts"], errors="coerce")
    mkp = pd.to_numeric(bt["market_line_pts"], errors="coerce")
    clv = np.abs(mkp) - np.abs(mlp)
    bt["clv_pts"] = clv
    return bt

def summarize(bt: pd.DataFrame) -> dict:
    out = {}
    placed = bt[bt["pick_tier"].isin(["LEAN","STRONG"])].copy()
    out["bets"] = int(len(placed))
    if out["bets"] == 0:
        return out
    out["win_rate"] = float(np.nanmean(placed["result"]))  # 1, 0.5, 0
    # ROI = average of 'roi' on placed bets
    out["roi"] = float(np.nanmean(pd.to_numeric(placed.get("roi", np.nan), errors="coerce")))
    out["avg_clv_pts"] = float(np.nanmean(pd.to_numeric(placed.get("clv_pts", np.nan), errors="coerce")))
    # Very simple equity curve & max drawdown from unit * roi approximation
    units = pd.to_numeric(placed.get("units", 0.0), errors="coerce").fillna(0.0).values
    rois  = pd.to_numeric(placed.get("roi", 0.0), errors="coerce").fillna(0.0).values
    # bankroll pct change per bet (approx): stake * roi; cumprod on (1 + stake*roi)
    curve = np.cumprod(1.0 + (units / max(units.max(), 1.0)) * rois)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    out["max_drawdown"] = float(np.min(dd)) if dd.size else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=str, default="2022-2024",
                    help="e.g. '2018-2024' or '2022,2023,2024'")
    ap.add_argument("--line_source", type=str, default="closing", choices=["closing","open","best_of"])
    ap.add_argument("--only_tiered", action="store_true", default=False,
                    help="If set, only count bets where pick_tier is LEAN/STRONG.")
    ap.add_argument("--fetch", action="store_true", default=False,
                    help="(Optional) If you wire metrics fetching here later.")
    args = ap.parse_args()

    if not BT_INPUT_CSV.exists():
        print(f"[backtest] Missing input CSV: {BT_INPUT_CSV}")
        return

    toggles, thresholds, weights = _read_controls()
    seasons = _parse_seasons(args.seasons)

    raw = pd.read_csv(BT_INPUT_CSV)
    # Filter by seasons if present
    if "season" in raw.columns and seasons:
        raw = raw[raw["season"].isin(seasons)].copy()

    # Choose which spread to test against (closing/open/best_of)
    raw["market_line_pts"] = build_market_line(raw, args.line_source)

    # Build features + model line/prob just like the picks pipeline
    gcols_needed = ["team_home","team_away","spread_home","market_line_pts","commence_time"]
    for c in gcols_needed:
        if c not in raw.columns:
            raw[c] = np.nan
    games = raw.rename(columns={"spread_home":"spread_home"})  # ensure column exists

    # Component lines
    feats = build_component_lines(games, metrics=pd.DataFrame())  # metrics optional; features.py fills means

    # Apply model blend (weights, meta_blend inside)
    mod  = apply_dynamic_component_weights(feats, weights, baseline_col="model_home_line")

    # Convert to probabilities
    mod  = finalize_lines_and_probs(mod, ats_sigma=float(thresholds.get("ats_sigma", 13.5)) if "ats_sigma" in thresholds else 13.5)

    # Compute edges vs chosen market line
    mod["model_line_pts"]  = -pd.to_numeric(mod["model_home_line"], errors="coerce")
    mod["market_line_pts"] = pd.to_numeric(mod["market_line_pts"], errors="coerce")
    mod["value_pts"] = np.abs(mod["market_line_pts"] - mod["model_line_pts"])
    mod["value_side"] = np.where(mod["model_line_pts"] < mod["market_line_pts"], "HOME", "AWAY")

    # Add tiers & units (uses thresholds & pricing)
    # If prices not present, function will fall back to default_spread_price from thresholds
    mod = add_tiers_and_units(
        mod,
        ats_sigma=float(thresholds.get("ats_sigma", 13.5)) if "ats_sigma" in thresholds else 13.5,
        default_spread_price=float(thresholds.get("default_spread_price", -110.0)),
        kelly_fraction=float(thresholds.get("kelly_fraction", 0.25)),
        max_units=float(thresholds.get("max_units", 3.0)),
        strong_ev_threshold=float(thresholds.get("strong_ev_threshold", 0.03)),
        strong_pts_threshold=float(thresholds.get("strong_pts_threshold", 3.0)),
        lean_ev_threshold=float(thresholds.get("lean_ev_threshold", 0.0125)),
        lean_pts_threshold=float(thresholds.get("lean_pts_threshold", 1.5)),
    )

    # Optionally keep only tiered bets
    if args.only_tiered:
        mod = mod[mod["pick_tier"].isin(["LEAN","STRONG"])].copy()

    # Merge back final scores to compute results & CLV
    needed_cols = ["home_score","away_score","market_line_pts"]
    for c in needed_cols:
        if c not in raw.columns:
            raw[c] = np.nan
    mod = mod.merge(raw[["team_home","team_away","season","week","home_score","away_score","commence_time","market_line_pts"]], 
                    on=["team_home","team_away","commence_time","market_line_pts"], how="left")

    # Outcomes & CLV
    mod = compute_outcomes(mod)

    # Save artifacts
    BT_RESULTS_CSV.write_text("")  # clear if needed
    mod.to_csv(BT_RESULTS_CSV, index=False)

    summary = summarize(mod)
    BT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[backtest] rows={len(mod)} bets={summary.get('bets',0)} roi={summary.get('roi',0):.4f} win%={summary.get('win_rate',0):.3f}")

if __name__ == "__main__":
    main()
