# models/predict_upcoming_plus.py  (DIAGNOSTIC VERSION)
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from models.fetch import fetch_odds_current_nfl, fetch_nfl_team_metrics, save_csv
from models.features import build_component_lines
from models.model import (
    apply_dynamic_component_weights,
    finalize_lines_and_probs,
    add_tiers_and_units,
)
# ----------------- constants & paths -----------------
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

CFG_FILE = ROOT / "config.yaml"
TOGGLES = RUNS / "controls_toggles.csv"
THRESH  = RUNS / "controls_thresholds.csv"
WEIGHTS = RUNS / "controls_weights.csv"

BANNER = "[predict_upcoming_plus NEW v1.2] dynamic components + EV tiering"

def load_controls() -> dict:
    import yaml, pandas as pd
    base = {}
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    def _csv_map(path: Path):
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            v = r["value"]
            try:
                v = float(v)
            except Exception:
                if str(v).strip().lower() in ("true", "false"):
                    v = (str(v).strip().lower() == "true")
            out[str(r["key"])] = v
        return out

    togg = _csv_map(TOGGLES)
    thr  = _csv_map(THRESH)
    wts  = _csv_map(WEIGHTS)

    # -------- defaults (used if nothing provided) --------
    DEFAULT_WEIGHTS = {
        "w_trench": 0.40,
        "w_qb": 0.30,
        "w_rest": 0.15,
        "w_travel": 0.05,
        "w_hfa": 0.05,
        "w_inj_cluster": 0.05,
        "meta_blend": 0.60,
    }
    DEFAULT_TIERS = {
        "strong_ev_threshold": 0.03,
        "strong_pts_threshold": 3.0,
        "lean_ev_threshold": 0.015,
        "lean_pts_threshold": 1.5,
    }

    params = {
        "ats_sigma": float(base.get("ats_sigma", 13.5)),
        "kelly_fraction": float(thr.get("kelly_fraction", base.get("kelly_fraction", 0.25))),
        "max_units": float(thr.get("max_units", base.get("max_units", 3.0))),
        "default_spread_price": float(thr.get("default_spread_price", base.get("default_spread_price", -110))),
        "tiers": DEFAULT_TIERS | (base.get("tiers", {}) or {})  # start with defaults, overlay config
    }

    # Build weights from config.yaml first, then overlay UI controls
    weights = (base.get("weights", {}) or {}).copy()
    for k, v in wts.items():
        if k.startswith("w_") or k == "meta_blend":
            try:
                weights[k] = float(v)
            except Exception:
                pass
    # If still empty, use hard defaults
    if not weights:
        weights = DEFAULT_WEIGHTS.copy()

    params["weights"] = weights
    return params


def build_market_now() -> pd.DataFrame:
    odds = fetch_odds_current_nfl().copy()
    # Make sure spreads are numeric
    for c in ["spread_home","spread_away","price_home_spread","price_away_spread","market_home_ml","market_away_ml"]:
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")

    # Fallbacks: if spread_home is NaN but spread_away present, infer
    if "spread_home" in odds.columns and "spread_away" in odds.columns:
        mask = odds["spread_home"].isna() & odds["spread_away"].notna()
        odds.loc[mask, "spread_home"] = -odds.loc[mask, "spread_away"]

    # Minimal columns required downstream
    out = odds[["game_id","commence_time","team_home","team_away","spread_home",
                "price_home_spread","price_away_spread","market_home_ml","market_away_ml"]].copy()
    out.rename(columns={
        "price_home_spread":"market_home_spread_price",
        "price_away_spread":"market_away_spread_price",
    }, inplace=True)

    # Helpful diagnostics
    n_nan = int(out["spread_home"].isna().sum())
    print(f"[market] rows={len(out)}; spread_home NaN={n_nan}")
    if n_nan == len(out):
        print("[WARN] spread_home is NaN for all rows — check fetch or book parsing.")
    return out

def main(fetch: bool):
    print(BANNER)
    params = load_controls()
    print("[params]", params)

    # ----- fetch or reuse -----
    if fetch:
        market = build_market_now()
        save_csv(market, RUNS/"_latest_odds.csv")
        # Team rolling metrics: use last 3 seasons
        seasons = [2023, 2024, 2025]
        metrics = fetch_nfl_team_metrics(seasons)
        save_csv(metrics, RUNS/"_metrics_team_rolling.csv")
    else:
        market = pd.read_csv(RUNS/"_latest_odds.csv")
        metrics = pd.read_csv(RUNS/"_metrics_team_rolling.csv")

    # ----- build features & components -----
    out = build_component_lines(market, metrics)

    # ----- dynamic weights -> final model line -----
    out = apply_dynamic_component_weights(out, weights=params["weights"], baseline_col="model_home_line")

    # ----- prob from spread vs model line -----
    out = finalize_lines_and_probs(out, ats_sigma=float(params["ats_sigma"]))

    # ----- value calc (aligned signs) -----
    out["model_line_pts"] = -out["model_home_line"]
    out["market_line_pts"] = -out["spread_home"]
    out["value_pts_signed"] = out["model_line_pts"] - out["market_line_pts"]
    out["value_pts"] = out["value_pts_signed"].abs()
    out["value_side"] = np.where(out["value_pts_signed"] >= 0, "HOME", "AWAY")

    # ----- tiering & units -----
    t = params["tiers"]
    out = add_tiers_and_units(
        out,
        ats_sigma=float(params["ats_sigma"]),
        default_spread_price=float(params["default_spread_price"]),
        kelly_fraction=float(params["kelly_fraction"]),
        max_units=float(params["max_units"]),
        strong_ev_threshold=float(t["strong_ev_threshold"]),
        strong_pts_threshold=float(t["strong_pts_threshold"]),
        lean_ev_threshold=float(t["lean_ev_threshold"]),
        lean_pts_threshold=float(t["lean_pts_threshold"]),
    )
    out["ats_ev"] = (out["roi"] * 100).round(2)

    # Minimal export
    cols = [c for c in [
        "team_home","team_away","commence_time",
        "spread_home","model_home_line","home_cover_prob",
        "market_line_pts","model_line_pts",
        "value_side","value_pts","ats_ev","units","pick_tier",
        "market_home_spread_price","market_away_spread_price",
        "market_home_ml","market_away_ml"
    ] if c in out.columns]
    out = out[cols].sort_values("commence_time")

    out.to_csv(RUNS/"upcoming_predictions_plus.csv", index=False)
    print("[ok] wrote runs/upcoming_predictions_plus.csv")
    # Quick peek
    print(out.head(8).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    main(fetch=args.fetch)
