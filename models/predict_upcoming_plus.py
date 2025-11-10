# models/predict_upcoming_plus.py — FULL REPLACEMENT (v2.1 export-14-cols)

from __future__ import annotations

# --- BEGIN ROBUST PACKAGE/IMPORT HEADER ---
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MODELS_DIR = _THIS_FILE.parent
_REPO_ROOT = _MODELS_DIR.parent  # .../sports-lab

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from models.injuries_online import merge_online_injuries_into_games
except Exception as _abs_err:
    try:
        from .injuries_online import merge_online_injuries_into_games
    except Exception as _rel_err:
        raise ImportError(
            "Unable to import injuries module via absolute or relative paths.\n"
            f"Absolute import error: {type(_abs_err).__name__}: {_abs_err}\n"
            f"Relative import error: {type(_rel_err).__name__}: {_rel_err}\n"
            f"sys.executable={sys.executable}\n"
            f"repo_root={_REPO_ROOT}"
        )
# --- END ROBUST PACKAGE/IMPORT HEADER ---

import argparse
import numpy as np
import pandas as pd
import yaml

from models.fetch import fetch_odds_current_nfl, fetch_nfl_team_metrics, save_csv
from models.features import build_component_lines
from models.model import (
    apply_dynamic_component_weights,
    finalize_lines_and_probs,
    add_tiers_and_units,
    add_pick_labels,
)

# ----------------- constants & paths -----------------
ROOT = _REPO_ROOT
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

CFG_FILE = ROOT / "config.yaml"
TOGGLES = RUNS / "controls_toggles.csv"
THRESH  = RUNS / "controls_thresholds.csv"
WEIGHTS = RUNS / "controls_weights.csv"

BANNER = "[predict_upcoming_plus] dynamic components + EV tiering (export-14-cols)"

# ----------------- helpers -----------------
def _csv_to_map(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        k = str(r.get("key", "")).strip()
        v = r.get("value", "")
        try:
            v = float(v)
        except Exception:
            s = str(v).strip().lower()
            if s in ("true", "false"):
                v = (s == "true")
        if k:
            out[k] = v
    return out

def round_to_half(x):
    if pd.isna(x):
        return np.nan
    try:
        return np.round(float(x) * 2) / 2.0
    except Exception:
        return np.nan

def load_controls() -> dict:
    base = {}
    if CFG_FILE.exists():
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    togg = _csv_to_map(TOGGLES)
    thr  = _csv_to_map(THRESH)
    wts  = _csv_to_map(WEIGHTS)

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
        "lean_ev_threshold":   0.015,
        "lean_pts_threshold":  1.5,
    }

    params = {
        "ats_sigma": float(base.get("ats_sigma", 13.5)),
        "kelly_fraction": float(thr.get("kelly_fraction", base.get("kelly_fraction", 0.25))),
        "max_units": float(thr.get("max_units", base.get("max_units", 3.0))),
        "default_spread_price": float(thr.get("default_spread_price", base.get("default_spread_price", -110))),
        "tiers": DEFAULT_TIERS | (base.get("tiers", {}) or {}),
    }

    weights = (base.get("weights", {}) or {}).copy()
    for k, v in wts.items():
        try:
            weights[k] = float(v)
        except Exception:
            pass
    if not weights:
        weights = DEFAULT_WEIGHTS.copy()

    params["weights"] = weights
    return params

def build_market_now() -> pd.DataFrame:
    odds = fetch_odds_current_nfl().copy()
    for c in ["spread_home","spread_away","price_home_spread","price_away_spread","market_home_ml","market_away_ml"]:
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")

    if "spread_home" in odds.columns and "spread_away" in odds.columns:
        mask = odds["spread_home"].isna() & odds["spread_away"].notna()
        odds.loc[mask, "spread_home"] = -odds.loc[mask, "spread_away"]

    out = odds[[
        "game_id","commence_time","team_home","team_away","spread_home",
        "price_home_spread","price_away_spread","market_home_ml","market_away_ml"
    ]].copy()
    out.rename(columns={
        "price_home_spread":"market_home_spread_price",
        "price_away_spread":"market_away_spread_price",
    }, inplace=True)

    n_nan = int(out["spread_home"].isna().sum())
    print(f"[market] rows={len(out)}; spread_home NaN={n_nan}")
    return out

# ----------------- main pipeline -----------------
def main(fetch: bool):
    print(BANNER)
    params = load_controls()
    print("[params]", params)

    # ----- fetch or reuse -----
    if fetch:
        market = build_market_now()
        save_csv(market, RUNS/"_latest_odds.csv")
        seasons = [2023, 2024, 2025]
        metrics = fetch_nfl_team_metrics(seasons)
        save_csv(metrics, RUNS/"_metrics_team_rolling.csv")
    else:
        market = pd.read_csv(RUNS/"_latest_odds.csv")
        metrics = pd.read_csv(RUNS/"_metrics_team_rolling.csv")

    # ----- features & components -----
    df = build_component_lines(market, metrics)

    # --- injuries (guarded) ---
    try:
        df = merge_online_injuries_into_games(df)
        print("[injuries_online] merged injuries from nfl_data_py")
    except Exception as e:
        print("[injuries_online] failed, continuing without injuries:", e)

    # ----- weights -> model line -----
    df = apply_dynamic_component_weights(df, weights=params["weights"], baseline_col="model_home_line")

    # ----- probs from spread vs model line -----
    df = finalize_lines_and_probs(df, ats_sigma=float(params["ats_sigma"]))

    # ----- edges & signs -----
    df["model_line_pts"]  = -pd.to_numeric(df["model_home_line"], errors="coerce")
    df["market_line_pts"] = -pd.to_numeric(df["spread_home"], errors="coerce")
    df["value_pts_signed"] = df["model_line_pts"] - df["market_line_pts"]
    df["value_pts"] = df["value_pts_signed"].abs()
    df["value_side"] = np.where(df["value_pts_signed"] >= 0, "HOME", "AWAY")

    # ----- tiers & units -----
    t = params["tiers"]
    df = add_tiers_and_units(
        df,
        ats_sigma=float(params["ats_sigma"]),
        default_spread_price=float(params["default_spread_price"]),
        kelly_fraction=float(params["kelly_fraction"]),
        max_units=float(params["max_units"]),
        strong_ev_threshold=float(t["strong_ev_threshold"]),
        strong_pts_threshold=float(t["strong_pts_threshold"]),
        lean_ev_threshold=float(t["lean_ev_threshold"]),
        lean_pts_threshold=float(t["lean_pts_threshold"]),
    )

    # ----- pick labeling -----
    df = add_pick_labels(df)

    # ----- derive pick_price if missing -----
    if "pick_price" not in df.columns or pd.isna(df["pick_price"]).all():
        mhml = pd.to_numeric(df.get("market_home_ml", np.nan), errors="coerce")
        maml = pd.to_numeric(df.get("market_away_ml", np.nan), errors="coerce")
        pick = df.get("pick_team", df.get("pick", ""))
        home = df.get("team_home", "")
        away = df.get("team_away", "")
        vals = []
        for p, h, a, hml, aml in zip(pick, home, away, mhml, maml):
            if isinstance(p, str) and p == h:
                vals.append(hml)
            elif isinstance(p, str) and p == a:
                vals.append(aml)
            else:
                vals.append(np.nan)
        df["pick_price"] = pd.to_numeric(pd.Series(vals, index=df.index), errors="coerce")

    # ---------- FRIENDLY ROUNDING ----------
    for col in ["spread_home", "model_home_line", "model_line_pts", "market_line_pts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").apply(round_to_half)
    if "value_pts" in df.columns:
        df["value_pts"] = pd.to_numeric(df["value_pts"], errors="coerce").round(1)
    if "ats_ev" in df.columns:
        df["ats_ev"] = pd.to_numeric(df["ats_ev"], errors="coerce").round(1)
    if "units" in df.columns:
        df["units"] = pd.to_numeric(df["units"], errors="coerce").round(1)

    # ---------- BUILD FINAL 14-COLUMN EXPORT ----------
    # Kickoff formatted: "MMM-DD-YYYY, HH:MM"
    kickoff = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    kickoff_local = kickoff.dt.tz_localize(None)  # leave naive string
    kickoff_fmt = kickoff_local.dt.strftime("%b-%d-%Y, %H:%M")

    # Percents as 0-100 numbers (rounded to 1 dp for CSV)
    home_cover_pct = (pd.to_numeric(df.get("home_cover_prob", np.nan), errors="coerce") * 100.0)
    ats_ev_pct     = (pd.to_numeric(df.get("ats_ev", np.nan),          errors="coerce") * 100.0)

    # Pick team: prefer pick_team, fallback pick
    pick_team = df.get("pick_team", df.get("pick", ""))

    final = pd.DataFrame({
        "Home Team":        df.get("team_home", ""),
        "Away Team":        df.get("team_away", ""),
        "Kickoff":          kickoff_fmt.fillna(""),
        "Home Spread":      pd.to_numeric(df.get("spread_home", np.nan),      errors="coerce"),
        "Model Home Line":  pd.to_numeric(df.get("model_home_line", np.nan),  errors="coerce"),
        "Home Cover %":     pd.to_numeric(home_cover_pct,                      errors="coerce"),
        "Model Line (pts)": pd.to_numeric(df.get("model_line_pts", np.nan),   errors="coerce"),
        "Edge (pts)":       pd.to_numeric(df.get("value_pts", np.nan),        errors="coerce"),
        "ATS EV (%)":       pd.to_numeric(ats_ev_pct,                         errors="coerce"),
        "Units":            pd.to_numeric(df.get("units", np.nan),            errors="coerce"),
        "Pick Tier":        df.get("pick_tier", ""),
        "Pick Team":        pick_team,
        "Pick Price":       pd.to_numeric(df.get("pick_price", np.nan),       errors="coerce"),
        "Market Home ML":   pd.to_numeric(df.get("market_home_ml", np.nan),   errors="coerce"),
    })

    # ---- ENFORCE ROUNDING FOR CSV (numbers, not strings) ----
    # Spreads/lines snapped to 0.5 and shown as 1 decimal
    for c in ["Home Spread", "Model Home Line", "Model Line (pts)"]:
        final[c] = final[c].apply(lambda x: np.round(x * 2) / 2.0 if pd.notna(x) else np.nan).round(1)

    # Percents to 1 decimal (fixes 'many decimals' in CSV)
    final["Home Cover %"] = final["Home Cover %"].round(1)
    final["ATS EV (%)"]   = final["ATS EV (%)"].round(1)

    # Edge/Units to 1 decimal
    final["Edge (pts)"] = final["Edge (pts)"].round(1)
    final["Units"]      = final["Units"].round(1)

    # Moneyline integers
    final["Pick Price"]     = final["Pick Price"].round(0)
    final["Market Home ML"] = final["Market Home ML"].round(0)

    # Ensure numeric dtypes are preserved (not strings)
    for c in ["Home Spread","Model Home Line","Model Line (pts)","Edge (pts)","Home Cover %","ATS EV (%)",
              "Units","Pick Price","Market Home ML"]:
        final[c] = pd.to_numeric(final[c], errors="coerce")

    # Order + sort by kickoff asc, units desc
    final = final[[
        "Home Team","Away Team","Kickoff","Home Spread","Model Home Line","Home Cover %",
        "Model Line (pts)","Edge (pts)","ATS EV (%)","Units","Pick Tier","Pick Team","Pick Price","Market Home ML",
    ]].sort_values(["Kickoff","Units"], ascending=[True, False])

    # ---------- WRITE CSV (ONLY 14 COLS, ROUNDED) ----------
    out_path = RUNS / "upcoming_predictions_plus.csv"
    final.to_csv(out_path, index=False)
    print("[predict_upcoming_plus] final shape:", final.shape)
    try:
        print(final.head(3).to_string(index=False))
    except Exception:
        pass
    print(f"[predict_upcoming_plus] wrote {len(final)} rows -> {out_path}")

# ----------------- cli -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Fetch fresh odds/metrics before running")
    args = ap.parse_args()
    main(fetch=args.fetch)
