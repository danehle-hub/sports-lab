# models/model.py — FULL REPLACEMENT
# - Dynamic component blending
# - Position-based injury multipliers (called automatically)
# - EV tiers + Kelly units
# - Normal ATS probability
# - Team-facing pick labels

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import american_to_decimal, ev_from_prob_and_american

# ----------------------------------------------------------------------
# Position injury multipliers
# ----------------------------------------------------------------------
def apply_position_injury_adjustments(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Creates df['injury_adjust_pts'] from per-position '..._out' columns and sliders in Streamlit.

    Expected (optional) columns (0/1 flags or counts):
      home_qb_out, away_qb_out
      home_rb_out, away_rb_out
      home_wr_out, away_wr_out
      home_te_out, away_te_out
      home_cb_out, away_cb_out
      home_edge_out, away_edge_out
      home_ol_out, away_ol_out

    Multipliers come from weights:
      pos_qb, pos_rb, pos_wr, pos_te, pos_cb, pos_edge, pos_ol

    Positive injury_adjust_pts = makes home LESS favored (moves line toward away).
    If columns are missing, adjustment = 0.
    """
    df = df.copy()

    def _col(side: str, pos: str):
        name = f"{side}_{pos}_out"
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
        return pd.Series(0.0, index=df.index, dtype=float)

    w = lambda k, d=0.0: float(weights.get(k, d))

    home_pen = (
        _col("home", "qb")   * w("pos_qb")   +
        _col("home", "rb")   * w("pos_rb")   +
        _col("home", "wr")   * w("pos_wr")   +
        _col("home", "te")   * w("pos_te")   +
        _col("home", "cb")   * w("pos_cb")   +
        _col("home", "edge") * w("pos_edge") +
        _col("home", "ol")   * w("pos_ol")
    )
    away_pen = (
        _col("away", "qb")   * w("pos_qb")   +
        _col("away", "rb")   * w("pos_rb")   +
        _col("away", "wr")   * w("pos_wr")   +
        _col("away", "te")   * w("pos_te")   +
        _col("away", "cb")   * w("pos_cb")   +
        _col("away", "edge") * w("pos_edge") +
        _col("away", "ol")   * w("pos_ol")
    )

    # Net effect expressed as home-line points
    df["injury_adjust_pts"] = (home_pen - away_pen).astype(float)
    return df


# ----------------------------------------------------------------------
# Component blending + calibration + shrink + injury call
# ----------------------------------------------------------------------
def apply_dynamic_component_weights(
    df: pd.DataFrame,
    weights: dict,
    baseline_col: str = "model_home_line",
    meta_blend: float = None,
    hard_cap: float = 14.0
) -> pd.DataFrame:
    """
    1) Blend component lines into raw model signal (points).
    2) Calibrate to spread space with OLS slope on (spread_home ~ raw).
    3) Apply meta_blend shrink toward zero.
    4) Hard-cap extremes.
    5) Apply per-position injury adjustments (called automatically here).
       Uses weights['w_inj_cluster'] to scale position-based penalty.

    Requires component columns (missing → 0):
      trench_line, qb_line, rest_line, travel_line, hfa_line, inj_cluster_line (optional)
    """
    df = df.copy()

    # Defaults
    if meta_blend is None:
        meta_blend = float(weights.get("meta_blend", 0.60))

    # Component weights
    w_trench      = float(weights.get("w_trench", 0.40))
    w_qb          = float(weights.get("w_qb", 0.30))
    w_rest        = float(weights.get("w_rest", 0.15))
    w_travel      = float(weights.get("w_travel", 0.05))
    w_hfa         = float(weights.get("w_hfa", 0.05))
    w_inj_cluster = float(weights.get("w_inj_cluster", 0.05))  # also scales position injuries

    # Safe getters for components
    def comp(name: str):
        return (
            pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
            if name in df.columns else
            pd.Series(0.0, index=df.index, dtype=float)
        )

    trench = comp("trench_line")
    qb     = comp("qb_line")
    rest   = comp("rest_line")
    travel = comp("travel_line")
    hfa    = comp("hfa_line")
    injc   = comp("inj_cluster_line")  # optional

    # 1) raw blend
    raw = (
        w_trench * trench +
        w_qb     * qb     +
        w_rest   * rest   +
        w_travel * travel +
        w_hfa    * hfa    +
        w_inj_cluster * injc
    ).astype(float)

    # 2) calibration slope (spread_home ~ raw)
    spread = pd.to_numeric(df.get("spread_home", np.nan), errors="coerce")
    valid = spread.notna() & raw.notna()
    if valid.sum() >= 3:
        X = raw[valid].values
        y = spread[valid].values
        denom = float(np.dot(X, X))
        slope = float(np.dot(X, y) / denom) if denom != 0.0 else 1.0
    else:
        slope = 1.0
    scaled = raw * slope

    # 3) shrink
    model_line = meta_blend * scaled

    # 4) cap
    model_line = np.clip(model_line, -hard_cap, +hard_cap)

    # Write into df
    df[baseline_col] = model_line.astype(float)

    # 5) APPLY POSITION INJURIES
    df = apply_position_injury_adjustments(df, weights)
    inj_scale = float(weights.get("w_inj_cluster", 1.0))  # reuse injury cluster slider
    df[baseline_col] = df[baseline_col].astype(float) + df["injury_adjust_pts"].astype(float) * inj_scale

    return df


# ----------------------------------------------------------------------
# EV-based tiering + Kelly units
# ----------------------------------------------------------------------
def add_tiers_and_units(
    df: pd.DataFrame,
    ats_sigma: float,
    default_spread_price: float = -110.0,
    kelly_fraction: float = 0.25,
    max_units: float = 3.0,
    strong_ev_threshold: float = 0.03,
    strong_pts_threshold: float = 2.5,
    lean_ev_threshold: float = 0.0125,
    lean_pts_threshold: float = 1.2,
) -> pd.DataFrame:
    """
    Tiering & unit sizing driven by BOTH price-adjusted EV and point edge.

    Tiers:
      STRONG if (value_pts >= strong_pts_threshold AND roi >= strong_ev_threshold)
      LEAN   if (value_pts >= lean_pts_threshold   AND roi >= lean_ev_threshold)
      else PASS

    Units via fractional Kelly on spread odds:
      f* = (b*p - q) / b; stake_units = max_units * max(0, kelly_fraction * f*)
      with b = (decimal-1), p = side cover prob, q = 1-p
      Unit floors: STRONG ≥ 0.50 u, LEAN ≥ 0.20 u
    """
    out = df.copy()

    # Market prices (may be NaN)
    home_price = pd.to_numeric(out.get("market_home_spread_price", np.nan), errors="coerce")
    away_price = pd.to_numeric(out.get("market_away_spread_price", np.nan), errors="coerce")

    # Cover probabilities from finalize_lines_and_probs
    p_home = pd.to_numeric(out.get("home_cover_prob", 0.5), errors="coerce").fillna(0.5)
    p_away = 1.0 - p_home

    # Which side are we backing?
    side_is_home = (out.get("value_side", pd.Series(["HOME"]*len(out))) == "HOME")
    p_side = np.where(side_is_home, p_home, p_away)

    # Use the correct price for that side; fallback to default if missing
    price_side = np.where(side_is_home, home_price, away_price)
    price_side = np.where(np.isfinite(price_side), price_side, default_spread_price)

    # EV / ROI per 1 unit risked (fraction, e.g., 0.028 = 2.8%)
    roi = ev_from_prob_and_american(p_side, price_side)  # vectorized
    out["roi"] = roi.astype(float)
    out["ats_ev"] = roi.astype(float)

    # Tier rules
    pts = pd.to_numeric(out.get("value_pts", 0.0), errors="coerce").fillna(0.0)
    is_strong = (pts >= strong_pts_threshold) & (roi >= strong_ev_threshold)
    is_lean   = (~is_strong) & (pts >= lean_pts_threshold) & (roi >= lean_ev_threshold)
    pick_tier = np.where(is_strong, "STRONG", np.where(is_lean, "LEAN", "PASS"))
    out["pick_tier"] = pick_tier

    # Kelly sizing
    dec = american_to_decimal(price_side)
    b = np.maximum(dec - 1.0, 1e-9)
    q = 1.0 - p_side
    f_star = (b * p_side - q) / b
    f_star = np.clip(f_star, 0.0, 1.0)

    units_raw = max_units * (kelly_fraction * f_star)

    # Unit floors by tier
    unit_floor_lean = 0.20
    unit_floor_strong = 0.50
    units = np.where(
        pick_tier == "STRONG",
        np.maximum(units_raw, unit_floor_strong),
        np.where(pick_tier == "LEAN", np.maximum(units_raw, unit_floor_lean), 0.0)
    )
    out["units"] = np.round(np.clip(units, 0.0, max_units), 2)

    return out


# ----------------------------------------------------------------------
# Normal ATS probability (spread vs model line)
# ----------------------------------------------------------------------
def finalize_lines_and_probs(df: pd.DataFrame, ats_sigma: float) -> pd.DataFrame:
    """
    Compute home_cover_prob from market spread vs model_home_line using a Normal ATS model.
    Uses scipy.special.erf if available; falls back to math.erf vectorized.
    """
    out = df.copy()

    try:
        from scipy.special import erf as _erf
    except Exception:
        from math import erf as _erf  # scalar; we vectorize below

    spread = pd.to_numeric(out.get("spread_home", np.nan), errors="coerce")
    model  = pd.to_numeric(out.get("model_home_line", np.nan), errors="coerce")

    # If either side missing, default to coin flip
    z = (spread - model) / float(ats_sigma)
    erf_vec = np.vectorize(_erf, otypes=[float])
    out["home_cover_prob"] = 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))
    out["home_cover_prob"] = out["home_cover_prob"].fillna(0.5).clip(0.0, 1.0)
    out["model_away_win_prob"] = 1.0 - out["home_cover_prob"]

    return out


# ----------------------------------------------------------------------
# Value side/points helper
# ----------------------------------------------------------------------
def compute_value_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires df['market_line_pts'] and df['model_line_pts'] (signed from HOME POV).
    Produces:
      - value_side: "HOME" if model >= market, else "AWAY"
      - value_pts: absolute point difference
    """
    out = df.copy()
    model_pts = pd.to_numeric(out.get("model_line_pts", np.nan), errors="coerce")
    market_pts = pd.to_numeric(out.get("market_line_pts", np.nan), errors="coerce")
    diff = (model_pts - market_pts)
    out["value_side"] = np.where(diff >= 0.0, "HOME", "AWAY")
    out["value_pts"] = np.abs(diff)
    return out


# ----------------------------------------------------------------------
# Team-facing pick labels (single definition)
# ----------------------------------------------------------------------
def add_pick_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts HOME/AWAY picks into real team names and readable strings.
    Creates:
      - pick_team   → actual team to bet
      - pick_spread → signed spread from that team's POV (numeric)
      - pick_price  → selected side's spread price (numeric)
      - pick_string → "TeamName -3.5 (-110)"
    Assumes:
      - spread_home signed from HOME POV (home -2.5 = home favored by 2.5)
    """
    out = df.copy()

    spread_home = pd.to_numeric(out.get("spread_home", np.nan), errors="coerce")
    price_home  = pd.to_numeric(out.get("market_home_spread_price", np.nan), errors="coerce")
    price_away  = pd.to_numeric(out.get("market_away_spread_price", np.nan), errors="coerce")

    side_home = (out.get("value_side", pd.Series(["HOME"]*len(out))) == "HOME")

    out["pick_team"]   = np.where(side_home, out.get("team_home"), out.get("team_away"))
    out["pick_spread"] = np.where(side_home, spread_home, -spread_home)
    out["pick_price"]  = np.where(side_home, price_home, price_away)

    # Display formatting
    # handle NaN spreads gracefully
    ps = pd.to_numeric(out["pick_spread"], errors="coerce")
    disp_spread = np.where(
        np.isfinite(ps),
        np.where(ps > 0, "+" + np.round(ps, 1).astype(str),
                 np.round(ps, 1).astype(str)),
        "N/A"
    )
    disp_price = np.where(np.isfinite(out["pick_price"]),
                          out["pick_price"].astype(int).astype(str),
                          "N/A")

    out["pick_string"] = out["pick_team"].astype(str) + " " + disp_spread + " (" + disp_price + ")"
    return out


# ----------------------------------------------------------------------
# YAML config loader
# ----------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
