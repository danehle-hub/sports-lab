import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import american_to_decimal, ev_from_prob_and_american, norm_cdf_np

def apply_dynamic_component_weights(
    df: pd.DataFrame,
    weights: dict,
    baseline_col: str = "model_home_line",
    meta_blend: float = None,
    hard_cap: float = 14.0
) -> pd.DataFrame:
    """
    1) Blend component lines into raw model signal (points).
    2) Fit linear regression: spread_home ~ raw
       → use slope to scale model into spread space.
    3) Apply meta_blend shrink toward zero.
    4) Hard cap to avoid extreme outliers.
    """
    import numpy as np
    import pandas as pd

    # Extract weights with safe defaults
    w_trench      = float(weights.get("w_trench", 0.40))
    w_qb          = float(weights.get("w_qb", 0.30))
    w_rest        = float(weights.get("w_rest", 0.15))
    w_travel      = float(weights.get("w_travel", 0.05))
    w_hfa         = float(weights.get("w_hfa", 0.05))
    w_inj_cluster = float(weights.get("w_inj_cluster", 0.05))
    if meta_blend is None:
        meta_blend = float(weights.get("meta_blend", 0.60))

    # ---- Step 1: Raw blend ----
    raw = (
        w_trench      * df["trench_line"]
      + w_qb          * df["qb_line"]
      + w_rest        * df["rest_line"]
      + w_travel      * df["travel_line"]
      + w_hfa         * df["hfa_line"]
      + w_inj_cluster * df["inj_cluster_line"]
    ).astype(float)

    # ---- Step 2: Linear regression calibration ----
    spread = pd.to_numeric(df["spread_home"], errors="coerce")
    valid = spread.notna() & raw.notna()

    if valid.sum() >= 3:
        X = raw[valid].values
        y = spread[valid].values
        slope = np.dot(X, y) / np.dot(X, X) if np.dot(X, X) != 0 else 1.0
    else:
        slope = 1.0

    scaled = raw * slope

    # ---- Step 3: Conservative shrink ----
    model_line = meta_blend * scaled

    # ---- Step 4: Hard cap safety ----
    model_line = np.clip(model_line, -hard_cap, +hard_cap)

    df[baseline_col] = model_line
    return df



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

    - We back the side indicated by value_side.
    - p_side = cover prob for that side (home: p; away: 1-p).
    - price_side from market columns, fallback to default_spread_price when NaN.
    - ROI/EV computed from p_side and price_side.
    - Tiers:
        STRONG if (value_pts >= strong_pts_threshold AND roi >= strong_ev_threshold)
        LEAN   if (value_pts >= lean_pts_threshold   AND roi >= lean_ev_threshold)
        else PASS
    - Units via fractional Kelly on *spread odds*:
        f* = (b*p - q) / b with b = (decimal-1), p = p_side, q = 1-p_side
        stake_units = max_units * max(0, kelly_fraction * f*)
      Then apply minimum unit floors by tier (0.20 for LEAN, 0.50 for STRONG).
    """

    # Choose price for the side we’re betting
    home_price = pd.to_numeric(df.get("market_home_spread_price", np.nan), errors="coerce")
    away_price = pd.to_numeric(df.get("market_away_spread_price", np.nan), errors="coerce")

    # Side probability to cover spread
    p_home = pd.to_numeric(df["home_cover_prob"], errors="coerce")
    p_away = 1.0 - p_home

    side_is_home = (df["value_side"] == "HOME")

    # Probability and price for the selected side
    p_side = np.where(side_is_home, p_home, p_away)
    price_side = np.where(side_is_home, home_price, away_price)

    # Fallback on missing juice
    price_side = np.where(np.isfinite(price_side), price_side, default_spread_price)

    # ROI per $ risked (expected value)
    roi = ev_from_prob_and_american(p_side, price_side)  # vectorized

    # --- Tier rules (AND across EV & points) ---
    pts = pd.to_numeric(df["value_pts"], errors="coerce").fillna(0.0)
    is_strong = (pts >= strong_pts_threshold) & (roi >= strong_ev_threshold)
    is_lean   = (~is_strong) & (pts >= lean_pts_threshold) & (roi >= lean_ev_threshold)

    pick_tier = np.where(is_strong, "STRONG", np.where(is_lean, "LEAN", "PASS"))

    # --- Kelly sizing (conservative) ---
    # b = decimal - 1
    dec = american_to_decimal(price_side)
    b = np.maximum(dec - 1.0, 1e-9)
    q = 1.0 - p_side
    f_star = (b * p_side - q) / b
    f_star = np.clip(f_star, 0.0, 1.0)  # no negative bets; cap at 100% of bankroll

    # Scale to unit system
    units_raw = max_units * (kelly_fraction * f_star)

    # Unit floors by tier (optional but helpful)
    unit_floor_lean = 0.20
    unit_floor_strong = 0.50
    units = np.where(
        pick_tier == "STRONG",
        np.maximum(units_raw, unit_floor_strong),
        np.where(pick_tier == "LEAN", np.maximum(units_raw, unit_floor_lean), 0.0)
    )
    units = np.clip(units, 0.0, max_units)

    out = df.copy()
    out["roi"] = roi
    out["units"] = np.round(units, 2)
    out["pick_tier"] = pick_tier
    return out


def finalize_lines_and_probs(df: pd.DataFrame, ats_sigma: float) -> pd.DataFrame:
    """
    Compute home_cover_prob from market spread vs model_home_line using a Normal ATS model.
    Uses scipy.special.erf if available; falls back to math.erf vectorized.
    """
    import numpy as np
    try:
        from scipy.special import erf as _erf
    except Exception:
        from math import erf as _erf  # scalar erf; we’ll vectorize below

    # z-score
    z = (df["spread_home"].astype(float) - df["model_home_line"].astype(float)) / float(ats_sigma)

    # vectorized erf (works for both scipy and math fallback)
    erf_vec = np.vectorize(_erf, otypes=[float])
    df["home_cover_prob"] = 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))

    # keep symmetry column if you want it
    df["model_away_win_prob"] = 1.0 - df["home_cover_prob"]
    return df


def compute_value_side(df: pd.DataFrame) -> pd.DataFrame:
    # Choose side where model disagrees with market by points
    df = df.copy()
    df["value_pts"] = (df["model_line_pts"].astype(float) - df["market_line_pts"].astype(float))
    df["value_side"] = np.where(df["value_pts"]>=0, "HOME", "AWAY")
    df["value_pts"] = np.abs(df["value_pts"])
    return df

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
