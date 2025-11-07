# models/build_trench_scores.py
# -----------------------------------------------------------------------------
# Build data-driven Offensive Line (OL) and Defensive Front (DL) scores
# from public play-by-play using nfl_data_py (nflfastR schema).
#
# Features:
# - Works across seasons with minor schema drift (pressure/qb_hit/sack flags)
# - Uses only pass plays for trench signal (where protection/pressure matter most)
# - Winsorizes outliers within each season to reduce skew
# - Per-season normalization (min-max) OR rank-based normalization (toggle)
# - Season-to-season smoothing (65% current, 35% prior) to avoid extreme 0/100
# - Minimum sample thresholds to drop ultra-small samples
#
# Outputs:
#   data/clean/ol_scores.csv  (season, team, ol_score_0_100)
#   data/clean/dl_scores.csv  (season, team, dl_score_0_100)
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import nfl_data_py as nfl

# ---------------------- Configuration ----------------------------------------

OUT_OL = Path("data/clean/ol_scores.csv")
OUT_DL = Path("data/clean/dl_scores.csv")

# Seasons to include (last 7 by default; adjust as needed)
SEASONS = list(range(2019, 2026))

# Minimum sample sizes (per season, per team)
MIN_DROPBACKS_OFFENSE = 150   # dropbacks for OL calc
MIN_PASS_PLAYS_DEF    = 300   # defensive pass plays for DL calc

# Outlier handling: winsorization within each season
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Normalization mode:
# - "minmax":  0..100 linear scaling within season
# - "rank":    percentile rank * 100 (more robust if season is skewed)
NORMALIZE_MODE = "minmax"

# Smoothing: blend current-season normalized score with prior-season score
SMOOTH_WEIGHT_CURRENT = 0.65   # 65% current
SMOOTH_WEIGHT_PRIOR   = 0.35   # 35% previous season (stabilizer)

# Weighting for OL/DL raw signals (tunable)
# For OL: lower pressure/sack allowed => better (negative weights)
W_PRESSURE_ALLOWED = -1.2
W_SACK_ALLOWED     = -1.5

# For DL: higher pressure/sack generated => better (positive weights)
W_PRESSURE_DEF =  1.3
W_SACK_DEF     =  1.4

# ---------------------- Helpers ----------------------------------------------

def pct_clip(s: pd.Series, lo=WINSOR_LO, hi=WINSOR_HI) -> pd.Series:
    """Winsorize a series to the given quantiles within a group."""
    if s.dropna().empty:
        return s
    low = s.quantile(lo)
    high = s.quantile(hi)
    return s.clip(lower=low, upper=high)

def normalize_per_season(df: pd.DataFrame, col: str, by: str, out: str) -> pd.DataFrame:
    """Normalize a column 0..100 within each season (or rank-based if selected)."""
    if NORMALIZE_MODE == "rank":
        def _rankpct(x):
            r = x.rank(pct=True)
            return (r * 100.0).round(6)
        df[out] = df.groupby(by)[col].transform(_rankpct)
        return df

    def _minmax(x):
        lo, hi = x.min(), x.max()
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            # flat/degenerate season → return 50 for all
            return pd.Series(np.full(len(x), 50.0), index=x.index)
        return 100.0 * (x - lo) / (hi - lo)

    df[out] = df.groupby(by)[col].transform(_minmax)
    return df

def ensure_flag(df: pd.DataFrame, col: str) -> None:
    """Ensure a 0/1 int column exists (fills missing with 0)."""
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)

# ---------------------- Main build -------------------------------------------

def main():
    # Load play-by-play (nflfastR schema via nfl_data_py)
    # Use pass plays only, where protection/pressure is meaningful.
    pbp = nfl.import_pbp_data(years=SEASONS)
    pbp = pbp[pbp["pass"] == 1].copy()

    # Handle schema differences:
    # Some seasons may lack 'pressure' flag; others may have it; always ensure ints.
    ensure_flag(pbp, "qb_hit")
    ensure_flag(pbp, "sack")
    if "pressure" in pbp.columns:
        pbp["pressure"] = pbp["pressure"].fillna(0).astype(int)
        pressure_flag = (pbp["pressure"] > 0).astype(int)
    else:
        pressure_flag = 0

    # Define a pressure event if any of: pressure, qb_hit, or sack
    pbp["pressure_event"] = ((pbp["qb_hit"] > 0) | (pbp["sack"] > 0) | (pressure_flag == 1)).astype(int)

    # ---------------- OL (Offense) ----------------
    # Dropbacks ~ count of pass plays (sacks are included in pass==1 rows in nflfastR)
    off = pbp.groupby(["season", "posteam"], as_index=False).agg(
        dropbacks=("play_id", "count"),
        pressures=("pressure_event", "sum"),
        sacks=("sack", "sum"),
    )

    # Remove tiny samples (set to NaN so they won't drive season scaling)
    mask_small_off = off["dropbacks"] < MIN_DROPBACKS_OFFENSE
    if mask_small_off.any():
        off.loc[mask_small_off, ["pressures", "sacks", "dropbacks"]] = np.nan

    # Rates
    off["pressure_rate"] = off["pressures"] / off["dropbacks"]
    off["sack_rate"]     = off["sacks"]     / off["dropbacks"]

    # Winsorize per season to reduce skew/outliers
    off["pressure_rate_w"] = off.groupby("season")["pressure_rate"].transform(pct_clip)
    off["sack_rate_w"]     = off.groupby("season")["sack_rate"].transform(pct_clip)

    # Raw OL signal (lower is better → negative weights)
    off["ol_raw"] = (
        W_PRESSURE_ALLOWED * off["pressure_rate_w"] +
        W_SACK_ALLOWED     * off["sack_rate_w"]
    )

    # Fill remaining NaNs in ol_raw with season median (keeps scale stable)
    off["ol_raw"] = off.groupby("season")["ol_raw"].transform(lambda s: s.fillna(s.median()))

    # Normalize current-season OL signal
    off = normalize_per_season(off, "ol_raw", by="season", out="ol_norm")

    # Season-to-season smoothing to avoid 0/100 extremes
    off = off.sort_values(["posteam", "season"]).reset_index(drop=True)
    off["prev_ol"] = off.groupby("posteam")["ol_norm"].shift(1)
    # Where prior season missing (first season in data), use that season's median as a neutral prior
    off["prev_ol"] = off.groupby("season")["prev_ol"].transform(lambda s: s.fillna(s.median()))
    off["ol_score_0_100"] = (
        SMOOTH_WEIGHT_CURRENT * off["ol_norm"] +
        SMOOTH_WEIGHT_PRIOR   * off["prev_ol"]
    )

    ol_out = off[["season", "posteam", "ol_score_0_100"]].rename(columns={"posteam": "team"})

    # ---------------- DL (Defense) ----------------
    # Defensive pass plays (same pass filter as above)
    df = pbp.groupby(["season", "defteam"], as_index=False).agg(
        plays=("play_id", "count"),
        pressures=("pressure_event", "sum"),
        sacks=("sack", "sum"),
    )

    mask_small_def = df["plays"] < MIN_PASS_PLAYS_DEF
    if mask_small_def.any():
        df.loc[mask_small_def, ["pressures", "sacks", "plays"]] = np.nan

    df["pressure_rate_def"] = df["pressures"] / df["plays"]
    df["sack_rate_def"]     = df["sacks"]     / df["plays"]

    df["pressure_rate_def_w"] = df.groupby("season")["pressure_rate_def"].transform(pct_clip)
    df["sack_rate_def_w"]     = df.groupby("season")["sack_rate_def"].transform(pct_clip)

    # Raw DL signal (higher is better → positive weights)
    df["dl_raw"] = (
        W_PRESSURE_DEF * df["pressure_rate_def_w"] +
        W_SACK_DEF     * df["sack_rate_def_w"]
    )

    df["dl_raw"] = df.groupby("season")["dl_raw"].transform(lambda s: s.fillna(s.median()))

    # Normalize current-season DL signal
    df = normalize_per_season(df, "dl_raw", by="season", out="dl_norm")

    # Season-to-season smoothing
    df = df.sort_values(["defteam", "season"]).reset_index(drop=True)
    df["prev_dl"] = df.groupby("defteam")["dl_norm"].shift(1)
    df["prev_dl"] = df.groupby("season")["prev_dl"].transform(lambda s: s.fillna(s.median()))
    df["dl_score_0_100"] = (
        SMOOTH_WEIGHT_CURRENT * df["dl_norm"] +
        SMOOTH_WEIGHT_PRIOR   * df["prev_dl"]
    )

    dl_out = df[["season", "defteam", "dl_score_0_100"]].rename(columns={"defteam": "team"})

    # ---------------- Save ----------------
    OUT_OL.parent.mkdir(parents=True, exist_ok=True)
    ol_out.to_csv(OUT_OL, index=False)
    dl_out.to_csv(OUT_DL, index=False)

    print("✅ Built trench scores (robust, smoothed)")
    print(f"OL → {OUT_OL}")
    print(f"DL → {OUT_DL}")

if __name__ == "__main__":
    main()
