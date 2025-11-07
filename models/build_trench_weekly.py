# models/build_trench_weekly.py
# -----------------------------------------------------------------------------
# Weekly Offensive Line (OL) and Defensive Front (DL) ratings with recency
# weighting. Produces ratings that exist BEFORE each game (no leakage).
#
# Output:
#   data/clean/trench_weekly.csv
#     columns:
#       season, week, team,
#       ol_week_raw, dl_week_raw,      # per-week signal (z-scored)
#       ol_rating_ema, dl_rating_ema,  # smoothed cumulative within season
#       ol_0_100, dl_0_100             # season-normalized 0..100 snapshot
#
# How it works:
#   1) From PBP pass plays, compute per-game pressure & sack rates.
#   2) Convert to weekly team signals (offense: allowed, defense: generated).
#   3) Z-score vs league that week to remove environment drift.
#   4) Update team EMA: new = (1-a)*old + a*week_signal (per season).
#   5) Min sample safeguards, winsorize outliers.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import nfl_data_py as nfl

OUT = Path("data/clean/trench_weekly.csv")

# Configure
YEARS_BACK = 7
SEASONS = list(range(2019, 2026))  # adjust as you like
MIN_DB_OFF_GM = 8                  # min dropbacks for an offense in a game to count
MIN_PP_DEF_GM = 12                 # min pass plays faced for a defense in a game
ALPHA = 0.35                       # EMA step (0.25–0.50 reasonable); higher = more reactive
WINSOR_LO, WINSOR_HI = 0.01, 0.99

def pct_clip(s, lo=WINSOR_LO, hi=WINSOR_HI):
    if s.dropna().empty:
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)

def main():
    pbp = nfl.import_pbp_data(years=SEASONS)
    # Keep pass plays only (protection/pressure lives here)
    pbp = pbp[pbp["pass"] == 1].copy()

    # Ensure flags
    for col in ["qb_hit", "sack"]:
        if col not in pbp.columns:
            pbp[col] = 0
        pbp[col] = pbp[col].fillna(0).astype(int)
    if "pressure" in pbp.columns:
        pbp["pressure"] = pbp["pressure"].fillna(0).astype(int)
        pressure_flag = (pbp["pressure"] > 0).astype(int)
    else:
        pressure_flag = 0

    pbp["pressure_event"] = ((pbp["qb_hit"] > 0) | (pbp["sack"] > 0) | (pressure_flag == 1)).astype(int)

    # ---------- Per-game team metrics ----------
    # Offense (allowed) keyed by (season, week, posteam)
    off = pbp.groupby(["season","week","game_id","posteam"], as_index=False).agg(
        dropbacks=("play_id","count"),
        pressures=("pressure_event","sum"),
        sacks=("sack","sum"),
    )
    off = off[off["dropbacks"] >= MIN_DB_OFF_GM].copy()
    off["pressure_rate_allowed"] = off["pressures"] / off["dropbacks"]
    off["sack_rate_allowed"]     = off["sacks"]     / off["dropbacks"]

    # Defense (generated) keyed by (season, week, game_id, defteam)
    dfn = pbp.groupby(["season","week","game_id","defteam"], as_index=False).agg(
        pass_plays=("play_id","count"),
        pressures=("pressure_event","sum"),
        sacks=("sack","sum"),
    )
    dfn = dfn[dfn["pass_plays"] >= MIN_PP_DEF_GM].copy()
    dfn["pressure_rate_def"] = dfn["pressures"] / dfn["pass_plays"]
    dfn["sack_rate_def"]     = dfn["sacks"]     / dfn["pass_plays"]

    # Winsorize game-level rates to reduce noise
    for col in ["pressure_rate_allowed","sack_rate_allowed"]:
        off[col] = off.groupby(["season","week"])[col].transform(pct_clip)
    for col in ["pressure_rate_def","sack_rate_def"]:
        dfn[col] = dfn.groupby(["season","week"])[col].transform(pct_clip)

    # Weekly team signal = average across that week's games (usually 1)
    off_w = off.groupby(["season","week","posteam"], as_index=False).agg(
        pr_allowed=("pressure_rate_allowed","mean"),
        sr_allowed=("sack_rate_allowed","mean"),
    ).rename(columns={"posteam":"team"})
    dfn_w = dfn.groupby(["season","week","defteam"], as_index=False).agg(
        pr_def=("pressure_rate_def","mean"),
        sr_def=("sack_rate_def","mean"),
    ).rename(columns={"defteam":"team"})

    # Convert to weekly z-scores vs league that week
    def zscore(df, cols):
        for c in cols:
            df[c+"_z"] = df.groupby(["season","week"])[c].transform(
                lambda s: (s - s.mean())/ (s.std(ddof=0) if s.std(ddof=0) > 1e-9 else 1.0)
            )
        return df

    off_w = zscore(off_w, ["pr_allowed","sr_allowed"])
    dfn_w = zscore(dfn_w, ["pr_def","sr_def"])

    # Offensive weekly raw signal: lower allowed rates is better → negative z
    off_w["ol_week_raw"] = -0.6*off_w["pr_allowed_z"] - 0.4*off_w["sr_allowed_z"]

    # Defensive weekly raw signal: higher generated rates is better → positive z
    dfn_w["dl_week_raw"] =  0.6*dfn_w["pr_def_z"]     + 0.4*dfn_w["sr_def_z"]

    # Full team list across seasons
    teams = sorted(set(off_w["team"]).union(set(dfn_w["team"])))

    # Build a calendar of (season, week) in chronological order
    cal = (
        pbp[["season","week"]]
        .drop_duplicates()
        .sort_values(["season","week"])
        .reset_index(drop=True)
    )

    # Initialize EMA states per (season, team) at neutral 0.0
    # (EMA is on z-scored units; we normalize to 0..100 at the end)
    records = []
    for season in sorted(cal["season"].unique()):
        season_weeks = cal[cal["season"]==season]["week"].sort_values().unique().tolist()
        ol_state = {t: 0.0 for t in teams}
        dl_state = {t: 0.0 for t in teams}
        for wk in season_weeks:
            # Merge available weekly raw signals
            ow = off_w[(off_w["season"]==season) & (off_w["week"]==wk)][["team","ol_week_raw"]]
            dw = dfn_w[(dfn_w["season"]==season) & (dfn_w["week"]==wk)][["team","dl_week_raw"]]
            ow = dict(zip(ow["team"], ow["ol_week_raw"]))
            dw = dict(zip(dw["team"], dw["dl_week_raw"]))

            for t in teams:
                # Update EMA if team played; otherwise carry forward unchanged
                if t in ow:
                    ol_state[t] = (1-ALPHA)*ol_state[t] + ALPHA*float(ow[t])
                if t in dw:
                    dl_state[t] = (1-ALPHA)*dl_state[t] + ALPHA*float(dw[t])

                records.append({
                    "season": season,
                    "week":   wk,
                    "team":   t,
                    "ol_week_raw": ow.get(t, np.nan),
                    "dl_week_raw": dw.get(t, np.nan),
                    "ol_rating_ema": ol_state[t],
                    "dl_rating_ema": dl_state[t],
                })

    wk = pd.DataFrame(records)

    # Normalize EMA within season to 0..100 for merge convenience
    def norm01(x):
        lo, hi = x.min(), x.max()
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return pd.Series(np.full(len(x), 50.0), index=x.index)
        return 100.0 * (x - lo) / (hi - lo)

    wk["ol_0_100"] = wk.groupby("season")["ol_rating_ema"].transform(norm01)
    wk["dl_0_100"] = wk.groupby("season")["dl_rating_ema"].transform(norm01)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wk.to_csv(OUT, index=False)
    print(f"✅ Wrote weekly trench ratings → {OUT}")

if __name__ == "__main__":
    main()
