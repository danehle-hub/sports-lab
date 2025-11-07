import numpy as np
import pandas as pd
from pathlib import Path

# INPUT / OUTPUT
CLEAN_GAMES   = Path("data/clean/games_clean.csv")
ELO_BY_GAME   = Path("data/clean/elo_by_game.csv")
TEAM_WEEK_ELO = Path("data/clean/team_week_elo.csv")

# --- SETTINGS ---
BASE_ELO = 1500.0
YEARS_BACK = 7               # only last 7 seasons influence ratings
SEASON_CARRY = 0.75          # mean reversion each season
K_REG, K_PO = 20.0, 24.0
HOME_BONUS_ELO = 55.0        # home advantage added to expectation only
MAX_MOV_MULT = 2.0
HALFLIFE_DAYS = 180          # recency half-life inside the 7-year window

def expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def mov_mult(margin, pre_diff):
    m = np.log1p(abs(margin)) * (2.2 / ((pre_diff * 0.001) + 2.2))
    return float(min(m, MAX_MOV_MULT))

def reseed(elo):
    return {t: SEASON_CARRY * r + (1 - SEASON_CARRY) * BASE_ELO for t, r in elo.items()}

def main():
    # -------- Load & validate ----------
    g = pd.read_csv(CLEAN_GAMES, parse_dates=["schedule_date"])
    need = {"schedule_date","schedule_season","schedule_week",
            "team_home","team_away","score_home","score_away","stadium_neutral"}
    miss = [c for c in need if c not in g.columns]
    if miss:
        raise ValueError(f"games_clean.csv missing columns: {miss}")

    # Strict chronological order (so we snapshot pre-game Elo, then update)
    g = g.sort_values(["schedule_date","team_home","team_away"]).reset_index(drop=True)

    # Only last N seasons can move ratings; older are zero-weighted
    last_season = int(g["schedule_season"].max())
    cutoff = last_season - (YEARS_BACK - 1)
    g["update_eligible"] = g["schedule_season"] >= cutoff

    # Recency decay within the window
    max_date = g["schedule_date"].max()
    g["age_days"] = (max_date - g["schedule_date"]).dt.days
    g["recency_wt"] = np.where(
        g["update_eligible"],
        0.5 ** (g["age_days"] / HALFLIFE_DAYS),
        0.0
    ).astype(float)

    # Initialize ratings
    teams = pd.unique(pd.concat([g["team_home"], g["team_away"]], ignore_index=True))
    elo = {t: BASE_ELO for t in teams}

    rows = []
    prev_season = None

    for _, row in g.iterrows():
        season = int(row["schedule_season"])
        week = str(row["schedule_week"])
        home, away = row["team_home"], row["team_away"]
        hs, as_ = float(row["score_home"]), float(row["score_away"])
        # PowerShell CSVs sometimes store booleans as 0/1 or strings; normalize:
        neutral = bool(int(row["stadium_neutral"])) if str(row["stadium_neutral"]).strip() in {"0","1"} else bool(row["stadium_neutral"])

        # New season → mean reversion
        if prev_season is None:
            prev_season = season
        elif season != prev_season:
            elo = reseed(elo)
            prev_season = season

        # -------- PRE-GAME SNAPSHOT (this is what your model should use) --------
        rh_pre, ra_pre = elo.get(home, BASE_ELO), elo.get(away, BASE_ELO)

        # Expectation (home bonus does not persist in rating, only in expectation)
        hb = 0.0 if neutral else HOME_BONUS_ELO
        p_home = expected(rh_pre + hb, ra_pre)
        p_away = 1 - p_home

        # Result to points
        if hs > as_:
            ah, aa = 1.0, 0.0
            margin = hs - as_
            pre_diff = rh_pre - ra_pre
        elif hs < as_:
            ah, aa = 0.0, 1.0
            margin = as_ - hs
            pre_diff = ra_pre - rh_pre
        else:
            ah, aa = 0.5, 0.5
            margin = 0.0
            pre_diff = 0.0

        # K with MOV multiplier and recency weight
        is_po = str(row.get("schedule_playoff", "")).strip().lower() in {"1","true","yes","wc","div","conf","sb","po","playoff","superbowl"}
        k_base = K_PO if is_po else K_REG
        k_eff = k_base * mov_mult(margin, pre_diff) * float(row["recency_wt"])

        # -------- POST-GAME UPDATE (happens AFTER we stored pre-game snapshot) --------
        rh_post = rh_pre + k_eff * (ah - p_home)
        ra_post = ra_pre + k_eff * (aa - p_away)

        elo[home], elo[away] = rh_post, ra_post

        rows.append({
            "schedule_date": row["schedule_date"],
            "schedule_season": season,
            "schedule_week": week,
            "home_team": home, "away_team": away,
            "neutral_site": neutral,
            "home_elo_pre": rh_pre, "away_elo_pre": ra_pre,
            "home_elo_post": rh_post, "away_elo_post": ra_post,
        })

    # -------- Write per-game Elo table --------
    elo_game = pd.DataFrame(rows)
    ELO_BY_GAME.parent.mkdir(parents=True, exist_ok=True)
    elo_game.to_csv(ELO_BY_GAME, index=False)

    # -------- Build weekly snapshot using PRE-GAME ratings --------
    # One row per (season, week, team) with elo_pre (primary) and elo_post (for reference)
    home_pre = elo_game[["schedule_season","schedule_week","home_team","home_elo_pre","home_elo_post"]] \
        .rename(columns={"home_team":"team","home_elo_pre":"elo_pre","home_elo_post":"elo_post"})
    away_pre = elo_game[["schedule_season","schedule_week","away_team","away_elo_pre","away_elo_post"]] \
        .rename(columns={"away_team":"team","away_elo_pre":"elo_pre","away_elo_post":"elo_post"})
    snap = pd.concat([home_pre, away_pre], ignore_index=True)

    # If a team appears multiple times in a week (rare), keep the FIRST pre snapshot
    # (the first occurrence in chronological order is the pre-game state for that week)
    snap = snap.sort_values(["schedule_season","schedule_week"]).groupby(
        ["schedule_season","schedule_week","team"], as_index=False
    ).first()

    TEAM_WEEK_ELO.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(TEAM_WEEK_ELO, index=False)

    print("✅ Recency-weighted (last 7 yrs) Elo complete — leak-free (walk-forward)")
    print("  Wrote:", ELO_BY_GAME)
    print("  Wrote:", TEAM_WEEK_ELO)
    print("  Header check:", list(snap.columns))

if __name__ == "__main__":
    main()
