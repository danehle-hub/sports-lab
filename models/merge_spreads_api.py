from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

GAMES = Path("data/clean/games_clean.csv")

# nflverse schedule codes we will target on OUR side (we’ll normalize nflverse to these)
SCHED_CODE = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO",
    "NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT",
    "SAN FRANCISCO 49ERS":"SF","SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN",
    "WASHINGTON COMMANDERS":"WAS","WASHINGTON FOOTBALL TEAM":"WAS",
}

def team_code(full_name: str, season: int) -> str:
    n = (full_name or "").upper().strip()
    # relocations using schedule codes
    if n in ("OAKLAND RAIDERS","LAS VEGAS RAIDERS"):
        return "OAK" if season <= 2019 else "LV"
    if n in ("LOS ANGELES CHARGERS","SAN DIEGO CHARGERS"):
        return "SD" if season <= 2016 else "LAC"
    if n in ("LOS ANGELES RAMS","ST. LOUIS RAMS"):
        return "STL" if season <= 2015 else "LAR"
    return SCHED_CODE.get(n)

def to_int_week(x):
    s = str(x).strip().upper()
    for p in ("WEEK ","WK ","W "):
        if s.startswith(p):
            s = s[len(p):]
    try:
        return int(s)
    except:
        return pd.NA  # playoffs etc.

def pick_first(cols, options):
    for c in options:
        if c in cols:
            return c
    return None

def normalize_sched_codes(df: pd.DataFrame) -> pd.DataFrame:
    # nflverse sometimes uses LA for Rams (we want LAR), SD for old Chargers (we want SD or LAC depending on year)
    rep = {"LA":"LAR", "SD":"SD", "OAK":"OAK", "LV":"LV"}  # minimalist; keep STL as-is for <=2015
    for col in ["home_team","away_team","home_code","away_code","team_favorite_id"]:
        if col in df.columns:
            df[col] = df[col].replace(rep)
    return df

def derive_home_spread(s: pd.DataFrame) -> pd.Series:
    """
    Return home-perspective spread:
      - If 'spread_line' exists, use it directly (it is already home perspective).
      - Else try to compute from 'team_favorite_id' + 'spread_favorite' (or similar).
        If favorite is home -> negative spread (home favored).
        If favorite is away -> positive spread.
        If pick'em or missing -> 0 or NaN.
    """
    cols = set(s.columns)
    if "spread_line" in cols:
        return s["spread_line"]

    fav_col = pick_first(cols, ["team_favorite_id", "favorite", "team_favorite"])
    spd_col = pick_first(cols, ["spread_favorite", "spread", "close_spread", "closing_spread"])

    if fav_col and spd_col:
        fav = s[fav_col].astype(str).str.upper().str.strip()
        val = pd.to_numeric(s[spd_col], errors="coerce").abs()
        # we need home_code to compare; if not present yet, use home_team field
        home_code = None
        if "home_code" in s.columns:
            home_code = s["home_code"].astype(str).str.upper().str.strip()
        elif "home_team" in s.columns:
            home_code = s["home_team"].astype(str).str.upper().str.strip()

        # If favorite == home team code -> home -spread; if favorite == away team -> +spread; else 0
        spread_home = np.where(fav.eq(home_code), -val,
                        np.where(~fav.eq(home_code) & fav.ne("") & fav.notna(), val, 0.0))
        return pd.to_numeric(spread_home, errors="coerce")
    # last resort: no data to compute
    return pd.Series([np.nan]*len(s), index=s.index)

def main():
    # ----- Load games -----
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    g = g[g["schedule_season"] >= 2010].copy()

    g["team_home"] = g["team_home"].str.upper().str.strip()
    g["team_away"] = g["team_away"].str.upper().str.strip()
    g["season"]    = g["schedule_season"].astype(int)
    g["week_int"]  = g["schedule_week"].apply(to_int_week)
    g["home_code"] = [team_code(h, int(s)) for h, s in zip(g["team_home"], g["season"])]
    g["away_code"] = [team_code(a, int(s)) for a, s in zip(g["team_away"], g["season"])]
    g = g.dropna(subset=["home_code","away_code"]).copy()

    g["game_dt"]   = pd.to_datetime(g["schedule_date"], utc=True, errors="coerce")
    g["game_date"] = g["game_dt"].dt.date

    # ----- Pull nflverse schedules -----
    seasons = sorted(g["season"].unique().tolist())
    s = nfl.import_schedules(seasons).copy()

    # Normalize/rename for consistent keys
    s.rename(columns={"home_team":"home_code", "away_team":"away_code"}, inplace=True)
    s["game_dt"] = pd.to_datetime(s.get("gameday", s.get("game_date")), utc=True, errors="coerce")
    s["game_date"] = s["game_dt"].dt.date

    s = normalize_sched_codes(s)

    # Create a home-perspective spread column (robust to schema changes)
    s["spread_home_calc"] = derive_home_spread(s)

    # Prefer REG for week-join; still keep full set for date fallbacks
    s_reg = s[s.get("game_type","REG") == "REG"].copy()

    # ----- PRIMARY MERGE: season + week + codes -----
    left = g.merge(
        s_reg[["season","week","home_code","away_code","spread_home_calc","game_date"]],
        left_on=["season","week_int","home_code","away_code"],
        right_on=["season","week","home_code","away_code"],
        how="left",
        validate="m:1"
    ).drop(columns=["week"], errors="ignore")

    # ----- FALLBACK A: swap home/away (just in case source rows are flipped) -----
    miss = left["spread_home_calc"].isna()
    if miss.any():
        sw = g.loc[miss].merge(
            s_reg[["season","week","home_code","away_code","spread_home_calc","game_date"]],
            left_on=["season","week_int","away_code","home_code"],
            right_on=["season","week","home_code","away_code"],
            how="left"
        )
        left.loc[miss, "spread_home_calc"] = sw["spread_home_calc"].values

    # ----- FALLBACK B: strict date + codes -----
    miss = left["spread_home_calc"].isna()
    if miss.any():
        fb = g.loc[miss, ["game_date","home_code","away_code"]].merge(
            s[["game_date","home_code","away_code","spread_home_calc"]],
            on=["game_date","home_code","away_code"], how="left"
        )
        left.loc[miss, "spread_home_calc"] = fb["spread_home_calc"].values

    # ----- FALLBACK C: nearest within ±2 days for same season & teams (both directions) -----
    miss = left["spread_home_calc"].isna()
    if miss.any():
        need = g.loc[miss, ["season","home_code","away_code","game_dt"]]
        cand = s.dropna(subset=["spread_home_calc"])[["season","home_code","away_code","game_dt","spread_home_calc"]]

        def nearest_fill(need_df, cand_df):
            m2 = need_df.merge(cand_df, on=["season","home_code","away_code"], how="left",
                               suffixes=("_g","_s"))
            m2["abs_days"] = (m2["game_dt_g"] - m2["game_dt_s"]).abs().dt.total_seconds() / 86400.0
            m2 = m2[m2["abs_days"] <= 2.0].sort_values(
                ["season","home_code","away_code","game_dt_g","abs_days"]
            ).groupby(["season","home_code","away_code","game_dt_g"], as_index=False).first()
            return m2

        same_dir = nearest_fill(need, cand)

        # swapped direction
        need_sw = need.rename(columns={"home_code":"away_code","away_code":"home_code"})
        swap_dir = nearest_fill(need_sw, cand)

        key = g.loc[miss, ["season","home_code","away_code","game_dt"]].rename(columns={"game_dt":"game_dt_g"})
        fill1 = key.merge(same_dir[["season","home_code","away_code","game_dt_g","spread_home_calc"]], how="left")
        fill2 = key.merge(swap_dir[["season","home_code","away_code","game_dt_g","spread_home_calc"]], how="left", suffixes=("","_swap"))

        vals = fill1["spread_home_calc"].where(~fill1["spread_home_calc"].isna(), fill2["spread_home_calc"])
        left.loc[miss, "spread_home_calc"] = vals.values

    left["spread_home"] = left["spread_home_calc"]
    if left["spread_home"].isna().all():
        raise RuntimeError("spread_home is all NaN after robust merging. Check mapping and dates.")

    # recompute ATS target strictly from spreads
    left["home_cover"] = (left["score_home"] - left["score_away"] + left["spread_home"] > 0).astype(int)

    # write back
    left.to_csv(GAMES, index=False)

    cov = 100 * (~left["spread_home"].isna()).mean()
    print(f"✅ Updated games_clean.csv with spreads (robust). Coverage: {cov:.1f}%")

if __name__ == "__main__":
    main()
