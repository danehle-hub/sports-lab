import pandas as pd
from pathlib import Path
from datetime import date

GAMES = Path("data/clean/games_clean.csv")
LINES = Path("data/raw/betting_lines.csv")

# Base map for teams whose codes don't change
BASE_CODE = {
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GNB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KAN",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NWE",
    "NEW ORLEANS SAINTS": "NOR",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SFO",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TAM",
    "TENNESSEE TITANS": "TEN",
    # You may also have WFT/Commanders in other datasets:
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
}

def team_code(full_name: str, season: int) -> str:
    """Map full team name from games_clean to nflverse 3-letter code, handling relocations by season."""
    full_name = (full_name or "").upper().strip()
    # Relocations / legacy names that depend on season:
    if full_name in ("OAKLAND RAIDERS", "LAS VEGAS RAIDERS"):
        # OAK through 2019; LVR from 2020 onward
        return "OAK" if season <= 2019 else "LVR"
    if full_name == "LOS ANGELES CHARGERS" or full_name == "SAN DIEGO CHARGERS":
        # SDG through 2016; LAC from 2017 onward
        return "SDG" if season <= 2016 else "LAC"
    if full_name == "LOS ANGELES RAMS" or full_name == "ST. LOUIS RAMS":
        # STL through 2015; LAR from 2016 onward
        return "STL" if season <= 2015 else "LAR"
    # Default
    return BASE_CODE.get(full_name, None)

def pick_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def main():
    # Load games (date & season needed)
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    # Work only with modern seasons where lines are reliable and nflverse has consistent codes
    g = g[g["schedule_season"] >= 2010].copy()

    # Build merge keys: codes and normalized date
    g["team_home"] = g["team_home"].str.upper().str.strip()
    g["team_away"] = g["team_away"].str.upper().str.strip()
    g["home_code"] = [team_code(n, int(s)) for n, s in zip(g["team_home"], g["schedule_season"])]
    g["away_code"] = [team_code(n, int(s)) for n, s in zip(g["team_away"], g["schedule_season"])]
    g = g.dropna(subset=["home_code","away_code"]).copy()
    g["game_date"] = pd.to_datetime(g["schedule_date"]).dt.date

    # Load betting lines and auto-detect columns
    l = pd.read_csv(LINES, parse_dates=True)
    lcols = set(l.columns)

    # Date column candidates
    date_col = pick_column(lcols, ["gameday","game_date","game_day","date"])
    if date_col is None:
        raise ValueError(f"betting_lines.csv missing a date column. Found columns: {sorted(lcols)}")
    l["game_date"] = pd.to_datetime(l[date_col]).dt.date

    # Home/away team code columns (nflverse schedules use 3-letter codes)
    home_col = pick_column(lcols, ["home_team","home","home_abbr","home_team_abbr","home_team_id"])
    away_col = pick_column(lcols, ["away_team","away","away_abbr","away_team_abbr","away_team_id"])
    if not home_col or not away_col:
        raise ValueError(f"betting_lines.csv missing home/away team columns. Found: {sorted(lcols)}")

    # Spread column candidates (home-perspective closing line)
    spread_col = pick_column(lcols, ["spread_line","home_spread","spread","spread_close","closing_spread"])
    if spread_col is None:
        raise ValueError(f"betting_lines.csv missing a spread column. Found columns: {sorted(lcols)}")

    l = l.rename(columns={home_col:"home_code", away_col:"away_code", spread_col:"spread_norm"})
    l["home_code"] = l["home_code"].astype(str).str.upper().str.strip()
    l["away_code"] = l["away_code"].astype(str).str.upper().str.strip()

    # Keep only necessary fields
    l = l[["game_date","home_code","away_code","spread_norm"]].dropna(subset=["spread_norm"]).copy()

    # Merge by date + codes
    merged = g.merge(
        l, on=["game_date","home_code","away_code"], how="left", validate="m:1"
    )

    # Assign spread_home (nflverse spread is home-perspective; negative means home favored)
    merged["spread_home"] = merged["spread_norm"]

    # Compute ATS: did home cover?
    merged["home_cover"] = (merged["score_home"] - merged["score_away"] + merged["spread_home"] > 0).astype(int)

    # Save back into games_clean.csv
    merged.to_csv(GAMES, index=False)

    # Coverage report
    cov = 100 * (~merged["spread_home"].isna()).mean()
    print(f"✅ Updated games_clean.csv with spreads & ATS.")
    print(f"   Seasons kept: {merged['schedule_season'].min():.0f}-{merged['schedule_season'].max():.0f}")
    print(f"   Spread coverage: {cov:.1f}% (aim for >95%)")

if __name__ == "__main__":
    main()
