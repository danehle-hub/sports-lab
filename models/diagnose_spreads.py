from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

GAMES = Path("data/clean/games_clean.csv")
DEBUG_OUT = Path("data/debug_unmatched_spreads.csv")

# nflverse schedule codes
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
def team_code(full_name:str, season:int)->str:
    n=(full_name or "").upper().strip()
    if n in ("OAKLAND RAIDERS","LAS VEGAS RAIDERS"): return "OAK" if season<=2019 else "LV"
    if n in ("LOS ANGELES CHARGERS","SAN DIEGO CHARGERS"): return "SD" if season<=2016 else "LAC"
    if n in ("LOS ANGELES RAMS","ST. LOUIS RAMS"): return "STL" if season<=2015 else "LAR"
    return SCHED_CODE.get(n)

def to_int_week(x):
    s=str(x).strip().upper()
    for p in ("WEEK ","WK ","W "):
        if s.startswith(p): s=s[len(p):]
    try: return int(s)
    except: return pd.NA

def main():
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    g = g[g["schedule_season"]>=2010].copy()
    g["team_home"]=g["team_home"].str.upper().str.strip()
    g["team_away"]=g["team_away"].str.upper().str.strip()
    g["season"]=g["schedule_season"].astype(int)
    g["week_int"]=g["schedule_week"].apply(to_int_week)
    g["home_code"]=[team_code(h,int(s)) for h,s in zip(g["team_home"],g["season"])]
    g["away_code"]=[team_code(a,int(s)) for a,s in zip(g["team_away"],g["season"])]
    g["game_dt"]=pd.to_datetime(g["schedule_date"], utc=True, errors="coerce")
    g["game_date"]=g["game_dt"].dt.date

    # show sample problematic rows (like your CHI vs GB opener)
    print("Sample (first 10 rows) with keys:")
    print(g[["season","week_int","team_home","team_away","home_code","away_code","game_date"]].head(10).to_string(index=False))

    seasons=sorted(g["season"].unique())
    s = nfl.import_schedules(seasons)
    print("\nSchedule codes seen for 2019 home_team:", sorted(s[s["season"]==2019]["home_team"].unique())[:20], "...")

    s = s[["season","week","gameday","home_team","away_team","spread_line","game_type"]].copy()
    s = s.rename(columns={"home_team":"home_code","away_team":"away_code"})
    s["game_dt"]=pd.to_datetime(s["gameday"], utc=True, errors="coerce")
    s["game_date"]=s["game_dt"].dt.date

    # REG only for primary
    s_reg = s[s["game_type"]=="REG"].copy()

    m = g.merge(
        s_reg[["season","week","home_code","away_code","spread_line","game_date"]],
        left_on=["season","week_int","home_code","away_code"],
        right_on=["season","week","home_code","away_code"],
        how="left"
    )

    miss = m[m["spread_line"].isna()].copy()
    print(f"\nPrimary season+week match coverage: {100*(1-len(miss)/len(g)):.1f}%  (want high)")
    # try strict date match
    fb = miss.merge(
        s[["game_date","home_code","away_code","spread_line"]],
        on=["game_date","home_code","away_code"], how="left"
    )
    got = fb[~fb["spread_line"].isna()].index
    miss.loc[got, "spread_line"] = fb.loc[got,"spread_line"]

    print(f"After strict date match coverage: {100*(1-miss['spread_line'].isna().mean()):.1f}%")

    # dump remaining unmatched for inspection
    unmatched = miss[miss["spread_line"].isna()][["season","week_int","team_home","team_away","home_code","away_code","game_date"]]
    DEBUG_OUT.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(DEBUG_OUT, index=False)
    print(f"\nWrote remaining unmatched rows → {DEBUG_OUT} (open this to see exactly what isn't matching)")

if __name__=="__main__":
    main()
