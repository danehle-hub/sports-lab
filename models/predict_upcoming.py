import pandas as pd
from pathlib import Path

UPCOMING = Path("data/raw/upcoming.csv")
ELO = Path("data/clean/team_week_elo.csv")
OUT = Path("runs/upcoming_predictions.csv")
DEBUG_DROPPED = Path("runs/upcoming_dropped_rows.csv")

TEAM_MAP = {
    "ARI":"ARIZONA CARDINALS","ATL":"ATLANTA FALCONS","BAL":"BALTIMORE RAVENS","BUF":"BUFFALO BILLS",
    "CAR":"CAROLINA PANTHERS","CHI":"CHICAGO BEARS","CIN":"CINCINNATI BENGALS","CLE":"CLEVELAND BROWNS",
    "DAL":"DALLAS COWBOYS","DEN":"DENVER BRONCOS","DET":"DETROIT LIONS","GB":"GREEN BAY PACKERS",
    "HOU":"HOUSTON TEXANS","IND":"INDIANAPOLIS COLTS","JAX":"JACKSONVILLE JAGUARS","KC":"KANSAS CITY CHIEFS",
    "LV":"LAS VEGAS RAIDERS","OAK":"OAKLAND RAIDERS","LAC":"LOS ANGELES CHARGERS","SD":"SAN DIEGO CHARGERS",
    "LAR":"LOS ANGELES RAMS","LA":"LOS ANGELES RAMS","STL":"ST. LOUIS RAMS","MIA":"MIAMI DOLPHINS","MIN":"MINNESOTA VIKINGS",
    "NE":"NEW ENGLAND PATRIOTS","NO":"NEW ORLEANS SAINTS","NYG":"NEW YORK GIANTS","NYJ":"NEW YORK JETS",
    "PHI":"PHILADELPHIA EAGLES","PIT":"PITTSBURGH STEELERS","SF":"SAN FRANCISCO 49ERS","SEA":"SEATTLE SEAHAWKS",
    "TB":"TAMPA BAY BUCCANEERS","TEN":"TENNESSEE TITANS","WAS":"WASHINGTON COMMANDERS","WSH":"WASHINGTON FOOTBALL TEAM"
}

def main():
    sched = pd.read_csv(UPCOMING)

    # If your upcoming.csv has game_type, keep REG only; otherwise this no-ops safely
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"].copy()

    # Identify rows with missing teams and log them
    dropped = sched[sched["home_team"].isna() | sched["away_team"].isna()].copy()
    if not dropped.empty:
        DEBUG_DROPPED.parent.mkdir(parents=True, exist_ok=True)
        dropped.to_csv(DEBUG_DROPPED, index=False)
        print(f"⚠️ Dropped {len(dropped)} row(s) with missing teams → {DEBUG_DROPPED}")

    # Keep only complete rows
    sched = sched.dropna(subset=["home_team","away_team"]).copy()

    # Map codes → full names for Elo merge
    sched["home_team"] = sched["home_team"].map(TEAM_MAP).fillna(sched["home_team"])
    sched["away_team"] = sched["away_team"].map(TEAM_MAP).fillna(sched["away_team"])

    # Load Elo snapshots and take latest pre-game rating per team
    elo = pd.read_csv(ELO)
    latest = elo.sort_values(["schedule_season","schedule_week"]).groupby("team", as_index=False).last()

    # Merge Elo
    sched = sched.merge(latest.rename(columns={"team":"home_team","elo_pre":"home_elo"}), on="home_team", how="left")
    sched = sched.merge(latest.rename(columns={"team":"away_team","elo_pre":"away_elo"}), on="away_team", how="left")

    # If any Elo still missing, log those (likely a naming mismatch)
    still = sched[sched["home_elo"].isna() | sched["away_elo"].isna()].copy()
    if not still.empty:
        print("⚠️ Elo missing for these rows (check team names / mapping):")
        print(still[["week","home_team","away_team","home_elo","away_elo"]].to_string(index=False))

    # Compute probs/spreads
    HOME_FIELD = 55
    sched["elo_diff"] = (sched["home_elo"] + HOME_FIELD) - sched["away_elo"]
    sched["home_win_prob"] = 1 / (1 + 10 ** (-sched["elo_diff"] / 400))
    sched["model_spread"] = -sched["elo_diff"] / 25
    if "spread_line" in sched.columns:
        sched["edge_vs_market"] = sched["model_spread"] - sched["spread_line"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(OUT, index=False)

    cols = ["week","home_team","away_team","home_win_prob","spread_line","model_spread","edge_vs_market"]
    print(f"✅ Predictions saved → {OUT}")
    print(sched[[c for c in cols if c in sched.columns]].to_string(index=False))

if __name__ == "__main__":
    main()
