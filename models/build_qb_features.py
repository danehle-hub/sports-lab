from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

GAMES = Path("data/clean/games_clean.csv")
OUT   = Path("data/clean/qb_team_week.csv")

YEARS_BACK = 7
MIN_DROPBACKS_GAME = 10

def main():
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    last_season = int(g["schedule_season"].max())
    cutoff = last_season - (YEARS_BACK - 1)
    seasons = sorted(g.loc[g["schedule_season"] >= cutoff, "schedule_season"].unique().tolist())

    print(f"Downloading PBP for seasons: {seasons}")
    # ✅ Correct call for your nfl_data_py version
    pbp = nfl.import_pbp_data(years=seasons)

    # Keep passing plays (dropbacks = passes + sacks)
    pbp = pbp[pbp["pass"] == 1].copy()

    # Ensure these exist (schema can vary slightly)
    needed = ["season", "week", "game_id", "posteam", "passer_player_id",
              "passer_player_name", "epa", "qb_epa"]
    missing = [c for c in needed if c not in pbp.columns]
    if missing:
        raise ValueError(f"Missing PBP columns: {missing}")

    pbp["dropback"] = 1  # pass = dropback in nflfastR convention

    # Aggregate per-team per-game QB stats
    grp = pbp.groupby(
        ["season", "week", "posteam", "passer_player_id", "passer_player_name"],
        as_index=False
    ).agg(
        dropbacks=("dropback", "sum"),
        epa_sum=("epa", "sum"),
        qb_epa_sum=("qb_epa", "sum")
    )

    # Filter out tiny samples
    grp = grp[grp["dropbacks"] >= MIN_DROPBACKS_GAME].copy()

    # Primary QB = most dropbacks that week
    grp["rank"] = grp.groupby(["season", "week", "posteam"])["dropbacks"].rank(
        ascending=False, method="first"
    )
    prim = grp[grp["rank"] == 1].drop(columns=["rank"])
    prim = prim.rename(columns={"posteam": "team"})

    # Rolling EPA per dropback over last 3 games (exclude current week)
    prim = prim.sort_values(["team", "season", "week"])
    prim["qb_epa_per_db"] = prim["qb_epa_sum"] / prim["dropbacks"]
    prim["qb_epa_roll3"] = (
        prim.groupby("team")["qb_epa_per_db"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    )

    # Save team-week QB rating feature
    feat = prim[["season", "week", "team", "passer_player_name", "qb_epa_per_db", "qb_epa_roll3"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(OUT, index=False)
    print(f"✅ Wrote QB features → {OUT}")

if __name__ == "__main__":
    main()
