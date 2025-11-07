import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

OUT = Path("data/raw/betting_lines.csv")

def main():
    # Adjust the range based on how many years you want.
    # Your model uses a 7-year recency window, so 2010+ is safe.
    seasons = list(range(2010, 2026))

    print("Downloading schedules (includes closing spreads) from nflverse...")
    df = nfl.import_schedules(seasons)

    # Keep only the columns we care about
    df = df[[
        "gameday",
        "home_team",
        "away_team",
        "spread_line",
        "total_line"
    ]]

    # Drop rows without spreads (preseason / old years)
    df = df.dropna(subset=["spread_line"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"✅ Saved betting lines → {OUT.resolve()}")

if __name__ == "__main__":
    main()
