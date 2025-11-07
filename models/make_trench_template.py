from pathlib import Path
import pandas as pd

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

teams = [
"ARIZONA CARDINALS","ATLANTA FALCONS","BALTIMORE RAVENS","BUFFALO BILLS","CAROLINA PANTHERS",
"CHICAGO BEARS","CINCINNATI BENGALS","CLEVELAND BROWNS","DALLAS COWBOYS","DENVER BRONCOS",
"DETROIT LIONS","GREEN BAY PACKERS","HOUSTON TEXANS","INDIANAPOLIS COLTS","JACKSONVILLE JAGUARS",
"KANSAS CITY CHIEFS","LAS VEGAS RAIDERS","LOS ANGELES CHARGERS","LOS ANGELES RAMS","MIAMI DOLPHINS",
"MINNESOTA VIKINGS","NEW ENGLAND PATRIOTS","NEW ORLEANS SAINTS","NEW YORK GIANTS","NEW YORK JETS",
"PHILADELPHIA EAGLES","PITTSBURGH STEELERS","SAN FRANCISCO 49ERS","SEATTLE SEAHAWKS",
"TAMPA BAY BUCCANEERS","TENNESSEE TITANS","WASHINGTON COMMANDERS"
]

# cover last 7 yrs by default; adjust if you want
seasons = list(range(2019, 2026))

def make_df(colname):
    rows = []
    for s in seasons:
        for t in teams:
            rows.append({"season": s, "team": t, colname: None})
    return pd.DataFrame(rows)

ol = make_df("ol_score_0_100")
f7 = make_df("front7_score_0_100")

ol_path = RAW / "ol_ratings.csv"
f7_path = RAW / "front7_ratings.csv"

if not ol_path.exists():
    ol.to_csv(ol_path, index=False)
if not f7_path.exists():
    f7.to_csv(f7_path, index=False)

print("✅ Wrote:")
print(" ", ol_path)
print(" ", f7_path)
print("Fill these 0–100 (higher=better). You can start with 2025 only.")
