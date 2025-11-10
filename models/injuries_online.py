# models/injuries_online.py
# Pull NFL injuries from nfl_data_py and convert to per-position OUT counts per team/day.
# Then merge into your games df as the home_* / away_* "..._out" columns your model uses.
# Changes in this version:
# - Robust guards so the pipeline never crashes if nfl_data_py returns None/malformed data
# - Fixed _season_range_from_games() to avoid ".dropna()" on a float when 'season' column is missing
# - Defensive conversions and defaults throughout; always returns zeroed columns if fetch fails

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import numpy as np
import nfl_data_py as nfl

# --- Position grouping your model/streamlit sliders expect ---
POS_MAP: Dict[str, str] = {
    # offense
    "QB": "qb",
    "RB": "rb", "FB": "rb",
    "WR": "wr",
    "TE": "te",
    "T": "ol", "LT": "ol", "RT": "ol",
    "G": "ol", "LG": "ol", "RG": "ol",
    "C": "ol",
    # defense (edge + cb)
    "DE": "edge", "EDGE": "edge", "OLB": "edge",
    "LB": "edge",  # many teams use OLB/EDGE interchangeably; treat as edge pressure
    "CB": "cb",
    # other defensive backs
    "DB": "cb", "S": "cb", "FS": "cb", "SS": "cb",
}

# Treat these game statuses as "out" for spread impact
OUT_STATUSES = {
    "Out", "Doubtful", "Inactive", "IR", "IR-R", "IR-Ret", "Physically Unable to Perform",
    "Suspension", "NFI", "COVID-19", "Reserve/COVID-19", "Did Not Play",
}
# Optional: include "Questionable" if you want to pre-bake uncertainty penalties.
# OUT_STATUSES |= {"Questionable"}

# Map full franchise names in your data -> team abbreviations used by nfl_data_py
FULLNAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "LA Rams": "LAR",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Tampa Bay Buccaneers": "TB",
    "Tampa Bay": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "San Diego Chargers": "LAC",
    # Common already-abbreviated variants
    "WSH": "WAS",
    "LA": "LAR",   # occasionally seen for Rams
}

def normalize_to_abbr(team: str) -> str:
    """Normalize a team label (full name or abbr) to a 2–3 letter abbreviation."""
    if not isinstance(team, str) or not team:
        return ""
    t = team.strip()
    # If a 2-4 char uppercase token, pass-through after mapping edge cases
    if len(t) <= 4 and t.isupper():
        return FULLNAME_TO_ABBR.get(t, t)
    return FULLNAME_TO_ABBR.get(t, t)

def _series_or_empty(games: pd.DataFrame, col: str) -> pd.Series:
    """Safely return a Series for a column or an empty Series if missing."""
    if isinstance(games, pd.DataFrame) and (col in games.columns):
        return games[col]
    return pd.Series(dtype="float64")

def _season_range_from_games(games: pd.DataFrame) -> List[int]:
    """
    Determine seasons present in the games df. Robust against missing columns:
    - If 'season' missing, tries 'commence_time'
    - If both missing, falls back to current UTC year
    """
    seasons: List[int] = []
    # Try explicit 'season' column
    season_s = _series_or_empty(games, "season")
    if not season_s.empty:
        s = pd.to_numeric(season_s, errors="coerce").dropna().astype(int)
        if not s.empty:
            seasons = sorted(s.unique().tolist())

    # Fallback: derive from commence_time
    if not seasons:
        ct = _series_or_empty(games, "commence_time")
        if not ct.empty:
            years = pd.to_datetime(ct, errors="coerce").dt.year.dropna().astype(int)
            if not years.empty:
                seasons = sorted(years.unique().tolist())

    # Last resort: current season
    if not seasons:
        import datetime as dt
        seasons = [dt.datetime.utcnow().year]

    return seasons

def _safe_to_dataframe(obj) -> pd.DataFrame:
    """Return a DataFrame or empty DataFrame if obj is not table-like."""
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    # Try coercing common structures (list[dict], dict[str, list], etc.)
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

def fetch_injury_counts(seasons: list[int]) -> pd.DataFrame:
    """
    Returns per-team-per-date injury counts by grouped position.
    Columns: date (YYYY-MM-DD), team (abbr), qb, rb, wr, te, cb, edge, ol
    Robust to API hiccups and schema variance; returns empty DF on failure.
    """
    try:
        inj_raw = nfl.import_injuries(seasons=seasons, downcast=True, cache=True)
    except Exception:
        # If the fetch itself throws, just return an empty shaped DF
        return pd.DataFrame(columns=["date", "team", "qb", "rb", "wr", "te", "cb", "edge", "ol"])

    inj = _safe_to_dataframe(inj_raw)
    if inj.empty:
        return pd.DataFrame(columns=["date", "team", "qb", "rb", "wr", "te", "cb", "edge", "ol"])

    inj = inj.copy()

    # Date column selection (prefer report_date, then game_date, else approximate)
    if "report_date" in inj.columns:
        inj["date"] = pd.to_datetime(inj["report_date"], errors="coerce")
    elif "game_date" in inj.columns:
        inj["date"] = pd.to_datetime(inj["game_date"], errors="coerce")
    else:
        # Approximate from season/week (rough mid-week)
        season_s = pd.to_numeric(inj.get("season", pd.Series(index=inj.index, dtype="float64")), errors="coerce")
        week_s = pd.to_numeric(inj.get("week", pd.Series(index=inj.index, dtype="float64")), errors="coerce").fillna(1)
        inj["date"] = pd.to_datetime(season_s.astype("Int64").astype(str) + "-09-01", errors="coerce") + pd.to_timedelta((week_s.astype(int) * 7), unit="D")

    inj["date"] = inj["date"].dt.strftime("%Y-%m-%d")

    # Normalize team/position/status fields
    inj["team"] = inj.get("team", "").astype(str).str.upper()
    inj["position"] = inj.get("position", "").astype(str).str.upper()
    # Keep game_status as-is (nfl_data_py usually uses Title Case), but force string
    inj["game_status"] = inj.get("game_status", "").astype(str)

    # Filter to OUT-like statuses
    inj = inj[inj["game_status"].isin(OUT_STATUSES)].copy()
    if inj.empty:
        return pd.DataFrame(columns=["date", "team", "qb", "rb", "wr", "te", "cb", "edge", "ol"])

    # Map positions to groups; drop un-mapped
    inj["pos_group"] = inj["position"].map(POS_MAP).fillna("")
    inj = inj[inj["pos_group"] != ""].copy()
    if inj.empty:
        return pd.DataFrame(columns=["date", "team", "qb", "rb", "wr", "te", "cb", "edge", "ol"])

    # Aggregate counts per team/date/pos_group
    grp = (
        inj.groupby(["date", "team", "pos_group"], as_index=False)
           .size()
           .rename(columns={"size": "count"})
    )

    # Pivot to wide
    wide = grp.pivot(index=["date", "team"], columns="pos_group", values="count").fillna(0.0)

    # Ensure all expected columns exist
    for need in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]:
        if need not in wide.columns:
            wide[need] = 0.0

    wide = wide.reset_index()

    # Ensure numeric dtypes
    for c in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0).astype(float)

    return wide[["date", "team", "qb", "rb", "wr", "te", "cb", "edge", "ol"]]

def merge_online_injuries_into_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      home_qb_out, away_qb_out, ..., home_ol_out, away_ol_out
    by auto-fetching injuries for the seasons present in `games`.
    Robust: if anything fails or no matches, returns zeros for all *_out columns.
    """
    out = games.copy()

    # Create a date key derived from commence_time (safe even if missing)
    commence = pd.to_datetime(out.get("commence_time", pd.NaT), errors="coerce")
    out["date"] = commence.dt.strftime("%Y-%m-%d")

    # Determine which seasons to query, guarded
    seasons = _season_range_from_games(out)

    # Fetch per-team/date injury counts; guarded
    counts = fetch_injury_counts(seasons)
    if counts is None or counts.empty:
        # Ensure required columns exist as zeros and return
        for pos in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]:
            out[f"home_{pos}_out"] = 0.0
            out[f"away_{pos}_out"] = 0.0
        return out

    # Normalize team names to abbreviations for matching
    out["home_abbr"] = out.get("team_home", "").map(normalize_to_abbr).str.upper()
    out["away_abbr"] = out.get("team_away", "").map(normalize_to_abbr).str.upper()

    # If there are rows without a valid date or abbr, merges will miss—still fine (we'll fill zeros)
    counts_home = counts.add_prefix("home_")
    counts_away = counts.add_prefix("away_")

    # Merge for home team (date + team)
    m_home = out.merge(
        counts_home,
        how="left",
        left_on=["date", "home_abbr"],
        right_on=["home_date", "home_team"],
        suffixes=("", ""),
    )

    # Merge for away team
    m_away = m_home.merge(
        counts_away,
        how="left",
        left_on=["date", "away_abbr"],
        right_on=["away_date", "away_team"],
        suffixes=("", ""),
    )

    # Fill NaNs -> 0 and rename columns to model schema
    for pos in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]:
        m_away[f"home_{pos}_out"] = pd.to_numeric(m_away.get(f"home_{pos}", 0.0), errors="coerce").fillna(0.0)
        m_away[f"away_{pos}_out"] = pd.to_numeric(m_away.get(f"away_{pos}", 0.0), errors="coerce").fillna(0.0)

    # Keep original game columns + the *_out columns
    keep_cols = list(out.columns) + \
                [f"home_{p}_out" for p in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]] + \
                [f"away_{p}_out" for p in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]]

    result = m_away[keep_cols].copy()

    # Final sanitize: ensure numeric *_out columns
    for pos in ["qb", "rb", "wr", "te", "cb", "edge", "ol"]:
        result[f"home_{pos}_out"] = pd.to_numeric(result[f"home_{pos}_out"], errors="coerce").fillna(0.0).astype(float)
        result[f"away_{pos}_out"] = pd.to_numeric(result[f"away_{pos}_out"], errors="coerce").fillna(0.0).astype(float)

    return result
