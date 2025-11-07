# models/fetch.py  (FULL REPLACEMENT)
from __future__ import annotations
import os
import time
import json
from pathlib import Path
from typing import List

import pandas as pd
import requests
from dotenv import load_dotenv

# ---- Setup paths & env
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ---- Small helpers
def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _prefer_book(df: pd.DataFrame) -> pd.DataFrame:
    """Pick a primary book per game (Pinnacle/DK/FD preferred), else the freshest update."""
    pref = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us"]
    df = df.copy()
    df["book_rank"] = df["book"].apply(lambda x: pref.index(x) if x in pref else 99)
    # Prefer best-ranked book; if tie, latest timestamp
    return (
        df.sort_values(["game_id", "book_rank", "last_update"], ascending=[True, True, False])
          .drop_duplicates(["game_id"], keep="first")
          .reset_index(drop=True)
    )

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ---- Public odds fetch
def fetch_odds_current_nfl() -> pd.DataFrame:
    """
    The Odds API – US region, spreads + moneyline. Robust to schema gaps.
    Returns one row per game (preferred book). Columns:
        game_id, commence_time, team_home, team_away,
        spread_home, price_home_spread, spread_away, price_away_spread,
        market_home_ml, market_away_ml, book, last_update
    """
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY missing. Add it to your .env file at project root.")

    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        "?regions=us&markets=spreads,h2h&oddsFormat=american&dateFormat=iso&apiKey="
        + ODDS_API_KEY
    )

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        games = r.json()
        if not isinstance(games, list):
            raise ValueError(f"Unexpected JSON from Odds API: {games}")
    except requests.HTTPError as e:
        # Helpful diagnostics
        msg = getattr(e.response, "text", "")
        raise RuntimeError(f"Odds API HTTP error: {e} :: {msg}") from e
    except Exception as e:
        raise RuntimeError(f"Odds API request failed: {e}") from e

    rows: List[dict] = []
    for g in games:
        gid = _safe_get(g, "id")
        commence = _safe_get(g, "commence_time")
        home = _safe_get(g, "home_team")
        away = _safe_get(g, "away_team")
        for bk in _safe_get(g, "bookmakers", default=[]) or []:
            book_key = _safe_get(bk, "key")
            last_update = _safe_get(bk, "last_update")
            markets = {m.get("key"): m for m in _safe_get(bk, "markets", default=[]) or []}

            # Extract spreads
            sp = markets.get("spreads", {})
            sp_out = {"spread_home": None, "price_home_spread": None, "spread_away": None, "price_away_spread": None}
            for o in _safe_get(sp, "outcomes", default=[]) or []:
                name = o.get("name")
                if name == home:
                    sp_out["spread_home"] = o.get("point")
                    sp_out["price_home_spread"] = o.get("price")
                elif name == away:
                    sp_out["spread_away"] = o.get("point")
                    sp_out["price_away_spread"] = o.get("price")

            # Extract moneyline
            ml = markets.get("h2h", {})
            ml_out = {"market_home_ml": None, "market_away_ml": None}
            for o in _safe_get(ml, "outcomes", default=[]) or []:
                name = o.get("name")
                if name == home:
                    ml_out["market_home_ml"] = o.get("price")
                elif name == away:
                    ml_out["market_away_ml"] = o.get("price")

            row = {
                "game_id": gid,
                "commence_time": commence,
                "team_home": home,
                "team_away": away,
                "book": book_key,
                "last_update": last_update,
                **sp_out,
                **ml_out,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        # No games returned is plausible in offseason; return empty shell with expected columns
        return pd.DataFrame(
            columns=[
                "game_id","commence_time","team_home","team_away","book","last_update",
                "spread_home","price_home_spread","spread_away","price_away_spread",
                "market_home_ml","market_away_ml",
            ]
        )

    # Coerce numeric columns defensively
    for col in ["spread_home","price_home_spread","spread_away","price_away_spread","market_home_ml","market_away_ml"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Choose preferred book per game
    df = _prefer_book(df)

    return df.reset_index(drop=True)

# ---- Team metrics via nfl_data_py (rolling)
def fetch_nfl_team_metrics(seasons: list[int]) -> pd.DataFrame:
    """
    Pulls nflfastR play-by-play via nfl_data_py and builds rolling team metrics (last 8).
    If the local cache is missing, automatically downloads from the source.
    """
    try:
        import nfl_data_py as nfl
    except Exception as e:
        raise RuntimeError(
            "nfl_data_py not installed or failed to import. "
            "Run:  pip install nfl_data_py"
        ) from e

    # Try using local cache first; if missing, pull from remote
    try:
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=True)
    except ValueError:
        # No cache present for one or more years — fetch from remote
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)

    # Keep only needed columns
    keep = pbp[[
        "season","week","posteam","defteam","epa","success","pass","rush","air_yards","qb_epa","qb_hit","sack"
    ]].copy()
    keep = keep.rename(columns={"posteam":"team","defteam":"opp"})

    # Offense aggregate per team/week
    off = keep.groupby(["season","week","team"], as_index=False).agg(
        off_epa=("epa","mean"),
        off_sr=("success","mean"),
        pass_rate=("pass","mean"),
        rush_rate=("rush","mean"),
        air_yds=("air_yards","mean"),
        qb_epa=("qb_epa","mean"),
        qb_hit=("qb_hit","sum"),
        sacks=("sack","sum"),
        plays=("epa","size")
    )

    # Defense aggregate (flip by opponent)
    keep_def = keep.rename(columns={"team":"opp_team","opp":"team"})
    deff = keep_def.groupby(["season","week","team"], as_index=False).agg(
        def_epa=("epa","mean"),
        def_sr=("success","mean")
    )

    m = off.merge(deff, on=["season","week","team"], how="left").sort_values(["team","season","week"])

    # Rolling last 8 per team (min 4)
    for col in ["off_epa","off_sr","def_epa","def_sr","pass_rate","rush_rate","qb_epa"]:
        m[f"{col}_r8"] = m.groupby("team")[col].transform(lambda s: s.rolling(8, min_periods=4).mean())

    return m

# ---- Convenience single-call (optional)
def fetch_all_current(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch odds + team metrics in one call. Returns (odds_df, metrics_df).
    """
    odds = fetch_odds_current_nfl()
    metrics = fetch_nfl_team_metrics(seasons)
    return odds, metrics
