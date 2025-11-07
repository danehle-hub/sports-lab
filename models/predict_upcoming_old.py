# models/predict_upcoming_plus.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------
# Paths & env
# ------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Load .env at repo root
load_dotenv(dotenv_path=ROOT / ".env", override=True)

ODDS_API_KEY = (os.getenv("ODDS_API_KEY") or "").strip()
print(f"[oddsapi] key loaded? {'yes' if ODDS_API_KEY else 'no'}; startswith='{ODDS_API_KEY[:3] + '***' if ODDS_API_KEY else ''}'")

# nfl_data_py is optional; only used to build an upcoming slate if data/raw/upcoming.csv is missing
try:
    import nfl_data_py as nfl
    HAS_NFL = True
except Exception:
    HAS_NFL = False

# ------------------------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------------------------
GAMES          = ROOT / "data/clean/games_clean.csv"
ELO            = ROOT / "data/clean/team_week_elo.csv"        # must include elo_pre
TRENCH_WEEKLY  = ROOT / "data/clean/trench_weekly.csv"        # weekly ol_0_100, dl_0_100
OL_SEASONAL    = ROOT / "data/clean/ol_scores.csv"            # fallback
DL_SEASONAL    = ROOT / "data/clean/dl_scores.csv"            # fallback
QB_FEATS       = ROOT / "data/clean/qb_team_week.csv"         # optional rolling QB EPA
UPCOMING_CSV   = ROOT / "data/raw/upcoming.csv"               # preferred source for upcoming
BETTING_LINES  = ROOT / "data/raw/betting_lines.csv"          # optional market lines
OUT_PICKS      = ROOT / "runs/upcoming_predictions_plus.csv"

YEARS_BACK     = 7
BASE_ELO       = 1500.0
HOME_FIELD_ELO = 55.0              # used only to translate prob -> spread (rough heuristic)
ALPHA_DEFAULT  = 0.50              # OL vs DL weight for trench composite

# ------------------------------------------------------------------------------------
# Team mapping (self-contained)
# ------------------------------------------------------------------------------------
TEAM_NAME_TO_CODE = {
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL", "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF", "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE", "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND", "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC", "LAS VEGAS RAIDERS": "LV", "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR", "MIAMI DOLPHINS": "MIA", "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI", "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF", "SEATTLE SEAHAWKS": "SEA", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",
    # historical
    "OAKLAND RAIDERS": "OAK", "SAN DIEGO CHARGERS": "SD", "ST. LOUIS RAMS": "STL",
}
LENIENT_CODE_FIX = { "LA": "LAR", "LVR": "LV", "WSH": "WAS", "WFT": "WAS" }
CODE_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_CODE.items()}

NAME_ALIASES = {
    "LA CHARGERS": "LOS ANGELES CHARGERS", "L.A. CHARGERS": "LOS ANGELES CHARGERS",
    "LA RAMS": "LOS ANGELES RAMS", "L.A. RAMS": "LOS ANGELES RAMS",
    "NY JETS": "NEW YORK JETS", "N.Y. JETS": "NEW YORK JETS",
    "NY GIANTS": "NEW YORK GIANTS", "N.Y. GIANTS": "NEW YORK GIANTS",
    "JACKSONVILLE": "JACKSONVILLE JAGUARS", "HOUSTON": "HOUSTON TEXANS",
    "INDIANAPOLIS": "INDIANAPOLIS COLTS", "ATLANTA": "ATLANTA FALCONS",
    "BALTIMORE": "BALTIMORE RAVENS", "BUFFALO": "BUFFALO BILLS",
    "CAROLINA": "CAROLINA PANTHERS", "CHICAGO": "CHICAGO BEARS",
    "CLEVELAND": "CLEVELAND BROWNS", "DENVER": "DENVER BRONCOS",
    "DETROIT": "DETROIT LIONS", "GREEN BAY": "GREEN BAY PACKERS",
    "LAS VEGAS": "LAS VEGAS RAIDERS", "MIAMI": "MIAMI DOLPHINS",
    "MINNESOTA": "MINNESOTA VIKINGS", "NEW ENGLAND": "NEW ENGLAND PATRIOTS",
    "NEW ORLEANS": "NEW ORLEANS SAINTS", "PHILADELPHIA": "PHILADELPHIA EAGLES",
    "PITTSBURGH": "PITTSBURGH STEELERS", "SAN FRANCISCO": "SAN FRANCISCO 49ERS",
    "SEATTLE": "SEATTLE SEAHAWKS", "TAMPA BAY": "TAMPA BAY BUCCANEERS",
    "TENNESSEE": "TENNESSEE TITANS", "WASHINGTON": "WASHINGTON COMMANDERS",
}

_NUMERIC_WEEK_RE = re.compile(r"^\d+$")
_CODE_RE = re.compile(r"^[A-Z]{2,4}$")

def looks_like_code(x) -> bool:
    return isinstance(x, str) and bool(_CODE_RE.match(x.strip().upper()))

def _canon_team_name(x: str) -> str:
    if x is None:
        return ""
    s = " ".join(str(x).upper().strip().split())
    # common fixes
    fixes = {
        "NY JETS": "NEW YORK JETS", "N.Y. JETS": "NEW YORK JETS",
        "NY GIANTS": "NEW YORK GIANTS", "N.Y. GIANTS": "NEW YORK GIANTS",
        "LA RAMS": "LOS ANGELES RAMS", "L.A. RAMS": "LOS ANGELES RAMS",
        "LA CHARGERS": "LOS ANGELES CHARGERS", "L.A. CHARGERS": "LOS ANGELES CHARGERS",
        "N.O. SAINTS": "NEW ORLEANS SAINTS", "SAN FRAN 49ERS": "SAN FRANCISCO 49ERS",
        "SAN FRAN": "SAN FRANCISCO 49ERS",
    }
    s = fixes.get(s, s)
    return NAME_ALIASES.get(s, s)

def _to_code(val) -> Optional[str]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    sval = str(val).upper().strip()
    if not sval or sval in {"<NA>", "NAN", "NA"}:
        return None
    if _CODE_RE.match(sval):
        return LENIENT_CODE_FIX.get(sval, sval)
    cname = _canon_team_name(sval)
    return TEAM_NAME_TO_CODE.get(cname)

def _force_str_codes(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("home_code", "away_code"):
        if c in df.columns:
            df[c] = df[c].where(pd.notna(df[c]), None).astype("string")
    return df

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def need_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def coerce_season_int64(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def coerce_week_str_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col].str.match(_NUMERIC_WEEK_RE, na=False)].copy()
    return df

def ensure_dirs():
    OUT_PICKS.parent.mkdir(parents=True, exist_ok=True)

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    for side in ["home", "away"]:
        tmp = df[["schedule_season", "schedule_week", f"team_{side}", "schedule_date"]].rename(
            columns={f"team_{side}": "team"}
        )
        tmp = tmp.sort_values(["team", "schedule_date"])
        tmp["prev_date"] = tmp.groupby("team")["schedule_date"].shift(1)
        tmp["rest_days"] = (tmp["schedule_date"] - tmp["prev_date"]).dt.days
        df = df.merge(
            tmp[["schedule_season", "schedule_week", "team", "rest_days"]].rename(
                columns={"team": f"team_{side}", "rest_days": f"{side}_rest_days"}
            ),
            on=["schedule_season", "schedule_week", f"team_{side}"],
            how="left",
        )
    df["rest_diff"] = df["home_rest_days"].fillna(7) - df["away_rest_days"].fillna(7)
    return df

# ------------------------------------------------------------------------------------
# Loaders & mergers for TRAINING (historical games)
# ------------------------------------------------------------------------------------
def load_games_train() -> Tuple[pd.DataFrame, int]:
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    need_cols(g, [
        "schedule_date", "schedule_season", "schedule_week",
        "team_home", "team_away", "score_home", "score_away", "spread_home"
    ], "games_clean.csv")

    last_season = int(g["schedule_season"].max())
    cutoff = last_season - (YEARS_BACK - 1)
    g = g[g["schedule_season"] >= cutoff].copy()

    g["team_home"] = g["team_home"].str.upper().str.strip()
    g["team_away"] = g["team_away"].str.upper().str.strip()
    g = coerce_week_str_numeric(g, "schedule_week")
    g = coerce_season_int64(g, "schedule_season")
    g = g.dropna(subset=["schedule_season", "schedule_week"])

    # Codes for trench merges
    g["home_code"] = g["team_home"].map(TEAM_NAME_TO_CODE)
    g["away_code"] = g["team_away"].map(TEAM_NAME_TO_CODE)

    # Label
    g["home_cover"] = (g["score_home"] - g["score_away"] + g["spread_home"] > 0).astype(int)
    return g, last_season

def load_elo() -> pd.DataFrame:
    e = pd.read_csv(ELO)
    need_cols(e, ["schedule_season", "schedule_week", "team", "elo_pre"], "team_week_elo.csv")
    e["team"] = e["team"].str.upper().str.strip()
    e = coerce_week_str_numeric(e, "schedule_week")
    e = coerce_season_int64(e, "schedule_season")
    return e

def merge_elo(df: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        elo.rename(columns={"team": "team_home", "elo_pre": "home_elo"}),
        on=["schedule_season", "schedule_week", "team_home"], how="left",
    ).merge(
        elo.rename(columns={"team": "team_away", "elo_pre": "away_elo"}),
        on=["schedule_season", "schedule_week", "team_away"], how="left",
    )
    df["home_elo"] = df["home_elo"].fillna(BASE_ELO)
    df["away_elo"] = df["away_elo"].fillna(BASE_ELO)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df

def merge_trench_weekly_or_fallback(df: pd.DataFrame, alpha_ol: float = ALPHA_DEFAULT) -> pd.DataFrame:
    if TRENCH_WEEKLY.exists():
        tw = pd.read_csv(TRENCH_WEEKLY)
        need_cols(tw, ["season", "week", "team", "ol_0_100", "dl_0_100"], "trench_weekly.csv")
        tw["team"] = tw["team"].str.upper().str.strip().replace(LENIENT_CODE_FIX)
        tw = coerce_week_str_numeric(tw, "week")
        tw = coerce_season_int64(tw, "season")
        tw = tw.dropna(subset=["season", "week"])

        df = df.merge(
            tw.rename(columns={
                "season": "schedule_season", "week": "schedule_week", "team": "home_code",
                "ol_0_100": "home_ol_wk", "dl_0_100": "home_dl_wk"
            }),
            on=["schedule_season", "schedule_week", "home_code"], how="left",
        ).merge(
            tw.rename(columns={
                "season": "schedule_season", "week": "schedule_week", "team": "away_code",
                "ol_0_100": "away_ol_wk", "dl_0_100": "away_dl_wk"
            }),
            on=["schedule_season", "schedule_week", "away_code"], how="left",
        )

        for c in ["home_ol_wk", "away_ol_wk", "home_dl_wk", "away_dl_wk"]:
            med_by_season = df.groupby("schedule_season")[c].transform(
                lambda s: s.median(skipna=True) if s.notna().any() else np.nan
            )
            df[c] = df[c].fillna(med_by_season)
            df[c] = df[c].fillna(df[c].median(skipna=True) if df[c].notna().any() else 0.0)

        df["trench_diff"] = (
            alpha_ol * (df["home_ol_wk"] - df["away_ol_wk"])
            - (1 - alpha_ol) * (df["away_dl_wk"] - df["home_dl_wk"])
        )
        return df

    # seasonal fallback
    if OL_SEASONAL.exists() and DL_SEASONAL.exists():
        ol = pd.read_csv(OL_SEASONAL)
        dl = pd.read_csv(DL_SEASONAL)
        need_cols(ol, ["season", "team", "ol_score_0_100"], "ol_scores.csv")
        need_cols(dl, ["season", "team", "dl_score_0_100"], "dl_scores.csv")

        for tdf in (ol, dl):
            tdf["team"] = tdf["team"].str.upper().str.strip()
            coerce_season_int64(tdf, "season")

        df = df.merge(
            ol.rename(columns={"season": "schedule_season", "team": "home_code"}),
            on=["schedule_season", "home_code"], how="left"
        ).rename(columns={"ol_score_0_100": "home_ol"})
        df = df.merge(
            ol.rename(columns={"season": "schedule_season", "team": "away_code"}),
            on=["schedule_season", "away_code"], how="left"
        ).rename(columns={"ol_score_0_100": "away_ol"})
        df = df.merge(
            dl.rename(columns={"season": "schedule_season", "team": "home_code"}),
            on=["schedule_season", "home_code"], how="left"
        ).rename(columns={"dl_score_0_100": "home_dl"})
        df = df.merge(
            dl.rename(columns={"season": "schedule_season", "team": "away_code"}),
            on=["schedule_season", "away_code"], how="left"
        ).rename(columns={"dl_score_0_100": "away_dl"})

        for c in ["home_ol", "away_ol", "home_dl", "away_dl"]:
            med_by_season = df.groupby("schedule_season")[c].transform(
                lambda s: s.median(skipna=True) if s.notna().any() else np.nan
            )
            df[c] = df[c].fillna(med_by_season)
            df[c] = df[c].fillna(df[c].median(skipna=True) if df[c].notna().any() else 0.0)

        df["home_unit"] = alpha_ol * df["home_ol"] - (1 - alpha_ol) * df["away_dl"]
        df["away_unit"] = alpha_ol * df["away_ol"] - (1 - alpha_ol) * df["home_dl"]
        df["trench_diff"] = df["home_unit"] - df["away_unit"]
        return df

    # no trench available
    df["trench_diff"] = 0.0
    return df

def load_qb_feats() -> Optional[pd.DataFrame]:
    if not QB_FEATS.exists():
        return None
    q = pd.read_csv(QB_FEATS)
    need_cols(q, ["season", "week", "team", "qb_epa_roll3"], "qb_team_week.csv")
    q["team"] = q["team"].str.upper().str.strip().replace(LENIENT_CODE_FIX)
    q = coerce_week_str_numeric(q, "week")
    q = coerce_season_int64(q, "season")
    return q

def merge_qb_edges(df: pd.DataFrame, q: Optional[pd.DataFrame]) -> pd.DataFrame:
    if q is None:
        df["qb_edge"] = 0.0
        return df
    df = df.merge(
        q.rename(columns={"season": "schedule_season", "week": "schedule_week", "team": "home_code",
                          "qb_epa_roll3": "home_qb_roll3"}),
        on=["schedule_season", "schedule_week", "home_code"], how="left",
    ).merge(
        q.rename(columns={"season": "schedule_season", "week": "schedule_week", "team": "away_code",
                          "qb_epa_roll3": "away_qb_roll3"}),
        on=["schedule_season", "schedule_week", "away_code"], how="left",
    )
    df["qb_edge"] = df["home_qb_roll3"].fillna(0) - df["away_qb_roll3"].fillna(0)
    return df

def add_rest(df: pd.DataFrame) -> pd.DataFrame:
    return add_rest_days(df)

# ------------------------------------------------------------------------------------
# Build TRAINING set & fit model (+ calibration)
# ------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

def build_training(alpha_ol: float = ALPHA_DEFAULT) -> Tuple[pd.DataFrame, int]:
    g, last_season = load_games_train()
    e = load_elo()
    q = load_qb_feats()

    g = merge_elo(g, e)
    g = merge_trench_weekly_or_fallback(g, alpha_ol=alpha_ol)
    g = merge_qb_edges(g, q)
    g = add_rest(g)

    g["week_int"] = g["schedule_week"].astype(int)
    g = g.sort_values(["schedule_season", "week_int"]).drop(columns=["week_int"])

    return g, last_season

def fit_model(g: pd.DataFrame, last_season: int):
    # Keep model behavior unchanged: DO NOT include market spread as a feature
    feats = ["elo_diff", "trench_diff", "qb_edge", "rest_diff"]

    train = g[g["schedule_season"] < last_season].copy()
    X = train[feats].values
    y = train["home_cover"].astype(int).values

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=2.0)),
    ])
    model.fit(X, y)

    # Bayesian-ish calibration on logits vs rolling empirical target
    train["model_raw_prob"] = model.predict_proba(train[feats].values)[:, 1]
    train = train.dropna(subset=["model_raw_prob", "home_cover"])
    target = train["home_cover"].rolling(7, min_periods=1).mean()

    Xcal = np.vstack([
        np.ones(len(train)),
        logit(np.clip(train["model_raw_prob"], 1e-6, 1 - 1e-6))
    ]).T
    ycal = logit(np.clip(target, 1e-6, 1 - 1e-6))
    a, b = np.linalg.lstsq(Xcal, ycal, rcond=None)[0]  # (bias, slope)

    return model, feats, (a, b)

# ------------------------------------------------------------------------------------
# Upcoming slate
# ------------------------------------------------------------------------------------
def infer_next_week_from_games(g: pd.DataFrame, last_season: int) -> int:
    w = g[g["schedule_season"] == last_season]["schedule_week"].astype(int)
    return int(w.max()) + 1 if len(w) else 1

def load_or_build_upcoming(last_season: int, next_week: Optional[int]) -> pd.DataFrame:
    # Prefer user-provided CSV
    if UPCOMING_CSV.exists():
        s = pd.read_csv(UPCOMING_CSV)
        for col in ["home_team", "away_team"]:
            if col not in s.columns:
                raise ValueError(f"'{col}' missing in {UPCOMING_CSV}")
        s["home_team"] = s["home_team"].astype(str).str.upper().str.strip()
        s["away_team"] = s["away_team"].astype(str).str.upper().str.strip()
        if "season" not in s.columns:
            s["season"] = last_season
        if "week" not in s.columns:
            s["week"] = next_week if next_week is not None else 1
        s = coerce_season_int64(s, "season")
        s = coerce_week_str_numeric(s, "week")
        s = s.dropna(subset=["season", "week"])
        s = s[["season", "week", "home_team", "away_team"]].copy()
        if len(s) == 0:
            raise ValueError(f"{UPCOMING_CSV} loaded but produced 0 rows.")
        return s

    if not HAS_NFL:
        raise FileNotFoundError(
            f"{UPCOMING_CSV} not found and nfl_data_py not available. "
            "Create data/raw/upcoming.csv with columns: home_team, away_team, season, week."
        )

    if next_week is None:
        next_week = 1

    sched = nfl.import_schedules([int(last_season)]).copy()
    sched["week"] = sched["week"].astype(str).str.strip()
    sched = sched[sched["week"].str.match(_NUMERIC_WEEK_RE, na=False)].copy()
    sched["week_int"] = sched["week"].astype(int)

    s = sched[sched["week_int"] == int(next_week)].copy()
    if s.empty:
        remaining = sched[sched["week_int"] > int(next_week)].copy()
        if remaining.empty:
            s = sched[sched["week_int"] == sched["week_int"].max()].copy()
        else:
            s = remaining[remaining["week_int"] == remaining["week_int"].min()].copy()

    s["home_team"] = s["home_team"].astype(str).str.upper().str.strip()
    s["away_team"] = s["away_team"].astype(str).str.upper().str.strip()
    s["season"] = int(last_season)
    s["week"] = s["week_int"].astype(str)
    s = s[["season", "week", "home_team", "away_team"]].copy()

    if len(s) == 0:
        raise RuntimeError("Could not build an upcoming slate from schedules; got 0 rows.")

    return s

# -------------------- Online odds (The Odds API) --------------------
def fetch_oddsapi() -> Optional[pd.DataFrame]:
    """
    Pull latest NFL spreads/totals/moneylines and emit canonical names.
    Returns columns:
      ['home_team','away_team','spread_home','total_points','market_home_ml','market_away_ml',
       'home_code','away_code']
    """
    api_key = ODDS_API_KEY
    if not api_key:
        print("[oddsapi] ODDS_API_KEY not set; skipping live odds.")
        return None
    if api_key.upper() == "YOUR_REAL_KEY":
        print("[oddsapi] placeholder ODDS_API_KEY; skipping live odds.")
        return None

    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        "?regions=us&markets=spreads,h2h,totals&oddsFormat=american&dateFormat=iso"
        f"&apiKey={api_key}"
    )

    try:
        r = requests.get(url, timeout=20)
        print(f"[oddsapi] GET {url} -> {r.status_code}")
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"[oddsapi] request failed: {e}")
        return None

    rows: List[dict] = []

    for ev in events:
        home = _canon_team_name(ev.get("home_team", ""))
        away = _canon_team_name(ev.get("away_team", ""))

        spreads, totals, h2hs = [], [], []
        for bk in ev.get("bookmakers", []) or []:
            for mk in bk.get("markets", []) or []:
                key = (mk.get("key") or "").lower()
                outs = mk.get("outcomes", []) or []
                if key == "spreads":
                    for outc in outs:
                        nm = _canon_team_name(outc.get("name", ""))
                        if nm in (home, away):
                            spreads.append((nm, outc.get("point"), outc.get("price")))
                elif key == "totals":
                    for outc in outs:
                        totals.append(outc.get("point"))
                elif key == "h2h":
                    for outc in outs:
                        h2hs.append((_canon_team_name(outc.get("name","")), outc.get("price")))

        spread_home = np.nan
        if spreads:
            home_sp = [p for (nm, p, _pr) in spreads if nm == home]
            if home_sp and home_sp[0] is not None:
                try:
                    spread_home = float(home_sp[0])
                except Exception:
                    spread_home = np.nan
            else:
                away_sp = [p for (nm, p, _pr) in spreads if nm == away]
                if away_sp and away_sp[0] is not None:
                    try:
                        spread_home = -float(away_sp[0])
                    except Exception:
                        spread_home = np.nan

        total_points = float(np.median([t for t in totals if t is not None])) if totals else np.nan

        ml_home = ml_away = np.nan
        if h2hs:
            for (nm, price) in h2hs:
                if nm == home and price is not None:
                    try:
                        ml_home = int(round(float(price)))
                    except Exception:
                        pass
                elif nm == away and price is not None:
                    try:
                        ml_away = int(round(float(price)))
                    except Exception:
                        pass

        rows.append({
            "home_team": home,
            "away_team": away,
            "spread_home": spread_home,
            "total_points": total_points,
            "market_home_ml": ml_home,
            "market_away_ml": ml_away,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["home_team", "away_team"])
    df["home_code"] = df["home_team"].map(lambda x: TEAM_NAME_TO_CODE.get(_canon_team_name(x)))
    df["away_code"] = df["away_team"].map(lambda x: TEAM_NAME_TO_CODE.get(_canon_team_name(x)))
    for c in ["home_code", "away_code"]:
        df[c] = df[c].astype("string")

    print(f"[oddsapi] odds rows built: {len(df)}")
    return df

def add_market_to_upcoming(s: pd.DataFrame) -> pd.DataFrame:
    """Merge market lines into the upcoming slate without dropping rows."""
    tmp = s.copy()

    # Ensure home_team/away_team exist
    if {"home_team", "away_team"} <= set(tmp.columns):
        pass
    elif {"team_home", "team_away"} <= set(tmp.columns):
        tmp["home_team"] = tmp["team_home"]
        tmp["away_team"] = tmp["team_away"]
    else:
        tmp["home_team"] = tmp.get("team_home", np.nan)
        tmp["away_team"] = tmp.get("team_away", np.nan)

    # Canonicalized names (for name-joins)
    tmp["home_team_c"] = tmp["home_team"].apply(_canon_team_name)
    tmp["away_team_c"] = tmp["away_team"].apply(_canon_team_name)

    # Codes in upcoming
    tmp["home_code"] = tmp["home_team"].apply(_to_code)
    tmp["away_code"] = tmp["away_team"].apply(_to_code)
    for c in ["home_code", "away_code"]:
        tmp[c] = tmp[c].astype("string")

    # Try live odds
    live = fetch_oddsapi()
    if live is not None and not live.empty:
        live_name = live.copy()
        live_name["home_team_c"] = live_name["home_team"].apply(_canon_team_name)
        live_name["away_team_c"] = live_name["away_team"].apply(_canon_team_name)

        live_code = live.copy()
        for c in ["home_code", "away_code"]:
            live_code[c] = live_code[c].astype("string")

        take = [c for c in ["spread_home", "total_points", "market_home_ml", "market_away_ml"] if c in live.columns]

        # NAME join
        name_join = tmp.merge(
            live_name[["home_team_c", "away_team_c"] + take],
            on=["home_team_c", "away_team_c"],
            how="left",
            suffixes=("", "_nm"),
        )
        name_hits = sum(name_join[c].notna().sum() for c in take)

        # CODE join (fill where missing)
        code_join = name_join.merge(
            live_code[["home_code", "away_code"] + take],
            on=["home_code", "away_code"],
            how="left",
            suffixes=("", "_cd"),
        )
        code_hits = 0
        for c in take:
            c_cd = f"{c}_cd"
            if c_cd in code_join.columns:
                before = code_join[c].notna().sum()
                code_join[c] = np.where(code_join[c].isna(), code_join[c_cd], code_join[c])
                after = code_join[c].notna().sum()
                code_hits += max(0, after - before)

        print(f"[oddsapi] merge preview: matched values ~ {name_hits} by names, + {code_hits} by codes, of {len(tmp)} rows")

        # Debug CSV
        try:
            (ROOT / "runs").mkdir(parents=True, exist_ok=True)
            debug_cols = [
                "home_team", "away_team", "home_team_c", "away_team_c",
                "home_code", "away_code",
            ] + take
            code_join[debug_cols].to_csv(ROOT / "runs" / "odds_debug_join.csv", index=False)
        except Exception as e:
            print(f"[oddsapi] (debug write skipped) {e}")

        drop_cols = [c for c in code_join.columns if c.endswith("_nm") or c.endswith("_cd")]
        code_join.drop(columns=drop_cols, inplace=True, errors="ignore")
        return code_join

    # Fallback: local CSV
    if BETTING_LINES.exists():
        bl = pd.read_csv(BETTING_LINES)
        for col in bl.columns:
            if bl[col].dtype == object:
                bl[col] = bl[col].astype(str).str.upper().str.strip()
        if "spread_line" in bl.columns:
            bl = bl.rename(columns={"spread_line": "spread_home"})
        if "total_line" in bl.columns:
            bl = bl.rename(columns={"total_line": "total_points"})
        bl["home_team_c"] = bl["home_team"].apply(_canon_team_name)
        bl["away_team_c"] = bl["away_team"].apply(_canon_team_name)
        merged = tmp.merge(
            bl[["home_team_c", "away_team_c", "spread_home", "total_points"]],
            on=["home_team_c", "away_team_c"],
            how="left",
        )
        return merged

    # Ensure columns exist
    for col in ["spread_home", "total_points", "market_home_ml", "market_away_ml"]:
        if col not in tmp.columns:
            tmp[col] = np.nan
    return tmp

# ------------------------------------------------------------------------------------
# Features for UPCOMING
# ------------------------------------------------------------------------------------
def prepare_upcoming_features(s: pd.DataFrame, alpha_ol: float = 0.5) -> pd.DataFrame:
    """
    Returns feature-ready df for upcoming games.
    Accepts either:
      - season, week, home_team, away_team
      - or schedule_season, schedule_week, team_home, team_away
    """
    # Normalize column names
    colmap = {}
    if "home_team" in s.columns and "team_home" not in s.columns:
        colmap["home_team"] = "team_home"
    if "away_team" in s.columns and "team_away" not in s.columns:
        colmap["away_team"] = "team_away"
    if "season" in s.columns and "schedule_season" not in s.columns:
        colmap["season"] = "schedule_season"
    if "week" in s.columns and "schedule_week" not in s.columns:
        colmap["week"] = "schedule_week"
    s = s.rename(columns=colmap)

    required = ["schedule_season", "schedule_week", "team_home", "team_away"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"Upcoming schedule is missing columns: {missing}")

    # Clean types & casing
    s["schedule_season"] = pd.to_numeric(s["schedule_season"], errors="coerce").astype("Int64")
    s["schedule_week"]   = s["schedule_week"].astype(str).str.strip()
    s["team_home"] = s["team_home"].apply(lambda v: v.strip().upper() if isinstance(v, str) else v)
    s["team_away"] = s["team_away"].apply(lambda v: v.strip().upper() if isinstance(v, str) else v)
    s = s.dropna(subset=["schedule_season"])

    # Detect code vs name
    home_is_code = s["team_home"].map(looks_like_code)
    away_is_code = s["team_away"].map(looks_like_code)

    # Codes for trenches
    s["home_code"] = np.where(home_is_code, s["team_home"], s["team_home"].map(TEAM_NAME_TO_CODE))
    s["away_code"] = np.where(away_is_code, s["team_away"], s["team_away"].map(TEAM_NAME_TO_CODE))
    s["home_code"] = s["home_code"].replace(LENIENT_CODE_FIX)
    s["away_code"] = s["away_code"].replace(LENIENT_CODE_FIX)

    # Full names for Elo merges
    s["team_home_full"] = np.where(
        home_is_code, s["team_home"].replace(LENIENT_CODE_FIX).map(CODE_TO_TEAM_NAME), s["team_home"]
    )
    s["team_away_full"] = np.where(
        away_is_code, s["team_away"].replace(LENIENT_CODE_FIX).map(CODE_TO_TEAM_NAME), s["team_away"]
    )

    # Elo (pre-game) via full names
    e = load_elo()
    df = s.merge(
        e.rename(columns={"team": "team_home", "elo_pre": "home_elo"}),
        left_on=["schedule_season", "schedule_week", "team_home_full"],
        right_on=["schedule_season", "schedule_week", "team_home"],
        how="left",
    ).merge(
        e.rename(columns={"team": "team_away", "elo_pre": "away_elo"}),
        left_on=["schedule_season", "schedule_week", "team_away_full"],
        right_on=["schedule_season", "schedule_week", "team_away"],
        how="left",
    )
    for c in ["team_home_x", "team_home_y", "team_away_x", "team_away_y"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors="ignore")

    df["home_elo"] = df["home_elo"].fillna(BASE_ELO)
    df["away_elo"] = df["away_elo"].fillna(BASE_ELO)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Trench by codes
    df = merge_trench_weekly_or_fallback(df, alpha_ol=alpha_ol)
    if "trench_diff" not in df.columns:
        df["trench_diff"] = 0.0

    # QB edges (optional)
    q = load_qb_feats() if "load_qb_feats" in globals() else None
    if q is not None:
        df = df.merge(
            q.rename(columns={"season":"schedule_season","week":"schedule_week","team":"home_code",
                              "qb_epa_roll3":"home_qb_roll3"}),
            on=["schedule_season","schedule_week","home_code"], how="left"
        ).merge(
            q.rename(columns={"season":"schedule_season","week":"schedule_week","team":"away_code",
                              "qb_epa_roll3":"away_qb_roll3"}),
            on=["schedule_season","schedule_week","away_code"], how="left"
        )
        df["qb_edge"] = df["home_qb_roll3"].fillna(0) - df["away_qb_roll3"].fillna(0)
    else:
        df["qb_edge"] = 0.0

    # Upcoming has no dates → set rest diff to 0
    df["rest_diff"] = 0.0

    # Ensure market spread column exists
    if "spread_home" not in df.columns:
        df["spread_home"] = np.nan

    # Final safety fills
    for c in ["elo_diff", "trench_diff", "qb_edge", "rest_diff"]:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)

    # Present full names
    df["team_home"] = s["team_home_full"]
    df["team_away"] = s["team_away_full"]

    assert isinstance(df, pd.DataFrame), f"prepare_upcoming_features must return DataFrame, got {type(df)}"
    return df

# ------------------------------------------------------------------------------------
# Prob ↔ spread helpers
# ------------------------------------------------------------------------------------
def prob_to_model_spread(p_home: np.ndarray) -> np.ndarray:
    """
    Convert calibrated home cover probability to an equivalent 'model spread' in points
    via an Elo heuristic (25 Elo ~ 1 point; 400 Elo ~ 10x odds).
    Positive model_spread => home stronger.
    """
    p = np.clip(p_home, 1e-4, 1 - 1e-4)
    elo_diff = -400.0 * np.log10((1.0 / p) - 1.0)      # Elo diff favoring HOME
    return elo_diff / 25.0                              # ~ points

def elo_from_spread(spread_points: np.ndarray) -> np.ndarray:
    return np.asarray(spread_points, dtype=float) * 25.0

def win_prob_from_elo_diff(elo_diff: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(elo_diff, dtype=float) / 400.0))

def win_prob_from_model_spread(model_spread: np.ndarray) -> np.ndarray:
    return win_prob_from_elo_diff(elo_from_spread(model_spread))

def prob_to_american_ml(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    am = np.where(
        p >= 0.5,
        -100.0 * p / (1.0 - p),
        100.0 * (1.0 - p) / p
    )
    return np.round(am).astype(int)

def american_to_decimal(ml: np.ndarray) -> np.ndarray:
    ml = np.asarray(ml, dtype=float)
    return np.where(ml >= 0, 1.0 + ml / 100.0, 1.0 + 100.0 / np.abs(ml))

def ev_from_prob_and_american(p: np.ndarray, ml: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    dec = american_to_decimal(ml)
    return p * (dec - 1.0) - (1.0 - p)

# ------------------------------------------------------------------------------------
# Bayesian calibration: map raw model probs -> calibrated probs
# ------------------------------------------------------------------------------------
def bayesian_calibrate(raw_prob: np.ndarray, calib: tuple) -> np.ndarray:
    a, b = calib
    rp = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    return expit(a + b * logit(rp))

def american_to_decimal(american):
    """
    Robustly convert American odds to decimal odds.
    Accepts scalars or array-like. Returns float(s), np.nan on failure.
    Handles strings, 'EVEN'/'EV' and 'PK'/'pick' (treated as +100 = 2.00).
    """
    import math
    import numpy as np

    def _one(x):
        # Missing?
        if x is None:
            return np.nan
        # Already numeric?
        try:
            # Keep strings out of this try first
            if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
                a = float(x)
                if a >= 100:
                    return 1.0 + (a / 100.0)
                if a <= -100:
                    return 1.0 + (100.0 / abs(a))
                return np.nan
        except Exception:
            pass

        # String cases
        s = str(x).strip().lower().replace(',', '')
        if s in {"even", "ev", "evs", "pk", "pick", "pickem", "pick’em"}:
            # Treat as +100 (even money)
            return 2.0
        try:
            a = float(s)
        except Exception:
            return np.nan

        if a >= 100:
            return 1.0 + (a / 100.0)
        if a <= -100:
            return 1.0 + (100.0 / abs(a))
        return np.nan

    # Vectorize safely
    if np.isscalar(american):
        return _one(american)
    arr = np.asarray(american, dtype=object)
    return np.vectorize(_one, otypes=[float])(arr)

def ev_from_prob_and_american(p, american):
    """
    Expected value per $1 stake from win prob p and American odds.
    Accepts scalars or array-like. Returns float or np.ndarray of floats.
    Returns np.nan when odds are invalid/missing.
    EV formula (decimal odds d): EV = p*(d-1) - (1-p)*1
    """
    import numpy as np

    p_arr = np.asarray(p, dtype=float)
    d_arr = american_to_decimal(american)

    # Ensure array shapes line up for vector math
    d_arr = np.asarray(d_arr, dtype=float)

    b = d_arr - 1.0
    # Valid where decimal odds are > 1 (i.e., have a real payout)
    valid = np.isfinite(d_arr) & (d_arr > 1.0) & np.isfinite(p_arr)

    ev = np.full_like(p_arr, np.nan, dtype=float)
    ev[valid] = p_arr[valid] * b[valid] - (1.0 - p_arr[valid])
    return ev if ev.shape != () else float(ev)  # scalar if scalar in, array if array in


# ------------------------------------------------------------------------------------
# Betting tiers & unit sizing (fractional Kelly for -110 spread bets)
# ------------------------------------------------------------------------------------
def add_tiers_and_units(
    df: pd.DataFrame,
    kelly_fraction: float = 0.25,
    price_american: int = -110,
) -> pd.DataFrame:
    """
    Tiering + unit sizing for spread bets at a fixed juice (default -110).
    Uses:
      - edge_vs_market = model_home_line - spread_home  (negative => model likes HOME)
      - p_home_cover   = P(home covers the MARKET spread) [already calibrated upstream]
    """
    out = df.copy()

    # --- Sanity: make sure needed columns exist ---
    for c in ["edge_vs_market", "home_cover_prob"]:
        if c not in out.columns:
            out[c] = np.nan

    # --- Decide side with value from edge ---
    # Negative edge means model_home_line < market => value on HOME (you get more points than fair)
    out["bet_side"] = np.where(pd.to_numeric(out["edge_vs_market"], errors="coerce") < 0, "HOME", "AWAY")

    # --- Probability of the chosen side covering the MARKET line ---
    p_home = pd.to_numeric(out.get("home_cover_prob", 0.5), errors="coerce").fillna(0.5).to_numpy()
    bet_side = out["bet_side"].to_numpy()
    p_chosen = np.where(bet_side == "HOME", p_home, 1.0 - p_home)

    # --- EV at given juice ---
    # b is decimal profit per 1u at the given American price
    b = (100.0 / abs(price_american)) if price_american < 0 else (price_american / 100.0)  # -110 -> 0.9091
    q = 1.0 - p_chosen
    ev = p_chosen * b - q

    # --- Edge magnitude (points) ---
    abs_edge = np.abs(pd.to_numeric(out["edge_vs_market"], errors="coerce").fillna(0.0).to_numpy())

    # ------------------------------------------------------------------
    # Tier rules (tuned to actually recommend when there’s real value)
    # STRONG:
    #   • abs_edge >= 3.0 AND EV >= +0.010    (about p_chosen ~0.525@-110)
    #   OR
    #   • abs_edge >= 5.0                     (massive line value even if calibration is conservative)
    #
    # LEAN:
    #   • abs_edge >= 1.5 AND EV >= +0.002    (very small +EV but decent line value)
    #
    # Otherwise: PASS
    # ------------------------------------------------------------------
    strong_mask = ((abs_edge >= 3.0) & (ev >= 0.010)) | (abs_edge >= 5.0)
    lean_mask   = (~strong_mask) & (abs_edge >= 1.5) & (ev >= 0.002)

    tier = np.where(strong_mask, "STRONG", np.where(lean_mask, "LEAN", "PASS"))

    # --- Kelly units on p_chosen ---
    full_kelly = (b * p_chosen - q) / b
    full_kelly = np.clip(full_kelly, 0.0, 1.0)
    units = full_kelly * float(kelly_fraction)

    # Tier minimums (optional but helpful)
    units = np.where(tier == "STRONG", np.maximum(units, 0.25), units)  # 0.25u minimum for STRONG
    units = np.where(tier == "LEAN",   np.maximum(units, 0.10), units)  # 0.10u minimum for LEAN
    units = np.where(tier == "PASS",   0.0,                    units)

    out["pick_tier"] = tier
    out["units"] = np.round(np.clip(units, 0.0, 2.0), 2)
    out["pick"] = np.where(out["pick_tier"] == "PASS", "PASS", out["bet_side"])
    out["pick_conf"] = np.round(p_chosen, 4)
    out["spread_ev"] = np.round(ev, 4)

    # --- Debugging columns so you can audit why something was PASS/LEAN/STRONG ---
    out["abs_edge"] = np.round(abs_edge, 2)
    out["tier_debug"] = np.where(
        out["pick_tier"].eq("STRONG"),
        "STRONG: (abs_edge>=3 & EV>=0.01) or abs_edge>=5",
        np.where(
            out["pick_tier"].eq("LEAN"),
            "LEAN: abs_edge>=1.5 & EV>=0.002",
            "PASS: below thresholds"
        )
    )
    out["price_used"] = price_american

    return out


# ------------------------------------------------------------------------------------
# Monte Carlo simulation of portfolio returns (spread bets)
# ------------------------------------------------------------------------------------
def run_monte_carlo(card: pd.DataFrame,
                    n_sims: int = 10000,
                    price_american: int = -110,
                    seed: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    bets = card.loc[card["pick_tier"].isin(["STRONG", "LEAN"])].copy()
    if bets.empty:
        return (
            pd.DataFrame({
                "n_bets": [0], "n_sims": [n_sims],
                "mean_roi": [0.0], "p_roi_pos": [0.0],
                "roi_p05": [0.0], "roi_p50": [0.0], "roi_p95": [0.0]
            }),
            pd.DataFrame({"sim_id": np.arange(n_sims), "roi": np.zeros(n_sims)})
        )

    b = (100.0 / abs(price_american)) if price_american < 0 else (price_american / 100.0)

    p = np.clip(bets["pick_conf"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    u = bets["units"].astype(float).to_numpy()

    wins = rng.binomial(1, p, size=(n_sims, p.size)).astype(float)

    profit_matrix = wins * (u * b) - (1.0 - wins) * u
    total_profit = profit_matrix.sum(axis=1)

    total_staked = np.full(n_sims, u.sum(), dtype=float)
    roi = np.where(total_staked > 0, total_profit / total_staked, 0.0)

    sim_paths = pd.DataFrame({"sim_id": np.arange(n_sims), "roi": roi})

    sim_summary = pd.DataFrame({
        "n_bets": [int(u.size)],
        "n_sims": [int(n_sims)],
        "mean_roi": [float(roi.mean())],
        "p_roi_pos": [float((roi > 0).mean())],
        "roi_p05": [float(np.percentile(roi, 5))],
        "roi_p50": [float(np.percentile(roi, 50))],
        "roi_p95": [float(np.percentile(roi, 95))],
    })

    return sim_summary, sim_paths

# ------------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------------
def _predict_and_export(kelly_fraction: float = 0.25, alpha_ol: float = ALPHA_DEFAULT):
    ensure_dirs()

    # 1) Train + calibrate
    base_g, last_season = build_training(alpha_ol=alpha_ol)
    model, feats, calib = fit_model(base_g, last_season)

    # 2) Guess next week
    next_week = infer_next_week_from_games(base_g, last_season)

    # 3) Upcoming slate
    s = load_or_build_upcoming(last_season=last_season, next_week=next_week)

    # 4) Attach market lines (if available)
    s = add_market_to_upcoming(s)

    # 5) Build features
    up = prepare_upcoming_features(s, alpha_ol=alpha_ol)
    if up is None or len(up) == 0:
        print("[warn] no upcoming features — nothing to predict.")
        OUT_PICKS.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "season","week","team_home","team_away","spread_home",
            "home_cover_prob_raw","home_cover_prob","model_spread","edge_vs_market",
            "model_home_win_prob","model_away_win_prob","model_home_ml","model_away_ml",
            "pick","pick_tier","units","pick_conf","market_home_ml","market_away_ml"
        ]).to_csv(OUT_PICKS, index=False)
        return

    # 6) Predict & calibrate
    Xf = up[feats].values
    raw_prob = model.predict_proba(Xf)[:, 1]
    p_home_cover = bayesian_calibrate(raw_prob, calib)

    out = up.copy()
    out["home_cover_prob_raw"] = raw_prob
    out["home_cover_prob"] = p_home_cover

    # --- Model line (no binary search): convert calibrated prob -> equivalent spread ---
    out["model_spread"] = prob_to_model_spread(out["home_cover_prob"].values)    # positive = home better
    out["model_home_line"] = -out["model_spread"]                                # market convention: home favs negative

    # Clamp to a sane NFL range for readability
    MAX_ABS_FAIR = 12.0
    out["model_home_line_raw"] = out["model_home_line"]
    out["clamped"] = (out["model_home_line"].abs() > MAX_ABS_FAIR)
    out["model_home_line"] = np.clip(out["model_home_line"], -MAX_ABS_FAIR, MAX_ABS_FAIR)

    # Edge (positive => model likes HOME; negative => model likes AWAY)
    out["edge_vs_market"] = out["model_home_line"] - out.get("spread_home", np.nan)

        # --- Edge vs market & friendly strings ---
    out["edge_vs_market"] = out["model_home_line"] - out.get("spread_home", np.nan)
    out["home_value_pts"] = out["spread_home"] - out["model_home_line"]

    out["market_line"] = out.apply(
        lambda r: f"{r['team_home']} {r['spread_home']:+.1f}" if pd.notna(r['spread_home']) else "N/A",
        axis=1
    )
    out["model_line"] = out.apply(
        lambda r: f"{r['team_home']} {r['model_home_line']:+.1f}" if pd.notna(r['model_home_line']) else "N/A",
        axis=1
    )
    out["value_side"] = np.where(out["home_value_pts"] > 0, out["team_home"], out["team_away"])
    out["value_pts"]  = out["home_value_pts"].abs().round(1)

    # --- ML fairs & EV (optional if market MLs present) ---
    out["model_home_win_prob"] = win_prob_from_model_spread(out["model_spread"].values)
    out["model_away_win_prob"] = 1.0 - out["model_home_win_prob"]
    out["model_home_ml"] = prob_to_american_ml(out["model_home_win_prob"].values)
    out["model_away_ml"] = prob_to_american_ml(out["model_away_win_prob"].values)

    if {"market_home_ml","market_away_ml"}.issubset(out.columns):
        out["home_ml_ev"] = ev_from_prob_and_american(out["model_home_win_prob"].values, out["market_home_ml"].values)
        out["away_ml_ev"] = ev_from_prob_and_american(out["model_away_win_prob"].values, out["market_away_ml"].values)

        # --- Convert fair line vs market line into P(home covers market) ---
    from math import erf, sqrt

    def norm_cdf(x: float) -> float:
        # Φ(x) without importing scipy here
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    # NFL ATS noise ~ 13–14 pts works well; tune if you like
    ATS_SIGMA = 13.5

    # delta > 0 means market is giving the home team more points than our fair → higher cover prob
    delta = pd.to_numeric(out["spread_home"], errors="coerce") - pd.to_numeric(out["model_home_line"], errors="coerce")
    out["p_home_cover_market"] = delta.apply(lambda d: norm_cdf(d / ATS_SIGMA) if pd.notna(d) else np.nan)

    # If you had an older 'home_cover_prob' that wasn't at the market number, replace it:
    out["home_cover_prob"] = out["p_home_cover_market"].fillna(out.get("home_cover_prob"))


    # --- Tiers/units (uses only spread info; safe to run here) ---
    out = add_tiers_and_units(out, kelly_fraction=0.25)

    # --- Save files ---
    OUT_PICKS.parent.mkdir(parents=True, exist_ok=True)
    final_cols = [
        "schedule_season","schedule_week","team_home","team_away","spread_home",
        "home_cover_prob_raw","home_cover_prob","model_spread","model_home_line","edge_vs_market",
        "market_line","model_line","value_side","value_pts",
        "model_home_win_prob","model_away_win_prob","model_home_ml","model_away_ml",
        "pick","pick_tier","units","pick_conf",
        "market_home_ml","market_away_ml","home_ml_ev","away_ml_ev"
    ]
    final_cols = [c for c in final_cols if c in out.columns]
    final = out[final_cols].rename(columns={"schedule_season":"season","schedule_week":"week"}).copy()
    final.to_csv(OUT_PICKS, index=False)

    features_out = ROOT / "runs" / "upcoming_features.csv"
    keep = ["schedule_season","schedule_week","team_home","team_away","spread_home","model_spread","home_cover_prob"]
    out[keep].to_csv(features_out, index=False)

    ml_cols = [
        "schedule_season","schedule_week","team_home","team_away",
        "model_home_win_prob","model_away_win_prob","model_home_ml","model_away_ml",
        "market_home_ml","market_away_ml","home_ml_ev","away_ml_ev"
    ]
    ml_cols = [c for c in ml_cols if c in out.columns]
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (out[ml_cols]).to_csv(ROOT / "runs" / "moneyline_card.csv", index=False)

    # --- Pretty print ---
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    to_show = final.copy()
    for c in ["home_cover_prob_raw","home_cover_prob","model_spread","model_home_line",
            "edge_vs_market","model_home_win_prob","model_away_win_prob","pick_conf"]:
        if c in to_show.columns and pd.api.types.is_numeric_dtype(to_show[c]):
            to_show[c] = to_show[c].round(4)
    print(to_show.to_string(index=False))


    # Friendly strings
    out["market_line"] = out.apply(
        lambda r: f"{r['team_home']} {r['spread_home']:+.1f}" if pd.notna(r['spread_home']) else "N/A", axis=1
    )
    out["model_line"] = out.apply(
        lambda r: f"{r['team_home']} {r['model_home_line']:+.1f}" if pd.notna(r['model_home_line']) else "N/A", axis=1
    )
    out["home_value_pts"] = out["spread_home"] - out["model_home_line"]
    out["value_side"]  = np.where(out["home_value_pts"] > 0, out["team_home"], out["team_away"])
    out["value_pts"]   = out["home_value_pts"].abs().round(1)

    # Moneyline fair probs/lines from model spread
    out["model_home_win_prob"] = win_prob_from_model_spread(out["model_spread"].values)
    out["model_away_win_prob"] = 1.0 - out["model_home_win_prob"]
    out["model_home_ml"] = prob_to_american_ml(out["model_home_win_prob"].values)
    out["model_away_ml"] = prob_to_american_ml(out["model_away_win_prob"].values)

    # Right before:
    # out["home_ml_ev"] = ev_from_prob_and_american(out["model_home_win_prob"].values, out["market_home_ml"].values)
    # add this normalization (won't hurt if already numeric):

    for col in ["market_home_ml", "market_away_ml"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(object)
                .map(lambda x: None if x in ("", "None", "nan", "NaN") else x)
            )


    # If market MLs exist, compute EV vs market (per 1u)
    if {"market_home_ml","market_away_ml"}.issubset(out.columns):
        out["home_ml_ev"] = ev_from_prob_and_american(out["model_home_win_prob"].values, out["market_home_ml"].values)
        out["away_ml_ev"] = ev_from_prob_and_american(out["model_away_win_prob"].values, out["market_away_ml"].values)

    # Tiers & unit sizing
    out = add_tiers_and_units(out, kelly_fraction=kelly_fraction)

    # ---- Exports ----
    cols_base = [
        "schedule_season","schedule_week","team_home","team_away","spread_home",
        "home_cover_prob_raw","home_cover_prob","model_spread","model_home_line","edge_vs_market",
        "market_line","model_line","value_side","value_pts",
        "model_home_win_prob","model_away_win_prob","model_home_ml","model_away_ml",
        "pick","pick_tier","units","pick_conf",
        "market_home_ml","market_away_ml","home_ml_ev","away_ml_ev"
    ]
    cols = [c for c in cols_base if c in out.columns]
    final = out[cols].rename(columns={"schedule_season":"season","schedule_week":"week"}).copy()

    OUT_PICKS.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PICKS, index=False)

    # Light features dump
    features_out = ROOT / "runs" / "upcoming_features.csv"
    keep = ["schedule_season","schedule_week","team_home","team_away","spread_home","model_spread","home_cover_prob"]
    out[keep].to_csv(features_out, index=False)

    # Moneyline card
    ml_card = out.copy()
    if {"market_home_ml","market_away_ml"}.issubset(ml_card.columns):
        ml_card = ml_card.assign(best_ml_ev=np.maximum(
            ml_card.get("home_ml_ev", np.nan),
            ml_card.get("away_ml_ev", np.nan)
        )).sort_values("best_ml_ev", ascending=False)
    else:
        ml_card = ml_card.sort_values("model_away_win_prob", ascending=False)

    ml_cols = [
        "schedule_season","schedule_week","team_home","team_away",
        "model_home_win_prob","model_away_win_prob","model_home_ml","model_away_ml",
        "market_home_ml","market_away_ml","home_ml_ev","away_ml_ev"
    ]
    ml_cols = [c for c in ml_cols if c in ml_card.columns]
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (ml_card[ml_cols]).to_csv(ROOT / "runs" / "moneyline_card.csv", index=False)

    # Monte Carlo portfolio simulation (spread bets)
    mc_summary, mc_paths = run_monte_carlo(out, n_sims=10000, price_american=-110, seed=42)
    mc_summary.to_csv(ROOT / "runs" / "mc_summary.csv", index=False)
    mc_paths.to_csv(ROOT / "runs" / "mc_paths.csv", index=False)

    print(f"✅ Saved predictions → {OUT_PICKS}")
    print(f"🧩 Saved features     → {features_out}")
    print(f"🗂  Saved moneyline card → {ROOT / 'runs' / 'moneyline_card.csv'}")

    # Pretty print for terminal
    show = final.copy()
    for c in ["home_cover_prob_raw", "home_cover_prob", "pick_conf",
              "model_spread", "model_home_line", "edge_vs_market", "spread_home",
              "model_home_win_prob", "model_away_win_prob"]:
        if c in show.columns and pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].round(4)
            pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(show.to_string(index=False))

def _is_bettable(df: pd.DataFrame) -> pd.Series:
    """
    A game is 'bettable' only if we have a market spread.
    (Add more conditions here if you want to filter further.)
    """
    return df["spread_home"].notna()

# ---------- CLI ----------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run upcoming predictions (+extras).")
    parser.add_argument("--alpha-ol", type=float, default=ALPHA_DEFAULT,
                        help="Trench weight for OL vs DL composite (0..1). Default 0.50.")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Fractional Kelly for units (default 0.25).")
    args = parser.parse_args()
    _predict_and_export(kelly_fraction=args.kelly_fraction, alpha_ol=args.alpha_ol)

if __name__ == "__main__":
    _cli()
