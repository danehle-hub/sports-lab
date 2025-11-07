# models/features.py  (FULL REPLACEMENT: adds team-name→code mapping and joins on codes)

import numpy as np
import pandas as pd

# Map The Odds API full names (and common variants) → nflfastR team codes
TEAM_NAME_TO_CODE = {
    # NFC
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "dallas cowboys": "DAL",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "los angeles rams": "LAR",
    "minnesota vikings": "MIN",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "philadelphia eagles": "PHI",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "washington commanders": "WAS",
    # AFC
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "denver broncos": "DEN",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "miami dolphins": "MIA",
    "new england patriots": "NE",
    "new york jets": "NYJ",
    "pittsburgh steelers": "PIT",
    "tennessee titans": "TEN",
    # Common legacy/alt spellings that sometimes appear
    "washington football team": "WAS",
    "oakland raiders": "LV",
    "san diego chargers": "LAC",
    "st. louis rams": "LAR",
}

def team_to_code(name: str) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    key = name.strip().lower()
    return TEAM_NAME_TO_CODE.get(key)

def _cap(s, lo, hi):
    return np.clip(s, lo, hi)

def _latest_team_rows(metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For each team code in nflfastR metrics, take the latest available row.
    """
    m = metrics.copy()
    # nfl_data_py uses three-letter codes in 'team'
    if "season" in m.columns and "week" in m.columns:
        m = m.sort_values(["team", "season", "week"])
    else:
        m = m.sort_values(["team"])
        m["season"] = np.nan
        m["week"] = np.nan
    wanted = ["team", "season", "week"] + [c for c in cols if c in m.columns]
    m = m[wanted]
    last = m.groupby("team", as_index=False).tail(1).reset_index(drop=True)
    return last

def build_component_lines(games: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
      games: from Odds API. Needs team_home, team_away, spread_home, (prices optional)
      metrics: team-week rolling metrics from nfl_data_py; 'team' uses nflfastR codes (e.g., ATL)
    Output:
      df with *_line component columns and initial model_home_line (baseline 0).
    """
    df = games.copy()

    # ------------ add team codes and join on them ------------
    df["team_home_code"] = df["team_home"].map(team_to_code)
    df["team_away_code"] = df["team_away"].map(team_to_code)

    # If any code is missing, log-friendly filler (won't crash)
    # We leave them as NaN; they’ll be filled by league means later.
    metric_cols = [
        "off_epa_r8", "off_sr_r8", "def_epa_r8", "def_sr_r8",
        "pass_rate_r8", "rush_rate_r8", "qb_epa_r8"
    ]
    latest = _latest_team_rows(metrics, metric_cols)

    # HOME join
    home = latest.rename(columns={
        "team": "team_home_code",
        "off_epa_r8": "off_epa_r8_home", "off_sr_r8": "off_sr_r8_home",
        "def_epa_r8": "def_epa_r8_home", "def_sr_r8": "def_sr_r8_home",
        "pass_rate_r8": "pass_rate_r8_home", "rush_rate_r8": "rush_rate_r8_home",
        "qb_epa_r8": "qb_epa_r8_home",
        "season": "season_home_metrics", "week": "week_home_metrics",
    })
    # AWAY join
    away = latest.rename(columns={
        "team": "team_away_code",
        "off_epa_r8": "off_epa_r8_away", "off_sr_r8": "off_sr_r8_away",
        "def_epa_r8": "def_epa_r8_away", "def_sr_r8": "def_sr_r8_away",
        "pass_rate_r8": "pass_rate_r8_away", "rush_rate_r8": "rush_rate_r8_away",
        "qb_epa_r8": "qb_epa_r8_away",
        "season": "season_away_metrics", "week": "week_away_metrics",
    })

    df = df.merge(
        home[["team_home_code","off_epa_r8_home","off_sr_r8_home","def_epa_r8_home","def_sr_r8_home","qb_epa_r8_home",
              "season_home_metrics","week_home_metrics"]],
        on="team_home_code", how="left"
    ).merge(
        away[["team_away_code","off_epa_r8_away","off_sr_r8_away","def_epa_r8_away","def_sr_r8_away","qb_epa_r8_away",
              "season_away_metrics","week_away_metrics"]],
        on="team_away_code", how="left"
    )

    # ------------ fill NaNs with league means so we always have numbers ------------
    metric_fill_cols = [
        "off_epa_r8_home","off_sr_r8_home","def_epa_r8_home","def_sr_r8_home","qb_epa_r8_home",
        "off_epa_r8_away","off_sr_r8_away","def_epa_r8_away","def_sr_r8_away","qb_epa_r8_away",
    ]
    if not metrics.empty:
        league_means = {}
        for base in ["off_epa_r8","off_sr_r8","def_epa_r8","def_sr_r8","qb_epa_r8"]:
            league_means[base] = float(metrics[base].mean(skipna=True)) if base in metrics.columns else np.nan
        lm = {
            "off_epa_r8_home": league_means.get("off_epa_r8", 0.0),
            "off_sr_r8_home":  league_means.get("off_sr_r8", 0.5),
            "def_epa_r8_home": league_means.get("def_epa_r8", 0.0),
            "def_sr_r8_home":  league_means.get("def_sr_r8", 0.5),
            "qb_epa_r8_home":  league_means.get("qb_epa_r8", 0.0),
            "off_epa_r8_away": league_means.get("off_epa_r8", 0.0),
            "off_sr_r8_away":  league_means.get("off_sr_r8", 0.5),
            "def_epa_r8_away": league_means.get("def_epa_r8", 0.0),
            "def_sr_r8_away":  league_means.get("def_sr_r8", 0.5),
            "qb_epa_r8_away":  league_means.get("qb_epa_r8", 0.0),
        }
        for c in metric_fill_cols:
            if c in df.columns:
                df[c] = df[c].fillna(lm.get(c, 0.0))
    else:
        for c in metric_fill_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

    # ------------ diffs (home - away) ------------
    df["off_epa_diff"] = df["off_epa_r8_home"] - df["off_epa_r8_away"]
    df["def_epa_diff"] = df["def_epa_r8_home"] - df["def_epa_r8_away"]
    df["qb_grade_diff"] = df["qb_epa_r8_home"] - df["qb_epa_r8_away"]

    # ------------ placeholders for flags (wire real sources later) ------------
    df["qb_status_risk"] = 1.0
    df["rest_days_diff"] = 0.0
    df["short_week_flag"] = 0.0
    df["bye_adv_flag"] = 0.0
    df["timezone_travel_hours"] = 0.0
    df["early_body_clock_flag"] = 0.0
    df["hfa_base"] = 1.1
    df["crowd_noise_proxy"] = 0.0
    df["qb_silent_count_risk"] = 0.0
    df["ol_starters_out"] = 0
    df["dl_starters_out"] = 0
    df["db_starters_out"] = 0

    # ------------ trench proxy from EPA diffs (placeholder until OL/DL grades) ------------
    df["trench_score_diff"] = _cap(-(df["def_epa_diff"]), -0.4, 0.4)

    # --- EPA → points scaling (bounded so one feature can't dominate)
    # Notes:
    # - off_epa_diff: higher is better for HOME offense vs AWAY offense
    # - def_epa_diff: more negative = better HOME defense (so we subtract it with a negative weight)
    # - qb_grade_diff: higher is better (HOME QB vs AWAY QB)
    # These multipliers put features roughly into "spread points" scale.

    # Trench/efficiency: offense vs defense balance
    # Heavier weight on defense (negative EPA is good), solid weight on offense.
    trench_pts = 10.0 * df["off_epa_diff"] - 12.0 * df["def_epa_diff"]
    df["trench_line"] = _cap(trench_pts, -10.0, 10.0)

    # Quarterback influence
    qb_pts = 6.0 * df["qb_grade_diff"] * df["qb_status_risk"]
    df["qb_line"] = _cap(qb_pts, -6.0, 6.0)

    # Rest / schedule advantages (points)
    # +0.10 per rest day, +1.0 off bye, -1.0 short week
    rest_pts = 0.10 * df["rest_days_diff"] + 1.0 * df["bye_adv_flag"] - 1.0 * df["short_week_flag"]
    df["rest_line"] = _cap(rest_pts, -2.0, 2.0)

    # Travel / body clock (points)
    # -0.10 per time-zone hour, -1.0 if early body clock disadvantage
    travel_pts = -0.10 * df["timezone_travel_hours"] - 1.0 * df["early_body_clock_flag"]
    df["travel_line"] = _cap(travel_pts, -3.0, 3.0)

    # Home-field advantage baseline (~1.5), plus noise proxy, minus silent-count risk
    hfa_pts = 1.5 + 0.3 * df["crowd_noise_proxy"] - 0.5 * df["qb_silent_count_risk"]
    df["hfa_line"] = _cap(hfa_pts, 0.0, 3.0)

    # Injury clusters (points)
    inj_pts = -0.5 * df["ol_starters_out"] - 0.4 * df["dl_starters_out"] - 0.3 * df["db_starters_out"]
    df["inj_cluster_line"] = _cap(inj_pts, -4.0, 0.0)


    # Baseline model line (keep if present; else start at 0)
    if "model_home_line" not in df.columns:
        df["model_home_line"] = 0.0

    # Ensure market spread numeric
    if "spread_home" in df.columns:
        df["spread_home"] = pd.to_numeric(df["spread_home"], errors="coerce")

    return df
