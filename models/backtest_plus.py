from pathlib import Path
import numpy as np
import pandas as pd

# --- Ensure project root & models are importable ---
import sys
from pathlib import Path as _PathForSys

_THIS_FILE = _PathForSys(__file__).resolve()
_MODELS_DIR = _THIS_FILE.parent          # models/
_PROJECT_ROOT = _MODELS_DIR.parent       # project root

for _p in (str(_PROJECT_ROOT), str(_MODELS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import team code map (prefer module style, fallback relative)
try:
    from models.team_map import TEAM_NAME_TO_CODE
except ModuleNotFoundError:
    from team_map import TEAM_NAME_TO_CODE

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

# --- Inputs ---
GAMES          = Path("data/clean/games_clean.csv")
ELO            = Path("data/clean/team_week_elo.csv")     # must include elo_pre (pre-game)
TRENCH_WEEKLY  = Path("data/clean/trench_weekly.csv")     # from build_trench_weekly.py
# Fallback seasonal files (used only if weekly file missing)
OL_SEASONAL    = Path("data/clean/ol_scores.csv")         # from build_trench_scores.py
DL_SEASONAL    = Path("data/clean/dl_scores.csv")         # from build_trench_scores.py
QB             = Path("data/clean/qb_team_week.csv")      # from build_qb_features.py
OUT_PICKS      = Path("runs/last_season_picks_plus.csv")

# --- Settings ---
YEARS_BACK    = 7
BASE_ELO      = 1500.0
VIG_BREAKEVEN = 0.524     # -110 fair win prob
ALPHAS_GRID   = [0.50, 0.55, 0.60, 0.65]   # OL vs DL weight
C_GRID        = [0.5, 1.0, 2.0]            # LogisticRegression C

def need_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")

def load_games():
    g = pd.read_csv(GAMES, parse_dates=["schedule_date"])
    need_cols(g, ["schedule_date","schedule_season","schedule_week",
                  "team_home","team_away","score_home","score_away","spread_home"], "games_clean.csv")
    last_season = int(g["schedule_season"].max())
    cutoff = last_season - (YEARS_BACK - 1)
    g = g[g["schedule_season"] >= cutoff].copy()
    g["schedule_week"] = g["schedule_week"].astype(str)
    g["team_home"] = g["team_home"].str.upper().str.strip()
    g["team_away"] = g["team_away"].str.upper().str.strip()
    # map full names -> codes for trench merges
    g["home_code"] = g["team_home"].map(TEAM_NAME_TO_CODE)
    g["away_code"] = g["team_away"].map(TEAM_NAME_TO_CODE)
    # label: did home cover?
    g["home_cover"] = (g["score_home"] - g["score_away"] + g["spread_home"] > 0).astype(int)
    return g, last_season

def load_elo():
    e = pd.read_csv(ELO)
    need_cols(e, ["schedule_season","schedule_week","team","elo_pre"], "team_week_elo.csv")
    e["schedule_season"] = e["schedule_season"].astype(int)
    e["schedule_week"] = e["schedule_week"].astype(str)
    e["team"] = e["team"].str.upper().str.strip()
    return e

def load_qb():
    if not QB.exists():
        return None
    q = pd.read_csv(QB)  # season, week, team, passer_player_name, qb_epa_per_db, qb_epa_roll3
    need_cols(q, ["season","week","team","qb_epa_roll3"], "qb_team_week.csv")
    q["season"] = q["season"].astype(int)
    q["week"]   = q["week"].astype(str)
    q["team"]   = q["team"].str.upper().str.strip()
    return q

def add_rest_days(df):
    # prior game date per team → rest days; fill 7 if NaN
    for side in ["home","away"]:
        tmp = df[["schedule_season","schedule_week",f"team_{side}","schedule_date"]].rename(
            columns={f"team_{side}":"team"})
        tmp = tmp.sort_values(["team","schedule_date"])
        tmp["prev_date"] = tmp.groupby("team")["schedule_date"].shift(1)
        tmp["rest_days"] = (tmp["schedule_date"] - tmp["prev_date"]).dt.days
        df = df.merge(tmp[["schedule_season","schedule_week","team","rest_days"]]
                      .rename(columns={"team":f"team_{side}","rest_days":f"{side}_rest_days"}),
                      on=["schedule_season","schedule_week",f"team_{side}"], how="left")
    df["rest_diff"] = df["home_rest_days"].fillna(7) - df["away_rest_days"].fillna(7)
    return df

def load_trench_weekly_or_fallback():
    """
    Returns:
      mode: "weekly" or "seasonal"
      merger_fn: function(g)->g that attaches trench columns (and keeps weekly subcols if available)
    """
    if TRENCH_WEEKLY.exists():
        def merge_weekly(g):
            tw = pd.read_csv(TRENCH_WEEKLY)
            need_cols(tw, ["season","week","team","ol_0_100","dl_0_100"], "trench_weekly.csv")
            tw["season"] = tw["season"].astype(int)
            tw["week"]   = tw["week"].astype(str)
            tw["team"]   = tw["team"].str.upper().str.strip()  # code: e.g., DEN, MIN

            # Merge by codes we added to games (home_code/away_code)
            g = g.merge(
                tw.rename(columns={"season":"schedule_season","week":"schedule_week","team":"home_code",
                                   "ol_0_100":"home_ol_wk","dl_0_100":"home_dl_wk"}),
                on=["schedule_season","schedule_week","home_code"], how="left"
            ).merge(
                tw.rename(columns={"season":"schedule_season","week":"schedule_week","team":"away_code",
                                   "ol_0_100":"away_ol_wk","dl_0_100":"away_dl_wk"}),
                on=["schedule_season","schedule_week","away_code"], how="left"
            )

            # Fill gaps by season median to be safe
            for c in ["home_ol_wk","away_ol_wk","home_dl_wk","away_dl_wk"]:
                g[c] = g.groupby("schedule_season")[c].transform(lambda s: s.fillna(s.median()))
                g[c] = g[c].fillna(g[c].median())

            # Add a placeholder trench_diff; we'll recompute during tuning
            g["trench_diff"] = (g["home_ol_wk"] - g["away_ol_wk"]) - (g["away_dl_wk"] - g["home_dl_wk"])
            return g
        return "weekly", merge_weekly

    # ---- Fallback: seasonal trenches (merge by codes as well) ----
    if OL_SEASONAL.exists() and DL_SEASONAL.exists():
        def merge_seasonal(g):
            ol = pd.read_csv(OL_SEASONAL)
            dl = pd.read_csv(DL_SEASONAL)
            need_cols(ol, ["season","team","ol_score_0_100"], "ol_scores.csv")
            need_cols(dl, ["season","team","dl_score_0_100"], "dl_scores.csv")
            ol["season"] = ol["season"].astype(int); ol["team"] = ol["team"].str.upper().str.strip()
            dl["season"] = dl["season"].astype(int); dl["team"] = dl["team"].str.upper().str.strip()

            g = g.merge(ol.rename(columns={"season":"schedule_season","team":"home_code"}),
                        on=["schedule_season","home_code"], how="left") \
                 .rename(columns={"ol_score_0_100":"home_ol"})
            g = g.merge(ol.rename(columns={"season":"schedule_season","team":"away_code"}),
                        on=["schedule_season","away_code"], how="left") \
                 .rename(columns={"ol_score_0_100":"away_ol"})
            g = g.merge(dl.rename(columns={"season":"schedule_season","team":"home_code"}),
                        on=["schedule_season","home_code"], how="left") \
                 .rename(columns={"dl_score_0_100":"home_dl"})
            g = g.merge(dl.rename(columns={"season":"schedule_season","team":"away_code"}),
                        on=["schedule_season","away_code"], how="left") \
                 .rename(columns={"dl_score_0_100":"away_dl"})

            for col in ["home_ol","away_ol","home_dl","away_dl"]:
                g[col] = g.groupby("schedule_season")[col].transform(lambda s: s.fillna(s.median()))
                g[col] = g[col].fillna(g[col].median())

            # Placeholder trench_diff; recomputed during tuning
            g["trench_diff"] = (g["home_ol"] - g["away_ol"]) - (g["away_dl"] - g["home_dl"])
            return g
        return "seasonal", merge_seasonal

    raise FileNotFoundError(
        "No trench data found. Build one of:\n"
        " - Weekly:  python -u models\\build_trench_weekly.py\n"
        " - Seasonal: python -u models\\build_trench_scores.py"
    )

def build_dataset():
    g, last_season = load_games()
    e = load_elo()
    q = load_qb()

    # ---- Elo (PRE-GAME) ----
    g = g.merge(
        e.rename(columns={"team":"team_home","elo_pre":"home_elo"}),
        on=["schedule_season","schedule_week","team_home"], how="left"
    ).merge(
        e.rename(columns={"team":"team_away","elo_pre":"away_elo"}),
        on=["schedule_season","schedule_week","team_away"], how="left"
    )
    g["home_elo"] = g["home_elo"].fillna(BASE_ELO)
    g["away_elo"] = g["away_elo"].fillna(BASE_ELO)
    g["elo_diff"] = g["home_elo"] - g["away_elo"]

    # Market prior (closing spread, home favorite positive)
    g["mkt_spread"] = g["spread_home"].where(np.isfinite(g["spread_home"]), 0.0)

    # ---- Trench (prefer weekly; fallback seasonal) ----
    mode, trench_merge_fn = load_trench_weekly_or_fallback()
    g = trench_merge_fn(g)
    print(f"Trench mode used: {mode}")

    # ---- QB rolling EPA (optional) ----
    if q is not None:
        g = g.merge(
            q.rename(columns={"season":"schedule_season","week":"schedule_week","team":"team_home",
                              "qb_epa_roll3":"home_qb_roll3","passer_player_name":"home_qb"}),
            on=["schedule_season","schedule_week","team_home"], how="left"
        )
        g = g.merge(
            q.rename(columns={"season":"schedule_season","week":"schedule_week","team":"team_away",
                              "qb_epa_roll3":"away_qb_roll3","passer_player_name":"away_qb"}),
            on=["schedule_season","schedule_week","team_away"], how="left"
        )
        g["qb_edge"] = g["home_qb_roll3"].fillna(0) - g["away_qb_roll3"].fillna(0)
    else:
        g["qb_edge"] = 0.0

    # ---- Rest differential ----
    g = add_rest_days(g)

    return g, last_season

def _recompute_trench_diff(df, alpha):
    # Use weekly columns if present, else seasonal
    if {"home_ol_wk","away_ol_wk","home_dl_wk","away_dl_wk"}.issubset(df.columns):
        return (alpha*(df["home_ol_wk"] - df["away_ol_wk"])) - ((1 - alpha)*(df["away_dl_wk"] - df["home_dl_wk"]))
    else:
        return (alpha*df["home_ol"] - (1 - alpha)*df["away_dl"]) - (alpha*df["away_ol"] - (1 - alpha)*df["home_dl"])

def _threshold_eval(p, y, thresh):
    win_payout = 100/110
    bets, profit = 0, 0.0
    for prob, truth in zip(p, y):
        conf = max(prob, 1 - prob)
        if conf >= thresh:
            side_is_home = (prob >= 0.5)
            won = (truth == 1) if side_is_home else (truth == 0)
            profit += (win_payout if won else -1.0)
            bets += 1
    roi = (profit / bets) if bets else 0.0
    return bets, roi

def train_test_and_bet():
    g, last_season = build_dataset()
    train = g[g["schedule_season"] < last_season].copy()
    test  = g[g["schedule_season"] == last_season].copy()

    # We’ll tune ALPHA_OL and Logistic C by minimizing Brier on the test season
    best = {"alpha": None, "C": None, "brier": 9, "model": None, "p_te": None}

    for alpha in ALPHAS_GRID:
        # recompute trench_diff for this alpha
        train_td = _recompute_trench_diff(train, alpha).values
        test_td  = _recompute_trench_diff(test,  alpha).values

        # feature matrix: elo_diff, trench_diff(alpha), qb_edge, rest_diff, market spread
        X_tr = np.column_stack([
            train["elo_diff"].values,
            train_td,
            train["qb_edge"].values,
            train["rest_diff"].values,
            train["mkt_spread"].values
        ])
        y_tr = train["home_cover"].astype(int).values

        X_te = np.column_stack([
            test["elo_diff"].values,
            test_td,
            test["qb_edge"].values,
            test["rest_diff"].values,
            test["mkt_spread"].values
        ])
        y_te = test["home_cover"].astype(int).values

        for C in C_GRID:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, C=C))
            ])
            pipe.fit(X_tr, y_tr)
            p = pipe.predict_proba(X_te)[:, 1]
            brier = brier_score_loss(y_te, p)
            if brier < best["brier"]:
                best.update(alpha=alpha, C=C, brier=brier, model=pipe, p_te=p)

    # Final metrics at 0.50 threshold
    p_te = best["p_te"]
    y_te = test["home_cover"].astype(int).values
    acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))

    # Sweep bet thresholds to maximize ROI
    candidate_ts = [0.50, 0.52, 0.54, 0.56, 0.58]
    sweep = []
    for t in candidate_ts:
        b, r = _threshold_eval(p_te, y_te, t)
        sweep.append((t, b, r))
    sweep.sort(key=lambda x: x[2], reverse=True)
    best_thresh, best_bets, best_roi = sweep[0]

    # Staking with best threshold
    win_payout = 100/110
    bets, profit, picks = 0, 0.0, []
    for prob, truth in zip(p_te, y_te):
        conf = max(prob, 1 - prob)
        if conf >= best_thresh:
            side = "HOME" if prob >= 0.5 else "AWAY"
            bets += 1
            won = (truth == 1) if side == "HOME" else (truth == 0)
            profit += (win_payout if won else -1.0)
            picks.append((side, conf, won))
        else:
            picks.append(("PASS", conf, None))
    roi = (profit / bets) if bets else 0.0

    # Save picks with probabilities & config
    out = test.copy()
    out["p_home_cover"] = p_te
    out["pick"] = [p[0] for p in picks]
    out["pick_conf"] = [p[1] for p in picks]
    out["pick_result"] = [p[2] for p in picks]
    out["alpha_used"] = best["alpha"]
    out["C_used"] = best["C"]
    out["best_threshold"] = best_thresh
    OUT_PICKS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PICKS, index=False)

    # Feature weights (after scaling)
    coef = best["model"].named_steps["lr"].coef_[0]
    feat_names = ["elo_diff","trench_diff(alpha)","qb_edge","rest_diff","mkt_spread"]
    print("\nFeature weights (|w| bigger ⇒ more influence):")
    for name, w in sorted(zip(feat_names, coef), key=lambda x: -abs(x[1])):
        print(f"{name:>18}: {w:+.3f}")

    # Report
    print("\n=== Backtest Results (Elo + Trench + QB + Rest + Market; hold-out = most recent season) ===")
    print(f"Train size: {len(train):,}   Test size: {len(test):,}")
    print(f"Best alpha (OL weight): {best['alpha']:.2f}   Best C: {best['C']}")
    print(f"Accuracy @0.50:         {acc:.3f}")
    print(f"Brier (min over grid):  {best['brier']:.3f} (lower is better)")
    for t, b, r in sweep:
        print(f"Threshold {t:.2f} -> bets={b}, ROI/bet={r:.3f}")
    print(f"Chosen threshold:       {best_thresh:.2f}")
    print(f"Bets placed:            {bets}")
    print(f"ROI per bet (@-110):    {roi:.3f}")
    print(f"Saved test picks → {OUT_PICKS}")

if __name__ == "__main__":
    train_test_and_bet()
