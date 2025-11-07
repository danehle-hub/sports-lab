# models/run_sims.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PICKS = ROOT / "runs" / "upcoming_predictions_plus.csv"

def simulate_season(n_sims=10000, seed=42):
    np.random.seed(seed)

    df = pd.read_csv(PICKS)
    
    # We simulate against spread outcomes (using home_cover_prob)
    probs = df["home_cover_prob"].values

    # Sim results: 1 = home covers, 0 = away covers
    sims = np.random.binomial(n=1, p=probs, size=(n_sims, len(df)))

    df["cover_rate_simulated"] = sims.mean(axis=0)   # empirical simulated success rate
    df["home_cover_prob"] = probs                    # model's calibrated probability
    
    df["difference_vs_model"] = df["cover_rate_simulated"] - df["home_cover_prob"]
    df = df.sort_values("difference_vs_model", ascending=False)

    out_path = ROOT / "runs" / "monte_carlo_results.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Monte Carlo results saved → {out_path}")

if __name__ == "__main__":
    simulate_season(n_sims=20000)
