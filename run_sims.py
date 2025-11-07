# models/run_sims.py
from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

from analysis_tools import (
    simulate_cover_probability,
    simulate_outcomes,
    BankrollSimConfig,
    load_predictions,
)

ROOT = Path(__file__).resolve().parents[1]

def ensure_latest_predictions(alpha_ol: float | None = None):
    cmd = ["python", str(ROOT / "models" / "predict_upcoming_plus.py")]
    if alpha_ol is not None:
        cmd += ["--alpha-ol", str(alpha_ol)]
    subprocess.run(cmd, check=True)

def main():
    # 1) ensure the base predictions exist (uses defaults)
    ensure_latest_predictions()

    # 2) load predictions and run per-game Monte Carlo cover sims
    preds = load_predictions()
    sim_cov = simulate_cover_probability(preds, sigma_points=13.5, n_sims=100_000)
    print("🧪 Monte Carlo cover probabilities saved → runs/sim_cover_probs.csv")
    print(sim_cov[["team_home","team_away","spread_home","model_spread","sim_cover_prob"]].to_string(index=False))

    # 3) optional outcomes simulation (doesn't change model, just shows distribution)
    summary = simulate_outcomes(preds, cfg=BankrollSimConfig(kelly_f=0.0), n_trials=5000)
    if "message" in summary.columns:
        print(summary["message"].iloc[0])
    else:
        print("\n🎲 Outcomes summary (units are abstract; kelly_f=0 disables staking):")
        print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
