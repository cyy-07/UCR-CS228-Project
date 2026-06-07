"""
src/exp_horizon.py — Workstream F: Forecast horizon sweep

Research question: How does each model's accuracy degrade as we predict
further into the future? Short-horizon prediction is dominated by persistence;
long-horizon prediction requires real temporal modeling.

Horizons tested (in 10-min steps):
   6   = 1 hour
   12  = 2 hours
   24  = 4 hours  (current default)
   48  = 8 hours
   96  = 16 hours
The look-back stays at 96 steps (16 h).

Models: Persistence, MLP, LSTM, DLinear, PatchTST  (vanilla, no residual)

Run
---
    python src/exp_horizon.py
    nohup python -u src/exp_horizon.py > results/log_horizon.txt 2>&1 &
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, make_config, get_loaders
from models import build_model
from training import (train, evaluate, persistence_reference,
                      save_csv, print_table, DEVICE)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# short-horizon task: 10-min steps (1h .. 16h)
HORIZONS        = [6, 12, 24, 48, 96]
# weekly task: hourly steps (1 day .. 7 days), look-back fixed at 1 week (168h)
HORIZONS_WEEKLY = [24, 48, 72, 120, 168]
MODELS   = ["persistence", "mlp", "lstm", "dlinear", "patchtst"]


def main(weekly=False):
    tag = " (WEEKLY: hourly, look-back 1wk)" if weekly else ""
    print("\n" + "=" * 60)
    print("WORKSTREAM F — Forecast horizon sweep" + tag)
    print("=" * 60)
    print(f"Device: {DEVICE}")

    horizons = HORIZONS_WEEKLY if weekly else HORIZONS
    rows = []
    for H in horizons:
        cfg = make_config(weekly, pred_len=H)
        tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
        if weekly:
            print(f"\n  -- horizon = {H} h ({H//24} day(s)) --")
        else:
            print(f"\n  -- horizon = {H} steps ({H*10} min = {H*10/60:.1f} h) --")
        persistence_reference(te_l, scaler, f"test@H={H}")

        for name in MODELS:
            t0 = time.time()
            model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                                use_residual=False)
            model = train(model, tr_l, vl_l, label=f"{name}@H={H}")
            m_te = evaluate(model, te_l, scaler)
            row = {"horizon": H, "model": name,
                   "test_MAE":  round(m_te["MAE"], 3),
                   "test_RMSE": round(m_te["RMSE"], 3),
                   "train_s":   round(time.time() - t0, 1)}
            rows.append(row)
            print(f"    {name:<12} test_MAE={m_te['MAE']:.3f}  "
                  f"test_RMSE={m_te['RMSE']:.3f}  ({row['train_s']}s)")

    csv_name = "horizon_weekly.csv" if weekly else "horizon.csv"
    png_name = "horizon_mae_weekly.png" if weekly else "horizon_mae.png"
    save_csv(rows, os.path.join(RESULTS, csv_name))
    print_table(rows, ["horizon","model","test_MAE","test_RMSE"])

    # quick plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(os.path.join(RESULTS, csv_name))
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, g in df.groupby("model"):
            g = g.sort_values("horizon")
            if weekly:
                xs, xlabel = g["horizon"] / 24.0, "Forecast horizon (days)"
                unit = "Wh/h"
            else:
                xs, xlabel = g["horizon"] * 10 / 60, "Forecast horizon (hours)"
                unit = "Wh"
            ax.plot(xs, g["test_MAE"], marker="o", label=name.upper())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Test MAE ({unit})")
        ax.set_title("MAE vs Forecast Horizon" + (" (weekly)" if weekly else ""))
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, png_name), dpi=120)
        plt.close()
        print(f"  -> saved results/{png_name}")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="run the hourly day-1..day-7 long-horizon sweep")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    main(weekly=args.weekly)
