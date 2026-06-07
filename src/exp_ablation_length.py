"""
exp_ablation_length.py — Ablation A2: how much history is enough?

We hold the model fixed (anchored MLP, α = 1) and vary the input window L:
    short  horizon (10-min ticks):  L ∈ {48, 96, 192}   = 8 h / 16 h / 32 h
    weekly horizon (hourly ticks):  L ∈ {84, 168, 240}  = 3.5 d / 7 d / 10 d
       (L=336 would exceed val-slice budget after hourly resample)

Outputs:
    results/ablation_length.csv

Run:
    python src/exp_ablation_length.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import MLPForecaster
from training import train, collect_predictions, save_csv, DEVICE


RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def metrics(preds, trues):
    err = preds - trues
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def run_one(L, weekly: bool):
    horizon = "weekly" if weekly else "short"
    cfg = make_config(weekly, feature_mode="no_rv",
                      add_time_features=True, log_target=True,
                      seq_len=L)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    torch.manual_seed(42); np.random.seed(42)

    model = MLPForecaster(cfg.seq_len, cfg.pred_len, n_feat,
                          use_residual=True)  # anchored, α = 1
    t0 = time.time()
    model = train(model, tr_l, vl_l, label=f"L={L}", loss_name="l1")
    preds, trues = collect_predictions(model, te_l, scaler,
                                       log_target=cfg.log_target)
    mae, rmse, r2 = metrics(preds, trues)
    row = dict(horizon=horizon, seq_len=L, pred_len=cfg.pred_len,
               n_feat=n_feat,
               test_MAE=round(mae, 3),
               test_RMSE=round(rmse, 3),
               test_R2=round(r2, 4),
               params=sum(p.numel() for p in model.parameters()),
               train_s=round(time.time() - t0, 1))
    print(f"  [{horizon} L={L:4d}] MAE={row['test_MAE']:.3f}  "
          f"R2={row['test_R2']:+.4f}  params={row['params']}  "
          f"t={row['train_s']}s")
    return row


def main():
    rows = []
    print(f"\n{'=' * 60}\n  Ablation A2 · input length L\n{'=' * 60}")
    print(f"  device={DEVICE}")

    print("\n[short horizon]")
    for L in [48, 96, 192]:
        rows.append(run_one(L, weekly=False))

    print("\n[weekly horizon]")
    for L in [84, 168, 240]:
        rows.append(run_one(L, weekly=True))

    save_csv(rows, os.path.join(RESULTS, "ablation_length.csv"))
    print("\nSummary:")
    for r in rows:
        print(f"  {r['horizon']:<7}  L={r['seq_len']:4d}  "
              f"MAE={r['test_MAE']:.3f}  R2={r['test_R2']:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    main()
