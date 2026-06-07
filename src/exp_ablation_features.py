"""
exp_ablation_features.py — Ablation A3: which input channels matter?

We hold the model fixed (anchored MLP, α = 1) and vary the feature subset:
    target-only     :  channel 0 only           ⇒ n_feat = 1
    target+time     :  channel 0 + 6 time enc.  ⇒ n_feat = 7
    all (no_rv)     :  default                  ⇒ n_feat = 32  (short)
                                                       n_feat = 32  (weekly)

Outputs:
    results/ablation_features.csv

Run:
    python src/exp_ablation_features.py
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


SUBSETS = [
    # (label, feature_mode, add_time_features)
    ("target_only",   "target_only", False),
    ("target_time",   "target_only", True),
    ("all_no_rv",     "no_rv",       True),
]


def metrics(preds, trues):
    err = preds - trues
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def run_one(label, feature_mode, add_time_features, weekly: bool):
    horizon = "weekly" if weekly else "short"
    cfg = make_config(weekly,
                      feature_mode=feature_mode,
                      add_time_features=add_time_features,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    torch.manual_seed(42); np.random.seed(42)

    model = MLPForecaster(cfg.seq_len, cfg.pred_len, n_feat,
                          use_residual=True)  # anchored, α = 1
    t0 = time.time()
    model = train(model, tr_l, vl_l, label=f"{horizon}-{label}", loss_name="l1")
    preds, trues = collect_predictions(model, te_l, scaler,
                                       log_target=cfg.log_target)
    mae, rmse, r2 = metrics(preds, trues)
    row = dict(horizon=horizon, subset=label, n_feat=n_feat,
               seq_len=cfg.seq_len, pred_len=cfg.pred_len,
               test_MAE=round(mae, 3),
               test_RMSE=round(rmse, 3),
               test_R2=round(r2, 4),
               params=sum(p.numel() for p in model.parameters()),
               train_s=round(time.time() - t0, 1))
    print(f"  [{horizon} {label:14}] n_feat={n_feat:2d}  "
          f"MAE={row['test_MAE']:.3f}  R2={row['test_R2']:+.4f}  "
          f"t={row['train_s']}s")
    return row


def main():
    rows = []
    print(f"\n{'=' * 60}\n  Ablation A3 · feature subsets\n{'=' * 60}")
    print(f"  device={DEVICE}")

    for weekly in [False, True]:
        print(f"\n[{'weekly' if weekly else 'short'} horizon]")
        for label, fmode, atf in SUBSETS:
            rows.append(run_one(label, fmode, atf, weekly))

    save_csv(rows, os.path.join(RESULTS, "ablation_features.csv"))
    print("\nSummary (lower MAE = better):")
    for r in rows:
        print(f"  {r['horizon']:<7}  {r['subset']:<13}  "
              f"n_feat={r['n_feat']:2d}  "
              f"MAE={r['test_MAE']:.3f}  R2={r['test_R2']:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    main()
