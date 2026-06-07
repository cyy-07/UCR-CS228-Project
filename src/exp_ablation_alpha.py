"""
exp_ablation_alpha.py — Ablation A1: does the anchor really help?

We sweep alpha in  ŷ = persistence(x) + α · MLP(x)  on both horizons:
    α = 0       (pure persistence — degenerates to PersistenceBaseline)
    α = 0.25    (light anchoring)
    α = 1.0     (default anchoring; matches exp_alpha_mlp.py's fixed-anchor)
    vanilla     (no anchor at all; MLP outputs ŷ directly)

Outputs:
    results/ablation_alpha.csv

Run:
    python src/exp_ablation_alpha.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import MLPForecaster, PersistenceBaseline, _persistence_anchor
from training import train, collect_predictions, save_csv, DEVICE


RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


class FixedAlphaMLP(nn.Module):
    """ŷ = persistence(x) + α · MLP(x), with α frozen at a constant."""
    def __init__(self, seq_len, pred_len, n_feat, alpha: float):
        super().__init__()
        self.pred_len = pred_len
        self.alpha = float(alpha)
        # use_residual=False so the inner MLP outputs the raw residual
        self.mlp = MLPForecaster(seq_len, pred_len, n_feat, use_residual=False)

    def forward(self, x):
        return _persistence_anchor(x, self.pred_len) + self.alpha * self.mlp(x)


def metrics(preds, trues):
    err = preds - trues
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def run_one(name, model, loaders, scaler, cfg):
    tr_l, vl_l, te_l = loaders
    t0 = time.time()
    if list(model.parameters()):
        model = train(model, tr_l, vl_l, label=name, loss_name="l1")
    p, y = collect_predictions(model, te_l, scaler, log_target=cfg.log_target)
    mae, rmse, r2 = metrics(p, y)
    return dict(model=name, test_MAE=round(mae, 3),
                test_RMSE=round(rmse, 3), test_R2=round(r2, 4),
                params=sum(p.numel() for p in model.parameters()),
                train_s=round(time.time() - t0, 1))


def run_horizon(weekly: bool):
    horizon = "weekly" if weekly else "short"
    cfg = make_config(weekly, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"\n{'=' * 60}\n  Ablation A1 · α  —  horizon={horizon}\n{'=' * 60}")
    print(f"  L={cfg.seq_len}  H={cfg.pred_len}  n_feat={n_feat}  "
          f"device={DEVICE}")

    rows = []
    configs = [
        ("alpha_0.00_persistence", PersistenceBaseline(cfg.pred_len), 0.0),
        ("alpha_0.25",
         FixedAlphaMLP(cfg.seq_len, cfg.pred_len, n_feat, 0.25), 0.25),
        ("alpha_1.00",
         FixedAlphaMLP(cfg.seq_len, cfg.pred_len, n_feat, 1.0), 1.0),
        ("vanilla_mlp",
         MLPForecaster(cfg.seq_len, cfg.pred_len, n_feat,
                       use_residual=False), float("nan")),
    ]
    for name, model, alpha in configs:
        torch.manual_seed(42); np.random.seed(42)
        row = run_one(name, model, (tr_l, vl_l, te_l), scaler, cfg)
        row["horizon"] = horizon
        row["alpha"] = alpha
        rows.append(row)
        print(f"  [{horizon} {name:24}] "
              f"MAE={row['test_MAE']:.3f}  R2={row['test_R2']:+.4f}  "
              f"params={row['params']}")
    return rows


def main():
    all_rows = []
    for weekly in [False, True]:
        all_rows.extend(run_horizon(weekly))
    save_csv(all_rows, os.path.join(RESULTS, "ablation_alpha.csv"))
    print("\nSummary:")
    for r in all_rows:
        print(f"  {r['horizon']:<7}  alpha={str(r['alpha']):<6}  "
              f"MAE={r['test_MAE']:.3f}  R2={r['test_R2']:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    main()
