"""
Professor-check experiment: can an MLP learn Persistence?

This script answers the concrete objection:
    "If Persistence just copies the last input target, an MLP should be able
     to learn it. What exactly is the input, and what are train/test R2?"

It reports:
  1. exact input tensor shape and feature columns
  2. Persistence train/val/test MAE/RMSE/R2
  3. vanilla MLP
  4. fixed anchored MLP: persistence(x) + MLP(x)
  5. alpha anchored MLP: persistence(x) + alpha * MLP(x), alpha initialized 0

Run:
    python src/exp_alpha_mlp.py
    python src/exp_alpha_mlp.py --weekly
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, make_config, get_loaders, load_and_prepare
from models import MLPForecaster, PersistenceBaseline
from training import train, collect_predictions, save_csv, print_table, DEVICE


RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


class AlphaResidualMLP(nn.Module):
    """persistence(x) + alpha * residual_mlp(x), with alpha initialized to 0."""

    def __init__(self, seq_len: int, pred_len: int, n_feat: int):
        super().__init__()
        self.pred_len = pred_len
        self.residual = MLPForecaster(seq_len, pred_len, n_feat,
                                      use_residual=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = x[:, -1, 0:1].expand(-1, self.pred_len)
        return anchor + self.alpha * self.residual(x)


def metrics_from_predictions(preds: np.ndarray, trues: np.ndarray) -> dict:
    err = preds - trues
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_split(model, loader, scaler, log_target: bool) -> dict:
    preds, trues = collect_predictions(model, loader, scaler,
                                       log_target=log_target)
    return metrics_from_predictions(preds, trues)


def describe_input(cfg: Config):
    """Reconstruct the selected feature names for a human-readable printout."""
    # load_and_prepare already verifies this config; here we only recover names.
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if cfg.resample is not None:
        # Keep this description short for weekly; exact aggregation is in
        # data_utils.py and README.
        df = df.resample(cfg.resample).mean().dropna()

    all_raw = [c for c in df.columns if c != cfg.target_col]
    if cfg.feature_mode == "all":
        feat_cols = all_raw
    elif cfg.feature_mode == "no_rv":
        feat_cols = [c for c in all_raw if c not in ("rv1", "rv2")]
    elif cfg.feature_mode == "target_only":
        feat_cols = []
    else:
        feat_cols = [f"(feature_mode={cfg.feature_mode}; see data_utils.py)"]
    if cfg.add_time_features or cfg.feature_mode == "target_time":
        feat_cols = feat_cols + [
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
            "is_weekend", "nsm"
        ]
    return [cfg.target_col] + feat_cols


def run_model(name, model, loaders, scaler, cfg):
    tr_l, vl_l, te_l = loaders
    t0 = time.time()
    if list(model.parameters()):
        model = train(model, tr_l, vl_l, label=name, loss_name="l1")
    train_m = evaluate_split(model, tr_l, scaler, cfg.log_target)
    val_m = evaluate_split(model, vl_l, scaler, cfg.log_target)
    test_m = evaluate_split(model, te_l, scaler, cfg.log_target)
    alpha = ""
    if hasattr(model, "alpha"):
        alpha = round(float(model.alpha.detach().cpu()), 6)
    return {
        "model": name,
        "train_MAE": round(train_m["MAE"], 3),
        "train_RMSE": round(train_m["RMSE"], 3),
        "train_R2": round(train_m["R2"], 4),
        "val_MAE": round(val_m["MAE"], 3),
        "val_RMSE": round(val_m["RMSE"], 3),
        "val_R2": round(val_m["R2"], 4),
        "test_MAE": round(test_m["MAE"], 3),
        "test_RMSE": round(test_m["RMSE"], 3),
        "test_R2": round(test_m["R2"], 4),
        "params": sum(p.numel() for p in model.parameters()),
        "alpha": alpha,
        "train_s": round(time.time() - t0, 1),
    }


def main(weekly=False):
    cfg = make_config(
        weekly,
        feature_mode="no_rv",
        add_time_features=True,
        log_target=True,
    )
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    _ = load_and_prepare(cfg)
    x, y = next(iter(tr_l))
    cols = describe_input(cfg)

    print("=" * 72)
    print("Professor-check: input definition + Persistence vs MLP")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Task: {'weekly hourly 1wk->1wk' if weekly else '10-min 16h->4h'}")
    print(f"Input X shape per batch: {tuple(x.shape)}")
    print(f"Target y shape per batch: {tuple(y.shape)}")
    print(f"Per-sample formula: X_i=data[i:i+{cfg.seq_len}, :], "
          f"y_i=Appliances[i+{cfg.seq_len}:i+{cfg.seq_len + cfg.pred_len}]")
    print(f"n_feat={n_feat}")
    print("Feature columns, in order:")
    for i, c in enumerate(cols):
        print(f"  {i:02d}: {c}")

    models = [
        ("persistence", PersistenceBaseline(cfg.pred_len)),
        ("mlp_vanilla", MLPForecaster(cfg.seq_len, cfg.pred_len, n_feat,
                                      use_residual=False)),
        ("mlp_fixed_anchor", MLPForecaster(cfg.seq_len, cfg.pred_len, n_feat,
                                           use_residual=True)),
        ("mlp_alpha_anchor", AlphaResidualMLP(cfg.seq_len, cfg.pred_len,
                                              n_feat)),
    ]

    rows = []
    for name, model in models:
        print(f"\n[{name}]")
        row = run_model(name, model, (tr_l, vl_l, te_l), scaler, cfg)
        rows.append(row)
        print(f"  train_R2={row['train_R2']:.4f}  test_R2={row['test_R2']:.4f}  "
              f"test_MAE={row['test_MAE']:.3f}  alpha={row['alpha']}")

    out = "alpha_mlp_weekly.csv" if weekly else "alpha_mlp.csv"
    save_csv(rows, os.path.join(RESULTS, out))
    print_table(rows, ["model", "train_R2", "test_R2", "test_MAE",
                       "test_RMSE", "params", "alpha"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    main(weekly=args.weekly)
