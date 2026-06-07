"""
src/exp_weekly.py — Long-horizon WEEKLY forecasting task.

Motivation
----------
The main study forecasts the next 4 h from a 16 h window at 10-min cadence.
A natural, more intuitive question for a CS audience is the WEEKLY one:

    "Given the past week of energy use, predict the next week."

To make this statistically sound we first AGGREGATE the 10-min series to
HOURLY total energy (Appliances/lights summed; covariates averaged). One step
is now one hour, so:

    seq_len  = 168   (past 1 week)
    pred_len = 168   (next 1 week)

The intuitive, strong baseline here is SEASONAL-NAIVE: "next week = last week
replayed verbatim" (next Monday 9am = last Monday 9am). Any useful model must
beat it. We re-run the full model zoo under the best training recipe found in
Workstream A (no_rv + time features + log target, L1 loss) and additionally
report how error GROWS across the 7-day horizon (day 1 .. day 7).

NOTE: MAE/RMSE here are in Wh-per-HOUR (energy summed over each hour), so the
absolute numbers are larger than the 10-min task and are NOT directly
comparable to it — compare models WITHIN this table.

Run
---
    python src/exp_weekly.py
    nohup python -u src/exp_weekly.py > results/log_weekly.txt 2>&1 &
"""

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, get_loaders
from models import build_model
from training import (train, evaluate, persistence_reference,
                      save_csv, print_table, DEVICE)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

PERIOD   = 168          # 1 week in hours
HORIZON_DAYS = 7


# ──────────────────────────────────────────────────────────────
#  Helper: collect predictions in Wh and compute per-day MAE
# ──────────────────────────────────────────────────────────────

def collect_predictions(model, loader, scaler, log_target):
    """Return (preds_wh, trues_wh) arrays of shape [N, H] in original units."""
    model.eval(); model.to(DEVICE)
    preds, trues = [], []
    use_median = hasattr(model, "predict_median")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            p = model.predict_median(x) if use_median else model(x)
            if p.dim() == 3:
                p = p[..., p.size(-1) // 2]
            preds.append(p.cpu()); trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    sc, mn = scaler.scale_[0], scaler.mean_[0]
    preds = preds * sc + mn
    trues = trues * sc + mn
    if log_target:
        preds = np.expm1(preds); trues = np.expm1(trues)
    return preds, trues


def per_day_mae(preds, trues):
    """MAE for each forecast day (day 1 = steps 0..23, ... day 7 = 144..167)."""
    out = []
    for d in range(HORIZON_DAYS):
        sl = slice(d * 24, (d + 1) * 24)
        out.append(float(np.mean(np.abs(preds[:, sl] - trues[:, sl]))))
    return out


# ──────────────────────────────────────────────────────────────
#  Main head-to-head on the weekly task
# ──────────────────────────────────────────────────────────────

CONTENDERS = [
    # name,            kwargs
    ("persistence",    {}),
    ("seasonalnaive",  {"period": PERIOD}),
    ("mlp",            {}),
    ("lstm",           {}),
    ("dlinear",        {}),
    ("patchtst",       {"d_model": 64,  "n_layers": 2}),
    ("itransformer",   {"d_model": 128, "n_layers": 2}),
    ("dpmixer",        {"d_channel": 64, "d_time": 128, "use_gating": True}),
    ("tcdpmixer",      {"time_feat_dim": 6}),
    ("ssdpmixer",      {}),
    ("tcssdpmixer",    {"time_feat_dim": 6}),
]


def main():
    print("\n" + "=" * 64)
    print("WEEKLY TASK — hourly aggregation, past 1 week → next 1 week")
    print("=" * 64)
    print(f"Device: {DEVICE}")

    cfg = Config(resample="1h",
                 seq_len=PERIOD, pred_len=PERIOD,
                 train_ratio=0.70, val_ratio=0.15,
                 feature_mode="no_rv",
                 add_time_features=True,
                 log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"\n  Setup: hourly, L={cfg.seq_len}h H={cfg.pred_len}h, "
          f"{cfg.feature_mode}+time+log, n_feat={n_feat}")
    print(f"  Windows — train {len(tr_l.dataset)}  val {len(vl_l.dataset)}  "
          f"test {len(te_l.dataset)}")
    persistence_reference(te_l, scaler, "test", log_target=cfg.log_target)

    rows, perday_rows = [], []
    for name, kwargs in CONTENDERS:
        print(f"\n  [{name}]")
        t0 = time.time()
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=True, **kwargs)
        model = train(model, tr_l, vl_l, label=name, loss_name="l1",
                      epochs=30, patience=6)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        preds, trues = collect_predictions(model, te_l, scaler, cfg.log_target)
        pday = per_day_mae(preds, trues)
        npar = sum(p.numel() for p in model.parameters())
        rows.append({"config": name,
                     "test_MAE":  round(m["MAE"], 3),
                     "test_RMSE": round(m["RMSE"], 3),
                     "params":    npar,
                     "train_s":   round(time.time() - t0, 1)})
        perday_rows.append({"config": name,
                            **{f"day{d+1}_MAE": round(pday[d], 3)
                               for d in range(HORIZON_DAYS)}})
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
        print("    per-day MAE: " +
              "  ".join(f"d{d+1}={pday[d]:.1f}" for d in range(HORIZON_DAYS)))

    save_csv(rows, os.path.join(RESULTS, "weekly.csv"))
    save_csv(perday_rows, os.path.join(RESULTS, "weekly_perday.csv"))
    print_table(rows, ["config", "test_MAE", "test_RMSE", "params"])
    print("\nDone.")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    main()
