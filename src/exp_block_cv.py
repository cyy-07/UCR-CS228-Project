"""
src/exp_block_cv.py  —  Reviewer R5: cross-month robustness check.

What it does
------------
All headline numbers in the report come from a single chronological
70/15/15 split. The reviewer's complaint was that this hides whether the
margin survives if the cut is moved. This script reruns the headline pair
(iTransformer vs TC-DPMixer; we also include DLinear as a cheap sanity
peg) at K=4 chronological cutoffs that emulate rolling-origin CV. The
test slab is the tail after each cut.

  Fold 1:  train 55 % | val 10 % | test 35 %   (earliest cut)
  Fold 2:  train 60 % | val 10 % | test 30 %
  Fold 3:  train 65 % | val 10 % | test 25 %
  Fold 4:  train 70 % | val 10 % | test 20 %   (reference cut)

Reports per-fold MAE for each model, plus mean ± std across folds.
The point is to give the reader a "if you move the cut by ~10 % the
ranking does/doesn't survive" picture.

Run
---
    python src/exp_block_cv.py
    python src/exp_block_cv.py --short
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, evaluate, save_csv, DEVICE

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

FOLDS = [
    ("fold1", 0.55),
    ("fold2", 0.60),
    ("fold3", 0.65),
    ("fold4", 0.70),
]
VAL_RATIO = 0.10
SEED = 42

CONTENDERS = [
    ("dlinear",      {}),
    ("itransformer", {"d_model": 128, "n_layers": 2}),
    ("tcdpmixer",    {"time_feat_dim": 6}),
]


def main(short=False):
    weekly = not short
    tag = "WEEKLY" if weekly else "SHORT"
    print("\n" + "=" * 60)
    print(f"BLOCK CV — rolling-origin across {len(FOLDS)} cuts  ({tag})")
    print("=" * 60)
    print(f"  Device: {DEVICE}   seed={SEED}   val_ratio={VAL_RATIO}")

    per_model = {n: [] for n, _ in CONTENDERS}
    rows = []

    for fold_name, tr_r in FOLDS:
        print(f"\n  ── {fold_name}:  train={tr_r:.0%}  val={VAL_RATIO:.0%}  "
              f"test={1 - tr_r - VAL_RATIO:.0%}")
        cfg = make_config(weekly,
                          feature_mode="no_rv",
                          add_time_features=True,
                          log_target=True,
                          train_ratio=tr_r,
                          val_ratio=VAL_RATIO)
        tr_l, vl_l, te_l, scaler, _ = get_loaders(cfg)

        for name, kw in CONTENDERS:
            torch.manual_seed(SEED); np.random.seed(SEED)
            model = build_model(name, cfg.seq_len, cfg.pred_len, cfg.n_feat,
                                use_residual=True, **kw)
            model = train(model, tr_l, vl_l, label=f"{name}@{fold_name}",
                          loss_name="l1", seed=SEED)
            m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
            mae = float(m["MAE"])
            per_model[name].append(mae)
            rows.append({"fold": fold_name, "train_ratio": tr_r,
                         "model": name, "test_MAE": mae,
                         "test_RMSE": float(m["RMSE"])})
            print(f"     {name:14s}  MAE = {mae:.3f}")

    print("\n-- Per-model mean ± std across folds --")
    for name, vals in per_model.items():
        v = np.array(vals)
        print(f"  {name:14s}  mean={v.mean():.3f}  std={v.std(ddof=1):.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}  range={v.ptp():.3f}")
        rows.append({"fold": "AGGREGATE", "train_ratio": float("nan"),
                     "model": name,
                     "test_MAE": float(v.mean()),
                     "test_RMSE": float(v.std(ddof=1))})  # std in RMSE col

    out = os.path.join(RESULTS,
                       f"block_cv{'_weekly' if weekly else ''}.csv")
    save_csv(rows, out)
    print(f"\n  → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--short", action="store_true")
    a = p.parse_args()
    main(short=a.short)
