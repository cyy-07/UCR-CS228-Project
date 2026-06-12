"""
src/exp_minimal.py  —  minimal-variant ablation.

What it does
------------
The paired-Wilcoxon ablation table shows that removing the persistence
anchor, the target-decomposition branch, or the channel mixer each gives
a *significantly smaller* MAE on 3829 windows; only removing the time
mixer is non-significant, and only-persistence is significantly worse.
Read literally, the additive backbone is over-parameterised on this
dataset. This script tests the natural simplification: a *minimal*
TC-DPMixer that keeps only the persistence anchor (cannot be removed
from the residual-learning recipe) and the time-mixer + TC gate, and
turns OFF the decomposition and channel branches.

  full       : persistence + decomp + channel + time + TC gate
  minimal-A  : persistence + time + TC gate          (kills decomp, channel)
  minimal-B  : persistence + time only (no gate)     (kills decomp, channel, gate)

5 seeds per variant.

Run
---
    python src/exp_minimal.py
    python src/exp_minimal.py --short
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

SEEDS = [42, 43, 44, 45, 46]

VARIANTS = [
    # (label,          model_name,    extra_kwargs)
    ("full",          "tcdpmixer",   {"time_feat_dim": 6}),
    ("minimal_A_tc",  "tcdpmixer",   {"time_feat_dim": 6,
                                      "use_decomp": False,
                                      "use_channel": False,
                                      "use_time":    True}),
    ("minimal_B_no_gate", "dpmixer", {"use_decomp": False,
                                      "use_channel": False,
                                      "use_time":    True,
                                      "use_gating":  False}),
]


def _run(name, kw, cfg, tr_l, vl_l, te_l, scaler, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name, cfg.seq_len, cfg.pred_len, cfg.n_feat,
                        use_residual=True, **kw)
    npar = sum(p.numel() for p in model.parameters())
    model = train(model, tr_l, vl_l, label=f"{name}@{seed}",
                  loss_name="l1", seed=seed)
    m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
    return float(m["MAE"]), float(m["RMSE"]), npar


def main(short=False):
    weekly = not short
    tag = "WEEKLY" if weekly else "SHORT"
    print("\n" + "=" * 60)
    print(f"MINIMAL DPMIXER  —  is the backbone earning its keep?  ({tag})")
    print("=" * 60)
    print(f"  Device: {DEVICE}   seeds={SEEDS}")

    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, _ = get_loaders(cfg)

    rows, per_variant = [], {v[0]: [] for v in VARIANTS}
    for seed in SEEDS:
        for label, name, kw in VARIANTS:
            mae, rmse, npar = _run(name, kw, cfg, tr_l, vl_l, te_l,
                                   scaler, seed)
            rows.append({"variant": label, "model": name, "seed": seed,
                         "test_MAE": mae, "test_RMSE": rmse, "params": npar})
            per_variant[label].append(mae)
            print(f"  {label:18s} seed={seed}  MAE={mae:.3f}  params={npar}")

    print("\n-- 5-seed summary --")
    for label, vals in per_variant.items():
        v = np.array(vals)
        print(f"  {label:18s}  MAE = {v.mean():.3f} ± {v.std(ddof=1):.3f}")

    out = os.path.join(RESULTS,
                       f"minimal{'_weekly' if weekly else ''}.csv")
    save_csv(rows, out)
    print(f"\n  → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--short", action="store_true")
    a = p.parse_args()
    main(short=a.short)
