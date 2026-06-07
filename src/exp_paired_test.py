"""
src/exp_paired_test.py  —  Reviewer R3: defensible paired statistics.

What it does
------------
Re-trains the weekly-task headline pair (iTransformer vs TC-DPMixer) for
SEEDS = [42..46] and writes a per-seed table. Then computes:

  * paired mean difference (TC − iTrans)
  * paired Wilcoxon signed-rank W and exact two-sided p
  * paired bootstrap 95 % CI on the mean difference (10 000 resamples)
  * paired Cohen's d   d = mean(diff) / sd(diff)
  * unpaired pooled-σ ratio (what we previously called "4.85 σ"),
    reported separately so the writer can compare and use the right one.

The point of this script is that nothing in the report should be sourced
to "4.85 σ" without the W / p / CI / d numbers next to it.

Run
---
    python src/exp_paired_test.py
    # short variant for sanity:
    python src/exp_paired_test.py --short
"""

import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, evaluate, save_csv, DEVICE

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

PAIR = [
    ("itransformer", {"d_model": 128, "n_layers": 2}),
    ("tcdpmixer",    {"time_feat_dim": 6}),
]


def _train_and_eval(name, kwargs, cfg, tr_l, vl_l, te_l, scaler, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name, cfg.seq_len, cfg.pred_len, cfg.n_feat,
                        use_residual=True, **kwargs)
    model = train(model, tr_l, vl_l, label=f"{name}@{seed}",
                  loss_name="l1", seed=seed)
    m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
    return float(m["MAE"]), float(m["RMSE"])


def _bootstrap_ci(diffs, n_boot=10_000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main(short=False):
    weekly = not short
    tag = "WEEKLY (1h, 1wk→1wk)" if weekly else "SHORT (10min, 16h→4h)"
    print("\n" + "=" * 60)
    print(f"PAIRED TEST  —  iTransformer vs TC-DPMixer  ({tag})")
    print("=" * 60)
    print(f"Device: {DEVICE}   seeds={SEEDS}")

    cfg = make_config(weekly, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, _ = get_loaders(cfg)

    rows = []
    per_seed = {n: [] for n, _ in PAIR}
    for seed in SEEDS:
        for name, kw in PAIR:
            mae, rmse = _train_and_eval(name, kw, cfg, tr_l, vl_l, te_l,
                                        scaler, seed)
            rows.append({"model": name, "seed": seed,
                         "MAE": mae, "RMSE": rmse})
            per_seed[name].append(mae)
            print(f"  {name:15s} seed={seed}  MAE={mae:.3f}")

    a = np.array(per_seed["itransformer"])
    b = np.array(per_seed["tcdpmixer"])
    diff = b - a                                # negative → ours better
    mean_diff = float(diff.mean())
    sd_diff   = float(diff.std(ddof=1))
    pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

    # paired Wilcoxon (scipy is on the server image)
    try:
        from scipy.stats import wilcoxon
        w_stat, w_p = wilcoxon(b, a, zero_method="wilcox",
                               alternative="two-sided", mode="exact")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  (wilcoxon failed: {e})")

    ci_lo, ci_hi = _bootstrap_ci(diff, n_boot=10_000)
    cohens_d_paired = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    sigma_unpaired  = mean_diff / pooled_sd if pooled_sd > 0 else float("nan")

    stats = {
        "n_seeds": len(SEEDS),
        "itrans_mean": float(a.mean()), "itrans_std": float(a.std(ddof=1)),
        "tcdp_mean":   float(b.mean()), "tcdp_std":   float(b.std(ddof=1)),
        "mean_diff (TC − iTrans)": mean_diff,
        "sd_of_diff":              sd_diff,
        "pooled_sd":               pooled_sd,
        "boot95_lo":               ci_lo,
        "boot95_hi":               ci_hi,
        "wilcoxon_W":              w_stat,
        "wilcoxon_p_two_sided":    w_p,
        "cohens_d_paired":         cohens_d_paired,
        "unpaired_sigma_ratio":    sigma_unpaired,
    }

    print("\n-- Stats --")
    for k, v in stats.items():
        print(f"  {k:30s} = {v}")

    save_csv(rows, os.path.join(
        RESULTS, f"paired_test_seeds{'_weekly' if weekly else ''}.csv"))
    with open(os.path.join(
            RESULTS, f"paired_test{'_weekly' if weekly else ''}.json"),
              "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  → results/paired_test{'_weekly' if weekly else ''}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--short", action="store_true",
                   help="run the short-horizon variant instead of weekly")
    a = p.parse_args()
    main(short=a.short)
