"""
src/exp_significance.py — Evidence A: multi-seed error bars + significance.

Research question (plain English):
    A single run reports e.g. iTransformer = 34.19 Wh and our TC-SS-DPMixer =
    34.83 Wh. That 0.64 Wh gap LOOKS like iTransformer wins. But neural nets
    are random — different seeds give different numbers. If the run-to-run
    noise of EACH model is bigger than 0.64 Wh, then the "gap" is just noise
    and the two models are statistically TIED.

What this does:
    Train every contender under SEEDS = [42, 43, 44, 45, 46] (5 runs each),
    record test MAE per seed, then report mean ± std. Zero-parameter baselines
    (persistence / seasonal-naive) are deterministic, so they run once.

    Auto-verdict: for the best-vs-ours pair, we compare the gap against the
    pooled std. If gap < pooled_std, we print "TIE (within noise)".

Output: significance.csv  (+  significance_weekly.csv with --weekly)
        fig_significance.png — bar chart with error bars.

Run
---
    python src/exp_significance.py
    python src/exp_significance.py --weekly
    nohup python -u src/exp_significance.py > results/log_significance.txt 2>&1 &
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

SEEDS = [42, 43, 44, 45, 46]

# Same recipe & contenders as the DPMixer head-to-head, so numbers line up.
CONTENDERS = [
    ("persistence",   {}),
    ("seasonalnaive", {}),
    ("dlinear",       {}),
    ("itransformer",  {"d_model": 128, "n_layers": 2}),
    ("patchtst",      {"d_model": 64,  "n_layers": 2}),
    ("dpmixer",       {"d_channel": 64, "d_time": 128, "use_gating": True}),
    ("tcdpmixer",     {"time_feat_dim": 6}),
    ("tcssdpmixer",   {"time_feat_dim": 6}),
]
DETERMINISTIC = {"persistence", "seasonalnaive"}


def _evaluate_seed(name, kwargs, cfg, tr_l, vl_l, te_l, scaler, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name, cfg.seq_len, cfg.pred_len,
                        cfg.n_feat, use_residual=True, **kwargs)
    model = train(model, tr_l, vl_l, label=f"{name}@{seed}",
                  loss_name="l1", seed=seed)
    m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
    npar = sum(p.numel() for p in model.parameters())
    return m["MAE"], m["RMSE"], npar


def main(weekly=False):
    tag = " (WEEKLY: hourly, 1wk→1wk)" if weekly else ""
    print("\n" + "=" * 60)
    print("EVIDENCE A — Multi-seed error bars / significance" + tag)
    print("=" * 60)
    print(f"Device: {DEVICE}   seeds={SEEDS}")

    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    cfg.n_feat = n_feat
    print(f"\n  Setup: {cfg.feature_mode} + time + log,  n_feat={n_feat}")
    persistence_reference(te_l, scaler, "test", log_target=cfg.log_target)

    rows, summary = [], []
    for name, kwargs in CONTENDERS:
        seeds = [SEEDS[0]] if name in DETERMINISTIC else SEEDS
        maes, rmses, npar = [], [], 0
        t0 = time.time()
        for s in seeds:
            mae, rmse, npar = _evaluate_seed(name, kwargs, cfg,
                                             tr_l, vl_l, te_l, scaler, s)
            maes.append(mae); rmses.append(rmse)
            rows.append({"config": name, "seed": s,
                         "test_MAE": round(mae, 3),
                         "test_RMSE": round(rmse, 3),
                         "params": npar})
        mae_arr = np.array(maes)
        mu, sd = float(mae_arr.mean()), float(mae_arr.std(ddof=0))
        summary.append({"config": name, "n_seeds": len(seeds),
                        "MAE_mean": round(mu, 3),
                        "MAE_std": round(sd, 3),
                        "RMSE_mean": round(float(np.mean(rmses)), 3),
                        "params": npar,
                        "wall_s": round(time.time() - t0, 1)})
        print(f"  {name:14s}  MAE = {mu:7.3f} ± {sd:.3f}  "
              f"(n={len(seeds)})  params={npar:,}")

    # ── auto-verdict: best learned model vs our flagship ──
    learned = [s for s in summary if s["config"] not in DETERMINISTIC]
    best = min(learned, key=lambda r: r["MAE_mean"])
    ours = next((s for s in summary if s["config"] == "tcssdpmixer"), best)
    gap = ours["MAE_mean"] - best["MAE_mean"]
    pooled = float(np.sqrt(best["MAE_std"] ** 2 + ours["MAE_std"] ** 2))
    verdict = ("TIE — gap is within run-to-run noise"
               if abs(gap) <= pooled else
               f"{'ours' if gap < 0 else best['config']} wins beyond noise")
    print("\n" + "-" * 60)
    print(f"  Best learned model : {best['config']} "
          f"= {best['MAE_mean']:.3f} ± {best['MAE_std']:.3f}")
    print(f"  Our flagship       : {ours['config']} "
          f"= {ours['MAE_mean']:.3f} ± {ours['MAE_std']:.3f}")
    print(f"  Gap = {gap:+.3f} Wh   pooled std = {pooled:.3f} Wh")
    print(f"  TAKEAWAY: {verdict}.")
    print("-" * 60)
    summary.append({"config": "VERDICT", "MAE_mean": round(gap, 3),
                    "MAE_std": round(pooled, 3), "n_seeds": verdict})

    sfx = "_weekly" if weekly else ""
    save_csv(rows,    os.path.join(RESULTS, f"significance_seeds{sfx}.csv"))
    save_csv(summary, os.path.join(RESULTS, f"significance{sfx}.csv"))
    print_table([s for s in summary if s["config"] != "VERDICT"],
                ["config", "MAE_mean", "MAE_std", "RMSE_mean", "params"])

    _plot(summary, weekly)
    print("\nDone.")


def _plot(summary, weekly):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    data = [s for s in summary if s["config"] != "VERDICT"]
    names = [s["config"] for s in data]
    mus   = [s["MAE_mean"] for s in data]
    sds   = [s.get("MAE_std", 0) for s in data]
    unit  = "Wh/h" if weekly else "Wh"
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if n == "tcssdpmixer" else "#7f8c8d" for n in names]
    ax.bar(range(len(names)), mus, yerr=sds, capsize=5, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"Test MAE ({unit})  — lower is better")
    ax.set_title(f"Multi-seed MAE ± std  ({len(SEEDS)} seeds)"
                 + (" — weekly" if weekly else ""))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = f"fig_significance{'_weekly' if weekly else ''}.png"
    fig.savefig(os.path.join(RESULTS, png), dpi=120)
    plt.close()
    print(f"  → saved {os.path.join(RESULTS, png)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="run the hourly 1-week→1-week long-horizon version")
    args = ap.parse_args()
    main(weekly=args.weekly)
